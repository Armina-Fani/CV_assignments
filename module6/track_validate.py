"""
================================================================================
 CSc 8830 - Computer Vision | Assignment 6, Part A 
 -------------------------------------------------------------------------

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   Takes two consecutive frames of a video and checks the tracking equations
   derived in the report against where the points really are.


 HOW TO USE:
   python track_validate.py --video my_video.mp4 --time 12.0     #frames at 12 s and the next one
   python track_validate.py --video my_video.mp4 --frame 300 --points 40
   # click your own points instead of automatic corners:
   python track_validate.py --video my_video.mp4 --time 12.0 --select

 OUTPUT (--out_dir, default ./tracking_output/<video>_f<frame>/):
   frame1.png, frame2.png        the two frames used
   tracking_validation.png       predicted vs actual positions on both frames
   tracking_table.csv            per point: start, LK prediction, actual, errors
   worked_example.txt            all numbers of the 2x2 system for one point
                                 (and one bilinear interpolation)
================================================================================
"""

import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIN = 21            # LK window size (odd)
LEVELS = 3          # pyramid levels (0 = no pyramid)
ITERS = 20          # Newton iterations per level
EPS = 0.01          # stop when the update is smaller than this (px)

def bilinear(img, x, y):
    """Sample a float image at real-valued positions x, y (arrays).
    Weights are the areas of the opposite sub-rectangles."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    j = np.floor(x).astype(int)
    i = np.floor(y).astype(int)
    a = x - j                     # horizontal fraction
    b = y - i                     # vertical fraction
    return ((1 - a) * (1 - b) * img[i, j] + a * (1 - b) * img[i, j + 1] +
            (1 - a) * b * img[i + 1, j] + a * b * img[i + 1, j + 1])

def gradients(img):
    """Central differences: Ix = (I[x+1]-I[x-1])/2, Iy likewise."""
    Ix = np.zeros_like(img); Iy = np.zeros_like(img)
    Ix[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    Iy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return Ix, Iy


def lk_single_level(I1, I2, Ix, Iy, x, y, u0, v0, log=None):
    """Iterative LK for one point on one pyramid level.
    Returns (u, v, ok, last_system) where last_system holds the 2x2 numbers."""
    r = WIN // 2
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    wx, wy = x + dx, y + dy                         # window in frame 1
    T = bilinear(I1, wx, wy)                        # template I1(window)
    gx = bilinear(Ix, wx, wy); gy = bilinear(Iy, wx, wy)
    A = np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                  [np.sum(gx * gy), np.sum(gy * gy)]])
    if np.linalg.eigvalsh(A)[0] < 1e-3 * WIN * WIN:  # flat / edge-only: aperture problem
        return u0, v0, False, None
    u, v = u0, v0
    system = None
    for k in range(ITERS):
        It = bilinear(I2, wx + u, wy + v) - T      # temporal difference
        b = -np.array([np.sum(gx * It), np.sum(gy * It)])
        d = np.linalg.solve(A, b)
        if system is None:
            system = {"A": A.copy(), "b": b.copy(), "d": d.copy(), "u_before": (u, v)}
        u += d[0]; v += d[1]
        if log is not None:
            log.append((k, u, v, float(np.hypot(*d))))
        if np.hypot(*d) < EPS:
            break
    return u, v, True, system


def lk_track(I1, I2, pts, levels=LEVELS, log_point=None):
    """Pyramidal LK: solve on a coarse half-size copy first, double the
    result, refine on the next finer level."""
    pyr1, pyr2 = [I1], [I2]
    for _ in range(levels):
        pyr1.append(cv2.pyrDown(pyr1[-1])); pyr2.append(cv2.pyrDown(pyr2[-1]))
    grads = [gradients(p) for p in pyr1]
    out, ok_all, systems = [], [], []
    for n, (x, y) in enumerate(pts):
        u = v = 0.0; ok = True; sys0 = None
        log = [] if n == log_point else None
        for L in range(levels, -1, -1):
            s = 2.0 ** L
            u, v, ok_l, sysL = lk_single_level(pyr1[L], pyr2[L], *grads[L], x / s, y / s, u, v,
                                               log if L == 0 else None)
            ok = ok and ok_l
            if L == 0:
                sys0 = sysL
            if L > 0:
                u, v = 2 * u, 2 * v
        out.append((x + u, y + v)); ok_all.append(ok); systems.append((sys0, log))
    return np.array(out), np.array(ok_all), systems

def ncc_locate(I1, I2, x, y, search=40, half=10):
    h, w = I1.shape
    xi, yi = int(round(x)), int(round(y))
    if not (half <= xi < w - half and half <= yi < h - half):
        return None, 0.0
    tpl = I1[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float32)
    x0, y0 = max(0, xi - half - search), max(0, yi - half - search)
    x1, y1 = min(w, xi + half + search + 1), min(h, yi + half + search + 1)
    roi = I2[y0:y1, x0:x1].astype(np.float32)
    if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
        return None, 0.0
    R = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(R)

    def sub(c_m, c_0, c_p):            # vertex of the parabola through 3 samples
        den = c_m - 2 * c_0 + c_p
        return 0.0 if abs(den) < 1e-9 else 0.5 * (c_m - c_p) / den

    ox = sub(R[my, mx - 1], R[my, mx], R[my, mx + 1]) if 0 < mx < R.shape[1] - 1 else 0.0
    oy = sub(R[my - 1, mx], R[my, mx], R[my + 1, mx]) if 0 < my < R.shape[0] - 1 else 0.0
    # patch centre in frame 2 + offset of where the patch was centred in frame 1
    return (x0 + mx + ox + half + (x - xi), y0 + my + oy + half + (y - yi)), float(score)


def read_pair(path, frame_idx, width):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok1, f1 = cap.read(); ok2, f2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        raise SystemExit("Could not read two consecutive frames there.")
    if width and f1.shape[1] > width:
        s = width / f1.shape[1]
        f1 = cv2.resize(f1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(f2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return f1, f2


def main():
    ap = argparse.ArgumentParser(description="Validate two-frame LK tracking against actual pixel locations")
    ap.add_argument("--video", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--frame", type=int, help="index of the first frame")
    g.add_argument("--time", type=float, help="time (s) of the first frame")
    ap.add_argument("--points", type=int, default=30, help="number of automatic corner points")
    ap.add_argument("--select", action="store_true", help="click points yourself (ENTER when done)")
    ap.add_argument("--width", type=int, default=640, help="processing width (match optical_flow.py)")
    ap.add_argument("--moving_only", action="store_true",
                    help="only use corners that actually move (> 1 px), e.g. on people")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()
    idx = args.frame if args.frame is not None else int(round((args.time or 0.0) * fps))
    f1, f2 = read_pair(args.video, idx, args.width)
    I1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    I2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out_dir or os.path.join("tracking_output", f"{name}_f{idx}")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "frame1.png"), f1)
    cv2.imwrite(os.path.join(out_dir, "frame2.png"), f2)

    # ---- choose points --------------------------------------------------
    if args.select:
        clicked = []
        def on_click(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicked.append((float(x), float(y)))
                cv2.circle(show, (x, y), 4, (0, 0, 255), -1); cv2.imshow(win, show)
        win = "click points on frame 1, ENTER when done"; show = f1.copy()
        cv2.imshow(win, show); cv2.setMouseCallback(win, on_click)
        while cv2.waitKey(20) not in (13, 10):
            pass
        cv2.destroyAllWindows()
        pts = np.array(clicked)
    else:
        n_try = args.points * (4 if args.moving_only else 1)
        c = cv2.goodFeaturesToTrack(I1.astype(np.float32), n_try, 0.01, 10, blockSize=7)
        pts = c.reshape(-1, 2).astype(np.float64)
        m = 30                                                  # keep away from the border
        pts = pts[(pts[:, 0] > m) & (pts[:, 0] < I1.shape[1] - m) &
                  (pts[:, 1] > m) & (pts[:, 1] < I1.shape[0] - m)]
        if args.moving_only:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(I1.astype(np.uint8), I2.astype(np.uint8),
                                                  pts.astype(np.float32).reshape(-1, 1, 2), None)
            mv = (st.ravel() == 1) & (np.hypot(*(nxt.reshape(-1, 2) - pts).T) > 1.0)
            pts = pts[mv]
        pts = pts[:args.points]
    if len(pts) == 0:
        raise SystemExit("No points to track.")

    pred, ok, systems = lk_track(I1, I2, pts, log_point=None)
    mot = np.hypot(*(pred - pts).T)
    lmin = np.array([np.linalg.eigvalsh(sy[0]["A"])[0] if sy[0] is not None else 0.0 for sy in systems])
    cand = np.where(ok & (mot >= np.median(mot[ok]) if ok.any() else ok))[0]
    ex = int(cand[np.argmax(lmin[cand])]) if len(cand) else 0
    pred, ok, systems = lk_track(I1, I2, pts, log_point=ex)

    cv_pts, cv_st, _ = cv2.calcOpticalFlowPyrLK(
        I1.astype(np.uint8), I2.astype(np.uint8), pts.astype(np.float32).reshape(-1, 1, 2), None,
        winSize=(WIN, WIN), maxLevel=LEVELS,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ITERS, EPS))
    cv_pts = cv_pts.reshape(-1, 2)
    actual, score = [], []
    for (x, y) in pts:
        a, sc = ncc_locate(I1, I2, x, y)
        actual.append(a if a is not None else (np.nan, np.nan)); score.append(sc)
    actual = np.array(actual, float); score = np.array(score)

    valid = ok & (score > 0.9) & np.isfinite(actual[:, 0])
    err = np.hypot(*(pred - actual).T)
    err_cv = np.hypot(*(cv_pts - actual).T)
    disp = np.hypot(*(actual - pts).T)

    with open(os.path.join(out_dir, "tracking_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point", "x1", "y1", "lk_x2", "lk_y2", "actual_x2", "actual_y2", "ncc_score",
                    "displacement_px", "lk_error_px", "opencv_lk_error_px", "used"])
        for i in range(len(pts)):
            w.writerow([i, *[f"{v:.2f}" for v in (*pts[i], *pred[i], *actual[i])], f"{score[i]:.3f}",
                        f"{disp[i]:.2f}", f"{err[i]:.3f}", f"{err_cv[i]:.3f}", int(valid[i])])


    fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    ax[0].imshow(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"frame {idx}: start points")
    ax[0].plot(pts[:, 0], pts[:, 1], "o", ms=5, mfc="none", mec="yellow")
    ax[1].imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)); ax[1].set_title(f"frame {idx + 1}: predicted vs actual")
    ax[1].plot(actual[valid, 0], actual[valid, 1], "o", ms=8, mfc="none", mec="lime", label="actual (NCC)")
    ax[1].plot(pred[valid, 0], pred[valid, 1], "+", ms=8, color="red", label="LK prediction (ours)")
    ax[1].legend(loc="lower right", fontsize=8)

    r = 12
    cx, cy = int(round(pts[ex, 0])), int(round(pts[ex, 1]))
    x0, y0 = max(0, cx - r), max(0, cy - r)
    crop = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)[y0:cy + r + 1, x0:cx + r + 1]
    ax[2].imshow(crop, interpolation="nearest", extent=(x0 - 0.5, x0 + crop.shape[1] - 0.5,
                                                        y0 + crop.shape[0] - 0.5, y0 - 0.5))
    ax[2].plot(*pts[ex], "o", ms=9, mfc="none", mec="yellow", mew=2, label="start (frame 1)")
    ax[2].plot(*actual[ex], "o", ms=12, mfc="none", mec="lime", mew=2, label="actual (frame 2)")
    ax[2].plot(*pred[ex], "+", ms=14, color="red", mew=2, label="LK prediction")
    ax[2].annotate("", xy=pred[ex], xytext=pts[ex], arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
    ax[2].set_title(f"zoom on point #{ex}: error {err[ex]:.2f} px")
    ax[2].legend(loc="lower right", fontsize=7)
    for a in ax[:2]:
        a.axis("off")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tracking_validation.png"), dpi=120); plt.close(fig)

    sys0, log = systems[ex]
    x, y = pts[ex]
    xq, yq = pred[ex]
    j, i = int(np.floor(xq)), int(np.floor(yq)); a, b = xq - j, yq - i
    lines = [f"Video {args.video}, frames {idx} -> {idx + 1}, image {I1.shape[1]}x{I1.shape[0]}",
             f"Window {WIN}x{WIN}, {LEVELS} pyramid levels, central-difference gradients", "",
             f"Point #{ex}: start (x, y) = ({x:.2f}, {y:.2f})"]
    if sys0 is not None:
        A, bb, d = sys0["A"], sys0["b"], sys0["d"]
        lines += [
            "Lucas-Kanade system at full resolution, first Newton step",
            f"  starting guess from coarser levels (u, v) = ({sys0['u_before'][0]:.3f}, {sys0['u_before'][1]:.3f})",
            f"  A = [[{A[0, 0]:.1f}, {A[0, 1]:.1f}], [{A[1, 0]:.1f}, {A[1, 1]:.1f}]]",
            f"  b = -[sum IxIt, sum IyIt] = [{bb[0]:.1f}, {bb[1]:.1f}]",
            f"  det A = {np.linalg.det(A):.1f},  eigenvalues = {np.linalg.eigvalsh(A).round(1).tolist()}",
            f"  update d = A^-1 b = ({d[0]:.4f}, {d[1]:.4f})",
            "  iterations at full resolution (k, u, v, |step|):"]
        lines += [f"    {k}: u={u:.4f}, v={v:.4f}, |d|={st:.4f}" for k, u, v, st in (log or [])]
    lines += [
        f"  final flow (u, v) = ({pred[ex, 0] - x:.3f}, {pred[ex, 1] - y:.3f})",
        f"  predicted position = ({xq:.3f}, {yq:.3f})",
        f"  actual position    = ({actual[ex, 0]:.3f}, {actual[ex, 1]:.3f})  (NCC score {score[ex]:.3f})",
        f"  error              = {err[ex]:.3f} px", "",
        "Bilinear interpolation of frame 2 at the predicted position:",
        f"  j = floor(x) = {j}, i = floor(y) = {i}, a = {a:.4f}, b = {b:.4f}",
        f"  I[i,j] = {I2[i, j]:.0f}, I[i,j+1] = {I2[i, j + 1]:.0f}, I[i+1,j] = {I2[i + 1, j]:.0f}, I[i+1,j+1] = {I2[i + 1, j + 1]:.0f}",
        f"  I(x,y) = (1-a)(1-b){I2[i, j]:.0f} + a(1-b){I2[i, j + 1]:.0f} + (1-a)b{I2[i + 1, j]:.0f} + ab{I2[i + 1, j + 1]:.0f}"
        f" = {bilinear(I2, np.array([xq]), np.array([yq]))[0]:.3f}", "",
        "Summary over all valid points (NCC score > 0.9, well-conditioned window):",
        f"  points used: {valid.sum()} / {len(pts)}",
        f"  mean displacement: {disp[valid].mean():.2f} px",
        f"  our LK error vs actual:    mean {err[valid].mean():.3f} px, median {np.median(err[valid]):.3f} px, max {err[valid].max():.3f} px",
        f"  OpenCV LK error vs actual: mean {err_cv[valid].mean():.3f} px, median {np.median(err_cv[valid]):.3f} px",
    ]
    with open(os.path.join(out_dir, "worked_example.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"Outputs in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
