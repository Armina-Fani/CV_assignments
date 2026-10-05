"""
================================================================================
 CSc 8830 - Computer Vision | Assignment 6, Part B
 -------------------------------------------------------------------------

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   From 4 photos of a flat object taken from different positions with the camera calibrated in Module 2, this script
   recovers (1) where each camera was and how it was turned, and (2) the 3-D
   positions of points on the object, so the object's BOUNDARY (outline) and
   its real size can be estimated.

   Steps (all numbers are written to sfm_workout.txt for the report):
     1. Camera parameters K and lens distortion from Module 2
        (module2/camera_calibration.npz); K is rescaled if your photos have a
        different resolution than the calibration photos.
     2. Feature matching: SIFT features matched between view 1 and each other
        view (ratio test).
     3. Homography between view 1 and view k, fitted with RANSAC. For points
        on one plane, x_k ~ H_1k x_1 with  H_1k = K (R_k + t_k n^T / d) K^-1.
     4. Camera motion: decompose each H_1k into rotation R_k, translation
        t_k / d and plane normal n. Of the 4 mathematical solutions, keep the
        ones that put the points in front of both cameras, then pick the one
        whose plane normal agrees across all three pairs.
     5. Scale: with the measured distance from camera 1 to the object
        (or a known object width) the translations become millimetres.
     6. Structure: every point seen in all 4 views is triangulated with the
        linear DLT method from the 4 projection matrices P_k = K [R_k | t_k].
     7. Boundary: the object's corners (clicked in view 1) are transferred to
        the other views, triangulated, and expressed in 2-D coordinates on
        the fitted plane -> outline, side lengths, corner angles, area.
        A top-down (rectified) picture of the object is also produced.


 HOW TO USE:
   python sfm_planar.py --images v1.jpg v2.jpg v3.jpg v4.jpg \\
          --calib ../module2/camera_calibration.npz \\
          --distance_mm 450 --select --true_size 152,229

   --select              click the object's corners in view 1 (in order around
                         the outline, ENTER when done)
   --corners "x,y;..."   give the corners as pixel coordinates instead
   --distance_mm D       straight-line distance from the phone (photo 1) to the
                         centre of the object, in mm (sets the metric scale), or
   --known_width_mm W    true length of the first clicked side (alternative scale)
   --true_size W,H       true width,height in mm, only used to report errors

 OUTPUT (--out_dir, default ./sfm_output):
   sfm_workout.txt       every matrix and intermediate number, step by step
   matches_v1_vk.jpg     feature matches (RANSAC inliers) for each pair
   reconstruction_3d.png cameras + reconstructed points + boundary in 3-D
   boundary_2d.png       recovered outline in mm, with side lengths
   rectified_object.jpg  top-down view of the object recreated from view 1
   cameras.csv           camera positions / orientations
   points_3d.csv         reconstructed 3-D points (mm, camera-1 frame)
================================================================================
"""

import argparse
import csv
import itertools
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FEATURE_MAX_DIM = 1600       # features are detected on a downscaled copy


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def fmt(M, p=4):
    return np.array2string(np.asarray(M, float), precision=p, suppress_small=True,
                           max_line_width=120)


def euler_deg(R):
    """Rotation as (rx, ry, rz) in degrees, R = Rz Ry Rx."""
    sy = np.hypot(R[0, 0], R[1, 0])
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])])


def load_calibration(path, image_size):
    d = np.load(path)
    K = d["camera_matrix"].astype(np.float64)
    dist = d["dist_coeffs"].astype(np.float64)
    cw, ch = [int(v) for v in d["image_size"]]
    w, h = image_size
    note = ""
    if (w, h) != (cw, ch):
        if abs(w / h - cw / ch) > 0.01:
            note = (f"WARNING: photos are {w}x{h} but the calibration was {cw}x{ch} with a different "
                    "aspect ratio (rotated or cropped?). Results will be unreliable.")
        s = w / cw
        K = K.copy(); K[0, :] *= s; K[1, :] *= h / ch
        note = note or f"K rescaled from {cw}x{ch} to {w}x{h} (factor {s:.4f})."
    return K, dist, note


def undistort_px(pts, K, dist):
    """Remove lens distortion, returning pixel coordinates of an ideal camera K."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def match_features(img1, img2, sift):
    def prep(img):
        s = min(1.0, FEATURE_MAX_DIM / max(img.shape[:2]))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g, s
    g1, s1 = prep(img1); g2, s2 = prep(img2)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    p1 = np.array([k1[m.queryIdx].pt for m in good]) / s1
    p2 = np.array([k2[m.trainIdx].pt for m in good]) / s2
    return p1, p2


def plane_depth_ok(R, t, n, m1):
    """Cheirality: plane points must be in front of both cameras.
    m1 = normalised rays of view 1 (3xN). With plane n.X = d (d = 1 here):
    X1 = m1 / (n.m1), X2 = R X1 + t."""
    nm = n.ravel() @ m1
    if np.any(nm <= 0):
        return False
    X1 = m1 / nm
    X2 = R @ X1 + t.reshape(3, 1)
    return bool(np.all(X2[2] > 0))


def triangulate(Ps, xs):
    """Linear (DLT) triangulation of one point seen in several views.
    Each view adds two rows:  x (p3.X) - (p1.X) = 0,  y (p3.X) - (p2.X) = 0."""
    A = []
    for P, (x, y) in zip(Ps, xs):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3], A


def project(P, X):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Planar structure from motion from 4 views")
    ap.add_argument("--images", nargs=4, required=True, help="the 4 photos; the first is the reference view")
    ap.add_argument("--calib", default=os.path.join(here, "..", "module2", "camera_calibration.npz"))
    ap.add_argument("--select", action="store_true", help="click the object's corners in view 1")
    ap.add_argument("--corners", help='corner pixels in view 1: "x1,y1;x2,y2;..." (in order around the outline)')
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--distance_mm", type=float, help="straight-line distance camera 1 -> object centre (mm)")
    sc.add_argument("--known_width_mm", type=float, help="true length of the first side (corner 1 -> corner 2)")
    ap.add_argument("--true_size", help="true width,height in mm (for the error report only)")
    ap.add_argument("--out_dir", default="sfm_output")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    W = []                                                  # workout lines

    imgs = [cv2.imread(p) for p in args.images]
    for p, im in zip(args.images, imgs):
        if im is None:
            raise SystemExit(f"Could not read {p}")
    h, w = imgs[0].shape[:2]
    K, dist, note = load_calibration(args.calib, (w, h))
    Kinv = np.linalg.inv(K)

    W += ["STEP 1 - Camera parameters (Module 2 calibration)",
          f"  image size: {w} x {h} px", f"  {note}" if note else "",
          f"  K =\n{fmt(K, 2)}",
          f"  fx = {K[0, 0]:.2f}, fy = {K[1, 1]:.2f}, cx = {K[0, 2]:.2f}, cy = {K[1, 2]:.2f} (px)",
          f"  horizontal field of view = {2 * np.degrees(np.arctan(w / 2 / K[0, 0])):.1f} deg",
          f"  distortion (k1,k2,p1,p2,k3) = {fmt(dist.ravel(), 5)}", ""]

    if args.select:
        disp_s = min(1.0, 1000 / max(h, w))
        show = cv2.resize(imgs[0], None, fx=disp_s, fy=disp_s)
        clicked = []
        def on_click(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicked.append((x / disp_s, y / disp_s))
                cv2.circle(show, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(show, str(len(clicked)), (x + 6, y - 6), 0, 0.7, (0, 0, 255), 2)
                cv2.imshow(win, show)
        win = "View 1: click the object's corners in order, ENTER when done"
        cv2.imshow(win, show); cv2.setMouseCallback(win, on_click)
        while cv2.waitKey(20) not in (13, 10):
            pass
        cv2.destroyAllWindows()
        corners_px = np.array(clicked, float)
    elif args.corners:
        corners_px = np.array([[float(v) for v in c.split(",")] for c in args.corners.split(";")])
    else:
        raise SystemExit("Give the object's corners with --select or --corners.")
    if len(corners_px) < 3:
        raise SystemExit("Need at least 3 corners.")

    sift = cv2.SIFT_create(nfeatures=6000)
    Hs, pairs = [], []
    W += ["STEP 2-3 - Feature matching and homographies  x_k ~ H_1k x_1  (undistorted pixels)"]
    for k in range(1, 4):
        p1, pk = match_features(imgs[0], imgs[k], sift)
        u1, uk = undistort_px(p1, K, dist), undistort_px(pk, K, dist)
        H, inl = cv2.findHomography(u1, uk, cv2.RANSAC, 3.0 * w / 1000)
        if H is None:
            raise SystemExit(f"Could not match view 1 with view {k + 1}.")
        inl = inl.ravel().astype(bool)
        err = np.hypot(*(cv2.perspectiveTransform(u1[inl].reshape(-1, 1, 2), H).reshape(-1, 2) - uk[inl]).T)
        Hs.append(H / H[2, 2]); pairs.append((u1[inl], uk[inl], p1[inl], pk[inl]))
        W += [f"  view 1 -> view {k + 1}: {len(p1)} matches, {inl.sum()} RANSAC inliers, "
              f"mean transfer error {err.mean():.2f} px",
              f"  H_1{k + 1} =\n{fmt(Hs[-1], 6)}"]
        vis = cv2.drawMatches(imgs[0], [cv2.KeyPoint(*p, 1) for p in p1[inl][::4]], imgs[k],
                              [cv2.KeyPoint(*p, 1) for p in pk[inl][::4]],
                              [cv2.DMatch(i, i, 0) for i in range(len(p1[inl][::4]))], None,
                              matchColor=(0, 255, 0), flags=2)
        s = 1600 / vis.shape[1]
        cv2.imwrite(os.path.join(args.out_dir, f"matches_v1_v{k + 1}.jpg"), cv2.resize(vis, None, fx=s, fy=s))
    W.append("")

    W += ["STEP 4 - Decompose H_1k = K (R + t n^T / d) K^-1 into rotation, translation/d, plane normal"]
    cands = []
    for k, H in enumerate(Hs):
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
        m1 = Kinv @ np.vstack([pairs[k][0].T, np.ones(len(pairs[k][0]))])
        ok = [(R, t, n) for R, t, n in zip(Rs, ts, ns) if plane_depth_ok(R, t, n, m1)]
        W.append(f"  view {k + 2}: {len(Rs)} solutions, {len(ok)} put the points in front of both cameras")
        if not ok:
            raise SystemExit(f"No physically valid decomposition for view {k + 2}.")
        cands.append(ok)
    # pick the combination whose plane normals (all in camera-1 frame) agree best
    best = min(itertools.product(*cands),
               key=lambda c: sum(1 - float(a[2].ravel() @ b[2].ravel()) for a, b in itertools.combinations(c, 2)))
    n_mean = np.mean([c[2].ravel() for c in best], axis=0); n_mean /= np.linalg.norm(n_mean)
    for k, (R, t, n) in enumerate(best):
        W += [f"  chosen solution for view {k + 2}:",
              f"    R_1{k + 2} =\n{fmt(R, 5)}",
              f"    rotation angles (x, y, z) = {fmt(euler_deg(R), 2)} deg, "
              f"total {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.2f} deg",
              f"    t / d = {fmt(t.ravel(), 5)}",
              f"    n     = {fmt(n.ravel(), 5)}"]
    spread = max(np.degrees(np.arccos(np.clip(c[2].ravel() @ n_mean, -1, 1))) for c in best)
    W += [f"  plane normals agree to within {spread:.2f} deg; mean n = {fmt(n_mean, 5)}", ""]

    corners_u = undistort_px(corners_px, K, dist)
    if args.distance_mm:
        # straight-line distance D from camera 1 to the centre of the object:
        # with d = 1 the centre is X = m / (n.m); scale so that |X| = D
        mc = Kinv @ np.append(corners_u.mean(axis=0), 1.0)
        d = args.distance_mm / np.linalg.norm(mc / (n_mean @ mc))
        how = (f"measured distance camera 1 -> object centre D = {args.distance_mm:.1f} mm "
               f"-> perpendicular distance to the plane d = {d:.1f} mm")
    elif args.known_width_mm:
        rays = Kinv @ np.vstack([corners_u[:2].T, np.ones(2)])
        X = rays / (n_mean @ rays)                          # on plane with d = 1
        d = args.known_width_mm / np.linalg.norm(X[:, 0] - X[:, 1])
        how = f"known width {args.known_width_mm:.1f} mm of side 1-2 -> d = {d:.1f} mm"
    else:
        d = 1.0
        how = "no scale given: units are 'distance from camera 1 to the plane' (d = 1)"
    unit = "mm" if d != 1.0 else "d"
    Ps = [K @ np.hstack([np.eye(3), np.zeros((3, 1))])]
    cams = [("view 1", np.eye(3), np.zeros(3))]
    W += ["STEP 5 - Metric scale and camera positions (camera-1 coordinate frame)", f"  {how}"]
    for k, (R, t, n) in enumerate(best):
        tk = t.ravel() * d
        Ps.append(K @ np.hstack([R, tk.reshape(3, 1)]))
        C = -R.T @ tk                                       # camera centre in the camera-1 frame
        cams.append((f"view {k + 2}", R, C))
    W.append(f"  {'view':7s} {'centre X':>10s} {'Y':>10s} {'Z':>10s}   dist. to cam 1   angle of view axis to plane normal")
    for name, R, C in cams:
        axis = R.T @ np.array([0, 0, 1.0])                  # viewing direction in camera-1 frame
        ang = np.degrees(np.arccos(abs(axis @ n_mean)))
        W.append(f"  {name:7s} {C[0]:10.1f} {C[1]:10.1f} {C[2]:10.1f}   {np.linalg.norm(C):10.1f} {unit}   {ang:8.1f} deg")
    for k, P in enumerate(Ps):
        W.append(f"  P_{k + 1} = K [R_{k + 1} | t_{k + 1}] =\n{fmt(P, 2)}")
    W.append("")
    with open(os.path.join(args.out_dir, "cameras.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["view", "X", "Y", "Z", "rot_x_deg", "rot_y_deg", "rot_z_deg", "unit"])
        for name, R, C in cams:
            wr.writerow([name, *[f"{v:.2f}" for v in C], *[f"{v:.2f}" for v in euler_deg(R)], unit])

    keys = [{tuple(np.round(p, 1)): i for i, p in enumerate(pr[0])} for pr in pairs]
    common = [p for p in keys[0] if p in keys[1] and p in keys[2]]
    feat3d, reproj = [], []
    for p in common:
        xs = [pairs[0][0][keys[0][p]]] + [pairs[k][1][keys[k][p]] for k in range(3)]
        X, _ = triangulate(Ps, xs)
        feat3d.append(X)
        reproj += [np.hypot(*(project(P, X) - x)) for P, x in zip(Ps, xs)]
    feat3d = np.array(feat3d)
    W += ["STEP 6 - Triangulation (linear DLT, 4 views)",
          f"  feature points seen in all 4 views: {len(feat3d)}",
          f"  reprojection error: mean {np.mean(reproj):.2f} px, 95th pct {np.percentile(reproj, 95):.2f} px"
          if reproj else "  (no common points)"]
    if len(feat3d) >= 3:
        cen = feat3d.mean(axis=0)
        _, sv, Vt = np.linalg.svd(feat3d - cen)
        n_fit = Vt[-1] * np.sign(Vt[-1] @ n_mean)
        flat = np.abs((feat3d - cen) @ n_fit)
        W.append(f"  plane fitted to them: normal {fmt(n_fit, 4)}, RMS distance from plane "
                 f"{np.sqrt(np.mean(flat ** 2)):.2f} {unit}  (flatness check)")
    W.append("")

    corner_views = [corners_u] + [cv2.perspectiveTransform(corners_u.reshape(-1, 1, 2), H).reshape(-1, 2) for H in Hs]
    corners3d = []
    W.append("STEP 7 - Object boundary (corners clicked in view 1, transferred with H_1k, triangulated)")
    for i in range(len(corners_u)):
        xs = [cv[i] for cv in corner_views]
        X, A = triangulate(Ps, xs)
        corners3d.append(X)
        if i == 0:
            W += [f"  worked example, corner 1:",
                  "    image positions (undistorted px): " + ", ".join(f"v{k + 1}=({x:.1f}, {y:.1f})" for k, (x, y) in enumerate(xs)),
                  "    DLT system A X = 0 (rows x*p3 - p1, y*p3 - p2 for each view):",
                  f"{fmt(A, 1)}",
                  f"    smallest singular vector -> X = {fmt(X, 2)} {unit}",
                  "    reprojection: " + ", ".join(f"v{k + 1} {np.hypot(*(project(P, X) - x)):.2f}px" for k, (P, x) in enumerate(zip(Ps, xs)))]
    corners3d = np.array(corners3d)
    W.append("  3-D corners (camera-1 frame):")
    W += [f"    corner {i + 1}: {fmt(X, 1)} {unit}" for i, X in enumerate(corners3d)]

    e1 = corners3d[1] - corners3d[0]; e1 -= (e1 @ n_mean) * n_mean; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_mean, e1)
    if np.mean((corners3d - corners3d[0]) @ e2) < 0:
        e2 = -e2
    uv = np.array([[(X - corners3d[0]) @ e1, (X - corners3d[0]) @ e2] for X in corners3d])
    sides = [np.linalg.norm(uv[(i + 1) % len(uv)] - uv[i]) for i in range(len(uv))]
    angles = []
    for i in range(len(uv)):
        a, b = uv[i - 1] - uv[i], uv[(i + 1) % len(uv)] - uv[i]
        angles.append(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    area = 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1)))
    W += ["  boundary in plane coordinates (origin = corner 1, x along side 1-2):"]
    W += [f"    corner {i + 1}: ({u:.1f}, {v:.1f}) {unit}" for i, (u, v) in enumerate(uv)]
    W += [f"  side lengths: " + ", ".join(f"{i + 1}-{(i + 1) % len(uv) + 1}: {s:.1f}" for i, s in enumerate(sides)) + f" {unit}",
          f"  corner angles: " + ", ".join(f"{a:.1f}" for a in angles) + " deg",
          f"  area: {area:.0f} {unit}^2"]
    if args.true_size and len(uv) == 4:
        tw, th = [float(v) for v in args.true_size.split(",")]
        est_w, est_h = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
        W += [f"  true size {tw:.1f} x {th:.1f} mm -> estimated {est_w:.1f} x {est_h:.1f} mm "
              f"(errors {100 * (est_w - tw) / tw:+.1f}%, {100 * (est_h - th) / th:+.1f}%)"]
    W.append("")

    with open(os.path.join(args.out_dir, "points_3d.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["kind", "X", "Y", "Z", "unit"])
        wr.writerows([["corner", *[f"{v:.2f}" for v in X], unit] for X in corners3d])
        wr.writerows([["feature", *[f"{v:.2f}" for v in X], unit] for X in feat3d])

    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    if len(feat3d):
        sub = feat3d[:: max(1, len(feat3d) // 1500)]
        ax.scatter(sub[:, 0], sub[:, 2], -sub[:, 1], s=1, c="grey", alpha=0.4, label="feature points")
    loop = np.vstack([corners3d, corners3d[:1]])
    ax.plot(loop[:, 0], loop[:, 2], -loop[:, 1], "r-", lw=2, label="object boundary")
    size = 0.15 * np.linalg.norm(corners3d.mean(axis=0))
    for (name, R, C), col in zip(cams, ["C0", "C1", "C2", "C3"]):
        axis = R.T @ np.array([0, 0, 1.0])
        ax.scatter(C[0], C[2], -C[1], color=col, s=40)
        ax.quiver(C[0], C[2], -C[1], axis[0], axis[2], -axis[1], length=size, color=col)
        ax.text(C[0], C[2], -C[1], " " + name, color=col)
    ax.set_xlabel(f"X ({unit})"); ax.set_ylabel(f"Z / depth ({unit})"); ax.set_zlabel(f"-Y ({unit})")
    ax.set_title("Recovered cameras and object"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "reconstruction_3d.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    lp = np.vstack([uv, uv[:1]])
    ax.plot(lp[:, 0], lp[:, 1], "r-o")
    for i in range(len(uv)):
        m = (uv[i] + uv[(i + 1) % len(uv)]) / 2
        ax.text(*m, f"{sides[i]:.1f}", color="b", ha="center")
        ax.text(*uv[i], f" {i + 1}", color="k")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.3)
    ax.set_xlabel(unit); ax.set_ylabel(unit); ax.set_title("Recovered boundary on the object plane")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "boundary_2d.png"), dpi=130); plt.close(fig)

    if unit == "mm":
        ppm = 2.0                                           # output pixels per mm
        lo, hi = uv.min(axis=0) - 10, uv.max(axis=0) + 10
        out_w, out_h = int((hi[0] - lo[0]) * ppm), int((hi[1] - lo[1]) * ppm)
        if 0 < out_w < 6000 and 0 < out_h < 6000:
            grid_uv = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            P3 = corners3d[0] + grid_uv[:, :1] * e1 + grid_uv[:, 1:] * e2
            src = cv2.projectPoints(P3, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
            dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float32)
            Hr = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
            cv2.imwrite(os.path.join(args.out_dir, "rectified_object.jpg"),
                        cv2.warpPerspective(imgs[0], Hr, (out_w, out_h)))

    with open(os.path.join(args.out_dir, "sfm_workout.txt"), "w") as f:
        f.write("\n".join(l for l in W if l is not None) + "\n")
    print("\n".join(W[-12:]))
    print(f"Full workout: {os.path.abspath(os.path.join(args.out_dir, 'sfm_workout.txt'))}")


if __name__ == "__main__":
    main()
