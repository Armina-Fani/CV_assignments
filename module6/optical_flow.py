"""
================================================================================
 CSc 8830 - Computer Vision | Assignment 6
 -------------------------------------------------------------------------

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   Computes dense optical flow between every pair of consecutive
   frames of a 30 s clip and writes it out as a video, side by side with the
   original. It also extracts the information that optical flow carries:

     - how fast things move -> flow magnitude (pixels / frame)
     - which way they move -> flow direction (colour wheel, arrows)
     - camera motion -> a global pan / zoom / rotation model
     - independently moving objects -> pixels whose flow differs from what the camera motion predicts 
     - when motion happens -> motion-over-time plot

 REQUIREMENTS:

 HOW TO USE:
   python optical_flow.py --video my_video.mp4                 #takes first 30 s
   python optical_flow.py --video my_video.mp4 --start 12      #30 s from 0:12
   python optical_flow.py --video my_video.mp4 --duration 45 --width 960

 OUTPUT (--out_dir, default ./flow_output/<video name>/):
   flow_video.mp4         original | colour-coded flow, with arrows + boxes
   flow_video_h264.mp4    same, H.264 for the web page (if ffmpeg is installed)
   flow_video.webm        same, WebM for the web page (if ffmpeg is NOT installed)
   motion_over_time.png   mean speed, moving area and camera motion vs time
   snapshots.png          4 frames: original, flow colours, moving objects
   colour_wheel.png       legend: colour = direction, brightness = speed
   flow_stats.csv         per-frame numbers behind the plot
   summary.txt            plain-language summary of what the flow shows
================================================================================
"""

import argparse
import csv
import os
import shutil
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MOVING_THRESHOLD = 1.0      #px/frame: slower than this counts as "not moving"
MIN_OBJECT_AREA = 0.002     #fraction of the frame a moving blob must cover

def flow_to_colour(flow, max_mag=None):
    """Standard optical-flow colour coding: hue = direction, value = speed."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    if max_mag is None:
        max_mag = max(np.percentile(mag, 99), 1e-3)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)                 # OpenCV hue is 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def colour_wheel(size=200):
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    flow = np.dstack([x, y]).astype(np.float32)
    img = flow_to_colour(flow, max_mag=1.0)
    img[np.hypot(x, y) > 1] = 255
    for txt, pos in [("right", (size - 48, size // 2)), ("left", (4, size // 2)),
                     ("down", (size // 2 - 20, size - 6)), ("up", (size // 2 - 10, 14))]:
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_arrows(img, flow, step=16, scale=3.0):
    """Arrow every `step` pixels, length = scale x flow (so motion is visible)."""
    out = img.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]
            if fx * fx + fy * fy < MOVING_THRESHOLD ** 2:
                continue
            cv2.arrowedLine(out, (x, y), (int(x + scale * fx), int(y + scale * fy)),
                            (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)
    return out


def direction_name(dx, dy):
    ang = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360     # image y points down
    names = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return names[int(((ang + 22.5) % 360) // 45)]


# ----------------------------------------------------------------------------
# Analysis of one flow field
# ----------------------------------------------------------------------------

def estimate_camera_motion(prev_gray, gray):
    """Global (camera) motion between two frames as a similarity transform
    x' = s R(theta) x + t, fitted with RANSAC to tracked corner features.
    RANSAC ignores features on independently moving objects, so what is left
    is the motion of the background, i.e. of the camera."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 10:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 10:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    M, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                         ransacReprojThreshold=1.5)
    if M is None or inl.sum() < 8:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    return M


def reliable_pixels(gray):
    """Flow can only be measured where the image has texture in two directions
    (otherwise the aperture problem makes it ambiguous, and flat regions read
    as 'not moving'). Keep pixels whose structure-tensor minimum eigenvalue is
    reasonably large."""
    lam = cv2.cornerMinEigenVal(gray, blockSize=7, ksize=3)
    return lam > max(1e-6, 0.05 * np.percentile(lam, 95))


def analyse_flow(flow, cam_M, reliable):
    """Returns per-frame statistics and the moving-object boxes."""
    h, w = flow.shape[:2]
    mag = np.hypot(flow[..., 0], flow[..., 1])

    # Flow the camera motion alone would produce at every pixel.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cam_flow = np.dstack([cam_M[0, 0] * xs + cam_M[0, 1] * ys + cam_M[0, 2] - xs,
                          cam_M[1, 0] * xs + cam_M[1, 1] * ys + cam_M[1, 2] - ys])
    zoom = float(np.hypot(cam_M[0, 0], cam_M[1, 0]))
    rot = float(np.degrees(np.arctan2(cam_M[1, 0], cam_M[0, 0])))
    centre_shift = cam_flow[h // 2, w // 2]

    # What is left after removing camera motion = independently moving things.
    rel = flow - cam_flow
    rel_mag = np.hypot(rel[..., 0], rel[..., 1])
    # Flow errors grow with speed, so the threshold grows with camera speed.
    thr = max(MOVING_THRESHOLD, 0.3 * float(np.hypot(*centre_shift)))
    moving = ((rel_mag > thr) & reliable).astype(np.uint8) * 255
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    moving = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    objects = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_OBJECT_AREA * h * w:
            continue
        v = np.median(rel[labels == i], axis=0)
        objects.append({"box": (x, y, bw, bh), "dx": float(v[0]), "dy": float(v[1]),
                        "speed": float(np.hypot(*v))})
    moving_px = mag > MOVING_THRESHOLD
    return {
        "mean_speed": float(mag.mean()),
        "moving_speed": float(mag[moving_px].mean()) if moving_px.any() else 0.0,
        "moving_fraction": float(moving_px.mean()),
        "camera_dx": float(centre_shift[0]), "camera_dy": float(centre_shift[1]),
        "zoom": zoom, "rotation_deg": rot,
        "objects": objects,
    }


def annotate(img, stats, fps):
    out = img.copy()
    for o in stats["objects"]:
        x, y, w, h = o["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
        label = f"{direction_name(o['dx'], o['dy'])} {o['speed'] * fps:.0f}px/s"
        cv2.putText(out, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    cam = np.hypot(stats["camera_dx"], stats["camera_dy"])
    parts = []
    if cam > 0.3:
        parts.append(f"pans {direction_name(stats['camera_dx'], stats['camera_dy'])} {cam * fps:.0f}px/s")
    if abs(stats["zoom"] - 1) > 0.002:
        parts.append("zooms in" if stats["zoom"] > 1 else "zooms out")
    if abs(stats["rotation_deg"]) > 0.1:
        parts.append(f"rotates {stats['rotation_deg'] * fps:+.0f} deg/s")
    txt = "camera: " + (", ".join(parts) if parts else "still")
    cv2.putText(out, txt, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# Video writing (codec support differs between OpenCV builds / operating systems)
# ----------------------------------------------------------------------------

def find_ffmpeg():
    """System ffmpeg, or the copy bundled with `pip install imageio-ffmpeg`."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class FFmpegPipeWriter:
    """Streams frames straight into ffmpeg -> H.264 .mp4 (plays in browsers).
    Used whenever ffmpeg is available, so the result never depends on which
    codecs this OpenCV build happens to support."""

    def __init__(self, exe, path, fps, frame_size):
        w, h = frame_size
        self.proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "26", path], stdin=subprocess.PIPE)

    def isOpened(self):
        return self.proc.poll() is None

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


def open_video_writer(out_dir, fps, frame_size):
    """ffmpeg if available, else try OpenCV codecs until one works.
    Returns (writer, path, browser_playable).
    avc1 = H.264 (plays in browsers; works on macOS), VP80 = WebM (plays in
    browsers), mp4v / MJPG = fall-backs that need converting for the web."""
    candidates = [("avc1", "flow_video_h264.mp4", True),
                  ("VP80", "flow_video.webm", True),
                  ("mp4v", "flow_video.mp4", False),
                  ("MJPG", "flow_video.avi", False)]
    exe = find_ffmpeg()
    if exe:
        path = os.path.join(out_dir, "flow_video_h264.mp4")
        try:
            w = FFmpegPipeWriter(exe, path, fps, frame_size)
            if w.isOpened():
                return w, path, True
        except OSError:
            pass
    for fourcc, fname, web in candidates:
        path = os.path.join(out_dir, fname)
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
        if w.isOpened():
            return w, path, web
        w.release()
        if os.path.exists(path) and os.path.getsize(path) == 0:
            os.remove(path)
    return None, None, False


def make_web_copy(src, out_dir):
    """Convert to H.264 with ffmpeg so the web page can play it."""
    exe = find_ffmpeg()
    if not exe:
        return None
    dst = os.path.join(out_dir, "flow_video_h264.mp4")
    r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", dst], check=False)
    return dst if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0 else None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Dense optical flow video + motion analysis")
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to process (assignment: >= 30)")
    ap.add_argument("--width", type=int, default=640, help="processing width in pixels")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(args.start * fps))
    n_frames = int(round(args.duration * fps))
    if total and first + n_frames > total:
        print(f"Note: video has only {(total - first) / fps:.1f} s after --start; using all of it.")
        n_frames = total - first
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out_dir or os.path.join("flow_output", name)
    os.makedirs(out_dir, exist_ok=True)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")
    scale = args.width / frame.shape[1]
    size = (args.width, int(round(frame.shape[0] * scale)) // 2 * 2)
    prev = cv2.resize(frame, size)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    frame_size = (size[0] * 2, size[1])
    writer, video_path, web_ok = open_video_writer(out_dir, fps, frame_size)
    if writer is None:
        print("WARNING: this OpenCV build could not open any video writer; "
              "statistics and figures will still be saved.")

    rows, snaps = [], []
    snap_at = set(np.linspace(1, max(1, n_frames - 2), 4).astype(int))
    for i in range(1, n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        cur = cv2.resize(frame, size)
        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # Farneback: fits a quadratic polynomial to each neighbourhood in both
        # frames and solves for the shift between them, on a 3-level pyramid.
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3, poly_n=5,
                                            poly_sigma=1.2, flags=0)
        stats = analyse_flow(flow, estimate_camera_motion(prev_gray, gray), reliable_pixels(prev_gray))
        left = annotate(draw_arrows(prev, flow), stats, fps)
        right = flow_to_colour(flow)
        if writer is not None:
            writer.write(np.hstack([left, right]))

        rows.append([i, i / fps, stats["mean_speed"], stats["moving_fraction"],
                     stats["camera_dx"], stats["camera_dy"], len(stats["objects"]),
                     stats["moving_speed"], stats["zoom"], stats["rotation_deg"]])
        if i in snap_at:
            snaps.append((i / fps, prev.copy(), right, left))
        prev, prev_gray = cur, gray
    cap.release()
    video_msgs = []
    if writer is not None:
        writer.release()
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            video_msgs.append(f"Flow video: {video_path}")
            if not web_ok:                                   # needs converting for browsers
                web = make_web_copy(video_path, out_dir)
                if web:
                    video_msgs.append(f"Web-playable copy: {web}")
                else:
                    video_msgs.append("Note: this video will not play on the web page. "
                                      "Run  pip install imageio-ffmpeg  and re-run to get an H.264 copy.")
        else:
            video_msgs.append("WARNING: the video file came out empty on this system. "
                              "Run  pip install imageio-ffmpeg  and re-run.")

    # --- statistics files ------------------------------------------------
    with open(os.path.join(out_dir, "flow_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "mean_speed_px_per_frame", "moving_fraction",
                    "camera_dx_px_per_frame", "camera_dy_px_per_frame", "moving_objects",
                    "speed_of_moving_pixels", "camera_zoom_per_frame", "camera_rotation_deg_per_frame"])
        w.writerows([[r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.3f}",
                      f"{r[5]:.3f}", r[6], f"{r[7]:.3f}", f"{r[8]:.5f}", f"{r[9]:.4f}"] for r in rows])
    R = np.array(rows, float)

    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(R[:, 1], R[:, 7] * fps, color="#2a6fdb"); ax[0].set_ylabel("speed of moving\npixels (px/s)")
    ax[1].plot(R[:, 1], 100 * R[:, 3], color="#d9822b"); ax[1].set_ylabel("moving area\n(% of frame)")
    ax[2].plot(R[:, 1], R[:, 4] * fps, label="horizontal (+ = right)", color="#2a9d5c")
    ax[2].plot(R[:, 1], R[:, 5] * fps, label="vertical (+ = down)", color="#8e44ad")
    ax[2].set_ylabel("camera pan\n(px/s)"); ax[2].set_xlabel("time in clip (s)"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Optical-flow evidence: {name}")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "motion_over_time.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(len(snaps), 3, figsize=(12, 2.9 * len(snaps)))
    ax = np.atleast_2d(ax)
    for r, (t, orig, col, ann) in enumerate(snaps):
        for c, (im, title) in enumerate([(orig, f"frame at {t:.1f} s"), (col, "flow (colour = direction)"),
                                         (ann, "arrows + moving objects")]):
            ax[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax[r, c].set_title(title, fontsize=9)
            ax[r, c].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "snapshots.png"), dpi=110); plt.close(fig)
    cv2.imwrite(os.path.join(out_dir, "colour_wheel.png"), colour_wheel())

    # --- plain-language summary ------------------------------------------
    cam_speed = np.hypot(R[:, 4], R[:, 5]) * fps
    busiest = R[np.argmax(R[:, 2]), 1]
    lines = [
        f"Video: {args.video}",
        f"Clip: {len(R) + 1} frames, {len(R) / fps:.1f} s at {fps:.1f} fps, processed at {size[0]}x{size[1]}",
        f"Average speed of the moving pixels: {R[:, 7].mean() * fps:.1f} px/s (busiest moment at {busiest:.1f} s)",
        f"Average share of the frame that moves: {100 * R[:, 3].mean():.1f}%",
        f"Camera pan speed: median {np.median(cam_speed):.1f} px/s "
        f"({'camera essentially still' if np.median(cam_speed) < 3 else 'camera is moving'}); "
        f"zoom changes by up to {100 * np.max(np.abs(R[:, 8] - 1)) * fps:.1f}%/s, "
        f"rotation up to {np.max(np.abs(R[:, 9])) * fps:.1f} deg/s",
        f"Frames with at least one independently moving object: {100 * (R[:, 6] > 0).mean():.0f}%",
        f"Average number of moving objects per frame: {R[:, 6].mean():.1f}",
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n".join(video_msgs))
    print(f"Outputs in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
