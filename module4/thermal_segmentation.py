"""
================================================================================
 CSc 8830 - Computer Vision | Module 4 Assignment
 Exact boundary of a human in a thermal image using only classical OpenCV

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   In a thermal camera image brightness = temperature. People (~37 C skin, ~30 C clothing surface) are usually warmer than their surroundings, so they show up as bright, compact, upright blobs.

   1. Grey-level + denoise ........... thermal sensors are noisy -> median filter (cv2.medianBlur)
   2. Background removal ............. morphological WHITE TOP-HAT with a large disk: img - opening(img) keeps the narrow warm objects
   3. Hysteresis threshold ........... HIGH = Otsu threshold of the top-hat
   4. Clean-up ....................... morphological open/close, hole filling
   5. Blob filtering ................. connected components
   6. Boundary snapping .............. marker-based watershed (cv2.watershed) so each outline sits on the strongest temperature edge
   7. Exact boundary ................. cv2.findContours, CHAIN_APPROX_NONE

 REQUIREMENTS:
   pip install opencv-python numpy

 HOW TO USE:
   python thermal_human_segmentation.py --image thermal.png

 OUTPUT:
   <name>_mask.png      binary mask, person = 255 (original image size)
   <name>_boundary.png  image with the exact boundary drawn in green
   <name>_steps.png     panel of the intermediate steps (for the report)
   <name>_contours.csv  x,y of every boundary pixel, per person
================================================================================
"""

import argparse
import csv
import os

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def ellipse_kernel(size):
    size = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def fill_holes(mask):
    h, w = mask.shape
    flood = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood)[1:-1, 1:-1])


def hysteresis(strength, low, high):
    weak = (strength > low).astype(np.uint8)
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(labels[strength > high])
    strong_labels = strong_labels[strong_labels > 0]
    return np.isin(labels, strong_labels).astype(np.uint8) * 255


def large_opening(gray, diameter):
    """Morphological opening with a big disk."""
    H, W = gray.shape
    f = min(1.0, 60.0 / diameter)                
    small = cv2.resize(gray, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA)
    opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, ellipse_kernel(diameter * f))
    return cv2.resize(opened, (W, H), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------
# Core algorithm
# ----------------------------------------------------------------------------

def segment_person_thermal(image, black_hot=False, rects=None, upright=True,
                           tophat_fraction=0.3, min_area_fraction=0.0008, rel_heat=0.8):
    """
    image  : thermal image (grey or 3-channel)
    returns: (mask, contours, debug)
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if black_hot:
        gray = 255 - gray
    H, W = gray.shape
    diag = np.hypot(H, W)

    den = cv2.medianBlur(gray, 5)

    disk = tophat_fraction * min(H, W)
    background = large_opening(den, disk)
    tophat = cv2.subtract(den, background)
    tophat = cv2.GaussianBlur(tophat, (0, 0), max(1.0, diag / 500))

    region = np.full((H, W), 255, np.uint8)
    if rects:
        region[:] = 0
        for (x, y, w, h) in rects:
            region[max(0, y):y + h, max(0, x):x + w] = 255

    vals = tophat[region > 0].reshape(-1, 1)
    high, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(high, 8.0)                     #never trust a near-zero Otsu 
    low = 0.5 * high
    initial = hysteresis(tophat, low, high)
    initial = cv2.bitwise_and(initial, region)

    k_small = ellipse_kernel(diag / 300)
    k_close = ellipse_kernel(diag / 120)
    min_area = max(30, min_area_fraction * H * W)

    def clean(m):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_small)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        return fill_holes(m)

    def person_like(stat):
        x, y, w, h, area = stat
        fill = area / float(w * h)                
        ok = (min_area <= area <= 0.25 * H * W    
              and fill > 0.25                     
              and y > 0)                         
        if upright:
            ok = ok and 1.0 <= h / float(w) <= 6.0 
        return ok

    kept = np.zeros((H, W), np.uint8)
    stats_log = {"rejected": 0, "split": 0}

    def accept_or_split(mask, depth):
        """Keep person-like blobs."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            comp = labels == i
            if person_like(stats[i]):
                kept[comp] = 255
            elif depth < 3 and stats[i, cv2.CC_STAT_AREA] >= 4 * min_area:
                vals = tophat[comp].reshape(-1, 1)
                t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                sub = np.where(comp & (tophat > t), 255, 0).astype(np.uint8)
                stats_log["split"] += 1
                accept_or_split(clean(sub), depth + 1)
            else:
                stats_log["rejected"] += 1

    accept_or_split(clean(initial), 0)

    n, labels = cv2.connectedComponents(kept, connectivity=8)
    if n > 2:
        means = np.array([np.percentile(tophat[labels == i], 90) for i in range(1, n)])
        for i, m in enumerate(means, start=1):
            if m < rel_heat * means.max():
                kept[labels == i] = 0
                stats_log["rejected"] += 1
    rejected = stats_log["rejected"]

    band = ellipse_kernel(diag / 90)
    sure_fg = cv2.erode(kept, ellipse_kernel(diag / 200))
    sure_bg = cv2.bitwise_not(cv2.dilate(kept, band))
    markers = np.zeros((H, W), np.int32)
    markers[sure_bg > 0] = 1

    nb, blob_labels = cv2.connectedComponents(sure_fg, connectivity=8)
    for i in range(1, nb):
        markers[blob_labels == i] = i + 1
    unknown = markers == 0
    cv2.watershed(cv2.cvtColor(den, cv2.COLOR_GRAY2BGR), markers)
    final = np.where(markers > 1, 255, 0).astype(np.uint8)
    final = fill_holes(final)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, k_small)

    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    debug = {
        "gray": gray, "background": background,
        "tophat": cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX),
        "initial": initial, "kept": kept,
        "sure_fg": sure_fg, "sure_bg": sure_bg, "unknown": unknown,
        "high": high, "low": low, "rejected": rejected, "split": stats_log["split"], "rects": rects or [],
    }
    return final, contours, debug


#Visualization

def draw_boundary(image, mask, contours, color=(0, 255, 0)):
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    sel = mask > 0
    out[sel] = (0.65 * out[sel] + 0.35 * np.array(color)).astype(np.uint8)
    cv2.drawContours(out, contours, -1, color, max(2, int(np.hypot(*mask.shape) / 450)), cv2.LINE_AA)
    return out


def labelled(img, text):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    s = max(0.5, img.shape[1] / 700)
    org = (8, int(28 * s))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), int(4 * s) + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
    return img


def steps_panel(image, final, contours, d):
    inp = cv2.cvtColor(d["gray"], cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in d["rects"]:
        cv2.rectangle(inp, (x, y), (x + w, y + h), (0, 200, 255), 2)
    trimap = np.full_like(inp, 30)
    trimap[d["sure_bg"] > 0] = (60, 60, 60)
    trimap[d["unknown"]] = (0, 200, 255)
    trimap[d["sure_fg"] > 0] = (255, 255, 255)
    tiles = [
        labelled(inp, "1. thermal input"),
        labelled(d["background"], "2. background (opening)"),
        labelled(cv2.applyColorMap(d["tophat"], cv2.COLORMAP_INFERNO), "3. top-hat = input - bg"),
        labelled(d["initial"], f"4. hysteresis {d['low']:.0f}/{d['high']:.0f}"),
        labelled(trimap, "5. filtered blobs -> markers"),
        labelled(draw_boundary(image, final, contours), "6. final boundary"),
    ]
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])


def save_contours_csv(path, contours, scale_back=1.0):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "x", "y"])
        for pid, c in enumerate(contours):
            for (px, py) in c.reshape(-1, 2):
                writer.writerow([pid, round(px * scale_back, 1), round(py * scale_back, 1)])

def main():
    ap = argparse.ArgumentParser(description="Classical (non-ML) human boundary extraction for thermal images")
    ap.add_argument("--image", required=True, help="thermal image (white-hot greyscale preferred)")
    ap.add_argument("--black_hot", action="store_true", help="use if hot objects appear DARK")
    ap.add_argument("--any_pose", action="store_true",
                    help="keep blobs of any shape (default keeps only upright, taller-than-wide blobs; "
                         "use this for people lying down / sitting)")
    ap.add_argument("--rect", action="append", default=None,
                    help="optional search box x,y,w,h (original pixels); repeatable")
    ap.add_argument("--tophat", type=float, default=0.3,
                    help="background disk diameter as a fraction of the short image side "
                         "(must be wider than a person; raise it for close-up people)")
    ap.add_argument("--rel_heat", type=float, default=0.8,
                    help="drop blobs whose peak warmth is below this fraction of the warmest blob "
                         "(lower it if a distant / cooler person is missed)")
    ap.add_argument("--max_dim", type=int, default=900)
    ap.add_argument("--out_dir", default="thermal_output")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    original = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if original is None:
        raise SystemExit(f"Could not read image: {args.image}")
    scale = min(1.0, args.max_dim / max(original.shape[:2]))
    image = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else original
    rects = [tuple(int(round(float(v) * scale)) for v in r.split(",")) for r in args.rect] if args.rect else None

    mask, contours, debug = segment_person_thermal(image, args.black_hot, rects, not args.any_pose,
                                                      args.tophat, rel_heat=args.rel_heat)

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    full_mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_mask.png"), full_mask)
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_boundary.png"), draw_boundary(image, mask, contours))
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_steps.png"), steps_panel(image, mask, contours, debug))
    save_contours_csv(os.path.join(args.out_dir, f"{name}_contours.csv"), contours, 1.0 / scale)

    print(f"Image: {args.image}  (processed at {image.shape[1]}x{image.shape[0]})")
    print(f"Hysteresis thresholds on top-hat: low={debug['low']:.1f}  high={debug['high']:.1f}")
    print(f"People / regions found: {len(contours)}  (blobs re-thresholded: {debug['split']}, "
          f"rejected by size/shape: {debug['rejected']})")
    print(f"Boundary length(s) in pixels: {[len(c) for c in contours]}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")

    if args.show:
        cv2.imshow("boundary", draw_boundary(image, mask, contours))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
