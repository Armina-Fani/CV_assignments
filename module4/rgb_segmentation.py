"""
================================================================================
 CSc 8830 - Computer Vision | Module 4 Assignment
 Exact boundary of a human in an RGB color image using only classical OpenCV

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   Finds the outline of the person in an rgb photograph using only classical image processing. 

 PIPELINE (per person box):
   1. Edge-preserving smoothing ....... cv2.bilateralFilter
   2. Seed regions from the box ....... person seed  = ellipse at the box centre
                                        background   = the 4 box corners plus a ring just outside the box
   3. Colour statistics ............... Lab colors histograms of both seeds (cv2.calcHist), lightly smoothed
   4. Per-pixel person score .......... histogram back-projection
   5. Threshold + clean-up ............ score > 0.5, morphological open/close, hole filling, keep the blob on the seed
   6. Re-estimate (x3) ................ recompute both histograms from the current mask and repeat 4-5, so the colors are learned from the person itself instead of a guess 
   7. Boundary snapping ............... marker-based watershed (cv2.watershed) on the colour image
   8. Exact boundary .................. cv2.findContours, CHAIN_APPROX_NONE


 HOW TO USE:
   run: python rgb_segmentation.py --image person.jpg --select
   draw the boxes by hand and press ENTER after done, ESC after all boxes selected

 OUTPUT:
   <name>_mask.png      binary mask, person = 255, background = 0
   <name>_boundary.png  image with the exact boundary drawn in green
   <name>_steps.png     panel of the intermediate steps
   <name>_contours.csv  x,y of every boundary pixel
================================================================================
"""

import argparse
import csv
import os

import cv2
import numpy as np

HIST_BINS = (8, 20, 20)     # Lab bins: lightness gets fewer bins because it
                            # changes with shadows; chroma (a,b) separates
                            # clothing/skin from background better.
N_ITERATIONS = 3            # colour-model re-estimation rounds


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def ellipse_kernel(size):
    size = max(3, int(size) | 1)  # odd, at least 3
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def fill_holes(mask):
    """Fill enclosed holes"""
    h, w = mask.shape
    flood = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def keep_seeded_component(mask, seed):
    """Keep the connected blob that overlaps the person seed the most
    (plus any blob at least 30 % as big that also touches the seed)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    overlap = np.bincount(labels[seed > 0].ravel(), minlength=n)
    overlap[0] = 0
    if overlap.max() == 0:                      # nothing on the seed -> biggest
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == best, 255, 0).astype(np.uint8)
    best = int(np.argmax(overlap))
    keep = [best] + [i for i in range(1, n) if i != best and overlap[i] > 0
                     and stats[i, cv2.CC_STAT_AREA] >= 0.3 * stats[best, cv2.CC_STAT_AREA]]
    return np.isin(labels, keep).astype(np.uint8) * 255


def colour_histogram(lab, mask):
    """Normalised 3-D Lab histogram of the pixels under `mask`."""
    h = cv2.calcHist([lab], [0, 1, 2], mask, list(HIST_BINS), [0, 256, 0, 256, 0, 256])
    for i in range(h.shape[0]):                                  # smooth a-b planes
        h[i] = cv2.GaussianBlur(h[i], (0, 0), 0.8)
    h = 0.25 * np.roll(h, 1, 0) + 0.5 * h + 0.25 * np.roll(h, -1, 0)  # smooth along L
    return h / max(float(h.sum()), 1e-9)


TEXTURE_BINS = 10


def texture_index(lab):
    L = lab[..., 0].astype(np.float32)
    mean = cv2.blur(L, (7, 7))
    var = np.maximum(cv2.blur(L * L, (7, 7)) - mean * mean, 0)
    std = np.sqrt(var)
    # log-spaced bins between std = 1 and std = 40 grey levels
    t = np.log1p(std) / np.log1p(40.0)
    return np.clip((t * TEXTURE_BINS).astype(np.int32), 0, TEXTURE_BINS - 1)


def texture_histogram(tex_idx, mask):
    h = np.bincount(tex_idx[mask > 0].ravel(), minlength=TEXTURE_BINS).astype(np.float64)
    h = np.convolve(h, [0.25, 0.5, 0.25], mode="same") + 1e-3
    return h / h.sum()


def back_project(lab, table):
    idx = [(lab[..., c].astype(np.int32) * HIST_BINS[c]) >> 8 for c in range(3)]
    return table[idx[0], idx[1], idx[2]]


# ----------------------------------------------------------------------------
# Core algorithm
# ----------------------------------------------------------------------------

def segment_one_box(smooth, lab, rect, debug):
    H, W = lab.shape[:2]
    x, y, w, h = rect
    size = np.hypot(w, h)

    box = np.zeros((H, W), np.uint8)
    box[y:y + h, x:x + w] = 255

    #Background seed 1: ring just outside the box
    ring_w = int(0.15 * max(w, h)) + 3
    ring = cv2.dilate(box, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * ring_w + 1, 2 * ring_w + 1)))
    ring = cv2.subtract(ring, box)

    #Background seed 2: the four corners of the box. A tight box around a standing person almost always has background in its corners.
    corners = np.zeros_like(box)
    cw, ch = int(0.22 * w), int(0.18 * h)
    for (px, py, dx, dy) in [(x, y, 1, 1), (x + w - 1, y, -1, 1),
                             (x, y + h - 1, 1, -1), (x + w - 1, y + h - 1, -1, -1)]:
        tri = np.array([[px, py], [px + dx * cw, py], [px, py + dy * ch]], np.int32)
        cv2.fillConvexPoly(corners, tri, 255)
    bg_seed = cv2.bitwise_or(ring, corners)

    #Person seed: an ellipse around the torso (centre of the box, slightly high)
    fg_seed = np.zeros_like(box)
    cv2.ellipse(fg_seed, (int(x + w / 2), int(y + 0.45 * h)),
                (max(2, int(0.16 * w)), max(2, int(0.28 * h))), 0, 0, 360, 255, -1)

    #Texture: local standard deviation of lightness. Foliage, gravel and brick are busy and skin and most clothing are smooth. 
    texture_idx = debug["texture_idx"]

    k_small = ellipse_kernel(size / 90)
    k_close = ellipse_kernel(size / 30)

    fg_sample = np.zeros_like(box)
    fg_sample[y:y + h, x + int(0.3 * w):x + int(0.7 * w)] = 255
    fg_sample = cv2.subtract(fg_sample, corners)
    bg_sample = bg_seed

    xs = (np.arange(W, dtype=np.float32) - (x + w / 2.0)) / (w / 2.0)
    prior_row = 0.65 - 0.30 * np.clip(np.abs(xs), 0, 1) ** 2
    prior = np.repeat(prior_row[None, :], H, axis=0)
    mask = fg_seed
    first_mask, score_u8 = None, None
    for it in range(N_ITERATIONS + 1):
        pc_fg, pc_bg = colour_histogram(lab, fg_sample), colour_histogram(lab, bg_sample)
        pt_fg = texture_histogram(texture_idx, fg_sample)
        pt_bg = texture_histogram(texture_idx, bg_sample)
        l_fg = back_project(lab, pc_fg) * pt_fg[texture_idx] + 1e-12
        l_bg = back_project(lab, pc_bg) * pt_bg[texture_idx] + 1e-12
        score = (l_fg * prior / (l_fg * prior + l_bg * (1 - prior))).astype(np.float32)
        score = cv2.GaussianBlur(score, (0, 0), max(1.0, size / 250))
        score[box == 0] = 0

        mask = np.where(score > 0.5, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = fill_holes(mask)
        mask = cv2.bitwise_and(mask, box)
        mask = keep_seeded_component(mask, fg_seed)
        if first_mask is None:
            first_mask = mask.copy()
        score_u8 = (np.clip(score, 0, 1) * 255).astype(np.uint8)

        if cv2.countNonZero(mask) > 50:
            fg_sample = cv2.erode(mask, k_small)
            away = cv2.bitwise_not(cv2.dilate(mask, k_close))
            bg_sample = cv2.bitwise_or(bg_seed, cv2.bitwise_and(box, away))

    band = ellipse_kernel(size / 22)
    sure_fg = cv2.erode(mask, band)
    if cv2.countNonZero(sure_fg) < 20:
        sure_fg = cv2.erode(mask, k_small)
    sure_bg = cv2.bitwise_or(cv2.bitwise_not(cv2.dilate(mask, band)), cv2.bitwise_not(box))

    markers = np.zeros((H, W), np.int32)
    markers[sure_bg > 0] = 1
    markers[sure_fg > 0] = 2
    unknown = (markers == 0)
    cv2.watershed(smooth, markers)
    final = np.where(markers == 2, 255, 0).astype(np.uint8)
    final = fill_holes(final)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, k_small)
    final = keep_seeded_component(final, sure_fg)

    # collect debug views (union over all boxes)
    debug["score"] = np.maximum(debug["score"], score_u8)
    debug["seeds"][bg_seed > 0] = (60, 60, 230)
    debug["seeds"][fg_seed > 0] = (60, 230, 60)
    debug["first"] = cv2.bitwise_or(debug["first"], first_mask)
    debug["iterated"] = cv2.bitwise_or(debug["iterated"], mask)
    debug["trimap"][(sure_bg > 0) & (box > 0)] = (60, 60, 60)
    debug["trimap"][unknown & (box > 0)] = (0, 200, 255)
    debug["trimap"][sure_fg > 0] = (255, 255, 255)
    return final


def segment_person_rgb(image_bgr, rects=None, border_fraction=0.05):
    """
    image_bgr : HxWx3 uint8 colour image
    rects     : list of (x, y, w, h) boxes, one per person, or None
    returns   : (mask, contours, debug)
    """
    H, W = image_bgr.shape[:2]
    if not rects:
        mx, my = int(border_fraction * W), int(border_fraction * H)
        rects = [(mx, my, W - 2 * mx, H - 2 * my)]
    clipped = []
    for (x, y, w, h) in rects:
        x, y = max(0, int(x)), max(0, int(y))
        w, h = min(int(w), W - x), min(int(h), H - y)
        if w > 4 and h > 4:
            clipped.append((x, y, w, h))

    smooth = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=40, sigmaSpace=7)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    tex = texture_index(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))

    debug = {
        "texture_idx": tex,
        "rects": clipped,
        "score": np.zeros((H, W), np.uint8),
        "seeds": image_bgr.copy() // 2,
        "first": np.zeros((H, W), np.uint8),
        "iterated": np.zeros((H, W), np.uint8),
        "trimap": np.full_like(image_bgr, 30),
    }
    final = np.zeros((H, W), np.uint8)
    for rect in clipped:
        final = cv2.bitwise_or(final, segment_one_box(smooth, lab, rect, debug))

    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return final, contours, debug


#Visualization section

def draw_boundary(image_bgr, mask, contours, color=(0, 255, 0)):
    out = image_bgr.copy()
    sel = mask > 0
    out[sel] = (0.65 * out[sel] + 0.35 * np.array(color)).astype(np.uint8)
    thick = max(2, int(np.hypot(*mask.shape) / 450))
    cv2.drawContours(out, contours, -1, color, thick, cv2.LINE_AA)
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


def steps_panel(image_bgr, final, contours, debug):
    boxed = image_bgr.copy()
    for (x, y, w, h) in debug["rects"]:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 200, 255), max(2, image_bgr.shape[1] // 300))
    tiles = [
        labelled(boxed, "1. input + box"),
        labelled(debug["seeds"], "2. seeds: person / bg"),
        labelled(cv2.applyColorMap(debug["score"], cv2.COLORMAP_JET), "3. P(person|colour)"),
        labelled(debug["iterated"], "4. after re-estimation"),
        labelled(debug["trimap"], "5. watershed markers"),
        labelled(draw_boundary(image_bgr, final, contours), "6. final boundary"),
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
    ap = argparse.ArgumentParser(description="Classical (non-ML) human boundary extraction for RGB images")
    ap.add_argument("--image", required=True, help="path to a colour photo of a person")
    ap.add_argument("--rect", action="append", default=None,
                    help="box around ONE person: x,y,w,h (original pixels). Repeat for several people.")
    ap.add_argument("--select", action="store_true", help="draw the box(es) with the mouse")
    ap.add_argument("--max_dim", type=int, default=900, help="processing size of the longest side")
    ap.add_argument("--out_dir", default="rgb_output")
    ap.add_argument("--show", action="store_true", help="display the result in a window")
    args = ap.parse_args()

    original = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if original is None:
        raise SystemExit(f"Could not read image: {args.image}")

    scale = min(1.0, args.max_dim / max(original.shape[:2]))
    image = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else original

    rects = []
    if args.select:
        boxes = cv2.selectROIs("Box each person (ENTER after each, ESC when done)", image, showCrosshair=False)
        cv2.destroyAllWindows()
        rects = [tuple(int(v) for v in b) for b in boxes]
    elif args.rect:
        for r in args.rect:
            rects.append(tuple(int(round(float(v) * scale)) for v in r.split(",")))

    mask, contours, debug = segment_person_rgb(image, rects)

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    full_mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_mask.png"), full_mask)
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_boundary.png"), draw_boundary(image, mask, contours))
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_steps.png"), steps_panel(image, mask, contours, debug))
    save_contours_csv(os.path.join(args.out_dir, f"{name}_contours.csv"), contours, 1.0 / scale)

    print(f"Image: {args.image}  (processed at {image.shape[1]}x{image.shape[0]})")
    print(f"Boxes used: {debug['rects']}")
    print(f"People / regions found: {len(contours)}   "
          f"foreground = {100.0 * cv2.countNonZero(mask) / mask.size:.1f}% of image")
    print(f"Boundary length(s) in pixels: {[len(c) for c in contours]}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")

    if args.show:
        cv2.imshow("boundary", draw_boundary(image, mask, contours))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
