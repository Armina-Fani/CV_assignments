"""
================================================================================
 CSc 8830 - Computer Vision | Module 2 Assignment
 Step 2: Real-World 2D Object Dimensions via Perspective Projection

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   Given a calibrated camera (from calibrateCamera.py) and a known
   object-to-camera distance Z, this script lets you click two points on an
   image of an object and computes the real-world distance between them
   (e.g. object width, height, or diagonal) using the perspective
   projection (pinhole camera) equations.

 HOW TO USE (single measurement, interactive click mode):
   python measureObject.py --image path/to/photo.jpg --calib camera_calibration.npz --distance_mm 2500
   Then click the two endpoints of the object in the popup window, in order,
   and press 'q'. The real-world distance (mm and cm) is printed and
   overlaid on the image.

 HOW TO USE (batch mode, 20 measurements):
   Prepare a CSV file "ground_truth.csv" with columns:
       image,distance_mm,true_length_mm
   One row per photo (20 rows for the validation experiment). Then run:
       python measureObject.py --batch ground_truth.csv --calib camera_calibration.npz
   For each image you will be prompted to click the two endpoints of the
   object being measured. The script computes the estimated length, compares
   it to the ground truth you supply, and writes validation_results.csv plus
   printed error statistics (mean error, std dev, RMSE, % error).

 OUTPUT:
   - Annotated image(s) with the measured line drawn and length labeled
   - validation_results.csv (batch mode) with per-trial and summary errors
================================================================================
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    """Records left-click coordinates for the two endpoints of the object."""
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))
        print(f"  Point {len(clicked_points)} clicked at: ({x}, {y})")


def load_calibration(calib_path):
    """Loads camera_matrix and dist_coeffs saved by calibrate_camera.py."""
    data = np.load(calib_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def pixel_to_world_xy(u, v, Z, camera_matrix):
    """
    Inverts the perspective projection equations to recover real-world
    (X, Y) at a known depth Z from a pixel coordinate (u, v).

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Returns (X, Y) in the same units as Z (millimeters, if Z is in mm).
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y


def real_world_distance(p1_px, p2_px, Z, camera_matrix):
    """
    Computes the real-world Euclidean distance between two image points,
    assuming both lie at the same depth Z from the camera.
    """
    X1, Y1 = pixel_to_world_xy(p1_px[0], p1_px[1], Z, camera_matrix)
    X2, Y2 = pixel_to_world_xy(p2_px[0], p2_px[1], Z, camera_matrix)
    dist = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
    return dist, (X1, Y1), (X2, Y2)


def get_two_clicks(image_path, max_display_dim=1000):
    """
    Opens an image window and collects two click points from the user.
    """
    global clicked_points
    clicked_points = []

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    scale = min(1.0, max_display_dim / max(orig_h, orig_w))
    display_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale))) if scale < 1.0 else img.copy()

    window_name = f"Click 2 endpoints of the object, then press 'q' | {os.path.basename(image_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # AUTOSIZE: no OS/user resizing, no coord ambiguity
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        vis = display_img.copy()
        for pt in clicked_points:
            cv2.circle(vis, pt, 5, (0, 0, 255), -1)
        if len(clicked_points) == 2:
            cv2.line(vis, clicked_points[0], clicked_points[1], (0, 255, 0), 2)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") and len(clicked_points) == 2:
            break
        if key == 27:  # ESC cancels
            cv2.destroyWindow(window_name)
            return None, img

    cv2.destroyWindow(window_name)

    full_res_points = [(int(x / scale), int(y / scale)) for (x, y) in clicked_points]
    print(f"  (Display was scaled {scale:.3f}x; full-resolution click coords: {full_res_points})")

    return full_res_points, img


def annotate_and_save(img, p1, p2, distance_mm, out_path):
    """Draws the measured line and label, then saves the annotated image."""
    vis = img.copy()
    cv2.circle(vis, p1, 6, (0, 0, 255), -1)
    cv2.circle(vis, p2, 6, (0, 0, 255), -1)
    cv2.line(vis, p1, p2, (0, 255, 0), 2)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    label = f"{distance_mm:.1f} mm ({distance_mm / 10:.2f} cm)"
    cv2.putText(vis, label, (mid[0] + 10, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)
    print(f"  Saved annotated image to: {out_path}")


def run_single(args, camera_matrix):
    print(f"\nMeasuring object in: {args.image}")
    print("Click the two endpoints of the object dimension you want to measure, then press 'q'.")
    points, img = get_two_clicks(args.image)
    if points is None:
        print("Cancelled.")
        return

    dist_mm, w1, w2 = real_world_distance(points[0], points[1], args.distance_mm, camera_matrix)
    print(f"\nEstimated real-world distance: {dist_mm:.2f} mm  ({dist_mm/10:.2f} cm)")
    print(f"World coords (mm) at Z={args.distance_mm}: P1={w1}, P2={w2}")

    out_path = os.path.splitext(args.image)[0] + "_measured.jpg"
    annotate_and_save(img, points[0], points[1], dist_mm, out_path)


def run_batch(args, camera_matrix):
    """
    Step 3 validation workflow: for each row in the ground-truth CSV,
    click the object's two endpoints, compute the estimated length, and
    compare it to the known true length. Reports error statistics across
    all 20 (or more) trials.
    """
    if not os.path.exists(args.batch):
        print(f"ERROR: ground truth CSV not found: {args.batch}")
        sys.exit(1)

    rows = []
    with open(args.batch, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        print("ERROR: ground truth CSV is empty.")
        sys.exit(1)

    results = []
    for i, row in enumerate(rows, start=1):
        image_path = row["image"]
        Z = float(row["distance_mm"])
        true_length = float(row["true_length_mm"])

        print(f"\n[{i}/{len(rows)}] Image: {image_path}  (Z = {Z} mm, true length = {true_length} mm)")
        points, img = get_two_clicks(image_path)
        if points is None:
            print("  Skipped (cancelled).")
            continue

        est_length, _, _ = real_world_distance(points[0], points[1], Z, camera_matrix)
        error_mm = est_length - true_length
        pct_error = 100.0 * abs(error_mm) / true_length

        print(f"  Estimated length: {est_length:.2f} mm | Error: {error_mm:+.2f} mm | % error: {pct_error:.2f}%")

        out_path = os.path.splitext(image_path)[0] + "_measured.jpg"
        annotate_and_save(img, points[0], points[1], est_length, out_path)

        results.append({
            "image": image_path,
            "distance_mm": Z,
            "true_length_mm": true_length,
            "estimated_length_mm": est_length,
            "error_mm": error_mm,
            "abs_error_mm": abs(error_mm),
            "pct_error": pct_error,
        })

    if len(results) == 0:
        print("No measurements were completed.")
        return

    errors = np.array([r["error_mm"] for r in results])
    abs_errors = np.array([r["abs_error_mm"] for r in results])
    pct_errors = np.array([r["pct_error"] for r in results])

    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_abs_error = float(np.mean(abs_errors))
    mean_pct_error = float(np.mean(pct_errors))

    print("\n================ VALIDATION SUMMARY (n={}) ================".format(len(results)))
    print(f"Mean error:          {mean_error:+.3f} mm")
    print(f"Std dev of error:    {std_error:.3f} mm")
    print(f"Mean absolute error: {mean_abs_error:.3f} mm")
    print(f"RMSE:                {rmse:.3f} mm")
    print(f"Mean % error:        {mean_pct_error:.2f} %")
    print("=============================================================\n")

    with open("validation_results.csv", "w", newline="") as f:
        fieldnames = ["image", "distance_mm", "true_length_mm", "estimated_length_mm",
                      "error_mm", "abs_error_mm", "pct_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        writer.writerow({})
        writer.writerow({"image": "SUMMARY"})
        writer.writerow({"image": "mean_error_mm", "distance_mm": mean_error})
        writer.writerow({"image": "std_error_mm", "distance_mm": std_error})
        writer.writerow({"image": "mean_abs_error_mm", "distance_mm": mean_abs_error})
        writer.writerow({"image": "rmse_mm", "distance_mm": rmse})
        writer.writerow({"image": "mean_pct_error", "distance_mm": mean_pct_error})

    print("Saved detailed results to: validation_results.csv")


def main():
    parser = argparse.ArgumentParser(description="Real-world 2D object measurement via perspective projection")
    parser.add_argument("--calib", required=True, help="Path to camera_calibration.npz from Step 1")
    parser.add_argument("--image", help="Single image to measure (single-measurement mode)")
    parser.add_argument("--distance_mm", type=float, help="Known camera-to-object distance Z, in mm (single mode)")
    parser.add_argument("--batch", help="CSV file with columns: image,distance_mm,true_length_mm (Step 3 validation mode)")
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.calib)
    print("Loaded calibration:")
    print(camera_matrix)

    if args.batch:
        run_batch(args, camera_matrix)
    elif args.image and args.distance_mm:
        run_single(args, camera_matrix)
    else:
        print("ERROR: Provide either (--image and --distance_mm) for a single measurement, "
              "or --batch ground_truth.csv for the Step 3 validation run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
