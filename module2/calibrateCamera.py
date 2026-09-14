"""
================================================================================
 CSc 8830 - Computer Vision | Module 2 Assignment
 Camera Calibration

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   Calibrates a smartphone camera using OpenCV's built-in chessboard
   calibration pipeline. Outputs the camera intrinsic matrix (K), the
   distortion coefficients, and per-image reprojection error, which are
   required for Step 2 (perspective projection / real-world measurement).

 HOW TO USE:
   1. Print a standard chessboard calibration pattern. I used the openCV 9x6 one.
   2. Take ~20 photos of the chessboard with smartphone camera at
      different angles, distances, and positions in the frame. I kept the same zoom and resolution throughout these images.
   3. Put all the chessboard photos in a single folder. In my case it is calibration images.
   4. Run this script: python calibrateCamera.py --images ./calibrationImages --cols 9 --rows 6 --square_size 24 (in mm, specifically for my board)
   5. The script saves camera_calibration.npz containing:
        - camera_matrix (K): 3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
        - dist_coeffs: lens distortion coefficients
        - reprojection error (printed to console, also saved)
      This .npz file is loaded directly by measureObject.py in Step 2.

 OUTPUT FILES:
   - camera_calibration.npz  (K, dist_coeffs, per-image errors, mean error)
   - calibration_report.txt  (human-readable summary)
================================================================================
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def find_chessboard_corners(image_paths, pattern_size, show=False):
    """
    Detects chessboard corners in each image.

    Args:
        image_paths: list of file paths to calibration images
        pattern_size: (cols, rows) = number of INTERNAL corners
        show: if True, display each image with detected corners overlaid

    Returns:
        obj_points: list of 3D points in real-world space (per image)
        img_points: list of 2D points in image plane (per image)
        image_size: (width, height) of the images used
        used_images: list of image paths where corners were found
    """
    #termination criteria 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    obj_points = []   #3D points in real world space
    img_points = []   #2D points in image plane
    image_size = None
    used_images = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] Could not read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  #(width, height)

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print(f"  [skip] Chessboard NOT found in: {os.path.basename(path)}")
            continue

        # Refine corner locations to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria
        )

        obj_points.append(objp.copy())
        img_points.append(corners_refined)
        used_images.append(path)

        print(f"  [ok]   Corners found in: {os.path.basename(path)}")

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
            cv2.imshow("Detected corners (press any key to continue)", vis)
            cv2.waitKey(0)

    if show:
        cv2.destroyAllWindows()

    return obj_points, img_points, image_size, used_images


def calibrate(obj_points, img_points, image_size, square_size_mm):
    """
    Runs OpenCV's calibrateCamera() and computes per-image reprojection error.
    Scales object points by the real square size so intrinsics are consistent
    with a metric (millimeter) world -- although the intrinsic matrix itself
    (fx, fy, cx, cy) is expressed in pixels regardless of the square size used.
    """
    #scaling the chessboard to real life mm
    scaled_obj_points = [op * square_size_mm for op in obj_points]

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        scaled_obj_points, img_points, image_size, None, None
    )

    per_image_errors = []
    for i in range(len(scaled_obj_points)):
        projected, _ = cv2.projectPoints(
            scaled_obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        detected_pts = np.asarray(img_points[i], dtype=np.float64).reshape(-1, 2)
        projected_pts = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        diffs = detected_pts - projected_pts
        per_point_dist = np.sqrt(np.sum(diffs ** 2, axis=1))
        error = float(np.mean(per_point_dist))
        per_image_errors.append(error)

    mean_error = float(np.mean(per_image_errors))

    return camera_matrix, dist_coeffs, rvecs, tvecs, per_image_errors, mean_error


def main():
    parser = argparse.ArgumentParser(description="Smartphone camera calibration (OpenCV)")
    parser.add_argument("--images", required=True, help="Folder containing chessboard calibration photos")
    parser.add_argument("--cols", type=int, default=9, help="Number of INTERNAL corners along width")
    parser.add_argument("--rows", type=int, default=6, help="Number of INTERNAL corners along height")
    parser.add_argument("--square_size", type=float, default=25.0, help="Chessboard square size in millimeters")
    parser.add_argument("--show", action="store_true", help="Visually display detected corners")
    parser.add_argument("--out", default="camera_calibration.npz", help="Output .npz filename")
    args = parser.parse_args()

    image_paths = sorted(
        glob.glob(os.path.join(args.images, "*.jpg"))
        + glob.glob(os.path.join(args.images, "*.jpeg"))
        + glob.glob(os.path.join(args.images, "*.png"))
    )

    if len(image_paths) < 5:
        print(f"ERROR: Found only {len(image_paths)} images in {args.images}. "
              f"Use at least 10-15 chessboard images for a reliable calibration.")
        sys.exit(1)

    print(f"Found {len(image_paths)} candidate images. Detecting chessboard corners...")
    pattern_size = (args.cols, args.rows)
    obj_points, img_points, image_size, used_images = find_chessboard_corners(
        image_paths, pattern_size, show=args.show
    )

    if len(obj_points) < 5:
        print(f"ERROR: Chessboard was only detected in {len(obj_points)} images. "
              f"Need at least 5-10 good detections.")
        sys.exit(1)

    print(f"\nRunning calibration using {len(obj_points)} valid images...")
    camera_matrix, dist_coeffs, rvecs, tvecs, per_image_errors, mean_error = calibrate(
        obj_points, img_points, image_size, args.square_size
    )

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    print("\n================ CALIBRATION RESULTS ================")
    print(f"Image size (px):        {image_size}")
    print(f"fx, fy (px):             {fx:.3f}, {fy:.3f}")
    print(f"cx, cy (px):             {cx:.3f}, {cy:.3f}")
    print(f"Distortion coeffs:       {dist_coeffs.ravel()}")
    print(f"Mean reprojection error: {mean_error:.4f} px  (lower is better; <0.5 px is very good)")
    print("=======================================================\n")

    #saving results for use in step 2
    np.savez(
        args.out,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        mean_reprojection_error=mean_error,
        per_image_errors=np.array(per_image_errors),
        used_images=np.array(used_images),
    )
    print(f"Saved calibration data to: {args.out}")

    #human-readable report
    with open("calibration_report.txt", "w") as f:
        f.write("Camera Calibration Report\n")
        f.write("==========================\n")
        f.write(f"Number of images used: {len(obj_points)} / {len(image_paths)}\n")
        f.write(f"Chessboard pattern (internal corners): {args.cols} x {args.rows}\n")
        f.write(f"Square size: {args.square_size} mm\n")
        f.write(f"Image size: {image_size[0]} x {image_size[1]} px\n\n")
        f.write("Intrinsic matrix K:\n")
        f.write(f"{camera_matrix}\n\n")
        f.write(f"fx = {fx:.4f} px, fy = {fy:.4f} px\n")
        f.write(f"cx = {cx:.4f} px, cy = {cy:.4f} px\n\n")
        f.write(f"Distortion coefficients (k1,k2,p1,p2,k3): {dist_coeffs.ravel()}\n\n")
        f.write(f"Mean reprojection error: {mean_error:.4f} px\n")
        f.write("Per-image reprojection errors:\n")
        for path, err in zip(used_images, per_image_errors):
            f.write(f"  {os.path.basename(path)}: {err:.4f} px\n")
    print("Saved human-readable report to: calibration_report.txt")


if __name__ == "__main__":
    main()
