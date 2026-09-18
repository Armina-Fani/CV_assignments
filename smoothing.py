"""
================================================================================
 CSc 8830 - Computer Vision | Module 3 Assignment
 Image Blurring: Spatial Filtering vs. Frequency-Domain (Fourier) Filtering

 README:
 -------------------------------------------------------------------------
 PURPOSE:
   Implements image blurring two independent ways -- (1) direct spatial
   convolution/correlation with a smoothing kernel (box or Gaussian), and
   (2) the Fourier-domain equivalent (multiply the image's FFT by the
   kernel's FFT, then inverse-FFT) and demonstrates numerically and
   visually that they produce the same result. 

 HOW TO USE:
   python blur_fourier_equivalence.py --image path/to/photo.jpg --filter gaussian --kernel_sizes 3,5,9,15,21

 OUTPUT:
   - comparison_kXX.png for each kernel size: original | spatial result |
     frequency-domain result 
   - equivalence_summary.csv: kernel size, filter type, mean/max absolute
     difference between the two methods, and runtime of each method
================================================================================
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np


def make_kernel(filter_type, k):
    """Builds a normalized k x k smoothing kernel (sums to 1)"""
    if filter_type == "box":
        return np.ones((k, k), dtype=np.float64) / (k * k)
    elif filter_type == "gaussian":
        g1d = cv2.getGaussianKernel(k, sigma=-1) 
        return (g1d @ g1d.T).astype(np.float64)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def spatial_blur(image, kernel):
    """Direct spatial-domain filtering"""
    return cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def frequency_blur(image, kernel):
    """
    Fourier-domain filtering: pad image and kernel, multiply their FFTs,
    inverse-transform, and crop back to the original image size. This is
    mathematically the same linear convolution that spatial_blur() computes
    (see the module docstring for why the padding matters).
    """
    H, W = image.shape
    kh, kw = kernel.shape

    out_h = H + kh - 1
    out_w = W + kw - 1

    N_h = next_pow2(out_h)
    N_w = next_pow2(out_w)

    padded_image = np.zeros((N_h, N_w), dtype=np.float64)
    padded_image[:H, :W] = image

    #placing the kernel so its center sits at index (0, 0) using circular (wraparound) indexing. 
    padded_kernel = np.zeros((N_h, N_w), dtype=np.float64)
    cy, cx = kh // 2, kw // 2
    for y in range(kh):
        for x in range(kw):
            yy = (y - cy) % N_h
            xx = (x - cx) % N_w
            padded_kernel[yy, xx] = kernel[y, x]

    F_image = np.fft.fft2(padded_image)
    F_kernel = np.fft.fft2(padded_kernel)
    F_result = F_image * F_kernel
    result_full = np.fft.ifft2(F_result).real

    return result_full[:H, :W]


def log_magnitude_spectrum(F, display_size):
    """
    Converts a complex FFT array F into a displayable log-magnitude spectrum:
    log(1 + |F|), fftshift'd so the DC term is centered, normalized to
    0-255, and resized to display_size (H, W) so it can sit in a comparison
    panel alongside the original image.
    """
    magnitude = np.log1p(np.abs(F))
    magnitude = np.fft.fftshift(magnitude)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    magnitude = cv2.resize(magnitude, (display_size[1], display_size[0]))
    return magnitude


def frequency_blur_with_spectra(image, kernel):
    """
    Same computation as frequency_blur(), but additionally returns the
    log-magnitude spectrum of the image's FFT before smoothing and after
    smoothing (i.e. after multiplying by the kernel's FFT, still in the
    frequency domain, before the inverse transform).
    """
    H, W = image.shape
    kh, kw = kernel.shape

    out_h = H + kh - 1
    out_w = W + kw - 1
    N_h = next_pow2(out_h)
    N_w = next_pow2(out_w)

    padded_image = np.zeros((N_h, N_w), dtype=np.float64)
    padded_image[:H, :W] = image

    padded_kernel = np.zeros((N_h, N_w), dtype=np.float64)
    cy, cx = kh // 2, kw // 2
    for y in range(kh):
        for x in range(kw):
            yy = (y - cy) % N_h
            xx = (x - cx) % N_w
            padded_kernel[yy, xx] = kernel[y, x]

    F_image = np.fft.fft2(padded_image)
    F_kernel = np.fft.fft2(padded_kernel)
    spec_before = log_magnitude_spectrum(F_image, (H, W))

    F_result = F_image * F_kernel
    spec_after = log_magnitude_spectrum(F_result, (H, W))

    result_full = np.fft.ifft2(F_result).real
    result = result_full[:H, :W]

    return result, spec_before, spec_after


def to_displayable(arr):
    """Clips and converts a float array to uint8 for saving/visual comparison."""
    return np.clip(arr, 0, 255).astype(np.uint8)


def run_for_kernel_size(gray, filter_type, k, out_dir):
    kernel = make_kernel(filter_type, k)

    spatial_result = spatial_blur(gray, kernel)

    freq_result, spec_before, spec_after = frequency_blur_with_spectra(gray, kernel)

    diff = np.abs(spatial_result - freq_result)
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())

    panel = np.hstack([
        to_displayable(gray),
        to_displayable(spatial_result),
        to_displayable(spec_before),
        to_displayable(spec_after),
        to_displayable(freq_result),
    ])
    out_path = os.path.join(out_dir, f"comparison_k{k:02d}_{filter_type}.png")
    cv2.imwrite(out_path, panel)

    print(f"[{filter_type}, k={k:2d}] mean|diff|={mean_diff:.3e}  max|diff|={max_diff:.3e} -> {out_path}")

    return {
        "filter_type": filter_type,
        "kernel_size": k,
        "mean_abs_diff": mean_diff,
        "max_abs_diff": max_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Spatial vs. frequency-domain blur equivalence")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--filter", choices=["box", "gaussian", "both"], default="both")
    parser.add_argument("--kernel_sizes", default="3,5,9,15,21", help="Comma-separated odd kernel sizes")
    parser.add_argument("--out_dir", default="module3_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    kernel_sizes = [int(k) for k in args.kernel_sizes.split(",")]
    filter_types = ["box", "gaussian"] if args.filter == "both" else [args.filter]

    rows = []
    for filter_type in filter_types:
        for k in kernel_sizes:
            if k % 2 == 0:
                print(f"Skipping even kernel size {k} (kernel sizes must be odd).")
                continue
            rows.append(run_for_kernel_size(gray, filter_type, k, args.out_dir))

    csv_path = os.path.join(args.out_dir, "equivalence_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary to {csv_path}")

if __name__ == "__main__":
    main()
