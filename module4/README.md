
CSc 8830 Computer Vision. Parts 1 and 2 use classical OpenCV only. SAM2 is used only as the comparison reference.

## Files

| File | Purpose |
|---|---|
| `rgb_segmentation.py` | Part 1 — person boundary in a colour photo (box seeds → colour/texture histograms → Bayes score → morphology → watershed → contours) |
| `thermal_segmentation.py` | Part 2 — person boundary in a thermal image (median → white top-hat → hysteresis → blob filtering / re-thresholding → watershed → contours) |
| `module4.js` | Browser port of Parts 1–2 used by the Module 4 tab of `index.html` (OpenCV.js) |
| `rgb_output/`, `thermal_output/` | Example outputs of the two scripts |

Every script has a README block at the top with full usage.

## Quick start

```bash

python rgb_human_segmentation.py --image image --select

python thermal_human_segmentation.py --image image

```


