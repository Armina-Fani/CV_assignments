# Assignment 6: Optical Flow, Tracking and Structure from Motion

CSc 8830 Computer Vision. Each script has a README block at the top with full details.

| File | Part | What it does |
|---|---|---|
| `optical_flow.py` | A | Dense optical flow of a 30 s clip as a video, plus what the flow shows: speed, direction, camera motion, moving objects |
| `track_validate.py` | A | Two consecutive frames: our own Lucas–Kanade tracker (derived equations + bilinear interpolation) vs. the actual pixel locations |
| `sfm_planar.py` | B | Structure from motion of a flat object from 4 photos: camera positions, 3-D points, object boundary and size |
| `module6.js` | A | Browser version of Part A for the Module 6 tab of `index.html` |

## Setup

```bash
pip install opencv-python numpy matplotlib
```

Optional: install [ffmpeg](https://ffmpeg.org/). If it is installed, the flow videos are also saved as H.264. If not, they are saved as WebM. Either format plays on the web page.

## Commands (output folders match the web page)

Put your two videos in `module6/videos/` and run these from inside `module6/`:

```bash
# Part A: optical flow (30 s starting at --start seconds)
python optical_flow.py --video videos/video1.mp4 --start 0 --out_dir flow_output/video1
python optical_flow.py --video videos/video2.mp4 --start 0 --out_dir flow_output/video2

# Part A: tracking check on two consecutive frames (pick a moment with motion)
python track_validate.py --video videos/video1.mp4 --time 10 --moving_only --out_dir tracking_output/video1
python track_validate.py --video videos/video2.mp4 --time 10 --out_dir tracking_output/video2
#   use --moving_only when the camera is still (tracks the people/cars, not the background)

# Part B: put the 4 photos in sfm_images/ (photo 1 = the one you measured the distance for)
python sfm_planar.py --images sfm_images/1.jpg sfm_images/2.jpg sfm_images/3.jpg sfm_images/4.jpg \
       --calib ../module2/camera_calibration.npz --distance_mm 450 --true_size 152,229 --select
#   --select opens view 1: click the object's corners in order around the outline, then press ENTER
```

## Photos for Part B

* Use the same phone and settings as the Module 2 calibration: portrait, 1× zoom, no "portrait mode", no cropping.
* Save as JPEG. iPhones save HEIC by default, which OpenCV cannot read. Either set Settings → Camera → Formats → Most Compatible, or export the photos as JPEG.
* Use a flat, textured object on a textured surface, filling about half the frame.
* Take the 4 photos from clearly different positions, moving about 20–40 cm between shots and turning the phone towards the object.
* For photo 1, measure the straight-line distance from the phone to the centre of the object. Also measure the object's real width and height (`--true_size`).

## Large files

Phone videos can exceed GitHub's 100 MB file limit. Either leave `videos/` out of the repo (the processed `flow_output` videos are much smaller), or shorten them to the 30 s you use first.
