# Camera Calibration — Windows .exe

A small, fast tool to **calibrate a camera** using a **chessboard**.  
It works **offline** and has a simple GUI

## What it does
- Finds the chessboard corners in many photos.
- Computes **intrinsics** (K, fx, fy, cx, cy, skew).
- Computes **distortion** (radial k1..k6, tangential p1, p2).
- Shows **extrinsics** per image (rvec/tvec) and **RMS error**.
- Exports a text **report** and **annotated images**.

## How to use
1. Run `Camera_calibration.exe`.
2. Click **Browse…** and pick the folder with your images.
3. Enter **inner corners** (cols × rows) and **square size** (mm).  
4. Click **Calibrate**.
5. Check results in the **Results** panel and the **Image viewer**.
6. Use **Save report** or **Export annotated images** if needed.

## Tips
- Use sharp, well-lit photos with the board large in the image.
- Take 10+ views at different angles and distances.
- If “No chessboard detected”, re-check inner corner counts and image quality.

## Requirements
- Windows 7/8/10/11. The `.exe` includes dependencies (OpenCV, Tkinter).
- From source: Python 3.9+, `opencv-python`, `numpy`, `Pillow`.

## License
**MIT**.  
Includes OpenCV (BSD-3-Clause).

<img width="1915" height="1018" alt="image" src="https://github.com/user-attachments/assets/f8de6ed1-6fd5-4033-adfb-fa396ee75a4f" />
