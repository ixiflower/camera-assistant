# 📷 Camera Assistant

A live computer-vision viewer built with Python 3, OpenCV, and Tkinter — powered by your phone camera via **scrcpy + v4l2loopback**.

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?style=flat&logo=opencv)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-orange?style=flat)

## Demo pipeline

```
Phone (USB/ADB) → scrcpy v4l2-sink → /dev/video0 → OpenCV → Tkinter canvas
```

## Features

| Toggle | Detection |
|--------|-----------|
| 👤 Face | Blue bounding box + "Face" label |
| 👁 Eye | Amber boxes inside detected faces |
| 😊 Smile | Yellow highlight on smiles |
| 🤚 Hand | **OpenCV** skin contour + convexity defects + finger count bar |
| 🧍 Body | Purple boxes (full + upper body) |
| ⚡ Edge | Canny edge overlay (teal tint) |

- **FPS counter** displayed at top-left
- **Finger counting via convexity defects** — YCrCb skin segmentation + contour analysis
- **Smart camera scan** — only checks existing `/dev/video*` devices
- **Dark theme** — Catppuccin Mocha inspired, glassmorphism aesthetic

## Requirements

```bash
# Arch / pacman
sudo pacman -S python-opencv python-pillow python-numpy v4l2loopback-dkms

# scrcpy (phone camera pipe)
sudo pacman -S scrcpy android-tools

# Load v4l2loopback
sudo modprobe v4l2loopback
```

| Package    | Why |
|------------|-----|
| OpenCV     | Face/Haar, YCrCb skin mask, contour detection, all CV |
| tkinter    | Built-in GUI |
| pillow     | Frame → Tkinter canvas |

> ⚠️ **No external ML libraries needed** — hand detection uses pure OpenCV
> (YCrCb colour-space skin segmentation + convexity-defect finger counting).

You also need `v4l2loopback-dkms` built for your kernel (pacman handles this).

## Usage

**1. Add yourself to the `video` group** (one-time):
```bash
sudo usermod -a -G video $USER
# Then log out and back in
```

**2. Start the phone camera stream** (keep this terminal open):
```bash
scrcpy --v4l2-sink=/dev/video0 --video-source=camera \
       --camera-id=0 --camera-size=1920x1080 \
       --camera-fps=30 --no-window
```

**3. Launch the app** (new terminal):
```bash
python3 camera_assistant.py
```

Select **Camera 0** from the dropdown → click **▶ Start**.

Toggle individual detections on/off while the stream is live.

## Controls

| Action | Button |
|--------|--------|
| ▶ Start / ■ Stop | Bottom-left |
| 🔄 Rescan | Re-scan camera devices |
| Face / Eye / Smile / Hand / Body / Edge | Checkboxes — toggle live |

## Notes

- Built for **Nothing A142P (Android 16)** but works with any phone camera accessible via ADB.
- The `v4l2loopback` kernel module creates a virtual video device that scrcpy writes to.
- Hand detection uses skin-colour HSV masking + convexity defects for finger counting.
