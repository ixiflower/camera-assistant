#!/usr/bin/env python3
"""
Camera Assistant — Tkinter GUI with real-time CV:
  • Face detection       • Eye detection
  • Hand / Finger(OCV)   • Finger counting
  • Full body detect     • Smile detection
  • Edge / Motion        • Gesture recognition

Hand detection uses improved OpenCV (YCrCb skin mask + convexity defects)
with multi-stage contour filtering. No external ML libs needed.
"""

from __future__ import annotations

import os
# Suppress OpenCV noisy logs (MUST be set BEFORE cv2 import)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from typing import Optional


# ── colour palette ──────────────────────────────────────────────────────────
BG        = "#1e1e2e"
FG        = "#cdd6f4"
SURFACE   = "#2a2a3e"
DARK      = "#11111b"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
AMBER     = "#fab387"
BLUE      = "#89b4fa"

# Convert hex RRGGBB → BGR tuple for OpenCV
def _bgr(hex_color: str) -> tuple:
    """'#a6e3a1' → (161, 227, 166)"""
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

ACCENTS   = {"face": _bgr(BLUE), "eye": _bgr(AMBER), "hand": _bgr(GREEN),
             "body": _bgr("#cba6f7"), "smile": _bgr("#f9e2af"), "edge": _bgr("#94e2d5")}
# Hex versions for Tkinter (tkinter doesn't understand BGR tuples)
HEX_COLORS = {"face": BLUE, "eye": AMBER, "hand": GREEN,
              "body": "#cba6f7", "smile": "#f9e2af", "edge": "#94e2d5"}
# OpenCV BGR constants for drawing code that uses bare colour names
_BGR_GREEN = _bgr(GREEN)
_BGR_AMBER = _bgr(AMBER)
_BGR_DARK  = _bgr(DARK)

# ── Haar cascade paths (OpenCV built-in) ───────────────────────────────────
CASCADE_DIR = cv2.data.haarcascades
CASCADES = {
    "face":       os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml"),
    "eye":        os.path.join(CASCADE_DIR, "haarcascade_eye.xml"),
    "smile":      os.path.join(CASCADE_DIR, "haarcascade_smile.xml"),
    "profile":    os.path.join(CASCADE_DIR, "haarcascade_profileface.xml"),
    "fullbody":   os.path.join(CASCADE_DIR, "haarcascade_fullbody.xml"),
    "upperbody":  os.path.join(CASCADE_DIR, "haarcascade_upperbody.xml"),
}


class CameraAssistant:
    """Tkinter camera app with computer-vision overlays."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("📷 Camera Assistant — CV")
        self.root.geometry("960x720")
        self.root.minsize(640, 480)
        self.root.configure(bg=BG)

        # ── state ──────────────────────────────────────────────────────
        self.capture: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.available_cameras: list[str] = []

        # Feature toggles
        self.features = {
            "face":     tk.BooleanVar(value=True),
            "eye":      tk.BooleanVar(value=False),
            "smile":    tk.BooleanVar(value=False),
            "hand":     tk.BooleanVar(value=True),   # YCrCb + contour
            "body":     tk.BooleanVar(value=False),
            "edge":     tk.BooleanVar(value=False),    # Canny edge
        }

        # ── classifiers ────────────────────────────────────────────────
        self.classifiers = {}
        for name, path in CASCADES.items():
            self.classifiers[name] = cv2.CascadeClassifier(path)

        # ── Hand detection config ──────────────────────────────────────
        # YCrCb ranges for skin detection
        self._skin_lower = (0, 115, 75)
        self._skin_upper = (255, 190, 145)
        # Minimum contour area as fraction of total frame area
        self._hand_min_area_ratio = 0.005
        # CLAHE for low-light enhancement
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # Background subtractor for motion-based hand detection
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=36, detectShadows=False
        )
        self._bg_learning_rate = -1  # auto-learn

        # ── build UI ───────────────────────────────────────────────────
        self._build_ui()
        self._scan_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  UI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _make_toggle(self, parent: tk.Widget, name: str, var: tk.BooleanVar,
                     colour: str) -> tk.Checkbutton:
        cb = tk.Checkbutton(
            parent, text=name.capitalize(), variable=var,
            fg=colour, bg=SURFACE, selectcolor=DARK,
            activebackground=SURFACE, activeforeground=colour,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        cb.pack(side=tk.LEFT, padx=2)
        return cb

    def _build_ui(self) -> None:
        # --- top bar ---
        top = tk.Frame(self.root, bg=SURFACE, padx=10, pady=6)
        top.pack(fill=tk.X)

        tk.Label(top, text="Camera:", fg=FG, bg=SURFACE,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        self.cam_combo = ttk.Combobox(
            top, state="readonly", width=16, font=("Segoe UI", 10),
        )
        self.cam_combo.pack(side=tk.LEFT, padx=6)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_select)

        self.scan_btn = tk.Button(
            top, text="⟳", command=self._scan_cameras,
            bg="#45475a", fg=FG, relief=tk.FLAT, padx=8, font=("Segoe UI", 10),
            cursor="hand2",
        )
        self.scan_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.start_btn = tk.Button(
            top, text="▶ Start", command=self._toggle,
            bg=GREEN, fg=BG, relief=tk.FLAT, padx=14, pady=2,
            font=("Segoe UI", 10, "bold"), cursor="hand2",
        )
        self.start_btn.pack(side=tk.LEFT)

        # feature toggles (packed right)
        tog_frame = tk.Frame(top, bg=SURFACE)
        tog_frame.pack(side=tk.RIGHT)
        for name, var in self.features.items():
            c = HEX_COLORS.get(name, FG)
            self._make_toggle(tog_frame, name, var, c)

        # status
        self.status_lbl = tk.Label(
            top, text="⏸ Stopped", fg="#a6adc8", bg=SURFACE,
            font=("Segoe UI", 9),
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=(10, 0))

        # --- video ---
        vframe = tk.Frame(self.root, bg=DARK)
        vframe.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.video_lbl = tk.Label(vframe, bg=DARK)
        self.video_lbl.pack(fill=tk.BOTH, expand=True)

        # bottom info bar
        self.info_lbl = tk.Label(
            self.root, text="", fg="#a6adc8", bg=BG,
            font=("Segoe UI", 9), anchor=tk.W, padx=10,
        )
        self.info_lbl.pack(fill=tk.X)

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        w = max(self.video_lbl.winfo_width(), 640)
        h = max(self.video_lbl.winfo_height(), 480)
        img = Image.new("RGB", (w, h), (17, 17, 27))
        self._tk_img = ImageTk.PhotoImage(img)
        self.video_lbl.configure(image=self._tk_img)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Camera scan
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _scan_cameras(self) -> None:
        self.available_cameras.clear()
        self.cam_combo["values"] = ()
        self.status_lbl.configure(text="⏳ Scanning…")

        import glob
        existing = sorted(glob.glob("/dev/video*"))

        def scan():
            found: list[str] = []
            # Open by DEVICE PATH — avoids V4L2 index warnings for
            # metadata channels (/dev/video1, etc.) and works with
            # devices that expect a specific device node.
            for dev in existing:
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        found.append(dev)
                    cap.release()
            # Fallback — scan indices if nothing found by path
            if not found:
                for i in range(6):
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ok, _ = cap.read()
                        if ok:
                            found.append(f"/dev/video{i}")
                        cap.release()
            self.root.after(0, self._scan_done, found)

        threading.Thread(target=scan, daemon=True).start()

    def _scan_done(self, found: list[str]) -> None:
        self.available_cameras = found
        labels = [dev for dev in found] or ["(none)"]
        self.cam_combo["values"] = labels
        if found:
            self.cam_combo.current(0)
            self.status_lbl.configure(text=f"✅ {len(found)} camera(s)")
        else:
            self.status_lbl.configure(text="❌ No camera")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Start / Stop
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _toggle(self) -> None:
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        if not self.available_cameras:
            messagebox.showwarning("No Camera", "No cameras found.")
            return
        idx = self.cam_combo.current()
        if idx < 0 or idx >= len(self.available_cameras):
            return
        dev_path = self.available_cameras[idx]
        self.capture = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not self.capture or not self.capture.isOpened():
            messagebox.showerror("Error", f"Failed {dev_path}")
            return

        # Reset background subtractor for new session
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=36, detectShadows=False
        )

        self.running = True
        self.start_btn.configure(text="■ Stop", bg=RED)
        self.status_lbl.configure(text=f"📷 LIVE — {dev_path}")
        self.cam_combo.configure(state="disabled")
        self.scan_btn.configure(state="disabled")

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
        if self.capture:
            self.capture.release()
            self.capture = None
        self.start_btn.configure(text="▶ Start", bg=GREEN)
        self.status_lbl.configure(text="⏸ Stopped")
        self.cam_combo.configure(state="readonly")
        self.scan_btn.configure(state="normal")
        self._show_placeholder()
        self.info_lbl.configure(text="")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  CV processing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _process_hands(self, frame: cv2.Mat, parts: list[str]) -> None:
        """Dual hand detection: YCrCb skin mask + MOG2 motion fallback.

        Pipeline:
          1. Skin mask → YCrCb inRange → morph → contours
          2. If skin finds nothing → MOG2 motion mask → contours
          3. Multi-stage contour filter (area, solidity, aspect)
          4. Convex hull → convexity defects → finger count
        """
        h, w, _ = frame.shape
        min_area = int(w * h * self._hand_min_area_ratio)

        # ── Method A: YCrCb skin mask ──────────────────────────────
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # Gaussian blur to reduce noise before inRange
        ycrcb = cv2.GaussianBlur(ycrcb, (5, 5), 0)
        skin_mask = cv2.inRange(ycrcb, self._skin_lower, self._skin_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hand_contours = self._filter_hand_contours(contours, min_area)

        method_tag = "SK"  # skin
        if not hand_contours:
            # ── Method B: motion-based fallback (color-independent) ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            fg_mask = self._bg_subtractor.apply(gray, learningRate=-1)
            # Threshold to binary
            _, fg_mask = cv2.threshold(fg_mask, 15, 255, cv2.THRESH_BINARY)
            # Morph cleanup
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

            motion_contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            hand_contours = self._filter_hand_contours(motion_contours, min_area)
            method_tag = "MO"

        if not hand_contours:
            return

        # Sort by area descending, keep top 2
        hand_contours.sort(key=cv2.contourArea, reverse=True)
        hand_contours = hand_contours[:2]

        accent = ACCENTS["hand"]

        for cnt in hand_contours:
            area = cv2.contourArea(cnt)
            rect = cv2.boundingRect(cnt)
            cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2

            # ── Convex hull ─────────────────────────────────────────
            hull = cv2.convexHull(cnt)
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            hull_area = cv2.contourArea(hull)

            # Draw hull outline
            cv2.drawContours(frame, [hull], -1, _bgr("#89b4fa"), 1, cv2.LINE_AA)

            # ── Convexity defects → finger counting ─────────────────
            n_fingers = 0
            deep_defects: list = []
            if hull_idx is not None and len(hull_idx) >= 3:
                defects = cv2.convexityDefects(cnt, hull_idx)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth_px = d / 256.0
                        if depth_px > max(12, rect[2] * 0.04):
                            deep_defects.append((s, e, f, depth_px))
                            far = tuple(cnt[f][0])
                            cv2.circle(frame, far, 3, _bgr("#f38ba8"), -1, cv2.LINE_AA)

                    n_fingers = min(len(deep_defects) + 1, 5)
                    if n_fingers <= 1 and hull_area > 0 and area / hull_area > 0.85:
                        n_fingers = 0

            # ── Draw contour outline ────────────────────────────────
            cv2.drawContours(frame, [cnt], -1, accent, 2, cv2.LINE_AA)

            # ── Info banner ─────────────────────────────────────────
            banner = f"✋ {n_fingers}/5 [{method_tag}]"
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.65, 2)
            bx = max(0, min(cx - bw // 2, w - bw))
            by = max(24, cy - rect[3] // 2 - 10)

            cv2.rectangle(frame, (bx - 6, by - 22), (bx + bw + 6, by + 6),
                          (17, 17, 27, 180), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx - 6, by - 22), (bx + bw + 6, by + 6),
                          _bgr("#2a2a3e"), 1, cv2.LINE_AA)
            cv2.putText(frame, banner, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        accent, 2, cv2.LINE_AA)

            # Position text below banner
            pos_label = f"pos ({cx},{cy})"
            (pw, ph), _ = cv2.getTextSize(pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            px = max(0, min(cx - pw // 2, w - pw))
            py = by + 20
            cv2.putText(frame, pos_label, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        _bgr("#a6adc8"), 1, cv2.LINE_AA)

            # Draw center dot
            cv2.circle(frame, (cx, cy), 4, accent, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, _bgr("#1e1e2e"), 1, cv2.LINE_AA)
            bar_x = max(0, min(cx - 30, w - 60))
            bar_y = max(24, cy - rect[3] // 2 + 12)
            seg_h, seg_w, gap = 8, 14, 3
            for fi in range(5):
                sx2 = bar_x + fi * (seg_w + gap)
                clr = accent if fi < n_fingers else _bgr("#45475a")
                cv2.rectangle(frame, (sx2, bar_y), (sx2 + seg_w, bar_y + seg_h),
                              clr, -1, cv2.LINE_AA)
                cv2.rectangle(frame, (sx2, bar_y), (sx2 + seg_w, bar_y + seg_h),
                              _bgr("#585b70"), 1, cv2.LINE_AA)

        parts.append(f"🤚 {len(hand_contours)} hand(s) [{method_tag}]")

    # ────────────────────────────────────────────────────────────────────
    def _filter_hand_contours(
        self,
        contours: tuple | list,
        min_area: int,
    ) -> list[cv2.Mat]:
        """Multi-stage contour filter: area, aspect ratio, solidity."""
        hands: list[cv2.Mat] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # Aspect ratio — hands are roughly 0.5–2.0
            _, _, cw, ch = cv2.boundingRect(cnt)
            if cw == 0 or ch == 0:
                continue
            aspect = max(cw, ch) / min(cw, ch)
            if aspect > 2.5:
                continue

            # Solidity — convex area vs actual area (~0.4–0.95 for hands)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.35 or solidity > 0.96:
                continue

            hands.append(cnt)
        return hands

    def _process(self, frame: cv2.Mat) -> tuple[cv2.Mat, str]:
        """Apply enabled CV detections. Returns (annotated_frame, info_line)."""
        # ── 1. Enhance low-light frames ────────────────────────────────
        mean_brightness = frame.mean()
        if mean_brightness < 120:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if mean_brightness < 80:
                gamma = 0.6
                look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
                frame = cv2.LUT(frame, look_up)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        parts: list[str] = []

        # ── 2. Hand detection FIRST (priority) ─────────────────────────
        hand_detected = False
        if self.features["hand"].get():
            self._process_hands(frame, parts)
            hand_detected = any("🤚" in p for p in parts)

        # ── 3. Other detection — ONLY when no hand is in frame ───────
        if not hand_detected:
            # ── Face ─────────────────────────────────────────────────
            if self.features["face"].get():
                faces = self.classifiers["face"].detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                faces_arr = faces if isinstance(faces, list) else (faces if hasattr(faces, 'shape') else [])
                for (x, y, fw, fh) in faces_arr:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh),
                                  ACCENTS["face"], 2, cv2.LINE_AA)
                    cv2.putText(frame, "Face", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENTS["face"], 1)

                    if self.features["eye"].get():
                        roi_gray = gray[y:y + fh, x:x + fw]
                        eyes = self.classifiers["eye"].detectMultiScale(
                            roi_gray, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20)
                        )
                        eyes_arr = eyes if hasattr(eyes, 'shape') else []
                        for (ex, ey, ew, eh) in eyes_arr:
                            cv2.rectangle(frame, (x + ex, y + ey),
                                          (x + ex + ew, y + ey + eh),
                                          ACCENTS["eye"], 1, cv2.LINE_AA)

                    if self.features["smile"].get():
                        roi_gray2 = gray[y:y + fh, x:x + fw]
                        smiles = self.classifiers["smile"].detectMultiScale(
                            roi_gray2, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
                        )
                        smiles_arr = smiles if hasattr(smiles, 'shape') else []
                        for (sx, sy, sw, sh) in smiles_arr:
                            cv2.rectangle(frame, (x + sx, y + sy),
                                          (x + sx + sw, y + sy + sh),
                                          ACCENTS["smile"], 1, cv2.LINE_AA)
                if hasattr(faces, 'any') and faces.any():
                    parts.append(f"👤 {len(faces)} face(s)")

            # ── Full / Upper body ────────────────────────────────────
            if self.features["body"].get():
                bodies = self.classifiers["fullbody"].detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 200)
                )
                for (bx, by, bw, bh) in bodies:
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                                  ACCENTS["body"], 2, cv2.LINE_AA)
                if hasattr(bodies, 'any') and bodies.any():
                    parts.append(f"👤 {len(bodies)} body/face(s)")
        else:
            # Hand is detected — only draw a clean status
            h_val, w_val = frame.shape[:2]
            pos_text = f"HAND ACTIVE — fingers open"
            cv2.putText(frame, pos_text, (w_val - 280, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENTS["hand"], 1, cv2.LINE_AA)

        # ── Edge (Canny) ─────────────────────────────────────────────────
        if self.features["edge"].get():
            edges = cv2.Canny(gray, 50, 150)
            edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.8, edge_bgr, 0.2, 0)

        return (frame, "  •  ".join(parts) if parts else "")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Video loop
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _loop(self) -> None:
        fps_counter = 0
        fps_timer = time.monotonic()
        fps_val = 0

        while self.running and self.capture and self.capture.isOpened():
            ok, frame = self.capture.read()
            if not ok:
                continue

            frame, info = self._process(frame)

            # FPS measurement
            fps_counter += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps_val = round(fps_counter / elapsed)
                fps_counter = 0
                fps_timer = time.monotonic()

            # FPS overlay
            cv2.putText(frame, f"{fps_val} FPS", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BGR_AMBER, 1, cv2.LINE_AA)

            # resize + display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = self._fit(img)
            tk_img = ImageTk.PhotoImage(img)

            self.root.after(0, self._update, tk_img, info)
            time.sleep(1 / 30)

    def _update(self, tk_img: ImageTk.PhotoImage, info: str) -> None:
        self._tk_img = tk_img
        self.video_lbl.configure(image=self._tk_img)
        self.info_lbl.configure(text=info)

    def _fit(self, img: Image.Image) -> Image.Image:
        w = self.video_lbl.winfo_width() or 640
        h = self.video_lbl.winfo_height() or 480
        img.thumbnail((w, h), Image.LANCZOS)
        return img

    def _on_select(self, _=None) -> None:
        if self.available_cameras:
            idx = self.cam_combo.current()
            dev_path = self.available_cameras[idx]
            self.status_lbl.configure(text=f"Ready — {dev_path}")

    def _on_close(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.capture:
            self.capture.release()
        self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    window = tk.Tk()
    app = CameraAssistant(window)
    window.mainloop()
