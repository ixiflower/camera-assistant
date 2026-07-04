#!/usr/bin/env python3
"""
Camera Assistant — Tkinter GUI with real-time CV:
  • Face detection     • Eye detection
  • Hand tracking      • Finger counting
  • Full body detect   • Smile detection
  • Edge / Motion      • Pose skeleton (optional)

Uses only OpenCV + Pillow — no MediaPipe needed.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
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

        # Suppress OpenCV noisy logs
        import os as _os
        _os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

        # ── state ──────────────────────────────────────────────────────
        self.capture: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.available_cameras: list[int] = []

        # Feature toggles
        self.features = {
            "face":     tk.BooleanVar(value=True),
            "eye":      tk.BooleanVar(value=False),
            "smile":    tk.BooleanVar(value=False),
            "hand":     tk.BooleanVar(value=True),   # skin + contour
            "body":     tk.BooleanVar(value=False),
            "edge":     tk.BooleanVar(value=False),    # Canny edge
        }

        # ── classifiers ────────────────────────────────────────────────
        self.classifiers = {}
        for name, path in CASCADES.items():
            self.classifiers[name] = cv2.CascadeClassifier(path)

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
            found = []
            # First try existing /dev/video* devices
            for dev in existing:
                # Extract index from /dev/videoN
                try:
                    idx = int(dev.replace("/dev/video", ""))
                except ValueError:
                    continue
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        found.append(idx)
                    cap.release()
            # Fallback: scan indices 0-5 if nothing found
            if not found:
                for i in range(6):
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ok, _ = cap.read()
                        if ok:
                            found.append(i)
                        cap.release()
            self.root.after(0, self._scan_done, found)

        threading.Thread(target=scan, daemon=True).start()

    def _scan_done(self, found: list[int]) -> None:
        self.available_cameras = found
        labels = [f"Camera {i}" for i in found] or ["(none)"]
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
        cam_id = self.available_cameras[idx]
        self.capture = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not self.capture or not self.capture.isOpened():
            messagebox.showerror("Error", f"Failed Camera {cam_id}")
            return

        self.running = True
        self.start_btn.configure(text="■ Stop", bg=RED)
        self.status_lbl.configure(text=f"📷 LIVE — cam {cam_id}")
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

    # HSV lower/upper bounds for skin detection
    SKIN_LOWER = (0, 15, 50)
    SKIN_UPPER = (30, 170, 255)

    def _process(self, frame: cv2.Mat) -> tuple[cv2.Mat, str]:
        """Apply enabled CV detections. Returns (annotated_frame, info_line)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        parts: list[str] = []

        # ── Face ───────────────────────────────────────────────────────
        if self.features["face"].get():
            faces = self.classifiers["face"].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            # detectMultiScale returns tuple when empty, ndarray when found
            faces_arr = faces if isinstance(faces, list) else (faces if hasattr(faces, 'shape') else [])
            for (x, y, fw, fh) in faces_arr:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh),
                              ACCENTS["face"], 2, cv2.LINE_AA)
                # label
                cv2.putText(frame, "Face", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENTS["face"], 1)

                # ── Eyes (inside each face) ──
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

                # ── Smile (inside face) ──
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

        # ── Hand / Finger detection via skin contour ───────────────────
        if self.features["hand"].get():
            mask = cv2.inRange(hsv, self.SKIN_LOWER, self.SKIN_UPPER)
            # blur + threshold
            mask = cv2.medianBlur(mask, 7)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            hand_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 2000 or area > h * w * 0.4:
                    continue
                # approximate polygon
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area < 500:
                    continue
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < 0.5:
                    continue

                # bounding rect aspect ratio (hand is roughly square-ish)
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if rw < 20 or rh < 20:
                    continue

                hand_count += 1

                # contour + hull
                cv2.drawContours(frame, [cnt], -1, ACCENTS["hand"], 1, cv2.LINE_AA)
                cv2.drawContours(frame, [hull], -1, ACCENTS["hand"], 1, cv2.LINE_AA)

                # convexity defects → finger counting
                hull_idx = cv2.convexHull(cnt, returnPoints=False)
                if hull_idx.shape[0] > 3:
                    defects = cv2.convexityDefects(cnt, hull_idx)
                    fingers = 0
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            _, _, fd, _ = defects[i, 0]
                            if fd > 15000:            # depth threshold
                                fingers += 1
                    # actual fingers = peaks + 1 (thumb heuristic)
                    n_fingers = min(fingers + 1, 5)
                    label = f"{n_fingers}"
                    lx, ly = rx + rw // 2 - 6, ry - 8
                    cv2.putText(frame, label, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                ACCENTS["hand"], 2, cv2.LINE_AA)

                    # finger count coloured bar
                    bar_x, bar_y = rx + rw + 4, ry
                    bar_h = rh
                    bar_w = 10
                    for fi in range(5):
                        by = bar_y + int(bar_h * (4 - fi) / 5)
                        end = bar_y + int(bar_h * (5 - fi) / 5)
                        col = _BGR_GREEN if fi < n_fingers else _BGR_DARK
                        cv2.rectangle(frame, (bar_x, by),
                                      (bar_x + bar_w, end), col, -1)

                # bounding box
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                              ACCENTS["hand"], 1, cv2.LINE_AA)

            if hand_count:
                parts.append(f"🤚 {hand_count} hand(s)")

        # ── Full / Upper body ──────────────────────────────────────────
        if self.features["body"].get():
            bodies = self.classifiers["fullbody"].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 200)
            )
            for (bx, by, bw, bh) in bodies:
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                              ACCENTS["body"], 2, cv2.LINE_AA)
            if isinstance(bodies, (list, tuple)):
                pass  # no bodies
            elif bodies.any():
                parts.append(f"🧍 {len(bodies)} body")

            # also upper body
            uppers = self.classifiers["upperbody"].detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 80)
            )
            for (ux, uy, uw, uh) in uppers:
                cv2.rectangle(frame, (ux, uy), (ux + uw, uy + uh),
                              ACCENTS["body"], 1, cv2.LINE_AA)
            if isinstance(uppers, (list, tuple)):
                pass  # no upper bodies
            elif uppers.any() and not (isinstance(bodies, (list, tuple)) or bodies.any()):
                parts.append(f"🧍 {len(uppers)} upper")

        # ── Edge (Canny) ───────────────────────────────────────────────
        if self.features["edge"].get():
            edges = cv2.Canny(gray, 80, 160)
            edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_col[:, :, 0] = 0   # keep only green channel tint
            edges_col[:, :, 2] = 0
            frame = cv2.addWeighted(frame, 0.7, edges_col, 0.3, 0)
            parts.append("⚡ Edge")

        # ── FPS meta ───────────────────────────────────────────────────
        label = " | ".join(parts) if parts else "No detections"
        return frame, label

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
            cid = self.available_cameras[idx]
            self.status_lbl.configure(text=f"Ready — camera {cid}")

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
