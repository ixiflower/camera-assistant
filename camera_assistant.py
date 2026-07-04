#!/usr/bin/env python3
"""
Camera Assistant — Tkinter GUI with real-time CV:
  • Face detection       • Eye detection
  • Hand tracking (MP)   • Finger counting
  • Full body detect     • Smile detection
  • Edge / Motion        • Gesture recognition

Powered by MediaPipe Hands (21 landmarks per hand) and OpenCV Haar cascades.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
from typing import Optional

# ── MediaPipe (0.10.x task API) ─────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base

# Hand landmark connections for custom skeleton drawing
mp_hand_connections = mp_vision.HandLandmarksConnections


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
            "hand":     tk.BooleanVar(value=True),   # MediaPipe Hands
            "body":     tk.BooleanVar(value=False),
            "edge":     tk.BooleanVar(value=False),    # Canny edge
        }

        # ── classifiers ────────────────────────────────────────────────
        self.classifiers = {}
        for name, path in CASCADES.items():
            self.classifiers[name] = cv2.CascadeClassifier(path)

        # ── MediaPipe Hands (0.10.x task API) ───────────────────────────
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if os.path.exists(model_path):
            self.mp_hand_detector = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_base.BaseOptions(model_asset_path=model_path),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                ),
            )
            self.mp_frame_ts = 0
        else:
            self.mp_hand_detector = None
            print(f"⚠️  Hand model not found at {model_path} — hand detection disabled")

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

    MP_HAND_IDX = {
        "THUMB_TIP": 4, "THUMB_IP": 3,
        "INDEX_FINGER_TIP": 8, "INDEX_FINGER_PIP": 6,
        "MIDDLE_FINGER_TIP": 12, "MIDDLE_FINGER_PIP": 10,
        "RING_FINGER_TIP": 16, "RING_FINGER_PIP": 14,
        "PINKY_TIP": 20, "PINKY_PIP": 18,
        "WRIST": 0,
    }

    def _count_fingers(self, landmarks: list, handedness_label: str) -> int:
        """Count extended fingers using 21-landmark positions.

        Each landmark has .x, .y, .z (normalized 0-1).
        Fingers are extended when tip.y < pip.y (above the PIP joint).
        Thumb uses x-axis comparison (handedness-aware).
        """
        idx = self.MP_HAND_IDX
        fingers = []

        # ── Thumb: compare tip.x vs IP.x (handedness-aware) ────────────
        thumb_tip = landmarks[idx["THUMB_TIP"]]
        thumb_ip  = landmarks[idx["THUMB_IP"]]
        if handedness_label == "Right":
            fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)
        else:
            fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)

        # ── Other 4 fingers: tip.y < pip.y → extended ──────────────────
        for tip_k, pip_k in [
            ("INDEX_FINGER_TIP", "INDEX_FINGER_PIP"),
            ("MIDDLE_FINGER_TIP", "MIDDLE_FINGER_PIP"),
            ("RING_FINGER_TIP", "RING_FINGER_PIP"),
            ("PINKY_TIP", "PINKY_PIP"),
        ]:
            fingers.append(
                1 if landmarks[idx[tip_k]].y < landmarks[idx[pip_k]].y else 0
            )

        return sum(fingers)

    def _process_hands(self, frame: cv2.Mat, parts: list[str]) -> None:
        """MediaPipe 0.10.x task-API hand detection with landmark skeleton."""
        if self.mp_hand_detector is None:
            return

        self.mp_frame_ts += 1
        h, w, _ = frame.shape

        # Convert BGR → RGB → mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection
        result = self.mp_hand_detector.detect_for_video(
            mp_img, timestamp_ms=self.mp_frame_ts,
        )

        if not result.hand_landmarks:
            return

        # Pair landmarks with handedness
        hands_meta: list[tuple[list, str]] = []
        for i, hand_lms in enumerate(result.hand_landmarks):
            label = "Right"
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
            hands_meta.append((hand_lms, label))

        # ── Draw skeleton connections (blue lines) ─────────────────────
        for hand_lms, _ in hands_meta:
            for conn in mp_hand_connections.HAND_CONNECTIONS:
                s_lm = hand_lms[conn.start]
                e_lm = hand_lms[conn.end]
                sx, sy = int(s_lm.x * w), int(s_lm.y * h)
                ex, ey = int(e_lm.x * w), int(e_lm.y * h)
                cv2.line(frame, (sx, sy), (ex, ey),
                         _bgr("#89b4fa"), 1, cv2.LINE_AA)

            # Landmark dots (green filled + outline)
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, _bgr("#a6e3a1"), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, _bgr("#a6e3a1"), 1, cv2.LINE_AA)

        # ── Per-hand info banner ───────────────────────────────────────
        for hand_lms, label in hands_meta:
            n_fingers = self._count_fingers(hand_lms, label)

            wrist = hand_lms[self.MP_HAND_IDX["WRIST"]]
            wx, wy = int(wrist.x * w), int(wrist.y * h)

            icon = "👈" if label == "Left" else "👉"
            banner = f"{icon} {label}  {n_fingers}/5"
            (bw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.65, 2)
            bx = max(0, min(wx - bw // 2, w - bw))
            by = max(24, wy - 40)

            # Background pill
            cv2.rectangle(frame, (bx - 6, by - 22), (bx + bw + 6, by + 6),
                          (17, 17, 27, 180), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx - 6, by - 22), (bx + bw + 6, by + 6),
                          _bgr("#2a2a3e"), 1, cv2.LINE_AA)
            cv2.putText(frame, banner, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        _bgr("#a6e3a1"), 2, cv2.LINE_AA)

            # Finger state bar
            bar_x = max(0, min(wx - 30, w - 60))
            bar_y = max(24, wy - 10)
            seg_h, seg_w, gap = 8, 14, 3
            for fi in range(5):
                sx2 = bar_x + fi * (seg_w + gap)
                clr = _bgr("#a6e3a1") if fi < n_fingers else _bgr("#45475a")
                cv2.rectangle(frame, (sx2, bar_y), (sx2 + seg_w, bar_y + seg_h),
                              clr, -1, cv2.LINE_AA)
                cv2.rectangle(frame, (sx2, bar_y), (sx2 + seg_w, bar_y + seg_h),
                              _bgr("#585b70"), 1, cv2.LINE_AA)

        parts.append(f"🤚 {len(hands_meta)} hand(s)")

    def _process(self, frame: cv2.Mat) -> tuple[cv2.Mat, str]:
        """Apply enabled CV detections. Returns (annotated_frame, info_line)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

        # ── Hand / Finger detection via MediaPipe ──────────────────────
        if self.features["hand"].get():
            self._process_hands(frame, parts)

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
