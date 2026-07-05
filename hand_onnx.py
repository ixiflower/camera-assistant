"""ONNX hand detection — MediaPipe palm-detection + hand-landmark models.

Provides ONNX-based hand detection compatible with Python 3.14+.
No MediaPipe dependency — uses onnxruntime with pre-converted ONNX models.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

LANDMARK_INDICES = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}

# Finger tip → PIP (second joint) for "extended" check
FINGER_PAIRS: List[Tuple[int, int, str]] = [
    (4, 3, "thumb"),
    (8, 6, "index"),
    (12, 10, "middle"),
    (16, 14, "ring"),
    (20, 18, "pinky"),
]

# Landmark connection pairs for skeleton drawing
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def _keep_aspect_resize_and_pad(
    image: np.ndarray,
    resize_w: int,
    resize_h: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Resize to fit within (resize_w, resize_h) keeping aspect, pad to exact size.

    Returns: (padded_image, resized_image, pad_ratio_w, pad_ratio_h)
      pad_ratio = half_pad_size / target_size  (for coordinate conversion)
    """
    h, w = image.shape[:2]
    scale = min(resize_w / w, resize_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
    pad_x = (resize_w - new_w) // 2
    pad_y = (resize_h - new_h) // 2
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    pad_ratio_w = pad_x / resize_w   # fraction of padding on left
    pad_ratio_h = pad_y / resize_h   # fraction of padding on top
    return padded, resized, pad_ratio_w, pad_ratio_h


def _rotate_and_crop(
    image: np.ndarray,
    cx: float, cy: float, size: float, angle_deg: float,
    output_size: int = 224,
) -> np.ndarray:
    """Extract a rotated hand region, output_size × output_size."""
    # Rotate the entire image around (cx, cy) by -angle_deg
    # so the hand becomes upright, then crop a square of side `size`.
    rot = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    # We need the output to be exactly output_size × output_size
    # containing the hand region. The hand occupies a square of side `size`
    # in the original image.
    # Scale factor to map `size` to `output_size`
    if size < 1:
        size = 1
    scale = output_size / size
    rot2 = cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale)
    # Shift so (cx, cy) maps to the centre of the output image
    rot2[0, 2] += output_size / 2 - cx
    rot2[1, 2] += output_size / 2 - cy
    cropped = cv2.warpAffine(
        image, rot2, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return cropped


# ── Hand Detection Result ────────────────────────────────────────────────────

class HandResult:
    """Result of hand detection for a single hand."""

    __slots__ = ("landmarks", "score", "is_right_hand", "bbox")

    def __init__(
        self,
        landmarks: np.ndarray,   # (21, 2) in image coordinates
        score: float,
        is_right_hand: bool,
        bbox: Tuple[int, int, int, int],  # (x, y, w, h)
    ):
        self.landmarks = landmarks
        self.score = score
        self.is_right_hand = is_right_hand
        self.bbox = bbox

    def count_fingers(self) -> int:
        """Count extended fingers using tip-to-PIP heuristics.

        Thumb: compare distance from tip to wrist vs knuckle to wrist.
        Other fingers: tip y < pip y (in landmark-normalised orientation).
        """
        lm = self.landmarks
        count = 0

        # Thumb: check if thumb tip is far from palm centre (index MCP)
        wrist = lm[0]
        index_mcp = lm[LANDMARK_INDICES["index_mcp"]]
        thumb_tip = lm[LANDMARK_INDICES["thumb_tip"]]
        thumb_ip = lm[LANDMARK_INDICES["thumb_ip"]]

        # Distance from thumb tip to index MCP vs thumb IP to index MCP
        d_tip = np.linalg.norm(thumb_tip - index_mcp)
        d_ip = np.linalg.norm(thumb_ip - index_mcp)
        if d_tip > d_ip * 1.1:
            count += 1

        # Other four fingers: compare tip y to pip y
        for tip_idx, pip_idx, _ in FINGER_PAIRS[1:]:
            tip = lm[tip_idx]
            pip = lm[pip_idx]
            if tip[1] < pip[1]:   # tip is above pip → extended
                count += 1

        return min(count, 5)

    def finger_states(self) -> List[bool]:
        """Returns [thumb, index, middle, ring, pinky] as booleans."""
        return [
            self.is_finger_extended("thumb"),
            self.is_finger_extended("index"),
            self.is_finger_extended("middle"),
            self.is_finger_extended("ring"),
            self.is_finger_extended("pinky"),
        ]

    def is_finger_extended(self, name: str) -> bool:
        if name == "thumb":
            wrist = self.landmarks[0]
            index_mcp = self.landmarks[LANDMARK_INDICES["index_mcp"]]
            tip = self.landmarks[LANDMARK_INDICES["thumb_tip"]]
            ip = self.landmarks[LANDMARK_INDICES["thumb_ip"]]
            d_tip = np.linalg.norm(tip - index_mcp)
            d_ip = np.linalg.norm(ip - index_mcp)
            return d_tip > d_ip * 1.1
        else:
            tip_idx, pip_idx, _ = [(t, p, n) for t, p, n in FINGER_PAIRS
                                   if n == name][0]
            return self.landmarks[tip_idx][1] < self.landmarks[pip_idx][1]

    def bounding_box(self) -> Tuple[int, int, int, int]:
        return self.bbox


# ── Detector ─────────────────────────────────────────────────────────────────

class ONNXHandDetector:
    """Two-stage ONNX hand detector (palm detection → hand landmark).

    Parameters
    ----------
    palm_model: str
        Path to palm detection ONNX model.
    landmark_model: str
        Path to hand landmark ONNX model.
    score_threshold: float
        Minimum detection score (default 0.6).
    """

    PALM_INPUT_SIZE = 192
    LANDMARK_INPUT_SIZE = 224

    def __init__(
        self,
        palm_model: str = "models/palm_detection_full_inf_post_192x192.onnx",
        landmark_model: str = "models/hand_landmark_sparse_Nx3x224x224.onnx",
        score_threshold: float = 0.6,
        landmark_threshold: float = 0.3,
    ):
        import onnxruntime
        self.score_threshold = score_threshold
        self.landmark_threshold = landmark_threshold

        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        self.palm_session = onnxruntime.InferenceSession(
            palm_model, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.landmark_session = onnxruntime.InferenceSession(
            landmark_model, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        self._palm_input_name = self.palm_session.get_inputs()[0].name
        self._palm_output_name = self.palm_session.get_outputs()[0].name

        self._lm_input_name = self.landmark_session.get_inputs()[0].name

    # ══════════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════════

    def detect(self, image: np.ndarray) -> List[HandResult]:
        """Run full hand detection pipeline on an image.

        Returns list of ``HandResult`` objects (one per hand), sorted by score
        descending. Empty list when no hands found.
        """
        img_h, img_w = image.shape[:2]

        # ── Stage 1: Palm detection ─────────────────────────────────────
        palm_boxes = self._detect_palms(image)  # (cx, cy, size, angle, pd_score)
        if len(palm_boxes) == 0:
            return []

        # Apply NMS to remove overlapping palm boxes
        keep = self._nms(palm_boxes, iou_threshold=0.5)
        palm_boxes = [palm_boxes[i] for i in keep]
        if len(palm_boxes) == 0:
            return []

        # Score on palm already checked — keep top 3 palms max
        palm_boxes = palm_boxes[:3]

        # ── Stage 2: Rotate & crop each hand ───────────────────────────
        rois, rects = [], []
        for box in palm_boxes:
            cx, cy, size, angle_deg, _ = box  # unpack, skip score
            cropped = _rotate_and_crop(
                image, cx, cy, size, angle_deg, self.LANDMARK_INPUT_SIZE
            )
            rois.append(cropped)
            rects.append((cx, cy, size, angle_deg))  # 4-elem for backward compat

        if not rois:
            return []

        # ── Stage 3: Hand landmark ─────────────────────────────────────
        lms_array, scores_array, handedness_array = self._detect_landmarks(rois)

        # ── Stage 4: Build results ─────────────────────────────────────
        results: List[HandResult] = []
        for i in range(len(rois)):
            if scores_array[i] < self.landmark_threshold:
                continue
            lm_21x2 = self._project_landmarks_to_image(
                lms_array[i], rects[i], img_w, img_h
            )
            cx, cy, size, _ = rects[i]
            x = int(max(0, cx - size / 2))
            y = int(max(0, cy - size / 2))
            bw = int(min(size, img_w - x))
            bh = int(min(size, img_h - y))

            result = HandResult(
                landmarks=lm_21x2,
                score=float(scores_array[i, 0]),
                is_right_hand=bool(handedness_array[i, 0] > 0.5),
                bbox=(x, y, bw, bh),
            )
            results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal — NMS
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _nms(
        boxes: List[Tuple[float, float, float, float, float]],
        iou_threshold: float = 0.5,
    ) -> List[int]:
        """Non-Maximum Suppression for palm boxes.

        Each box: (cx, cy, size, angle_deg, score).
        Returns list of kept indices sorted by score desc.
        """
        if not boxes:
            return []
        # Sort by score desc
        indices = list(range(len(boxes)))
        indices.sort(key=lambda i: boxes[i][4], reverse=True)

        keep: List[int] = []
        while indices:
            i = indices.pop(0)
            keep.append(i)
            # Compute IoU with remaining
            x1_i = boxes[i][0] - boxes[i][2] / 2
            y1_i = boxes[i][1] - boxes[i][2] / 2
            x2_i = boxes[i][0] + boxes[i][2] / 2
            y2_i = boxes[i][1] + boxes[i][2] / 2
            area_i = boxes[i][2] ** 2

            to_remove: List[int] = []
            for j in indices:
                x1_j = boxes[j][0] - boxes[j][2] / 2
                y1_j = boxes[j][1] - boxes[j][2] / 2
                x2_j = boxes[j][0] + boxes[j][2] / 2
                y2_j = boxes[j][1] + boxes[j][2] / 2
                area_j = boxes[j][2] ** 2

                inter_x1 = max(x1_i, x1_j)
                inter_y1 = max(y1_i, y1_j)
                inter_x2 = min(x2_i, x2_j)
                inter_y2 = min(y2_i, y2_j)
                inter = max(0, (inter_x2 - inter_x1)) * max(0, (inter_y2 - inter_y1))
                iou = inter / (area_i + area_j - inter + 1e-6)

                if iou > iou_threshold:
                    to_remove.append(j)

            for j in to_remove:
                indices.remove(j)

        return keep

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal — Palm detection
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_palms(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Run palm detection, return list of (cx, cy, size, angle_deg, score)."""
        img_h, img_w = image.shape[:2]

        # Pad to square and resize to 192×192
        max_side = max(img_h, img_w)
        pad_h = int((max_side - img_h) / 2)
        pad_w = int((max_side - img_w) / 2)

        padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        padded[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = image

        resized = cv2.resize(padded, (self.PALM_INPUT_SIZE, self.PALM_INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        # Convert BGR → RGB, normalise to [0, 1], CHW
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)    # → NCHW

        # Inference
        raw_output = self.palm_session.run(
            [self._palm_output_name],
            {self._palm_input_name: blob},
        )[0]  # (N, 8)

        if raw_output is None or len(raw_output) == 0:
            return []

        # Post-process
        boxes: List[Tuple[float, float, float, float]] = []
        for row in raw_output:
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp1_x, kp1_y = row
            if pd_score < self.score_threshold:
                continue
            if box_size <= 0:
                continue

            # Convert from 192×192 padded-norm space to original image space
            # box_x, box_y, box_size are in [0, 1] relative to 192 padding
            # kp0, kp1 are also in [0, 1] relative to 192 padding
            def denorm(v, max_side_v):
                return v * max_side_v

            b_x = denorm(box_x, max_side)
            b_y = denorm(box_y, max_side)
            b_size = denorm(box_size, max_side)
            k_x0 = denorm(kp0_x, max_side)
            k_y0 = denorm(kp0_y, max_side)
            k_x1 = denorm(kp1_x, max_side)
            k_y1 = denorm(kp1_y, max_side)

            # Compute rotation from keypoints (0-based → index finger base)
            dx = k_x1 - k_x0
            dy = k_y1 - k_y0
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            rotation = 0.5 * math.pi - math.atan2(-dy, dx)
            rotation = _normalize_radians(rotation)

            # Compute rotated rect center
            rr_size = 2.9 * b_size
            rr_cx = b_x + 0.5 * b_size * math.sin(rotation)
            rr_cy = b_y - 0.5 * b_size * math.cos(rotation)

            # Shift from padded space to original image space
            rr_cx_orig = rr_cx - pad_w
            rr_cy_orig = rr_cy - pad_h

            # Clamp
            rr_cx_orig = max(0, min(rr_cx_orig, img_w - 1))
            rr_cy_orig = max(0, min(rr_cy_orig, img_h - 1))
            rr_size = max(10, min(rr_size, max(img_w, img_h) * 2))

            boxes.append((rr_cx_orig, rr_cy_orig, rr_size, math.degrees(rotation), pd_score))

        boxes.sort(key=lambda x: x[4], reverse=True)
        return boxes

    # ══════════════════════════════════════════════════════════════════════════
    #  Internal — Hand landmark
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_landmarks(
        self, rois: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run hand landmark model on cropped ROIs.

        Returns: (xyz_x21, hand_scores, leftright) arrays.
        """
        if not rois:
            return np.empty((0, 63)), np.empty((0, 1)), np.empty((0, 1))

        # Preprocess each ROI: pad to square, resize to 224×224, normalise
        batch = []
        for roi in rois:
            h, w = roi.shape[:2]
            max_s = max(h, w)
            # Pad to square
            sq = np.zeros((max_s, max_s, 3), dtype=np.uint8)
            ph = (max_s - h) // 2
            pw = (max_s - w) // 2
            sq[ph:ph + h, pw:pw + w] = roi
            resized = cv2.resize(sq, (self.LANDMARK_INPUT_SIZE, self.LANDMARK_INPUT_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            # BGR→RGB, normalise to [0, 1], CHW
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            blob = rgb.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))
            batch.append(blob)

        inp = np.stack(batch, axis=0).astype(np.float32)

        out = self.landmark_session.run(None, {self._lm_input_name: inp})
        if not out:
            return np.empty((0, 63)), np.empty((0, 1)), np.empty((0, 1))

        xyz = out[0]  # (N, 63)
        scores = out[1]  # (N, 1)
        handedness = out[2]  # (N, 1)
        return xyz, scores, handedness

    def _project_landmarks_to_image(
        self,
        xyz_21: np.ndarray,
        rect: Tuple[float, float, float, float],
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Convert 63-float landmark output to (21, 2) in image coordinates.

        The model outputs xyz in normalized [-1, 1] relative to the rotated crop.
        We need to map them back using the inverse of the rotation + crop.
        """
        cx, cy, size, angle_deg = rect
        half = size / 2

        # The model outputs 21 (x, y, z) where x, y are in normalized [-1, 1]
        # covering the rotated crop area (side = size in original image).
        # z is depth (not used here).
        pts = xyz_21.reshape(21, 3)
        xy = pts[:, :2]  # (21, 2) in [-1, 1]

        # Denormalize to original image coordinates
        # First, convert from [-1, 1] → [0, size]
        xy = (xy + 1.0) * 0.5 * size  # (21, 2)

        # Now we have coordinates in the rotated-rect centred at (0, 0)
        # with the hand pointing up (rotation applied by palm detection).
        # We need to rotate back by -angle_deg and shift to (cx, cy).
        # Build rotation matrix
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rotate points: R * pt + centre
        rotated = np.empty_like(xy)
        rotated[:, 0] = xy[:, 0] * cos_a - xy[:, 1] * sin_a + cx
        rotated[:, 1] = xy[:, 0] * sin_a + xy[:, 1] * cos_a + cy

        return rotated.astype(np.int32)
