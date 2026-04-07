"""
hand/tracker.py
MediaPipe HandLandmarker wrapper.
Downloads model automatically, runs in VIDEO mode for low latency.
Supports up to 2 hands.
"""

import os
import urllib.request
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe landmark indices
WRIST       = 0
THUMB_TIP   = 4;  THUMB_IP   = 3
INDEX_TIP   = 8;  INDEX_PIP  = 6
MIDDLE_TIP  = 12; MIDDLE_PIP = 10
RING_TIP    = 16; RING_PIP   = 14
PINKY_TIP   = 20; PINKY_PIP  = 18

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
FINGERTIP_INDICES = {4, 8, 12, 16, 20}

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"


def _ensure_model():
    # Check multiple locations
    candidates = [
        MODEL_PATH,
        os.path.join(os.path.dirname(__file__), "..", MODEL_PATH),
        os.path.join(os.path.dirname(__file__), "..", "..", MODEL_PATH),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    # Download to current working directory
    print("[HandTracker] Downloading hand_landmarker.task (~8 MB)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[HandTracker] Download complete.")
    return os.path.abspath(MODEL_PATH)


@dataclass
class HandResult:
    landmarks:  object           # list of 21 NormalizedLandmark
    handedness: str              # "Left" | "Right"
    index_tip:  Tuple[int, int]  # pixel coords of index fingertip
    thumb_tip:  Tuple[int, int]
    pinch_dist: float            # normalised thumb↔index distance


class HandTracker:
    """
    Runs MediaPipe HandLandmarker and returns HandResult objects.
    Handles model download, VIDEO-mode timestamp management.
    Falls back to legacy mp.solutions.hands if Tasks API is unavailable.
    """

    def __init__(self, max_hands: int = 2, min_conf: float = 0.48):
        self._max_hands = max_hands
        self._backend   = None
        self._ts_ms     = 0   # monotonic timestamp for VIDEO mode
        self._try_tasks(min_conf)
        if self._backend is None:
            self._try_solutions(max_hands, min_conf)

    # ── Init ──────────────────────────────────────────────────────────────────

    def _try_tasks(self, conf: float):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision import (
                HandLandmarker, HandLandmarkerOptions, RunningMode,
            )
            from mediapipe import Image, ImageFormat

            model_path = _ensure_model()
            base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
            opts = HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=RunningMode.VIDEO,
                num_hands=self._max_hands,
                min_hand_detection_confidence=conf,
                min_hand_presence_confidence=conf,
                min_tracking_confidence=0.40,
            )
            self._detector    = HandLandmarker.create_from_options(opts)
            self._Image       = Image
            self._ImageFormat = ImageFormat
            self._backend     = "tasks"
            print("[HandTracker] Backend: MediaPipe Tasks (VIDEO)")
        except Exception as e:
            print(f"[HandTracker] Tasks unavailable: {e}")

    def _try_solutions(self, max_hands, conf):
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands.Hands(
                max_num_hands=max_hands,
                min_detection_confidence=conf,
                min_tracking_confidence=0.35,
            )
            self._backend = "solutions"
            print("[HandTracker] Backend: MediaPipe Solutions (legacy)")
        except Exception as e:
            print(f"[HandTracker] Solutions also unavailable: {e}")

    # ── Detect ────────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[HandResult]:
        if self._backend == "tasks":
            return self._detect_tasks(frame_bgr)
        if self._backend == "solutions":
            return self._detect_solutions(frame_bgr)
        return []

    def _detect_tasks(self, frame_bgr: np.ndarray) -> List[HandResult]:
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img  = self._Image(image_format=self._ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33   # ~30 FPS monotonic timestamp
        res  = self._detector.detect_for_video(img, self._ts_ms)
        return self._parse_tasks_results(res.hand_landmarks, res.handedness, w, h)

    def _detect_solutions(self, frame_bgr: np.ndarray) -> List[HandResult]:
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res  = self._mp_hands.process(rgb)
        if not res.multi_hand_landmarks:
            return []
        hands   = []
        lm_list = res.multi_hand_landmarks
        hd_list = res.multi_handedness or []
        for i, lm in enumerate(lm_list[:self._max_hands]):
            label = "Right"
            if i < len(hd_list):
                try:
                    label = hd_list[i].classification[0].label
                except Exception:
                    pass
            hands.append(self._build_result(lm.landmark, label, w, h))
        return hands

    # ── Parse helpers ─────────────────────────────────────────────────────────

    def _parse_tasks_results(self, lm_list, hd_list, w, h) -> List[HandResult]:
        """
        MediaPipe Tasks API (>= 0.10):
          res.hand_landmarks[i]  → list of 21 NormalizedLandmark
          res.handedness[i]      → list of Category
                                   Category has .category_name  (newer builds)
                                              or .label          (older builds)
        """
        results = []
        for i in range(min(len(lm_list), self._max_hands)):
            lm    = lm_list[i]
            label = "Right"
            if i < len(hd_list):
                label = self._extract_handedness_label(hd_list[i])
            results.append(self._build_result(lm, label, w, h))
        return results

    @staticmethod
    def _extract_handedness_label(hd_entry) -> str:
        """
        Robustly extract 'Left'/'Right' from whatever structure
        MediaPipe returns for a single hand's handedness.

        Known structures across MediaPipe versions:
          A) list[Category]  where Category has .category_name  (Tasks >= 0.10.3)
          B) list[Category]  where Category has .label           (Tasks 0.10.0-0.10.2)
          C) ClassificationList  with .classifications[0].label  (very old Tasks)
        """
        # Case A / B — it's a list
        if isinstance(hd_entry, list):
            if not hd_entry:
                return "Right"
            first = hd_entry[0]
            return (getattr(first, "category_name", None)
                    or getattr(first, "label", None)
                    or "Right")

        # Case C — object with .classifications
        cats = getattr(hd_entry, "classifications", None)
        if cats:
            return getattr(cats[0], "label", "Right")

        # Last resort: str coercion sometimes works
        s = str(hd_entry)
        if "Left" in s:
            return "Left"
        return "Right"

    def _build_result(self, landmarks, label: str, w: int, h: int) -> HandResult:
        def px(i):
            lm = landmarks[i]
            return int(lm.x * w), int(lm.y * h)

        idx_tip = px(INDEX_TIP)
        thm_tip = px(THUMB_TIP)
        dx = idx_tip[0] - thm_tip[0]
        dy = idx_tip[1] - thm_tip[1]
        pinch = (dx * dx + dy * dy) ** 0.5 / max(w, h)

        return HandResult(
            landmarks=landmarks,
            handedness=label,
            index_tip=idx_tip,
            thumb_tip=thm_tip,
            pinch_dist=pinch,
        )

    def release(self):
        try:
            if self._backend == "tasks":
                self._detector.close()
            elif self._backend == "solutions":
                self._mp_hands.close()
        except Exception:
            pass
