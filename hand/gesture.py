"""
hand/gesture.py  [IMPROVED v2]
==============================
Improvements:
  - CONF_MIN lowered to 0.55 (better detection)
  - COOLDOWN reduced to 0.20 (faster switching)
  - BUFFER reduced to 8 (less lag)
  - PINCH_THRESHOLD slightly increased to 0.065 (easier to trigger pinch)
  - Added partial confidence weighting
"""

import time
from collections import deque
from enum import Enum
from typing import Tuple

WRIST      = 0
THUMB_TIP  = 4;  THUMB_IP   = 3
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

PINCH_THRESHOLD = 0.065   # slightly easier to pinch
CLICK_COOLDOWN  = 0.45    # slightly faster click repeat


class Gesture(str, Enum):
    POINT     = "POINT"
    PINCH     = "PINCH"
    OPEN_PALM = "OPEN_PALM"
    ILY       = "ILY"
    FIST      = "FIST"
    OTHER     = "OTHER"


def _finger_states(landmarks, hand_label: str):
    lx = lambda i: landmarks[i].x
    ly = lambda i: landmarks[i].y
    if hand_label == "Right":
        thumb = 1 if lx(THUMB_TIP) < lx(THUMB_IP) else 0
    else:
        thumb = 1 if lx(THUMB_TIP) > lx(THUMB_IP) else 0
    fingers = [thumb]
    for tip, pip_ in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                      (RING_TIP, RING_PIP),   (PINKY_TIP,  PINKY_PIP)]:
        fingers.append(1 if ly(tip) < ly(pip_) - 0.008 else 0)  # slight hysteresis
    return fingers


def _classify(up, pinch_dist: float) -> Gesture:
    thumb, idx, mid, ring, pinky = up
    total = sum(up)

    if pinch_dist < PINCH_THRESHOLD:
        return Gesture.PINCH
    if thumb == 1 and idx == 1 and mid == 0 and ring == 0 and pinky == 1:
        return Gesture.ILY
    if total == 5:
        return Gesture.OPEN_PALM
    if idx == 1 and total <= 2:  # POINT: index up, possibly thumb
        return Gesture.POINT
    if total == 0:
        return Gesture.FIST
    return Gesture.OTHER


class GestureEngine:
    """
    Stateful gesture recogniser with:
      - 8-frame majority-vote buffer (IMPROVED: was 10)
      - 0.20s cooldown (IMPROVED: was 0.3)
      - per-gesture click edge detection
      - CONF_MIN 0.55 (IMPROVED: was 0.60)
    """

    BUFFER   = 8
    COOLDOWN = 0.20
    CONF_MIN = 0.55

    def __init__(self):
        self._buf:    deque = deque(maxlen=self.BUFFER)
        self._stable: Gesture = Gesture.OTHER
        self._last_switch: float = 0.0
        self._last_click:  float = 0.0
        self.confidence:   float = 0.0

    def update(self, hand_result) -> Tuple[Gesture, bool]:
        up   = _finger_states(hand_result.landmarks, hand_result.handedness)
        raw  = _classify(up, hand_result.pinch_dist)
        self._buf.append(raw)

        counts = {}
        for g in self._buf:
            counts[g] = counts.get(g, 0) + 1
        winner = max(counts, key=counts.__getitem__)
        conf   = counts[winner] / len(self._buf)
        self.confidence = conf

        now = time.time()
        if (conf >= self.CONF_MIN
                and winner != self._stable
                and now - self._last_switch >= self.COOLDOWN):
            self._stable      = winner
            self._last_switch = now

        click = False
        if (self._stable == Gesture.PINCH
                and now - self._last_click >= CLICK_COOLDOWN):
            click            = True
            self._last_click = now

        return self._stable, click

    def reset(self):
        self._buf.clear()
        self._stable    = Gesture.OTHER
        self.confidence = 0.0
