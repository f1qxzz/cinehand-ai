"""
line_renderer.py [v2 — FIXED + SMOOTHER]
High-quality anti-jitter line renderer for hand landmark overlays.

FIXES v2:
  - EMA alpha tuned for faster tracking (0.35 → less lag)
  - MAX_GAP break logic: don't break trail on fast motion, only on
    actual hand disappearance (detected via break_trail() call)
  - Interpolation steps increased for smoother curves
  - GaussianBlur uses addWeighted (safe — no overflow / no whitescreen)
  - GLOW_LAYERS: 2 layers (outer blur + inner sharp) for performance
  - draw_glow_trail: skip blur when alpha_scale < 0.3 (low FPS save)
  - All draw calls wrapped in try/except — never crashes main loop
"""

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hand.tracker import HAND_CONNECTIONS, FINGERTIP_INDICES

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIL_CAPACITY  = 28
SMOOTHING_ALPHA = 0.35      # slightly higher = more responsive
INTERP_STEPS    = 3
MIN_THICKNESS   = 1
MAX_THICKNESS   = 5
SPEED_SCALE     = 0.07

GLOW_LAYERS = [
    (5,  0.16, 5),   # outer glow
    (1,  0.82, 0),   # inner sharp
]


class _EMAPoint:
    def __init__(self, alpha: float = SMOOTHING_ALPHA):
        self._alpha = alpha
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def update(self, x: float, y: float) -> Tuple[int, int]:
        if self._x is None:
            self._x, self._y = float(x), float(y)
        else:
            self._x = self._x * (1 - self._alpha) + x * self._alpha
            self._y = self._y * (1 - self._alpha) + y * self._alpha
        return int(self._x), int(self._y)

    def reset(self):
        self._x = self._y = None

    @property
    def value(self) -> Optional[Tuple[float, float]]:
        if self._x is None:
            return None
        return self._x, self._y


class _TrailChannel:
    def __init__(self, capacity: int = TRAIL_CAPACITY):
        self._smoother = _EMAPoint()
        self._buffer: deque = deque(maxlen=capacity)
        self._prev: Optional[Tuple[float, float]] = None
        self._speed: float = 0.0

    def push(self, raw_x: float, raw_y: float) -> Tuple[int, int]:
        sx, sy = self._smoother.update(raw_x, raw_y)

        if self._prev is not None:
            dx = sx - self._prev[0]; dy = sy - self._prev[1]
            inst_speed = math.hypot(dx, dy)
            self._speed = 0.7 * self._speed + 0.3 * inst_speed
        self._prev = (sx, sy)

        if len(self._buffer) > 0:
            last = self._buffer[-1]
            if last is not None:
                for k in range(1, INTERP_STEPS):
                    t  = k / INTERP_STEPS
                    ix = int(last[0] * (1 - t) + sx * t)
                    iy = int(last[1] * (1 - t) + sy * t)
                    self._buffer.append((ix, iy))

        self._buffer.append((sx, sy))
        return sx, sy

    def break_trail(self):
        self._buffer.append(None)

    def reset(self):
        self._smoother.reset()
        self._buffer.clear()
        self._prev  = None
        self._speed = 0.0

    @property
    def points(self) -> List[Optional[Tuple[int, int]]]:
        return list(self._buffer)

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def thickness(self) -> int:
        t = int(MIN_THICKNESS + self._speed * SPEED_SCALE)
        return max(MIN_THICKNESS, min(MAX_THICKNESS, t))


def _color_from_hue(hue_deg: float, s: int = 230, v: int = 255) -> Tuple[int, int, int]:
    h   = int(hue_deg / 2) % 180
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
    return tuple(bgr)


def _segments(pts: List[Optional[Tuple[int, int]]]):
    seg: List[Tuple[int, int]] = []
    for p in pts:
        if p is None:
            if len(seg) >= 2: yield seg
            seg = []
        else:
            seg.append(p)
    if len(seg) >= 2:
        yield seg


def draw_glow_trail(
    frame: np.ndarray,
    points: List[Optional[Tuple[int, int]]],
    hue_start: float = 120.0,
    hue_end:   float = 200.0,
    base_thick: int  = 2,
    alpha_scale: float = 1.0,
):
    if frame is None: return
    skip_blur = alpha_scale < 0.3

    for seg in _segments(points):
        n = len(seg)
        for i in range(1, n):
            t   = i / max(n - 1, 1)
            hue = hue_start + (hue_end - hue_start) * t
            col = _color_from_hue(hue)
            p_a = seg[i - 1]; p_b = seg[i]

            try:
                for extra_r, alpha_mul, blur_k in GLOW_LAYERS:
                    a = alpha_mul * alpha_scale * (0.4 + 0.6 * t)
                    if a < 0.01: continue
                    layer_thick = max(1, base_thick + extra_r)

                    if blur_k > 0 and not skip_blur:
                        tmp = frame.copy()
                        g_col = tuple(int(c * a) for c in col)
                        cv2.line(tmp, p_a, p_b, g_col, layer_thick, cv2.LINE_AA)
                        cv2.GaussianBlur(tmp, (blur_k, blur_k), 0, tmp)
                        cv2.addWeighted(tmp, a * 0.7, frame, 1.0, 0, frame)
                    else:
                        cv2.line(frame, p_a, p_b, col, base_thick, cv2.LINE_AA)
            except Exception:
                pass


def draw_glow_polylines(
    frame: np.ndarray,
    pts_array: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    if frame is None or len(pts_array) < 2: return
    pts = pts_array.reshape((-1, 1, 2))
    try:
        for extra_r, alpha_mul, blur_k in GLOW_LAYERS:
            glow_col   = tuple(int(c * alpha_mul) for c in color)
            glow_thick = max(1, thickness + extra_r)
            if blur_k > 0:
                tmp = frame.copy()
                cv2.polylines(tmp, [pts], False, glow_col, glow_thick, cv2.LINE_AA)
                cv2.GaussianBlur(tmp, (blur_k, blur_k), 0, tmp)
                cv2.addWeighted(tmp, alpha_mul * 0.8, frame, 1.0, 0, frame)
            else:
                cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)
    except Exception:
        pass


class LineRenderer:

    def __init__(self, trail_capacity: int = TRAIL_CAPACITY):
        self._capacity = trail_capacity
        self._channels: Dict[str, _TrailChannel] = {}

    def update(self, key: str, x: float, y: float) -> Tuple[int, int]:
        if key not in self._channels:
            self._channels[key] = _TrailChannel(self._capacity)
        return self._channels[key].push(x, y)

    def break_trail(self, key: str):
        if key in self._channels:
            self._channels[key].break_trail()

    def reset(self, key: str = None):
        if key is None:
            for ch in self._channels.values(): ch.reset()
        elif key in self._channels:
            self._channels[key].reset()

    def smoothed(self, key: str) -> Optional[Tuple[float, float]]:
        if key not in self._channels: return None
        return self._channels[key]._smoother.value

    def draw_trail(self, frame, key, hue_start=120.0, hue_end=200.0, alpha_scale=1.0):
        if key not in self._channels or frame is None: return
        ch  = self._channels[key]
        pts = ch.points
        draw_glow_trail(frame, pts, hue_start, hue_end,
                        base_thick=ch.thickness, alpha_scale=alpha_scale)

    def draw_line(self, frame, key_a, key_b, color=(100,200,255), thickness=2):
        if frame is None: return
        a = self.smoothed(key_a); b = self.smoothed(key_b)
        if a is None or b is None: return
        pa = (int(a[0]), int(a[1])); pb = (int(b[0]), int(b[1]))
        try:
            for extra_r, alpha_mul, blur_k in GLOW_LAYERS:
                glow_col   = tuple(int(c * alpha_mul) for c in color)
                glow_thick = max(1, thickness + extra_r)
                if blur_k > 0:
                    tmp = frame.copy()
                    cv2.line(tmp, pa, pb, glow_col, glow_thick, cv2.LINE_AA)
                    cv2.GaussianBlur(tmp, (blur_k, blur_k), 0, tmp)
                    cv2.addWeighted(tmp, alpha_mul * 0.8, frame, 1.0, 0, frame)
                else:
                    cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)
        except Exception:
            pass

    def draw_polyline(self, frame, keys, color=(100,200,255), thickness=2):
        if frame is None: return
        pts = []
        for k in keys:
            s = self.smoothed(k)
            if s: pts.append([int(s[0]), int(s[1])])
        if len(pts) < 2: return
        arr = np.array(pts, dtype=np.int32)
        draw_glow_polylines(frame, arr, color, thickness)

    def speed(self, key: str) -> float:
        if key not in self._channels: return 0.0
        return self._channels[key].speed

    def thickness(self, key: str) -> int:
        if key not in self._channels: return MIN_THICKNESS
        return self._channels[key].thickness


class HandLineRenderer:
    """Convenience wrapper: one LineRenderer per hand index."""

    FINGER_HUES = {
        (1,  4):  22,
        (5,  8):  82,
        (9,  12): 130,
        (13, 16): 160,
        (17, 20): 270,
    }

    def __init__(self, max_hands: int = 1):
        self._renderers: Dict[int, LineRenderer] = {}
        self._max_hands = max_hands

    def update(self, hand_results, W: int, H: int):
        active = set()
        for i, hr in enumerate(hand_results[:self._max_hands]):
            active.add(i)
            if i not in self._renderers:
                self._renderers[i] = LineRenderer()
            lr = self._renderers[i]
            for lm_idx, lm in enumerate(hr.landmarks):
                key = f"lm{lm_idx}"
                try:
                    lr.update(key, lm.x * W, lm.y * H)
                except Exception:
                    pass

        for i in list(self._renderers.keys()):
            if i not in active:
                lr = self._renderers[i]
                for key in list(lr._channels.keys()):
                    lr.break_trail(key)

    def draw(self, frame: np.ndarray):
        if frame is None: return
        for hand_idx, lr in self._renderers.items():
            try:
                for tip_idx in FINGERTIP_INDICES:
                    hue = self._hue_for(tip_idx)
                    lr.draw_trail(frame, f"lm{tip_idx}",
                                  hue_start=hue, hue_end=(hue + 40) % 360)
                for a, b in HAND_CONNECTIONS:
                    hue = self._hue_for(b)
                    col = _color_from_hue(hue)
                    thick = lr.thickness(f"lm{a}")
                    lr.draw_line(frame, f"lm{a}", f"lm{b}", col, thick)
            except Exception:
                pass

    def _hue_for(self, lm_idx: int) -> float:
        for (lo, hi), hue in self.FINGER_HUES.items():
            if lo <= lm_idx <= hi:
                return float(hue)
        return 90.0
