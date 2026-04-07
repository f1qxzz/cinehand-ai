"""
hand/fx.py
Cinematic hand effects: finger trail, fingertip particles,
pinch-click flash, ILY rainbow burst.
Optimised for 30–60 FPS:
  - particle pool capped at MAX_PARTICLES
  - heavy effects update every 2 frames
  - post-verification mode halves particle count
"""

import math
import random
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .tracker import (HAND_CONNECTIONS, FINGERTIP_INDICES,
                      INDEX_TIP, THUMB_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
                      WRIST)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_PARTICLES    = 600
TRAIL_LEN        = 36
GRAVITY          = 0.20
PINCH_FLASH_DUR  = 0.18    # seconds
ILY_COOLDOWN     = 2.0     # seconds
ILY_BURST        = 50


# ══════════════════════════════════════════════════════════════════════════════
class _Particle:
    __slots__ = ('x','y','vx','vy','life','decay','size','color')

    def __init__(self, x, y, color):
        self.x = float(x);  self.y = float(y)
        a      = random.uniform(0, 2*math.pi)
        spd    = random.uniform(2, 8)
        self.vx    = math.cos(a)*spd
        self.vy    = math.sin(a)*spd - random.uniform(0, 3)
        self.life  = 1.0
        self.decay = random.uniform(0.018, 0.04)
        self.size  = random.randint(3, 8)
        self.color = color

    def step(self):
        self.x  += self.vx;  self.y  += self.vy
        self.vy += GRAVITY
        self.vx *= 0.97
        self.life -= self.decay

    @property
    def alive(self):
        return self.life > 0.01


# ══════════════════════════════════════════════════════════════════════════════
class HandFX:
    """
    Manages all visual hand effects.
    Call .update(hand_results, frame) once per frame.
    Call .trigger_pinch(x, y) on click events.
    Call .trigger_ily(x, y)   on ILY gestures.
    """

    def __init__(self, low_particle_mode: bool = False):
        self._particles:  List[_Particle] = []
        self._trails:     dict            = {}   # hand_idx → deque
        self._frame_count = 0
        self._pinch_flash: dict           = {}   # hand_idx → expire_time
        self._last_ily    = 0.0
        self._low_mode    = low_particle_mode

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, hand_results, frame: np.ndarray,
               frame_w: int, frame_h: int):
        self._frame_count += 1
        heavy = (self._frame_count % 2 == 0)   # heavy effects every 2 frames

        # Draw skeleton + trails
        for i, hr in enumerate(hand_results):
            self._ensure_trail(i)
            self._draw_skeleton(frame, hr.landmarks, frame_w, frame_h)
            tip = self._lm_px(hr.landmarks[INDEX_TIP], frame_w, frame_h)
            self._trails[i].append(tip)
            self._draw_trail(frame, i)

            if heavy:
                self._emit_fingertips(hr.landmarks, frame_w, frame_h)

            # Pinch flash
            if i in self._pinch_flash and time.time() < self._pinch_flash[i]:
                self._draw_pinch_flash(frame, hr.thumb_tip, hr.index_tip)

        # Evict stale trails
        active = set(range(len(hand_results)))
        for idx in list(self._trails.keys()):
            if idx not in active:
                self._trails[idx].append(None)   # break trail

        if heavy:
            self._update_particles(frame)

    # ── Triggers ─────────────────────────────────────────────────────────────

    def trigger_pinch(self, hand_idx: int, x: int, y: int):
        self._pinch_flash[hand_idx] = time.time() + PINCH_FLASH_DUR
        count = ILY_BURST // 3 if self._low_mode else ILY_BURST // 2
        self._emit(x, y, count, hue_range=(90, 130))

    def trigger_ily(self, x: int, y: int):
        now = time.time()
        if now - self._last_ily < ILY_COOLDOWN:
            return False
        self._last_ily = now
        count = ILY_BURST // 2 if self._low_mode else ILY_BURST
        self._emit(x, y, count, hue_range=(0, 179))
        return True

    def set_low_mode(self, low: bool):
        self._low_mode = low

    # ── Internal ─────────────────────────────────────────────────────────────

    def _ensure_trail(self, idx):
        if idx not in self._trails:
            self._trails[idx] = deque(maxlen=TRAIL_LEN)

    @staticmethod
    def _lm_px(lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def _draw_skeleton(self, frame, landmarks, w, h):
        pts = [self._lm_px(landmarks[i], w, h) for i in range(21)]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (60, 60, 60), 1, cv2.LINE_AA)
        for idx in FINGERTIP_INDICES:
            cv2.circle(frame, pts[idx], 5, (200, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pts[idx], 7, (80, 100, 180), 1,  cv2.LINE_AA)

    def _draw_trail(self, frame, idx):
        pts  = list(self._trails[idx])
        n    = len(pts)
        if n < 2:
            return
        for i in range(1, n):
            if pts[i] is None or pts[i-1] is None:
                continue
            alpha = (i / n) ** 1.4
            hue   = int(120 * alpha) % 180
            bgr   = cv2.cvtColor(
                np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0].tolist()
            thick = max(1, int(3 * alpha))
            glow  = [c//3 for c in bgr]
            cv2.line(frame, pts[i-1], pts[i], glow,  thick+4, cv2.LINE_AA)
            cv2.line(frame, pts[i-1], pts[i], bgr,   thick,   cv2.LINE_AA)

    def _emit_fingertips(self, landmarks, w, h):
        # Finger particles disabled — removed for cleaner look and better FPS
        pass

    def _emit(self, x, y, count, hue_range=(0, 179)):
        for _ in range(count):
            if len(self._particles) >= MAX_PARTICLES:
                break
            hue = random.randint(*hue_range)
            bgr = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0].tolist()
            self._particles.append(_Particle(x, y, bgr))

    def _update_particles(self, frame):
        if not self._particles:
            return
        layer = np.zeros_like(frame)
        alive = []
        for p in self._particles:
            p.step()
            if p.alive:
                a    = max(0, p.life)
                col  = [int(c*a) for c in p.color]
                sz   = max(1, int(p.size * a))
                cx, cy = int(p.x), int(p.y)
                if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                    cv2.circle(layer, (cx,cy), sz+3, [c//3 for c in col], -1)
                    cv2.circle(layer, (cx,cy), sz,   col,                 -1)
                alive.append(p)
        self._particles = alive
        cv2.add(frame, layer, frame)

    def _draw_pinch_flash(self, frame, thumb_tip, index_tip):
        mid = ((thumb_tip[0]+index_tip[0])//2,
               (thumb_tip[1]+index_tip[1])//2)
        overlay = frame.copy()
        cv2.circle(overlay, mid, 30, (200, 230, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.circle(frame, mid, 12, (255, 255, 255), 2, cv2.LINE_AA)
