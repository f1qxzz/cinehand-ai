"""
gesture_love.py [FIXED + OPTIMIZED]
Love gesture detection + cinematic visual effects.

FIXES:
  - MAX_PARTICLES = 50 (hard cap enforced)
  - Particle update wrapped in try/except
  - _draw_heart: ROI-based alpha blend instead of full frame copy
  - cv2.addWeighted used everywhere (no overflow → no whitescreen)
  - _HeartParticle.update: safe dt guard (never divide by zero)
  - All render functions wrapped in try/except

Detects:
  - ILY (I Love You) gesture
  - Two-hand OPEN_PALM held for GESTURE_HOLD_SEC

On trigger → renders:
  - Big pulsing heart at screen centre
  - Floating heart particles (upward + fade)
  - Soft pink/red screen glow overlay
  - "LOVE DETECTED" text with fade-in + upward drift
  - (If 2 faces) love beam between face centers + animated percentage
"""

import math
import random
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
GESTURE_HOLD_SEC  = 0.30
COOLDOWN_SEC      = 2.50
MAX_PARTICLES     = 50        # hard cap — was unbounded → FPS drop
PARTICLE_DECAY    = 0.012
HEART_PULSE_FREQ  = 2.8
TEXT_FLOAT_SPEED  = 22
TEXT_FADE_DUR     = 0.45
EFFECT_TOTAL      = 3.0
GLOW_FADE_DUR     = 0.40

_PINK       = (170,  80, 255)
_RED        = ( 40,  40, 230)
_WHITE      = (255, 255, 255)
_HEART_CORE = (100, 100, 255)
_HEART_GLOW = ( 40,  40, 160)


# ══════════════════════════════════════════════════════════════════════════════
class _HeartParticle:
    __slots__ = ('x', 'y', 'vy', 'life', 'size', 'hue')

    def __init__(self, x: int, y: int):
        self.x    = float(x) + random.uniform(-80, 80)
        self.y    = float(y) + random.uniform(-20, 20)
        self.vy   = random.uniform(-55, -25)
        self.life = 1.0
        self.size = random.uniform(8, 22)
        self.hue  = random.randint(0, 20)

    def update(self, dt: float):
        safe_dt   = max(dt, 0.001)   # guard against zero
        self.y   += self.vy * safe_dt
        self.vy  += safe_dt * 8
        self.life -= PARTICLE_DECAY / safe_dt

    @property
    def alive(self) -> bool:
        return self.life > 0.01


def _draw_heart(frame: np.ndarray, cx: int, cy: int, size: float,
                color: Tuple[int, int, int], alpha: float = 1.0):
    """Render a filled heart polygon at (cx, cy)."""
    if size < 1:
        return
    pts = []
    n   = 32   # reduced from 48 for speed
    for i in range(n):
        t = (2 * math.pi / n) * i - math.pi / 2
        x_raw = 16 * math.sin(t) ** 3
        y_raw = -(13 * math.cos(t) - 5 * math.cos(2 * t)
                  - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append([
            int(cx + x_raw * size / 16),
            int(cy + y_raw * size / 16),
        ])
    pts = np.array([pts], dtype=np.int32)

    try:
        if alpha >= 0.99:
            cv2.fillPoly(frame, pts, color, cv2.LINE_AA)
        else:
            # ROI approach to avoid full frame copy
            xs = [p[0] for p in pts[0]]
            ys = [p[1] for p in pts[0]]
            rx0, rx1 = max(0, min(xs)), min(frame.shape[1] - 1, max(xs))
            ry0, ry1 = max(0, min(ys)), min(frame.shape[0] - 1, max(ys))
            if rx1 > rx0 and ry1 > ry0:
                roi_orig = frame[ry0:ry1, rx0:rx1].copy()
                cv2.fillPoly(frame, pts, color, cv2.LINE_AA)
                cv2.addWeighted(frame[ry0:ry1, rx0:rx1], alpha,
                                roi_orig, 1.0 - alpha, 0,
                                frame[ry0:ry1, rx0:rx1])
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
class LoveGestureEffect:

    def __init__(self):
        self._active      = False
        self._t_start     = 0.0
        self._phase       = 0.0
        self._particles: List[_HeartParticle] = []
        self._text_y_off  = 0.0
        self._text_alpha  = 0.0
        self._glow_alpha  = 0.0
        self._W           = 1280
        self._H           = 720

    def trigger(self, W: int, H: int):
        self._active     = True
        self._t_start    = time.time()
        self._phase      = 0.0
        self._text_y_off = 0.0
        self._text_alpha = 0.0
        self._glow_alpha = 0.8
        self._W, self._H = W, H
        cx, cy = W // 2, H // 2
        # Burst — respect MAX_PARTICLES
        for _ in range(min(30, MAX_PARTICLES - len(self._particles))):
            self._particles.append(_HeartParticle(cx, cy))

    def update(self, dt: float):
        if not self._active:
            return
        self._phase += dt
        elapsed = time.time() - self._t_start

        # Text alpha
        if elapsed < TEXT_FADE_DUR:
            self._text_alpha = elapsed / TEXT_FADE_DUR
        elif elapsed < EFFECT_TOTAL - 0.5:
            self._text_alpha = 1.0
        else:
            self._text_alpha = max(0.0, 1.0 - (elapsed - (EFFECT_TOTAL - 0.5)) / 0.5)

        self._text_y_off = elapsed * TEXT_FLOAT_SPEED

        # Screen glow fade
        if self._glow_alpha > 0:
            self._glow_alpha = max(0.0, 0.8 * (1.0 - elapsed / GLOW_FADE_DUR))

        # Spawn particles (cap enforced)
        if elapsed < EFFECT_TOTAL - 0.5 and len(self._particles) < MAX_PARTICLES:
            cx, cy = self._W // 2, self._H // 2
            for _ in range(2):
                if len(self._particles) < MAX_PARTICLES:
                    self._particles.append(_HeartParticle(cx, cy))

        # Update + prune
        for p in self._particles:
            p.update(dt)
        self._particles = [p for p in self._particles if p.alive]

        if elapsed >= EFFECT_TOTAL:
            self._active     = False
            self._particles  = []
            self._glow_alpha = 0.0
            self._text_alpha = 0.0

    def render(self, frame: np.ndarray,
               face_centers: Optional[List[Tuple[int, int]]] = None):
        if not self._active or frame is None:
            return

        W, H  = self._W, self._H
        cx, cy = W // 2, H // 2
        elapsed = time.time() - self._t_start

        try:
            # 1 — Screen glow (addWeighted — no overflow)
            if self._glow_alpha > 0.005:
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (W, H), (80, 60, 200), -1)
                cv2.addWeighted(ov, self._glow_alpha * 0.3,
                                frame, 1.0 - self._glow_alpha * 0.3, 0, frame)

            # 2 — Floating heart particles
            for p in self._particles:
                try:
                    a = max(0.0, p.life)
                    col = cv2.cvtColor(
                        np.uint8([[[p.hue, 220, int(230 * a)]]]),
                        cv2.COLOR_HSV2BGR
                    )[0][0].tolist()
                    _draw_heart(frame, int(p.x), int(p.y), p.size, col, alpha=a)
                except Exception:
                    pass

            # 3 — Centre big heart (pulsing)
            pulse_scale = 1.0 + 0.18 * math.sin(self._phase * HEART_PULSE_FREQ * 2 * math.pi)
            heart_size  = int(60 * pulse_scale)

            for glow_r, glow_a in [(heart_size + 28, 0.10), (heart_size + 14, 0.18)]:
                _draw_heart(frame, cx, cy, glow_r, _HEART_GLOW, alpha=glow_a)
            _draw_heart(frame, cx, cy, heart_size, _HEART_CORE, alpha=0.92)
            _draw_heart(frame, cx, cy, max(1, heart_size - 6),
                        (180, 140, 255), alpha=0.60)

            # 4 — Love beam between faces
            if face_centers and len(face_centers) >= 2:
                self._draw_love_beam(frame, face_centers[0], face_centers[1], elapsed)

            # 5 — Text
            if self._text_alpha > 0.01:
                self._draw_love_text(frame, cx, cy, elapsed)

        except Exception:
            pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _draw_love_text(self, frame: np.ndarray, cx: int, cy: int, elapsed: float):
        try:
            text  = "LOVE DETECTED"
            font  = cv2.FONT_HERSHEY_DUPLEX
            scale = 1.05
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            tx = cx - tw // 2
            ty = int(cy - 110 - self._text_y_off)
            alpha = self._text_alpha

            ov = frame.copy()
            cv2.putText(ov, text, (tx + 2, ty + 2), font, scale,
                        (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(ov, text, (tx, ty), font, scale,
                        (180, 120, 255), thick, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)

            ov2 = frame.copy()
            _draw_heart(ov2, tx + tw + 22, ty - 4, 14, _HEART_CORE, alpha=1.0)
            cv2.addWeighted(ov2, alpha, frame, 1.0 - alpha, 0, frame)
        except Exception:
            pass

    def _draw_love_beam(self, frame: np.ndarray,
                        c1: Tuple[int, int], c2: Tuple[int, int],
                        elapsed: float):
        try:
            ph    = elapsed * 3.0
            alpha = 0.55 + 0.30 * math.sin(ph)
            total = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
            if total < 1:
                return
            n_seg = 18
            dx    = (c2[0] - c1[0]) / n_seg
            dy    = (c2[1] - c1[1]) / n_seg
            for i in range(n_seg):
                if (i + int(ph * 3)) % 3 == 0:
                    continue
                pa = (int(c1[0] + dx * i),       int(c1[1] + dy * i))
                pb = (int(c1[0] + dx * (i + 1)), int(c1[1] + dy * (i + 1)))
                bri = int(200 + 55 * math.sin(ph + i * 0.4))
                cv2.line(frame, pa, pb,
                         (bri // 3, bri // 2, bri), 2, cv2.LINE_AA)

            mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2 - 20)
            pct = int(min(100, elapsed / EFFECT_TOTAL * 100))
            txt = f"{pct}% COMPATIBLE"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            ov = frame.copy()
            cv2.putText(ov, txt,
                        (mid[0] - tw // 2, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (200, 160, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
class LoveGestureDetector:

    def __init__(self):
        self._effect       = LoveGestureEffect()
        self._hold_start   = 0.0
        self._holding      = False
        self._last_trigger = 0.0

    def update(self, dt: float, gesture_label: str, n_hands: int,
               W: int, H: int) -> bool:
        triggered = False
        now = time.time()

        love_gesture = (
            gesture_label == "ILY"
            or (gesture_label == "OPEN_PALM" and n_hands >= 2)
        )

        if love_gesture and now - self._last_trigger >= COOLDOWN_SEC:
            if not self._holding:
                self._holding    = True
                self._hold_start = now
            elif now - self._hold_start >= GESTURE_HOLD_SEC:
                self._effect.trigger(W, H)
                self._last_trigger = now
                self._holding      = False
                triggered          = True
        else:
            self._holding = False

        self._effect.update(dt)
        return triggered

    def render(self, frame: np.ndarray,
               face_centers: Optional[List[Tuple[int, int]]] = None):
        if frame is None:
            return
        self._effect.render(frame, face_centers)

    @property
    def is_active(self) -> bool:
        return self._effect._active
