"""
face_state.py
Cinematic face state machine: SCANNING → VERIFYING → VERIFIED
All animation is dt-based (no frame-rate dependency).
Zero flicker — state transitions use eased interpolation and cooldowns.
"""

import math
import time
import cv2
import numpy as np
from enum import Enum, auto
from typing import Dict, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
SCAN_TO_VERIFY_DELAY = 0.8   # seconds in SCANNING before auto-advance
VERIFY_DURATION      = 1.2   # seconds for the VERIFYING ring to fill
VERIFIED_FADE_START  = 0.0   # opacity at verified transition
VERIFIED_FADE_END    = 1.0
VERIFIED_FADE_DUR    = 0.45  # seconds for circle fade-out
COOLDOWN_RESET       = 1.5   # min seconds before re-entering SCANNING for same id

# BGR palette
_CYAN  = (255, 230, 60)
_BLUE  = (255, 180, 40)
_GREEN = (80,  230, 80)
_GOLD  = (30,  200, 255)
_PINK  = (180, 100, 255)


# ══════════════════════════════════════════════════════════════════════════════
class FSMState(Enum):
    SCANNING  = auto()
    VERIFYING = auto()
    VERIFIED  = auto()


def _ease_in_out(t: float) -> float:
    """Smooth hermite easing: 3t² - 2t³"""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


# ══════════════════════════════════════════════════════════════════════════════
class FaceStateAnimator:
    """
    Per-face state machine + renderer.
    Manages SCANNING / VERIFYING / VERIFIED states with smooth dt-based animation.
    Call update(dt, is_verified) each frame, then draw(frame, cx, cy, rx, ry).
    """

    def __init__(self, track_id: int):
        self.track_id   = track_id
        self._state     = FSMState.SCANNING
        self._phase     = 0.0        # generic animation phase accumulator
        self._t_entered = time.time()

        # VERIFYING
        self._verify_progress = 0.0  # 0 → 1

        # VERIFIED circle fade
        self._circle_alpha = 1.0    # 1=fully visible, 0=gone
        self._bracket_alpha = 0.0   # 0=hidden, 1=fully shown

        # Pulse scale
        self._pulse = 1.0

        # track last verified push
        self._verified_locked = False

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, dt: float, is_verified: bool):
        """Advance animation by dt seconds. is_verified comes from IdentityCache."""
        self._phase += dt

        if self._state == FSMState.SCANNING:
            self._update_scanning(dt, is_verified)
        elif self._state == FSMState.VERIFYING:
            self._update_verifying(dt, is_verified)
        elif self._state == FSMState.VERIFIED:
            self._update_verified(dt)

    def draw(self, frame: np.ndarray, cx: int, cy: int, rx: int, ry: int):
        """Render state-specific aura around face center (cx,cy) with radii rx,ry."""
        if self._state == FSMState.SCANNING:
            self._draw_scanning(frame, cx, cy, rx, ry)
        elif self._state == FSMState.VERIFYING:
            self._draw_verifying(frame, cx, cy, rx, ry)
        elif self._state == FSMState.VERIFIED:
            self._draw_verified(frame, cx, cy, rx, ry)

    @property
    def state(self) -> FSMState:
        return self._state

    # ── State transitions ─────────────────────────────────────────────────────

    def _update_scanning(self, dt: float, is_verified: bool):
        elapsed = time.time() - self._t_entered
        if is_verified:
            self._transition(FSMState.VERIFIED)
        elif elapsed >= SCAN_TO_VERIFY_DELAY:
            self._transition(FSMState.VERIFYING)

    def _update_verifying(self, dt: float, is_verified: bool):
        if is_verified:
            self._verify_progress = 1.0
            self._transition(FSMState.VERIFIED)
            return
        self._verify_progress = min(
            1.0,
            self._verify_progress + dt / VERIFY_DURATION
        )
        if self._verify_progress >= 1.0:
            self._transition(FSMState.VERIFIED)

    def _update_verified(self, dt: float):
        # Fade circle out
        if self._circle_alpha > 0.0:
            self._circle_alpha = max(0.0, self._circle_alpha - dt / VERIFIED_FADE_DUR)
        # Fade brackets in
        if self._bracket_alpha < 1.0:
            self._bracket_alpha = min(1.0, self._bracket_alpha + dt / VERIFIED_FADE_DUR * 1.5)

    def _transition(self, new_state: FSMState):
        self._state     = new_state
        self._t_entered = time.time()
        if new_state == FSMState.SCANNING:
            self._verify_progress = 0.0
            self._circle_alpha    = 1.0
            self._bracket_alpha   = 0.0
        elif new_state == FSMState.VERIFYING:
            self._verify_progress = 0.0
        elif new_state == FSMState.VERIFIED:
            self._circle_alpha  = 1.0
            self._bracket_alpha = 0.0

    # ── Draw: SCANNING ────────────────────────────────────────────────────────

    def _draw_scanning(self, frame: np.ndarray, cx: int, cy: int, rx: int, ry: int):
        ph = self._phase
        ov = frame.copy()

        # Outer pulse ring
        pulse    = 1.0 + 0.06 * math.sin(ph * 3.5)
        prx, pry = int(rx * pulse), int(ry * pulse)
        bright   = int(160 + 80 * math.sin(ph * 2.5))
        cv2.ellipse(ov, (cx, cy), (prx, pry), 0, 0, 360,
                    (bright // 3, bright, bright), 1, cv2.LINE_AA)

        # Rotating dashed arc (16 segments, skip every 4th)
        n_seg = 20
        rot   = (ph * 72) % 360   # 72°/sec
        for i in range(n_seg):
            if i % 5 == 0:
                continue
            a1 = rot + 360 / n_seg * i
            a2 = rot + 360 / n_seg * (i + 1)
            bri = 0.55 + 0.35 * math.sin(ph * 3.0 + i * 0.4)
            color = (int(60 * bri), int(220 * bri), int(255 * bri))
            cv2.ellipse(ov, (cx, cy), (rx, ry), 0,
                        a1 % 360, a2 % 360, color, 2, cv2.LINE_AA)

        # Inner dot pulse
        inner = max(4, rx // 3)
        cv2.circle(ov, (cx, cy), inner,
                   (40, 180, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    # ── Draw: VERIFYING ───────────────────────────────────────────────────────

    def _draw_verifying(self, frame: np.ndarray, cx: int, cy: int, rx: int, ry: int):
        ph   = self._phase
        prog = _ease_in_out(self._verify_progress)
        ov   = frame.copy()

        # Background ring (dim)
        cv2.ellipse(ov, (cx, cy), (rx, ry), 0, 0, 360,
                    (20, 60, 80), 2, cv2.LINE_AA)

        # Progress arc (0 → 360°)
        end_angle = int(prog * 360)
        if end_angle > 0:
            bri = int(180 + 70 * math.sin(ph * 4))
            # Glow (thick, dim)
            cv2.ellipse(ov, (cx, cy), (rx + 4, ry + 4),
                        -90, 0, end_angle,
                        (bri // 6, bri // 2, bri // 2), 10, cv2.LINE_AA)
            # Core arc
            cv2.ellipse(ov, (cx, cy), (rx, ry),
                        -90, 0, end_angle,
                        (bri // 3, bri, bri), 3, cv2.LINE_AA)

        # Percentage text
        pct = int(prog * 100)
        txt = f"{pct}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(ov, txt,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 220, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(ov, 0.90, frame, 0.10, 0, frame)

    # ── Draw: VERIFIED ────────────────────────────────────────────────────────

    def _draw_verified(self, frame: np.ndarray, cx: int, cy: int, rx: int, ry: int):
        ph = self._phase

        # Fading circle
        if self._circle_alpha > 0.01:
            ov = frame.copy()
            bri = int(255 * self._circle_alpha)
            pulse = 1.0 + 0.04 * math.sin(ph * 4)
            prx, pry = int(rx * pulse), int(ry * pulse)

            # Outer glow
            cv2.ellipse(ov, (cx, cy), (prx + 8, pry + 8), 0, 0, 360,
                        (bri // 8, bri // 3, bri // 3), 14, cv2.LINE_AA)
            # Core ring
            cv2.ellipse(ov, (cx, cy), (prx, pry), 0, 0, 360,
                        (bri // 4, bri, bri), 3, cv2.LINE_AA)

            cv2.addWeighted(ov,
                            self._circle_alpha * 0.9,
                            frame,
                            1.0 - self._circle_alpha * 0.9,
                            0, frame)

        # Fading-in corner brackets
        if self._bracket_alpha > 0.01:
            self._draw_brackets(frame, cx, cy, rx, ry, self._bracket_alpha, ph)

    def _draw_brackets(self, frame, cx, cy, rx, ry, alpha, ph):
        """Minimal corner brackets — Apple Face ID style."""
        bx, by   = int(rx * 0.80), int(ry * 0.80)
        blen     = max(12, rx // 3)
        thick    = 2
        bri      = int(220 * alpha)
        pulse    = 0.85 + 0.15 * math.sin(ph * 3.0)
        col      = (int(bri * pulse * 0.3), int(bri * pulse), int(bri * pulse))

        corners = [
            (cx - bx, cy - by),  # TL
            (cx + bx, cy - by),  # TR
            (cx + bx, cy + by),  # BR
            (cx - bx, cy + by),  # BL
        ]
        dirs = [
            (( 1, 0), ( 0,  1)),  # TL → right & down
            ((-1, 0), ( 0,  1)),  # TR → left  & down
            ((-1, 0), ( 0, -1)),  # BR → left  & up
            (( 1, 0), ( 0, -1)),  # BL → right & up
        ]
        ov = frame.copy()
        for (px, py), ((dx, dy), (ex, ey)) in zip(corners, dirs):
            h_end = (px + dx * blen, py + dy * 0)
            v_end = (px + ex * 0,   py + ey * blen)
            cv2.line(ov, (px, py), h_end, col, thick, cv2.LINE_AA)
            cv2.line(ov, (px, py), v_end, col, thick, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)


# ══════════════════════════════════════════════════════════════════════════════
class FaceStateManager:
    """
    Manages FaceStateAnimator instances per track_id.
    Evicts stale entries automatically.
    Drop-in for FaceAuraFX in main.py.
    """

    MAX_AGE = 4.0  # seconds before eviction

    def __init__(self):
        self._animators: Dict[int, FaceStateAnimator] = {}
        self._last_seen: Dict[int, float]             = {}

    def update_and_draw(self, frame: np.ndarray, active_faces: list, dt: float):
        """
        active_faces: list of dicts with keys:
            track_id, bbox, state (FaceUIState), display_name, role, similarity
        """
        from core.identity_cache import FaceUIState

        now   = time.time()
        seen  = set()

        for face in active_faces:
            tid  = face["track_id"]
            seen.add(tid)
            self._last_seen[tid] = now

            if tid not in self._animators:
                self._animators[tid] = FaceStateAnimator(tid)

            anim        = self._animators[tid]
            is_verified = (face["state"] == FaceUIState.VERIFIED)

            anim.update(dt, is_verified)

            x1, y1, x2, y2 = face["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rx = max(22, (x2 - x1) // 2 + 18)
            ry = max(26, (y2 - y1) // 2 + 22)

            anim.draw(frame, cx, cy, rx, ry)

        # Evict stale
        stale = [tid for tid, t in self._last_seen.items()
                 if now - t > self.MAX_AGE]
        for tid in stale:
            self._animators.pop(tid, None)
            self._last_seen.pop(tid, None)
