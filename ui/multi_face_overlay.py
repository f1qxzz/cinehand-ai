"""
ui/multi_face_overlay.py
Renders Face ID–style UI for up to 2 faces simultaneously.

Per-face states:
  SCANNING  → animated oval frame + scan line
  VERIFIED  → fade-out frame, fade-in identity card with name/role
  UNKNOWN   → red oval + "Unknown User"

Smooth transitions:
  - alpha blend between scanning frame and info card
  - slight scale animation on verification
"""

import cv2
import numpy as np
import math
import time
from typing import Optional, Tuple

# FaceUIState is defined in core.identity_cache to avoid circular imports
# Import it here for convenience
try:
    from core.identity_cache import FaceUIState
except ImportError:
    from enum import Enum, auto
    class FaceUIState(Enum):
        SCANNING = auto()
        VERIFIED = auto()
        UNKNOWN  = auto()
C_WHITE  = (255, 255, 255)
C_BLUE   = (220, 160,  60)   # soft blue scanning
C_GREEN  = ( 80, 210, 100)   # verified green
C_RED    = ( 60,  60, 220)   # unknown / denied
C_GRAY   = (120, 120, 120)
C_SCAN   = (200, 230, 255)
C_ACCENT = (180, 200, 255)

FONT     = cv2.FONT_HERSHEY_DUPLEX
FONT_SM  = cv2.FONT_HERSHEY_SIMPLEX

TRANSITION_SPEED = 2.5   # alpha units/sec




class FaceUI:
    """Per-face UI state and renderer."""

    def __init__(self, frame_w: int, frame_h: int):
        self.W = frame_w
        self.H = frame_h
        # Animation
        self._scan_y    = 0.0
        self._glow_ph   = 0.0
        self._alpha     = 0.0    # 0 = scanning frame visible, 1 = info card
        self._scale     = 1.0
        self._prev_t    = time.time()

    def render(self,
               frame:       np.ndarray,
               bbox:        Tuple[int,int,int,int],   # x1,y1,x2,y2
               state:       FaceUIState,
               display_name: str = "",
               role:         str = "",
               similarity:   float = 0.0):

        now = time.time()
        dt  = now - self._prev_t
        self._prev_t = now

        # Animate alpha transition
        target_alpha = 1.0 if state == FaceUIState.VERIFIED else 0.0
        self._alpha  = _lerp(self._alpha, target_alpha, dt * TRANSITION_SPEED)

        # Animate scale
        target_scale = 1.04 if state == FaceUIState.VERIFIED else 1.0
        self._scale  = _lerp(self._scale, target_scale, dt * 4.0)

        # Glow phase
        self._glow_ph = (self._glow_ph + dt * 2.0) % (2 * math.pi)

        # Scan progress
        if state == FaceUIState.SCANNING:
            self._scan_y = (self._scan_y + dt * 0.6) % 1.0

        x1, y1, x2, y2 = bbox
        cx  = (x1 + x2) // 2
        cy  = (y1 + y2) // 2
        bw  = x2 - x1
        bh  = y2 - y1
        rx  = int(bw * 0.55 * self._scale)
        ry  = int(bh * 0.65 * self._scale)

        color = _state_color(state)

        # ── Scanning frame (alpha-faded) ──
        if self._alpha < 0.98:
            frame_a = 1.0 - self._alpha
            self._draw_oval(frame, cx, cy, rx, ry, color, frame_a)
            if state == FaceUIState.SCANNING:
                self._draw_scan_line(frame, cx, cy, rx, ry, frame_a)
            self._draw_scan_label(frame, cx, cy, ry, frame_a)

        # ── Info card (alpha-faded in) ──
        if self._alpha > 0.02 and state == FaceUIState.VERIFIED:
            self._draw_info_card(frame, cx, cy, ry,
                                 display_name, role, self._alpha)

        # ── Unknown label ──
        if state == FaceUIState.UNKNOWN:
            self._draw_unknown_label(frame, cx, cy, ry)

        # ── Similarity bar ──
        if similarity > 0.05:
            self._draw_sim_bar(frame, cx, cy, ry, similarity, state)

    # ── Internal drawing helpers ──────────────────────────────────────────────

    def _draw_oval(self, frame, cx, cy, rx, ry, color, alpha):
        glow_a   = alpha * (0.4 + 0.3 * math.sin(self._glow_ph))
        glow_ov  = frame.copy()
        for thick, gap in [(12, 5), (7, 2), (3, 0)]:
            cv2.ellipse(glow_ov, (cx,cy), (rx+gap, ry+gap),
                        0, 0, 360, color, thick)
        cv2.addWeighted(glow_ov, glow_a * 0.22, frame,
                        1 - glow_a * 0.22, 0, frame)
        cv2.ellipse(frame, (cx,cy), (rx, ry), 0, 0, 360,
                    [int(c*alpha) for c in color], 2, cv2.LINE_AA)

    def _draw_scan_line(self, frame, cx, cy, rx, ry, alpha):
        y_local = -ry + int(self._scan_y * 2 * ry)
        y_abs   = cy + y_local
        if abs(y_local) >= ry:
            return
        x_span  = int(rx * math.sqrt(max(0, 1 - (y_local/ry)**2)))
        x1, x2  = cx - x_span, cx + x_span
        for off, a_mul in [(-1, 0.12), (0, alpha*0.7), (1, 0.12)]:
            yr = y_abs + off
            if 0 <= yr < frame.shape[0]:
                ov = frame.copy()
                cv2.line(ov, (x1, yr), (x2, yr), C_SCAN, 1)
                cv2.addWeighted(ov, a_mul, frame, 1-a_mul, 0, frame)

    def _draw_scan_label(self, frame, cx, cy, ry, alpha):
        text = "Scanning..."
        (tw, _), _ = cv2.getTextSize(text, FONT_SM, 0.55, 1)
        x = cx - tw // 2
        y = cy + ry + 28
        col = [int(c * alpha) for c in C_BLUE]
        _shadow_text(frame, text, x, y, FONT_SM, 0.55, tuple(col), 1)

    def _draw_info_card(self, frame, cx, cy, ry,
                        name, role, alpha):
        name_disp = name or "Verified"
        role_disp = role or ""
        H, W = frame.shape[:2]

        # ── Name card panel (improved) ────────────────────────────────────
        yn = cy + ry + 18

        font_scale = 0.72
        (tw, th), _ = cv2.getTextSize(name_disp, FONT, font_scale, 2)
        pad = 10
        px0 = cx - tw//2 - pad
        py0 = yn - th - 4
        px1 = cx + tw//2 + pad
        py1 = yn + 8

        if px0 >= 0 and py0 >= 0 and px1 < W and py1 < H:
            roi = frame[py0:py1, px0:px1]
            bg  = np.full_like(roi, (6, 8, 18))
            cv2.addWeighted(bg, alpha*0.85, roi, 1-alpha*0.85, 0, roi)
            frame[py0:py1, px0:px1] = roi
            # Animated border
            border_col = [int(c * alpha) for c in C_GREEN]
            cv2.rectangle(frame, (px0, py0), (px1, py1), border_col, 1)
            cv2.line(frame, (px0+2, py0+1), (px1-2, py0+1), border_col, 1)

        nc = [int(c * alpha) for c in C_WHITE]
        _shadow_text(frame, name_disp, cx - tw//2, yn,
                     FONT, font_scale, tuple(nc), 2)

        if role_disp:
            (tw2, _), _ = cv2.getTextSize(role_disp, FONT_SM, 0.44, 1)
            rc = [int(c * alpha) for c in (160, 170, 200)]
            yr = yn + 22
            _shadow_text(frame, role_disp, cx - tw2//2, yr,
                         FONT_SM, 0.44, tuple(rc), 1)

        # "VERIFIED" badge above oval — improved with glow box
        badge = "✓ VERIFIED"
        (bw2, bh2), _ = cv2.getTextSize(badge, FONT_SM, 0.40, 1)
        bx0 = cx - bw2//2 - 6
        by0 = cy - ry - 26
        bx1 = cx + bw2//2 + 6
        by1 = cy - ry - 6

        if bx0 >= 0 and by0 >= 0 and bx1 < W and by1 < H:
            gc = [int(c * alpha) for c in C_GREEN]
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), [c//4 for c in gc], -1)
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), gc, 1)
        _shadow_text(frame, badge, cx-bw2//2, by1-2, FONT_SM, 0.40,
                     tuple([int(c*alpha) for c in C_GREEN]), 1)

    def _draw_unknown_label(self, frame, cx, cy, ry):
        text = "Unknown"
        (tw, _), _ = cv2.getTextSize(text, FONT_SM, 0.55, 1)
        _shadow_text(frame, text, cx-tw//2, cy+ry+28,
                     FONT_SM, 0.55, C_RED, 1)

    def _draw_sim_bar(self, frame, cx, cy, ry, sim, state):
        bw = 160; bh = 3
        bx = cx - bw // 2
        by = cy - ry - 30
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40,40,40), -1)
        fill = int(bw * min(sim, 1.0))
        col  = C_GREEN if state == FaceUIState.VERIFIED else C_BLUE
        if fill > 0:
            cv2.rectangle(frame, (bx, by), (bx+fill, by+bh), col, -1)
        label = f"{sim:.0%}"
        cv2.putText(frame, label, (bx+bw+6, by+bh+2),
                    FONT_SM, 0.34, C_GRAY, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
class MultiFaceOverlay:
    """
    Manages FaceUI instances for up to 2 faces.
    Call .render(frame, active_faces) each frame.
    """

    MAX_FACES = 2

    def __init__(self, frame_w: int, frame_h: int):
        self._W = frame_w
        self._H = frame_h
        self._uis: dict = {}    # track_id → FaceUI

    def render(self, frame: np.ndarray, active_faces: list):
        """
        active_faces : list of dicts with keys:
          track_id, bbox, state (FaceUIState), display_name, role, similarity
        """
        H, W = frame.shape[:2]
        active_ids = set()
        count = min(len(active_faces), self.MAX_FACES)
        for face in active_faces[:self.MAX_FACES]:
            tid  = face["track_id"]
            active_ids.add(tid)
            if tid not in self._uis:
                self._uis[tid] = FaceUI(self._W, self._H)
            self._uis[tid].render(
                frame,
                face["bbox"],
                face["state"],
                face.get("display_name", ""),
                face.get("role", ""),
                face.get("similarity", 0.0),
            )

        # Show dual-user indicator when 2 faces are present
        if count == 2:
            label = "DUAL USER MODE"
            (lw, _), _ = cv2.getTextSize(label, FONT_SM, 0.42, 1)
            lx = (W - lw) // 2
            ly = 28
            cv2.putText(frame, label, (lx + 1, ly + 1), FONT_SM, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (lx, ly), FONT_SM, 0.42, (180, 255, 200), 1, cv2.LINE_AA)

        # Evict inactive
        for tid in list(self._uis.keys()):
            if tid not in active_ids:
                del self._uis[tid]


# ── Utility ───────────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def _state_color(state: FaceUIState):
    return {
        FaceUIState.SCANNING: C_BLUE,
        FaceUIState.VERIFIED: C_GREEN,
        FaceUIState.UNKNOWN:  C_RED,
    }.get(state, C_WHITE)


def _shadow_text(frame, text, x, y, font, scale, color, thickness):
    cv2.putText(frame, text, (x+1, y+1), font, scale,
                (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y),   font, scale,
                color, thickness, cv2.LINE_AA)
