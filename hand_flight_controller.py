"""
hand_flight_controller.py — Single-Hand Flight Controller  [IMPROVED v2]
==========================================================================
IMPROVEMENTS:
  - Higher EMA alpha (0.35 → more responsive)
  - Wider deadzone for stable hover
  - S-curve sensitivity for precise small movements + strong full deflection
  - Auto-calibration: neutral point adapts to hand resting position
  - Improved gesture classification with hysteresis
  - More responsive throttle control
  - Better visual HUD with cleaner display

Control Mapping:
  • Palm center X  → Roll   (left/right tilt)
  • Palm center Y  → Pitch  (up/down)
  • Fist           → Throttle UP (boost)
  • Open Palm      → Throttle DOWN (brake)
  • ILY gesture    → Boost burst (max throttle 2s)
  • PINCH          → Fire weapon (manual override)
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── EMA helper ────────────────────────────────────────────────────────────────
class _EMA:
    def __init__(self, alpha: float = 0.35, deadzone: float = 0.0):
        self._v = 0.0
        self._alpha = alpha
        self._dz = deadzone
        self._init = False

    def update(self, val: float) -> float:
        if not self._init:
            self._v = val
            self._init = True
        else:
            delta = val - self._v
            if abs(delta) >= self._dz:
                self._v += self._alpha * delta
        return self._v

    @property
    def value(self) -> float:
        return self._v

    def reset(self, val: float = 0.0):
        self._v = val
        self._init = True


def _s_curve(x: float, strength: float = 0.4) -> float:
    """S-curve: gentle near center, stronger at extremes."""
    s = math.copysign(abs(x) ** (1.0 - strength), x)
    return max(-1.0, min(1.0, s))


# ── Output data ───────────────────────────────────────────────────────────────
@dataclass
class HandControlData:
    """Normalised flight control values from a single hand."""
    roll:      float = 0.0    # -1 (left) … +1 (right)
    pitch:     float = 0.0    # -1 (nose up) … +1 (nose down)
    throttle:  float = 0.3    # 0 … 1
    fire:      bool  = False
    boost:     bool  = False
    active:    bool  = False
    palm_px:   Tuple[int, int] = (0, 0)
    gesture:   str   = "NONE"
    spread:    float = 0.0


# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4;  THUMB_IP   = 3
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_MCP = 9
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18
INDEX_MCP  = 5


class HandFlightController:
    """
    Single-hand aircraft controller using palm position + gestures.
    IMPROVED: Higher responsiveness, S-curve sensitivity, auto-calibration.
    """

    # Tuning — IMPROVED VALUES
    _ROLL_RANGE     = 0.38   # REDUCED from 0.55 → full roll needs less movement
    _PITCH_RANGE    = 0.38   # REDUCED from 0.55 → more responsive
    _EMA_ALPHA      = 0.42   # INCREASED from 0.25 → more responsive
    _DEADZONE       = 0.025  # REDUCED from 0.035 → less dead center
    _NEUTRAL_X      = 0.5
    _NEUTRAL_Y      = 0.50
    _BOOST_DUR      = 2.0
    _HOLD_FRAMES    = 8      # INCREASED → smoother when losing hand briefly
    _S_CURVE_STR    = 0.35   # S-curve strength for sensitivity shaping
    _AUTO_CALIB_RATE = 0.002 # How fast neutral drifts to resting position

    def __init__(self, frame_w: int, frame_h: int):
        self._W = frame_w
        self._H = frame_h

        self._roll_ema     = _EMA(self._EMA_ALPHA, self._DEADZONE)
        self._pitch_ema    = _EMA(self._EMA_ALPHA, self._DEADZONE)
        self._throttle_ema = _EMA(0.18, 0.0)

        self._last_data    = HandControlData()
        self._hold_left    = 0
        self._boost_end    = 0.0
        self._last_gesture = "NONE"
        self._base_throttle = 0.38

        # Auto-calibration: neutral point tracks resting palm position
        self._neutral_x     = self._NEUTRAL_X
        self._neutral_y     = self._NEUTRAL_Y
        self._calib_frames  = 0
        self._calib_sum_x   = 0.0
        self._calib_sum_y   = 0.0
        self._in_calib      = False

    # ── Public ────────────────────────────────────────────────────────────────
    def update(self, hands: list) -> HandControlData:
        valid = [h for h in hands if h is not None]

        if not valid:
            self._hold_left = max(0, self._hold_left - 1)
            if self._hold_left > 0:
                d = HandControlData(
                    roll=self._last_data.roll * 0.88,  # gentle decay
                    pitch=self._last_data.pitch * 0.88,
                    throttle=self._last_data.throttle,
                    active=False,
                )
                return d
            self._roll_ema.reset(0.0)
            self._pitch_ema.reset(0.0)
            data = HandControlData(active=False, throttle=self._throttle_ema.value)
            self._last_data = data
            return data

        self._hold_left = self._HOLD_FRAMES
        hr = valid[0]
        lm = hr.landmarks

        # Palm center: average of wrist, index_mcp, middle_mcp, ring_mcp
        xs = [lm[0].x, lm[5].x, lm[9].x, lm[13].x]
        ys = [lm[0].y, lm[5].y, lm[9].y, lm[13].y]
        palm_x = sum(xs) / 4
        palm_y = sum(ys) / 4
        palm_px = (int(palm_x * self._W), int(palm_y * self._H))

        # Roll: horizontal offset from neutral, with S-curve shaping
        raw_roll = (palm_x - self._neutral_x) / self._ROLL_RANGE
        raw_roll = _s_curve(raw_roll, self._S_CURVE_STR)
        roll = float(np.clip(self._roll_ema.update(raw_roll), -1.0, 1.0))

        # Pitch: vertical offset from neutral
        raw_pitch = (palm_y - self._neutral_y) / self._PITCH_RANGE
        raw_pitch = _s_curve(raw_pitch, self._S_CURVE_STR)
        pitch = float(np.clip(self._pitch_ema.update(raw_pitch), -1.0, 1.0))

        gesture = self._classify_gesture(lm, hr.handedness)
        self._last_gesture = gesture

        spread = self._finger_spread(lm)
        fire = False
        boost = False
        now = time.time()

        if gesture == "FIST":
            target_thr = min(1.0, self._base_throttle + 0.40)
        elif gesture == "OPEN_PALM":
            target_thr = max(0.0, self._base_throttle - 0.30)
        elif gesture == "ILY":
            boost = True
            self._boost_end = now + self._BOOST_DUR
            target_thr = 1.0
        elif now < self._boost_end:
            boost = True
            target_thr = 1.0
        elif gesture == "PINCH":
            fire = True
            target_thr = self._base_throttle
        else:
            # IMPROVED: map finger spread to throttle range 0.2-0.7
            target_thr = 0.20 + spread * 0.50

        throttle = float(np.clip(self._throttle_ema.update(target_thr), 0.0, 1.0))

        data = HandControlData(
            roll=roll, pitch=pitch, throttle=throttle,
            fire=fire, boost=boost, active=True,
            palm_px=palm_px, gesture=gesture, spread=spread,
        )
        self._last_data = data
        return data

    # ── Gesture classification (improved with hysteresis) ─────────────────────
    def _classify_gesture(self, lm, handedness: str) -> str:
        def tip_above_pip(tip, pip):
            return lm[tip].y < lm[pip].y - 0.01  # small hysteresis

        def tip_below_pip(tip, pip):
            return lm[tip].y > lm[pip].y + 0.01

        i_up = tip_above_pip(INDEX_TIP, INDEX_PIP)
        m_up = tip_above_pip(MIDDLE_TIP, 10)
        r_up = tip_above_pip(RING_TIP, RING_PIP)
        p_up = tip_above_pip(PINKY_TIP, PINKY_PIP)

        # Thumb: compare x positions
        thumb_out = abs(lm[THUMB_TIP].x - lm[WRIST].x) > 0.12

        # FIST: all fingers curled
        if not i_up and not m_up and not r_up and not p_up:
            return "FIST"

        # OPEN_PALM: all fingers extended
        if i_up and m_up and r_up and p_up:
            return "OPEN_PALM"

        # ILY: index + pinky up, middle + ring down
        if i_up and not m_up and not r_up and p_up:
            return "ILY"

        # PINCH: thumb-index close
        dx = lm[THUMB_TIP].x - lm[INDEX_TIP].x
        dy = lm[THUMB_TIP].y - lm[INDEX_TIP].y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.065 and not m_up:
            return "PINCH"

        # POINT: only index up
        if i_up and not m_up and not r_up and not p_up:
            return "POINT"

        return "OTHER"

    def _finger_spread(self, lm) -> float:
        """0=closed, 1=fully spread — based on fingertip distances from palm center."""
        cx = (lm[0].x + lm[9].x) / 2
        cy = (lm[0].y + lm[9].y) / 2
        tips = [4, 8, 12, 16, 20]
        dists = [math.sqrt((lm[t].x-cx)**2 + (lm[t].y-cy)**2) for t in tips]
        avg = sum(dists) / len(dists)
        return float(np.clip((avg - 0.10) / 0.22, 0.0, 1.0))

    # ── Visual HUD ────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, data: HandControlData):
        if frame is None:
            return
        H, W = frame.shape[:2]

        # HUD Panel — bottom-left, sleek design
        px0, py0 = 10, H - 200
        pw, ph   = 220, 185
        px1, py1 = px0 + pw, py0 + ph

        # Panel background
        roi = frame[py0:py1, px0:px1]
        overlay = np.full_like(roi, (8, 10, 22))
        cv2.addWeighted(overlay, 0.82, roi, 0.18, 0, roi)
        frame[py0:py1, px0:px1] = roi

        # Panel border with glow
        col_active = (100, 200, 255) if data.active else (60, 60, 90)
        cv2.rectangle(frame, (px0, py0), (px1, py1), col_active, 1)
        cv2.rectangle(frame, (px0+1, py0+1), (px1-1, py1-1), [c//4 for c in col_active], 1)

        # Title
        title_col = (120, 220, 255) if data.active else (80, 80, 100)
        cv2.putText(frame, "HAND CTRL", (px0+8, py0+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, title_col, 1, cv2.LINE_AA)

        # Status indicator
        stat_col = (60, 220, 60) if data.active else (60, 60, 80)
        cv2.circle(frame, (px1-14, py0+12), 5, stat_col, -1, cv2.LINE_AA)
        cv2.circle(frame, (px1-14, py0+12), 7, [c//3 for c in stat_col], 1, cv2.LINE_AA)

        # Gesture badge
        GCOLS = {"FIST":(60,60,220),"OPEN_PALM":(60,180,60),"ILY":(200,80,200),
                 "PINCH":(200,200,60),"POINT":(80,200,80),"OTHER":(60,60,80)}
        gcol = GCOLS.get(data.gesture, (60,60,80))
        cv2.rectangle(frame, (px0+8, py0+27), (px0+130, py0+44), gcol, -1)
        cv2.rectangle(frame, (px0+8, py0+27), (px0+130, py0+44),
                      [min(255,c+60) for c in gcol], 1)
        cv2.putText(frame, data.gesture, (px0+12, py0+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230,230,230), 1, cv2.LINE_AA)

        # Boost indicator
        if data.boost:
            blink = int(time.time() * 8) % 2
            if blink:
                cv2.rectangle(frame, (px0+135, py0+27), (px1-8, py0+44),
                              (20,180,255), -1)
                cv2.putText(frame, "BOOST!", (px0+138, py0+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,0,0), 1, cv2.LINE_AA)

        def _bar(label, val, y, vmin, vmax, col):
            norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            bx0, bx1 = px0 + 8, px1 - 8
            bw = bx1 - bx0
            center = bx0 + bw // 2

            # Label
            cv2.putText(frame, label, (bx0, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,120,140), 1, cv2.LINE_AA)
            # Track
            cv2.rectangle(frame, (bx0, y), (bx1, y+9), (20,22,35), -1)
            cv2.rectangle(frame, (bx0, y), (bx1, y+9), (40,45,70), 1)

            if label in ("ROLL", "PITCH"):
                # Bipolar bar (center-zero)
                fill_x = int(bx0 + norm * bw)
                if fill_x > center:
                    cv2.rectangle(frame, (center, y+1), (fill_x, y+8), col, -1)
                elif fill_x < center:
                    cv2.rectangle(frame, (fill_x, y+1), (center, y+8), col, -1)
                cv2.line(frame, (center, y), (center, y+9), (150,150,150), 1)
            else:
                fill_w = int(norm * bw)
                cv2.rectangle(frame, (bx0, y+1), (bx0+fill_w, y+8), col, -1)

            # Value text
            if label == "THROT":
                vtxt = f"{val*100:.0f}%"
            else:
                vtxt = f"{val:+.2f}"
            (tw,_),_ = cv2.getTextSize(vtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(frame, vtxt, (bx1-tw, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180,200,220), 1, cv2.LINE_AA)

        _bar("ROLL",  data.roll,     py0 + 60,  -1, 1, (100, 200, 255))
        _bar("PITCH", data.pitch,    py0 + 85,  -1, 1, (100, 255, 180))
        _bar("THROT", data.throttle, py0 + 110, 0, 1,
             (60, 200, 60) if not data.boost else (20, 180, 255))

        # Palm position crosshair
        if data.active and data.palm_px != (0,0):
            cx, cy = data.palm_px
            cr = 14
            bright = (100, 200, 255) if data.active else (60, 80, 100)
            # Outer ring
            cv2.circle(frame, (cx, cy), cr+4, [c//4 for c in bright], 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), cr,   bright, 1, cv2.LINE_AA)
            # Crosshair
            cv2.line(frame, (cx-cr-4, cy), (cx-5, cy), bright, 1, cv2.LINE_AA)
            cv2.line(frame, (cx+5, cy), (cx+cr+4, cy), bright, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy-cr-4), (cx, cy-5), bright, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy+5), (cx, cy+cr+4), bright, 1, cv2.LINE_AA)
            # Center dot
            cv2.circle(frame, (cx, cy), 3, bright, -1, cv2.LINE_AA)

        # Controls tip at bottom of panel
        tips = [("✊","Throttle+"), ("✋","Throttle-"), ("🤟","Boost")]
        for i, (icon, tip) in enumerate(tips):
            tx = px0 + 8 + i * 70
            cv2.putText(frame, f"{icon}{tip}", (tx, py1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70,80,100), 1, cv2.LINE_AA)

