"""
hand/cursor.py
Smoothed cursor driven by index fingertip position.
Uses rolling average + EMA for jitter-free movement.
"""

from collections import deque
from typing import Optional, Tuple

HISTORY = 8     # rolling average window
EMA_A   = 0.30  # EMA alpha for velocity-aware blending


class CursorController:
    """
    Tracks a single smoothed cursor point.
    Call .update(x, y) each frame hand is detected.
    Call .reset()      when hand is lost.
    Read .position     for current (x, y).
    """

    def __init__(self):
        self._history:   deque          = deque(maxlen=HISTORY)
        self._ema_pos:   Optional[Tuple[float, float]] = None
        self.position:   Optional[Tuple[int, int]]     = None

    def update(self, x: int, y: int):
        self._history.append((x, y))
        # Rolling average
        avg_x = sum(p[0] for p in self._history) / len(self._history)
        avg_y = sum(p[1] for p in self._history) / len(self._history)

        # EMA on top of rolling average
        if self._ema_pos is None:
            self._ema_pos = (avg_x, avg_y)
        else:
            ex, ey = self._ema_pos
            self._ema_pos = (EMA_A*avg_x + (1-EMA_A)*ex,
                             EMA_A*avg_y + (1-EMA_A)*ey)
        self.position = (int(self._ema_pos[0]), int(self._ema_pos[1]))

    def reset(self):
        self._history.clear()
        self._ema_pos = None
        self.position = None

    def draw_cursor(self, frame, gesture_label: str = "OTHER"):
        """Draw a subtle cursor dot at the current position."""
        if self.position is None:
            return
        x, y = self.position
        import cv2
        colors = {
            "PINCH":     (255, 255, 180),
            "POINT":     (180, 255, 180),
            "OPEN_PALM": (180, 200, 255),
            "ILY":       (255, 100, 255),
        }
        col = colors.get(gesture_label, (200, 200, 200))
        cv2.circle(frame, (x, y), 8,  col,       -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 12, col,        1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2,  (255,255,255), -1, cv2.LINE_AA)
