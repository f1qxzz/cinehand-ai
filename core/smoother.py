"""
smoother.py
Per-track smoothing utilities:
  - EMA bounding-box smoother (reduce jitter)
  - IdentityBuffer (majority-vote across N frames, debounced)
"""

import time
import numpy as np
from collections import deque, Counter
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
class EMABox:
    """
    Exponential Moving Average smoothing for a bounding box [x1,y1,x2,y2].

    alpha : smoothing factor (0 = no update, 1 = no smoothing)
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        b = np.array(box, dtype=np.float32)
        if self._state is None:
            self._state = b
        else:
            self._state = self.alpha * b + (1 - self.alpha) * self._state
        return tuple(self._state.astype(int))

    def reset(self):
        self._state = None


# ══════════════════════════════════════════════════════════════════════
class IdentityBuffer:
    """
    Majority-vote buffer for identity decisions.

    - Accumulates identity_id over a sliding window of `window` frames.
    - Returns the winning identity only if it reaches `min_votes` votes
      AND at least `debounce_sec` seconds have passed since the last change.
    - Prevents flickering between unknown / known / different persons.
    """

    def __init__(
        self,
        window: int = 7,
        min_votes: int = 4,
        debounce_sec: float = 0.35,
    ):
        self._window = window
        self._min_votes = min_votes
        self._debounce_sec = debounce_sec
        self._buf: deque = deque(maxlen=window)
        self._current_id: Optional[str] = None
        self._last_change: float = 0.0

    # ------------------------------------------------------------------
    def push(self, identity_id: str) -> str:
        """Add a raw identity prediction; return the stabilised identity."""
        self._buf.append(identity_id)
        if len(self._buf) < self._window // 2:
            return self._current_id or "unknown"

        counts = Counter(self._buf)
        winner, votes = counts.most_common(1)[0]

        if votes < self._min_votes:
            return self._current_id or "unknown"

        # Check debounce
        now = time.time()
        if winner != self._current_id:
            if now - self._last_change >= self._debounce_sec:
                self._current_id = winner
                self._last_change = now

        return self._current_id or "unknown"

    # ------------------------------------------------------------------
    def reset(self):
        self._buf.clear()
        self._current_id = None
        self._last_change = 0.0

    @property
    def current(self) -> Optional[str]:
        return self._current_id


# ══════════════════════════════════════════════════════════════════════
class SimilarityEMA:
    """
    EMA smoother for cosine similarity score (avoids confidence flicker).
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._val: float = 0.0

    def update(self, sim: float) -> float:
        self._val = self.alpha * sim + (1 - self.alpha) * self._val
        return self._val

    def reset(self):
        self._val = 0.0
