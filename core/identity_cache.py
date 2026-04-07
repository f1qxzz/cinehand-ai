"""
core/identity_cache.py
Stable identity cache keyed by tracking_id.

Key design rules:
  - Once a tracking_id is inserted, its identity is NEVER overwritten.
  - Verification requires VERIFY_FRAMES consistent frames before verified=True.
  - Fallback identity is "Unknown User / Guest" for new / unrecognised faces.
  - After verification, face recognition is skipped (use cache only).
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


# ── UI State (also imported by ui layer) ──────────────────────────────────────
class FaceUIState(Enum):
    SCANNING  = auto()
    VERIFIED  = auto()
    UNKNOWN   = auto()

from .matcher  import MatchResult, UNKNOWN_IDENTITY
from .smoother import EMABox, IdentityBuffer, SimilarityEMA


# ── Constants ─────────────────────────────────────────────────────────────────
VERIFY_FRAMES   = 5      # post-warmup stable frames to flip verified=True
                         # buffer warmup = 4 frames → total ~9-12 frames (~0.4 s)
ENCODE_INTERVAL = 0.15   # seconds between re-encodes while unverified
MAX_AGE_SEC     = 4.0    # remove cache entry if track unseen this long


# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrackState:
    """All mutable per-tracking-id state."""

    # Bounding box smoothing (EMA)
    box_smoother: EMABox             = field(default_factory=lambda: EMABox(alpha=0.35))
    # Identity majority-vote buffer
    id_buffer:    IdentityBuffer     = field(default_factory=lambda: IdentityBuffer(
                                          window=7, min_votes=4, debounce_sec=0.35))
    # Similarity EMA
    sim_smoother: SimilarityEMA      = field(default_factory=lambda: SimilarityEMA(alpha=0.3))

    # Locked-in identity (set once, never overwritten)
    identity:      Optional[MatchResult] = None
    verified:      bool                  = False
    verify_count:  int                   = 0        # consecutive consistent frames
    last_seen:     float                 = field(default_factory=time.time)
    last_encoded:  float                 = 0.0
    ui_alpha:      float                 = 0.0      # 0=scanning frame, 1=info card
    ui_scale:      float                 = 1.0

    # Aux attributes from multitask model
    gender:  str = ""
    emotion: str = ""

    def touch(self):
        self.last_seen = time.time()

    def needs_encode(self) -> bool:
        if self.verified:
            return False
        return (time.time() - self.last_encoded) >= ENCODE_INTERVAL

    def set_encoded(self):
        self.last_encoded = time.time()

    def push_result(self, result: MatchResult):
        """Feed a recognition result; lock in identity after VERIFY_FRAMES."""
        if self.verified:
            return  # already locked — ignore

        stable_id = self.id_buffer.push(result.identity_id)
        self.sim_smoother.update(result.similarity)

        if stable_id == result.identity_id:
            self.verify_count += 1
        else:
            self.verify_count = max(0, self.verify_count - 1)

        if self.verify_count >= VERIFY_FRAMES:
            # Lock in
            self.identity = result
            self.verified = True

    @property
    def display_result(self) -> MatchResult:
        if self.identity is not None:
            return self.identity
        return UNKNOWN_IDENTITY


# ══════════════════════════════════════════════════════════════════════════════
class IdentityCache:
    """
    Dict[track_id → TrackState] with automatic eviction of stale entries.
    At most MAX_FACES active tracks are retained.
    """

    MAX_FACES = 2

    def __init__(self):
        self._cache: Dict[int, TrackState] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def get_or_create(self, track_id: int) -> TrackState:
        if track_id not in self._cache:
            self._cache[track_id] = TrackState()
        state = self._cache[track_id]
        state.touch()
        return state

    def update_box(self, track_id: int, bbox) -> tuple:
        state = self.get_or_create(track_id)
        return state.box_smoother.update(bbox)

    def evict_stale(self):
        now  = time.time()
        dead = [tid for tid, s in self._cache.items()
                if now - s.last_seen > MAX_AGE_SEC]
        for tid in dead:
            del self._cache[tid]

    def active_ids(self):
        return list(self._cache.keys())

    def __getitem__(self, track_id: int) -> TrackState:
        return self._cache[track_id]

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._cache

    def __len__(self):
        return len(self._cache)
