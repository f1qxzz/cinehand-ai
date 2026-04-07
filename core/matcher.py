"""
matcher.py
Cosine-similarity based identity matcher.
Loads known embeddings from data/encodings.pkl and compares
against incoming query embeddings.
"""

import os
import pickle
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MatchResult:
    identity_id: str           # folder name  (e.g. "afna")
    display_name: str          # pretty name  (e.g. "Afna Feyza Chalisa P")
    role: str                  # e.g. "Special Person"
    similarity: float          # 0–1
    is_known: bool


# ══════════════════════════════════════════════════════════════════════
IDENTITY_MAP = {
    "afna": {
        "display_name": "Afna Feyza Chalisa P",
        "role": "Special Person",
    },
    "f1qxzz": {
        "display_name": "f1qxzz (Developer)",
        "role": "System Owner",
    },
}

UNKNOWN_IDENTITY = MatchResult(
    identity_id="unknown",
    display_name="Unknown User",
    role="Guest",
    similarity=0.0,
    is_known=False,
)


# ══════════════════════════════════════════════════════════════════════
class IdentityMatcher:
    """
    Nearest-neighbour cosine similarity matcher.

    Attributes
    ----------
    threshold_accept : float
        Minimum similarity to accept a match.
    threshold_reject : float
        Below this → definite reject (no soft zone).
    """

    def __init__(
        self,
        encodings_path: str = "data/encodings.pkl",
        identities_path: str = "data/identities.json",
        threshold_accept: float = 0.70,
        threshold_reject: float = 0.55,
    ):
        self.threshold_accept = threshold_accept
        self.threshold_reject = threshold_reject
        self._db: Dict[str, List[np.ndarray]] = {}
        self._identity_meta: Dict[str, dict] = {}

        self._load_identities(identities_path)
        self._load_encodings(encodings_path)

    # ------------------------------------------------------------------
    def _load_identities(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self._identity_meta = data
        else:
            self._identity_meta = IDENTITY_MAP

    # ------------------------------------------------------------------
    def _load_encodings(self, path: str):
        if not os.path.exists(path):
            print(f"[Matcher] No encodings file at '{path}'. Run build_dataset.py first.")
            return
        with open(path, "rb") as f:
            raw: Dict[str, List[np.ndarray]] = pickle.load(f)
        self._db = raw
        total = sum(len(v) for v in raw.values())
        print(f"[Matcher] Loaded {total} embeddings for {len(raw)} identities.")

    # ------------------------------------------------------------------
    def match(self, embedding: np.ndarray) -> MatchResult:
        """
        Compare query embedding against every stored embedding.
        Returns the best MatchResult.
        """
        if not self._db:
            return UNKNOWN_IDENTITY

        best_id = None
        best_sim = -1.0

        for identity_id, embs in self._db.items():
            sims = [float(np.dot(embedding, e)) for e in embs]
            sim = float(np.mean(sorted(sims)[-3:]))   # top-3 average
            if sim > best_sim:
                best_sim = sim
                best_id = identity_id

        if best_sim < self.threshold_reject:
            result = UNKNOWN_IDENTITY
            result = MatchResult("unknown", "Unknown User", "Guest", best_sim, False)
            return result

        if best_sim >= self.threshold_accept and best_id:
            meta = self._identity_meta.get(best_id, IDENTITY_MAP.get(best_id, {}))
            return MatchResult(
                identity_id=best_id,
                display_name=meta.get("display_name", best_id),
                role=meta.get("role", ""),
                similarity=best_sim,
                is_known=True,
            )

        # Soft zone → unknown
        return MatchResult("unknown", "Unknown User", "Guest", best_sim, False)

    # ------------------------------------------------------------------
    def add_embedding(self, identity_id: str, embedding: np.ndarray,
                      save_path: str = "data/encodings.pkl"):
        """Incrementally add an embedding and persist."""
        if identity_id not in self._db:
            self._db[identity_id] = []
        self._db[identity_id].append(embedding)
        with open(save_path, "wb") as f:
            pickle.dump(self._db, f)
