"""
tracker.py
Minimal SORT (Simple Online and Realtime Tracking) implementation.
Zero external dependencies beyond numpy — no scipy, no filterpy.
Kalman filter implemented from scratch using numpy only.

Reference: Bewley et al., 2016 – https://arxiv.org/abs/1602.00763
"""

import numpy as np
from typing import List, Tuple, Dict


# ══════════════════════════════════════════════════════════════════════
# Pure-numpy Hungarian algorithm (no scipy needed)
# ══════════════════════════════════════════════════════════════════════
def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve linear assignment on a cost matrix using the Hungarian method.
    Returns (row_indices, col_indices) like scipy.optimize.linear_sum_assignment.
    Works for non-square matrices. O(n^3).
    """
    # Pad to square
    n, m = cost.shape
    size = max(n, m)
    C = np.full((size, size), fill_value=cost.max() + 1 if cost.size else 0.0)
    C[:n, :m] = cost

    # Row reduction
    C -= C.min(axis=1, keepdims=True)
    # Col reduction
    C -= C.min(axis=0, keepdims=True)

    covered_rows = np.zeros(size, dtype=bool)
    covered_cols = np.zeros(size, dtype=bool)
    starred = np.zeros((size, size), dtype=bool)
    primed  = np.zeros((size, size), dtype=bool)

    # Star zeros (greedy)
    for r in range(size):
        for c in range(size):
            if C[r, c] == 0 and not covered_rows[r] and not covered_cols[c]:
                starred[r, c] = True
                covered_rows[r] = True
                covered_cols[c] = True
    covered_rows[:] = False
    covered_cols[:] = False

    # Cover starred columns
    covered_cols = starred.any(axis=0)

    while not covered_cols.all() and covered_cols[:m].sum() < min(n, m):
        # Find uncovered zero
        found = False
        Z0 = (-1, -1)
        for r in range(size):
            if covered_rows[r]: continue
            for c in range(size):
                if covered_cols[c]: continue
                if C[r, c] == 0:
                    Z0 = (r, c)
                    found = True
                    break
            if found: break

        if not found:
            # Augment
            uncov_vals = C[~covered_rows][:, ~covered_cols]
            if uncov_vals.size == 0: break
            minval = uncov_vals.min()
            C[~covered_rows] -= np.where(covered_cols, 0, minval)
            C[:, covered_cols] += np.where(covered_rows[:, None], minval, 0)
            continue

        r0, c0 = Z0
        primed[r0, c0] = True

        star_in_row = starred[r0, :].argmax() if starred[r0, :].any() else -1
        if starred[r0, :].any():
            covered_rows[r0] = True
            covered_cols[star_in_row] = False
        else:
            # Build alternating path
            path = [(r0, c0)]
            while True:
                r_cur, c_cur = path[-1]
                star_row = -1
                for rr in range(size):
                    if starred[rr, c_cur]:
                        star_row = rr; break
                if star_row == -1: break
                path.append((star_row, c_cur))
                prime_col = primed[star_row, :].argmax()
                path.append((star_row, prime_col))
            for (pr, pc) in path:
                starred[pr, pc] = not starred[pr, pc]
            primed[:] = False
            covered_rows[:] = False
            covered_cols = starred.any(axis=0)

    rows, cols = np.where(starred[:n, :m])
    return rows, cols


# ══════════════════════════════════════════════════════════════════════
def _iou(bb_a: np.ndarray, bb_b: np.ndarray) -> float:
    """IoU between two bboxes [x1,y1,x2,y2]."""
    xa1 = max(bb_a[0], bb_b[0])
    ya1 = max(bb_a[1], bb_b[1])
    xa2 = min(bb_a[2], bb_b[2])
    ya2 = min(bb_a[3], bb_b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (bb_a[2] - bb_a[0]) * (bb_a[3] - bb_a[1])
    area_b = (bb_b[2] - bb_b[0]) * (bb_b[3] - bb_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → [cx,cy,s,r] where s=area, r=aspect."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h else 1.0
    return np.array([cx, cy, s, r], dtype=np.float32).reshape(4, 1)


def _z_to_bbox(z: np.ndarray, score: float = 0.0) -> np.ndarray:
    """[cx,cy,s,r] → [x1,y1,x2,y2,score].

    Safe extraction: z may be shaped (4,), (4,1), or (7,1) depending on
    whether it comes from _bbox_to_z() or the Kalman state self.x.
    np.squeeze + float() guarantees every value is a plain Python scalar
    before arithmetic, preventing the 'setting an array element with a
    sequence' ValueError.
    """
    z = np.asarray(z).flatten()          # (4,) or (7,) — always 1-D

    cx = float(z[0])
    cy = float(z[1])
    s  = float(z[2])                      # area  (must stay positive)
    r  = float(z[3])                      # aspect ratio w/h

    s = max(s, 1e-6)                      # guard against negative/zero area
    r = max(r, 1e-6)

    w = float(np.sqrt(s * r))
    h = float(s / w)

    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
        float(score),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
class KalmanBoxTracker:
    """
    Single-object tracker using a pure-numpy Kalman filter.
    State: [cx, cy, s, r, vx, vy, vs]  (s=area, r=aspect ratio)
    """

    _count = 0

    # Transition matrix F (7x7)
    _F = np.array([
        [1,0,0,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,0,1,0,0,0,1],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1],
    ], dtype=np.float64)

    # Observation matrix H (4x7)
    _H = np.array([
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0],
    ], dtype=np.float64)

    def __init__(self, bbox: np.ndarray):
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4] = _bbox_to_z(bbox).astype(np.float64)

        self.P = np.eye(7, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6]  *= 0.01

        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.history: List[np.ndarray] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0

    def update(self, bbox: np.ndarray):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = _bbox_to_z(bbox).astype(np.float64)
        # Kalman update
        H, R, P, x = self._H, self.R, self.P, self.x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        self.x = x + K @ (z - H @ x)
        self.P = (np.eye(7) - K @ H) @ P

    def predict(self) -> np.ndarray:
        F, Q, P, x = self._F, self.Q, self.P, self.x
        if (x[6] + x[2]) <= 0:
            x[6] *= 0.0
        self.x = F @ x
        self.P = F @ P @ F.T + Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(_z_to_bbox(self.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Return [x1,y1,x2,y2,score] as a flat float32 array of shape (5,)."""
        return np.asarray(_z_to_bbox(self.x), dtype=np.float32).flatten()


# ══════════════════════════════════════════════════════════════════════
class SORTTracker:
    """
    SORT multi-object tracker.

    Returns per-frame list of (x1, y1, x2, y2, track_id).
    """

    def __init__(self, max_age: int = 10, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker._count = 0

    # ------------------------------------------------------------------
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        detections : np.ndarray  shape (N, 5) – [x1,y1,x2,y2,conf]
                     or (0, 5) when no detections.

        Returns
        -------
        np.ndarray  shape (M, 5) – [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # ── Sanitise input ────────────────────────────────────────────
        # Guarantee shape (N, 5) float32 — never let bad input reach Kalman.
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)
            if detections.ndim == 1:
                detections = detections.reshape(1, -1)
            if detections.shape[1] < 5:
                # Pad missing confidence column with 1.0
                pad = np.ones((detections.shape[0], 5 - detections.shape[1]),
                              dtype=np.float32)
                detections = np.hstack([detections, pad])
            # Remove degenerate boxes (w<=0 or h<=0)
            valid = ((detections[:, 2] > detections[:, 0]) &
                     (detections[:, 3] > detections[:, 1]))
            detections = detections[valid]
            # Cap at MAX_FACES (2) — sort by confidence desc
            if len(detections) > 2:
                order = np.argsort(-detections[:, 4])
                detections = detections[order[:2]]

        # ── Predict existing trackers ─────────────────────────────────
        trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
        to_del = []
        for i, t in enumerate(self.trackers):
            raw = t.predict()                        # shape (5,) float32
            pos = np.asarray(raw, dtype=np.float32).flatten()[:4]
            trks[i] = [*pos, 0]
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                to_del.append(i)
        for i in reversed(to_del):
            self.trackers.pop(i)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # Match detections → trackers
        matched, unmatched_dets, unmatched_trks = self._associate(detections, trks)

        # Update matched
        for d, t in matched:
            self.trackers[t].update(detections[d, :4])

        # Create new trackers for unmatched detections
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d, :4]))

        # Collect active tracks
        ret = []
        for t in reversed(self.trackers):
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or
                                              self.frame_count <= self.min_hits):
                state = np.asarray(t.get_state(), dtype=np.float32).flatten()
                box = state[:4]
                if np.any(np.isnan(box)) or np.any(np.isinf(box)):
                    continue
                ret.append([float(box[0]), float(box[1]),
                             float(box[2]), float(box[3]), float(t.id)])
            if t.time_since_update > self.max_age:
                self.trackers.remove(t)

        return np.array(ret, dtype=np.float32) if ret else np.empty((0, 5), dtype=np.float32)

    # ------------------------------------------------------------------
    def _associate(
        self,
        dets: np.ndarray,
        trks: np.ndarray,
        iou_thresh: float = None,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        if iou_thresh is None:
            iou_thresh = self.iou_threshold

        if len(trks) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(dets))), []
        if len(dets) == 0:
            return np.empty((0, 2), dtype=int), [], list(range(len(trks)))

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d in range(len(dets)):
            for t in range(len(trks)):
                iou_matrix[d, t] = _iou(dets[d, :4], trks[t, :4])

        row_ind, col_ind = _hungarian(-iou_matrix.astype(np.float64))
        matched_indices = np.column_stack([row_ind, col_ind]) if len(row_ind) else np.empty((0,2),dtype=int)

        unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_thresh:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
        return matches, unmatched_dets, unmatched_trks
