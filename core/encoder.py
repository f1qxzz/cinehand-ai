"""
encoder.py
Extracts 128-d face embeddings from cropped face images.
Uses a lightweight MobileNet-based model trained via TF/Keras.
Falls back to HOG+LBP descriptor if the deep model is absent.
"""

import os
import numpy as np
import cv2
from typing import Optional

# ── optional deep model ────────────────────────────────────────────────
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    tf = None


# ══════════════════════════════════════════════════════════════════════
class FaceEncoder:
    """
    Produces L2-normalised 128-d embeddings for face crops.

    Priority:
      1. Deep model  (models/face_encoder.h5) – highest accuracy
      2. HOG + LBP   descriptor              – fallback, no GPU needed
    """

    EMBED_DIM = 128
    INPUT_SIZE = (112, 112)

    # ------------------------------------------------------------------
    def __init__(self, model_path: str = "models/face_encoder.h5"):
        self._model = None
        self._use_deep = False

        if _TF_AVAILABLE and os.path.exists(model_path):
            try:
                self._model = tf.keras.models.load_model(model_path, compile=False)
                self._use_deep = True
                print(f"[Encoder] Deep model loaded from {model_path}")
            except Exception as e:
                print(f"[Encoder] Could not load model: {e}. Using HOG+LBP fallback.")
        else:
            print("[Encoder] Using HOG+LBP descriptor (lightweight fallback).")

    # ------------------------------------------------------------------
    def encode(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns a unit-norm 128-d embedding, or None on failure.
        face_crop_bgr : BGR image of any size (will be resized internally).
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        resized = cv2.resize(face_crop_bgr, self.INPUT_SIZE)

        if self._use_deep:
            return self._deep_encode(resized)
        return self._hog_encode(resized)

    # ------------------------------------------------------------------
    def _deep_encode(self, img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(rgb, 0)
        emb = self._model.predict(tensor, verbose=0)[0]
        return self._l2_norm(emb)

    # ------------------------------------------------------------------
    def _hog_encode(self, img: np.ndarray) -> np.ndarray:
        """
        HOG (64-d) + LBP histogram (59-d) concatenated → 123-d → padded to 128-d.
        Deterministic, CPU-only, good enough for small private datasets.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HOG descriptor
        hog = cv2.HOGDescriptor(
            _winSize=(112, 112),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        hog_feat = hog.compute(gray).flatten()
        # Take first 64 PCA-like dims via mean-pooling chunks
        chunk = max(1, len(hog_feat) // 64)
        hog64 = np.array([hog_feat[i * chunk:(i + 1) * chunk].mean()
                          for i in range(64)], dtype=np.float32)

        # LBP uniform histogram (59 bins)
        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        lbp59 = hist.astype(np.float32)

        feat = np.concatenate([hog64, lbp59])          # 123-d
        feat = np.pad(feat, (0, self.EMBED_DIM - len(feat)))  # → 128-d
        return self._l2_norm(feat)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_lbp(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for p in range(n_points):
            angle = 2 * np.pi * p / n_points
            xp = int(round(radius * np.cos(angle)))
            yp = int(round(-radius * np.sin(angle)))
            shifted = np.roll(np.roll(gray, yp, axis=0), xp, axis=1)
            lbp += ((shifted >= gray).astype(np.uint8)) << p
        return lbp

    # ------------------------------------------------------------------
    @staticmethod
    def _l2_norm(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm
