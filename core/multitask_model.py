"""
multitask_model.py
Multi-task CNN: gender classification + emotion recognition.
Builds / loads a lightweight MobileNetV2-based model.
"""

import os
import numpy as np
import cv2
from typing import Tuple, Optional, Dict

try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_OK = True
except ImportError:
    _TF_OK = False
    tf = None
    keras = None


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
GENDERS  = ["Female", "Male"]


# ══════════════════════════════════════════════════════════════════════
class MultitaskModel:
    """
    Gender + Emotion multi-task model.

    If the saved model exists it is loaded; otherwise a fresh architecture
    is built and random weights are used (training required).

    Inference input  : 96×96 BGR crop
    Inference output : (gender_str, gender_conf, emotion_str, emotion_conf)
    """

    INPUT_SIZE = (96, 96)

    def __init__(self, model_path: str = "models/multitask_model.h5"):
        self._model = None
        self._model_path = model_path

        if _TF_OK:
            if os.path.exists(model_path):
                self._load(model_path)
            else:
                print("[Multitask] No pretrained model found. Building architecture…")
                self._build()
                print("[Multitask] Using random weights — train the model for real results.")
        else:
            print("[Multitask] TensorFlow not available. Using rule-based fallback.")

    # ------------------------------------------------------------------
    def _build(self):
        """Lightweight dual-head MobileNetV2."""
        inp = keras.Input(shape=(96, 96, 3), name="face_input")

        base = keras.applications.MobileNetV2(
            input_tensor=inp,
            include_top=False,
            weights=None,           # random for demo; load pretrained for production
            alpha=0.35,
        )
        x = keras.layers.GlobalAveragePooling2D()(base.output)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)

        # Gender head
        gender_out = keras.layers.Dense(len(GENDERS), activation="softmax",
                                        name="gender")(x)
        # Emotion head
        emotion_out = keras.layers.Dense(len(EMOTIONS), activation="softmax",
                                         name="emotion")(x)

        self._model = keras.Model(inputs=inp, outputs=[gender_out, emotion_out])

    # ------------------------------------------------------------------
    def _load(self, path: str):
        try:
            self._model = keras.models.load_model(path, compile=False)
            print(f"[Multitask] Model loaded from {path}")
        except Exception as e:
            print(f"[Multitask] Load failed: {e}. Building fresh architecture.")
            self._build()

    # ------------------------------------------------------------------
    def save(self):
        if self._model and self._model_path:
            os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
            self._model.save(self._model_path)
            print(f"[Multitask] Saved to {self._model_path}")

    # ------------------------------------------------------------------
    def predict(
        self, face_crop_bgr: np.ndarray
    ) -> Tuple[str, float, str, float]:
        """
        Returns (gender, gender_conf, emotion, emotion_conf).
        Falls back to ("Unknown", 0.0, "Neutral", 0.0) on error.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return "Unknown", 0.0, "Neutral", 0.0

        if self._model is None:
            return self._rule_based(face_crop_bgr)

        try:
            img = cv2.resize(face_crop_bgr, self.INPUT_SIZE)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.expand_dims(rgb, 0)

            g_logits, e_logits = self._model.predict(tensor, verbose=0)

            g_idx  = int(np.argmax(g_logits[0]))
            g_conf = float(g_logits[0][g_idx])
            e_idx  = int(np.argmax(e_logits[0]))
            e_conf = float(e_logits[0][e_idx])

            return GENDERS[g_idx], g_conf, EMOTIONS[e_idx], e_conf

        except Exception as ex:
            print(f"[Multitask] Inference error: {ex}")
            return "Unknown", 0.0, "Neutral", 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _rule_based(img: np.ndarray) -> Tuple[str, float, str, float]:
        """
        Very rough heuristic when TF is unavailable.
        Based on skin-tone brightness and simple stats.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_bright = float(gray.mean())
        std_bright  = float(gray.std())

        # Crude gender proxy (not accurate — demo only)
        gender = "Female" if mean_bright > 128 else "Male"
        g_conf = 0.5

        # Crude emotion proxy
        if std_bright < 30:
            emotion, e_conf = "Neutral", 0.6
        elif mean_bright > 150:
            emotion, e_conf = "Happy", 0.55
        else:
            emotion, e_conf = "Neutral", 0.5

        return gender, g_conf, emotion, e_conf
