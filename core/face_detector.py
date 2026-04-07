"""
face_detector.py
Face detection with automatic backend fallback chain:

  1. OpenCV DNN SSD     — works on ANY Python version, no extra deps
  2. MediaPipe Tasks    — mediapipe >= 0.10.x  (new API, Python <= 3.12)
  3. MediaPipe Solutions — mediapipe old-style  (legacy mp.solutions)

The correct backend is chosen at __init__ time so nothing else needs
to change regardless of mediapipe version or Python version.
"""

import cv2
import numpy as np
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
@dataclass
class FaceDetection:
    """Single face detection result."""
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2  (pixels)
    confidence: float
    landmarks:  Optional[np.ndarray]        # 6x2 keypoints or None
    center:     Tuple[int, int]


# ══════════════════════════════════════════════════════════════════════
# OpenCV DNN model paths / URLs
# ══════════════════════════════════════════════════════════════════════
_MODEL_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))
_PROTO_PATH = os.path.join(_MODEL_DIR, "deploy.prototxt")
_CAFFE_PATH = os.path.join(_MODEL_DIR, "res10_300x300_ssd.caffemodel")
_PROTO_URL  = ("https://raw.githubusercontent.com/opencv/opencv/master/"
               "samples/dnn/face_detector/deploy.prototxt")
_CAFFE_URL  = ("https://github.com/opencv/opencv_3rdparty/raw/"
               "dnn_samples_face_detector_20170830/"
               "res10_300x300_ssd_iter_140000.caffemodel")


def _download_opencv_models() -> bool:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for url, path in [(_PROTO_URL, _PROTO_PATH), (_CAFFE_URL, _CAFFE_PATH)]:
        if os.path.exists(path):
            continue
        print(f"[FaceDetector] Downloading {os.path.basename(path)} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"[FaceDetector] Download failed: {e}")
            return False
    return os.path.exists(_PROTO_PATH) and os.path.exists(_CAFFE_PATH)


# ══════════════════════════════════════════════════════════════════════
class FaceDetector:
    """
    Multi-backend face detector.
    Priority: OpenCV DNN → MediaPipe Tasks → MediaPipe Solutions → Haar
    """

    def __init__(
        self,
        model_selection: int = 1,
        min_detection_confidence: float = 0.5,
        max_faces: int = 10,
    ):
        self.max_faces  = max_faces
        self.confidence = min_detection_confidence
        self._backend   = None

        # Try each backend in order
        if self._try_opencv_dnn():
            return
        if self._try_mp_tasks(min_detection_confidence):
            return
        if self._try_mp_solutions(model_selection, min_detection_confidence):
            return
        self._try_haar()

    # ── Backend initialisers ──────────────────────────────────────────

    def _try_opencv_dnn(self) -> bool:
        if not _download_opencv_models():
            return False
        try:
            net = cv2.dnn.readNetFromCaffe(_PROTO_PATH, _CAFFE_PATH)
            self._dnn_net = net
            self._backend = "opencv_dnn"
            print("[FaceDetector] Backend: OpenCV DNN (SSD)")
            return True
        except Exception as e:
            print(f"[FaceDetector] OpenCV DNN failed: {e}")
            return False

    def _try_mp_tasks(self, conf: float) -> bool:
        try:
            import mediapipe as mp
            if not hasattr(mp, "tasks"):
                return False
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks import python as mp_python

            tflite_path = os.path.join(_MODEL_DIR, "blaze_face_short_range.tflite")
            if not os.path.exists(tflite_path):
                url = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                       "blaze_face_short_range/float16/1/blaze_face_short_range.task")
                print("[FaceDetector] Downloading BlazeFace TFLite model ...")
                urllib.request.urlretrieve(url, tflite_path)

            base_opts = mp_python.BaseOptions(model_asset_path=tflite_path)
            opts = mp_vision.FaceDetectorOptions(
                base_options=base_opts,
                min_detection_confidence=conf,
            )
            self._mp_detector   = mp_vision.FaceDetector.create_from_options(opts)
            self._mp_image_cls  = mp.Image
            self._mp_fmt        = mp.ImageFormat.SRGB
            self._backend       = "mp_tasks"
            print("[FaceDetector] Backend: MediaPipe Tasks API")
            return True
        except Exception as e:
            print(f"[FaceDetector] MediaPipe Tasks unavailable: {e}")
            return False

    def _try_mp_solutions(self, model_sel: int, conf: float) -> bool:
        try:
            import mediapipe as mp
            if not (hasattr(mp, "solutions") and
                    hasattr(mp.solutions, "face_detection")):
                return False
            mp_fd = mp.solutions.face_detection
            self._mp_detector = mp_fd.FaceDetection(
                model_selection=model_sel,
                min_detection_confidence=conf,
            )
            self._backend = "mp_solutions"
            print("[FaceDetector] Backend: MediaPipe Solutions API")
            return True
        except Exception as e:
            print(f"[FaceDetector] MediaPipe Solutions unavailable: {e}")
            return False

    def _try_haar(self):
        cc = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar    = cv2.CascadeClassifier(cc)
        self._backend = "haar"
        print("[FaceDetector] Backend: Haar Cascade (fallback — lower accuracy)")

    # ── Unified detect ────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        if self._backend == "opencv_dnn":
            return self._detect_dnn(frame_bgr)
        if self._backend == "mp_tasks":
            return self._detect_mp_tasks(frame_bgr)
        if self._backend == "mp_solutions":
            return self._detect_mp_solutions(frame_bgr)
        return self._detect_haar(frame_bgr)

    # ── Per-backend detect ────────────────────────────────────────────

    def _detect_dnn(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0),
        )
        self._dnn_net.setInput(blob)
        out  = self._dnn_net.forward()
        dets = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < self.confidence:
                continue
            x1 = max(0, int(out[0,0,i,3] * w))
            y1 = max(0, int(out[0,0,i,4] * h))
            x2 = min(w, int(out[0,0,i,5] * w))
            y2 = min(h, int(out[0,0,i,6] * h))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(FaceDetection(
                bbox=(x1,y1,x2,y2), confidence=conf,
                landmarks=None, center=((x1+x2)//2,(y1+y2)//2),
            ))
        return sorted(dets, key=lambda d: d.confidence, reverse=True)[:self.max_faces]

    def _detect_mp_tasks(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp_image_cls(image_format=self._mp_fmt, data=rgb)
        result = self._mp_detector.detect(mp_img)
        dets   = []
        for det in (result.detections or [])[:self.max_faces]:
            bb = det.bounding_box
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(w, bb.origin_x + bb.width)
            y2 = min(h, bb.origin_y + bb.height)
            if x2 <= x1 or y2 <= y1:
                continue
            conf = det.categories[0].score if det.categories else 0.5
            kps  = det.keypoints or []
            lm   = (np.array([[kp.x*w, kp.y*h] for kp in kps], dtype=np.float32)
                    if kps else None)
            dets.append(FaceDetection(
                bbox=(int(x1),int(y1),int(x2),int(y2)),
                confidence=float(conf), landmarks=lm,
                center=((int(x1)+int(x2))//2, (int(y1)+int(y2))//2),
            ))
        return sorted(dets, key=lambda d: d.confidence, reverse=True)

    def _detect_mp_solutions(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        h, w = frame_bgr.shape[:2]
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mp_detector.process(rgb)
        dets    = []
        if not results.detections:
            return dets
        for det in results.detections[:self.max_faces]:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            if x2 <= x1 or y2 <= y1:
                continue
            kps = det.location_data.relative_keypoints
            lm  = np.array([[kp.x*w, kp.y*h] for kp in kps], dtype=np.float32)
            dets.append(FaceDetection(
                bbox=(x1,y1,x2,y2), confidence=float(det.score[0]),
                landmarks=lm, center=((x1+x2)//2,(y1+y2)//2),
            ))
        return sorted(dets, key=lambda d: d.confidence, reverse=True)

    def _detect_haar(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        dets  = []
        for (x, y, fw, fh) in list(faces)[:self.max_faces]:
            dets.append(FaceDetection(
                bbox=(x,y,x+fw,y+fh), confidence=0.8,
                landmarks=None, center=(x+fw//2, y+fh//2),
            ))
        return dets

    # ── crop_face ─────────────────────────────────────────────────────

    def crop_face(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        target_size: Tuple[int, int] = (112, 112),
        margin: float = 0.25,
    ) -> Optional[np.ndarray]:
        h, w   = frame_bgr.shape[:2]
        x1,y1,x2,y2 = detection.bbox
        bw, bh = x2-x1, y2-y1
        mx, my = int(bw*margin), int(bh*margin)
        x1 = max(0,x1-mx); y1 = max(0,y1-my)
        x2 = min(w,x2+mx); y2 = min(h,y2+my)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    # ── release ───────────────────────────────────────────────────────

    def release(self):
        try:
            det = getattr(self, "_mp_detector", None)
            if det and hasattr(det, "close"):
                det.close()
        except Exception:
            pass
