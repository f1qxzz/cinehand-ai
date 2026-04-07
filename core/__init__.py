"""
core/ — Face recognition pipeline components.
Pulled from ai_face_system and adapted for unified system.
"""
from .face_detector import FaceDetector, FaceDetection
from .encoder       import FaceEncoder
from .matcher       import IdentityMatcher, MatchResult, UNKNOWN_IDENTITY
from .tracker       import SORTTracker
from .smoother      import EMABox, IdentityBuffer, SimilarityEMA
from .identity_cache import IdentityCache, TrackState, FaceUIState

__all__ = [
    "FaceDetector", "FaceDetection",
    "FaceEncoder",
    "IdentityMatcher", "MatchResult", "UNKNOWN_IDENTITY",
    "SORTTracker",
    "EMABox", "IdentityBuffer", "SimilarityEMA",
    "IdentityCache", "TrackState", "FaceUIState",
]
