"""
scripts/build_dataset.py
Encode all face images in data/faces/<identity>/*.jpg
and save embeddings to data/encodings.pkl + data/identities.json.

Run from project root:
    python scripts/build_dataset.py
"""

import os
import sys
import json
import pickle
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── resolve project root ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.face_detector import FaceDetector
from core.encoder import FaceEncoder


# ── Registered users ──────────────────────────────────────────────────────────
# Add new users here before running this script.
# "folder_name" must match the subfolder name under data/faces/
# ──────────────────────────────────────────────────────────────────────────────
IDENTITY_CONFIG = {
    "f1qxzz": {
        "display_name": "f1qxzz",
        "role": "Developer",
    },
    # "afna": {
    #     "display_name": "Afna Feyza",
    #     "role": "Member",
    # },
}


def build_dataset(
    faces_dir:     str = "data/faces",
    encodings_out: str = "data/encodings.pkl",
    identities_out:str = "data/identities.json",
    margin:        float = 0.25,
    augment:       bool  = True,
):
    faces_path = ROOT / faces_dir
    if not faces_path.exists():
        print(f"[Build] Faces directory not found: {faces_path}")
        return

    detector = FaceDetector(model_selection=0, min_detection_confidence=0.5)
    encoder  = FaceEncoder(model_path=str(ROOT / "models" / "face_encoder.h5"))

    db     = {}
    errors = []

    identity_dirs = [d for d in sorted(faces_path.iterdir()) if d.is_dir()]
    if not identity_dirs:
        print("[Build] No identity subfolders found in", faces_path)
        return

    for id_dir in identity_dirs:
        identity_id = id_dir.name

        images      = list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png")) + \
                      list(id_dir.glob("*.jpeg"))

        if not images:
            print(f"  [skip] {identity_id}: no images")
            continue

        print(f"\n[Build] Processing '{identity_id}' ({len(images)} images)…")
        embeddings = []

        for img_path in tqdm(images, desc=f"  {identity_id}"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                errors.append(str(img_path))
                continue

            dets = detector.detect(frame)
            if not dets:
                # Fall back: use whole image as face crop
                crop = cv2.resize(frame, (112, 112))
            else:
                crop = detector.crop_face(frame, dets[0], margin=margin)
                if crop is None:
                    crop = cv2.resize(frame, (112, 112))

            emb = encoder.encode(crop)
            if emb is None:
                continue
            embeddings.append(emb)

            # ── Augmentation ────────────────────────────────────────
            if augment:
                for aug_crop in _augment(crop):
                    ae = encoder.encode(aug_crop)
                    if ae is not None:
                        embeddings.append(ae)

        if embeddings:
            db[identity_id] = embeddings
            print(f"  → {len(embeddings)} embeddings stored for '{identity_id}'")
        else:
            print(f"  ✗ No valid embeddings for '{identity_id}'")

    # ── Save ─────────────────────────────────────────────────────────
    out_enc = ROOT / encodings_out
    out_id  = ROOT / identities_out
    out_enc.parent.mkdir(parents=True, exist_ok=True)

    with open(out_enc, "wb") as f:
        pickle.dump(db, f)
    print(f"\n[Build] Saved {sum(len(v) for v in db.values())} embeddings → {out_enc}")

    # Merge with IDENTITY_CONFIG (fallback for unlisted users)
    meta = {}
    for k in db:
        meta[k] = IDENTITY_CONFIG.get(k, {"display_name": k, "role": "Member"})
    with open(out_id, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Build] Saved identities metadata → {out_id}")

    if errors:
        print(f"[Build] Could not load {len(errors)} image(s):", errors[:5])

    detector.release()
    print("\n[Build] Done.")


# ── Simple augmentations ─────────────────────────────────────────────
def _augment(crop: np.ndarray):
    """Yield a few augmented variants of a face crop."""
    # Horizontal flip
    yield cv2.flip(crop, 1)
    # Brightness up / down
    for alpha in (1.15, 0.85):
        yield np.clip(crop.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    # Slight Gaussian blur
    yield cv2.GaussianBlur(crop, (3, 3), 0)


if __name__ == "__main__":
    build_dataset()
