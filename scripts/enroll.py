"""
scripts/enroll.py
=================
Daftarkan wajah baru langsung dari webcam — hasilnya langsung masuk
ke data/encodings.pkl + data/identities.json tanpa perlu foto manual.

Cara pakai:
    python scripts/enroll.py --identity f1qxzz
    python scripts/enroll.py --identity user1 --display "Username" --role Member
    python scripts/enroll.py --identity user1 --count 80 --camera 1

Kontrol saat jendela terbuka:
    SPASI  → mulai/jeda capture
    Q / ESC → selesai & simpan

Isi IDENTITY_CONFIG di sini untuk mendaftarkan nama & role secara permanen.
Jika --display / --role tidak diisi, program otomatis cek IDENTITY_CONFIG.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── resolve project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.face_detector import FaceDetector
from core.encoder import FaceEncoder

# ── Registered identity metadata ──────────────────────────────────────────────
# Tambah user baru di sini (opsional — bisa juga pakai --display & --role)
IDENTITY_CONFIG = {
    "f1qxzz": {
        "display_name": "f1qxzz",
        "role": "Developer",
    },
    # "user1": {
    #     "display_name": "Username",
    #     "role": "Member",
    # },
}

# ── Paths ─────────────────────────────────────────────────────────────────────
ENCODINGS_PATH  = ROOT / "data" / "encodings.pkl"
IDENTITIES_PATH = ROOT / "data" / "identities.json"
FACES_DIR       = ROOT / "data" / "faces"

# ── Augmentation ──────────────────────────────────────────────────────────────
def _augment(crop: np.ndarray):
    yield cv2.flip(crop, 1)
    for alpha in (1.15, 0.85):
        yield np.clip(crop.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    yield cv2.GaussianBlur(crop, (3, 3), 0)


# ── Load existing DB ──────────────────────────────────────────────────────────
def _load_db():
    if ENCODINGS_PATH.exists():
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _load_meta():
    if IDENTITIES_PATH.exists():
        with open(IDENTITIES_PATH) as f:
            return json.load(f)
    return {}


# ── Save DB ───────────────────────────────────────────────────────────────────
def _save(db, meta):
    ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(db, f)
    with open(IDENTITIES_PATH, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ── Main enroll ───────────────────────────────────────────────────────────────
def enroll(identity_id: str, display_name: str, role: str,
           target_count: int, camera_idx: int, augment: bool):

    print(f"\n[Enroll] Identity : {identity_id}")
    print(f"         Display  : {display_name}")
    print(f"         Role     : {role}")
    print(f"         Target   : {target_count} samples")
    print(f"         Camera   : {camera_idx}\n")

    detector = FaceDetector(model_selection=0, min_detection_confidence=0.5)
    encoder  = FaceEncoder(model_path=str(ROOT / "models" / "face_encoder.h5"))

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[Enroll] ERROR: Cannot open camera.")
        return

    db   = _load_db()
    meta = _load_meta()

    existing = len(db.get(identity_id, []))
    print(f"[Enroll] Existing embeddings for '{identity_id}': {existing}")

    embeddings = list(db.get(identity_id, []))

    capturing = False
    captured  = 0
    last_cap  = 0.0
    CAP_INTERVAL = 0.12   # seconds between captures

    # Save face images too
    face_dir = FACES_DIR / identity_id
    face_dir.mkdir(parents=True, exist_ok=True)
    img_idx = len(list(face_dir.glob("*.jpg")))

    print("[Enroll] Tekan SPASI untuk mulai capture, Q/ESC untuk selesai & simpan.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)   # mirror
        dets  = detector.detect(frame)
        disp  = frame.copy()

        H, W = disp.shape[:2]

        # ── Status panel ─────────────────────────────────────────────
        total_so_far = len(embeddings)
        pct  = min(100, int(total_so_far / target_count * 100))
        bar_w = 300
        bar_h = 14
        bx = W // 2 - bar_w // 2
        by = H - 60
        cv2.rectangle(disp, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
        fill = int(bar_w * pct / 100)
        col_bar = (80, 220, 80) if capturing else (100, 180, 255)
        if fill > 0:
            cv2.rectangle(disp, (bx, by), (bx + fill, by + bar_h), col_bar, -1)
        cv2.rectangle(disp, (bx, by), (bx + bar_w, by + bar_h), (120, 120, 120), 1)

        status_txt = f"{'● CAPTURING' if capturing else '○ PAUSED'}   {total_so_far}/{target_count} ({pct}%)"
        (tw, _), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(disp, status_txt,
                    (W // 2 - tw // 2, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_bar, 1, cv2.LINE_AA)

        hint = "SPASI: mulai/jeda  |  Q / ESC: selesai & simpan"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(disp, hint,
                    (W // 2 - hw // 2, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # ── Draw detections ───────────────────────────────────────────
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            color = (80, 220, 80) if capturing else (100, 180, 255)
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            label = f"{identity_id}  {det.confidence:.0%}"
            cv2.putText(disp, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # ── Capture ───────────────────────────────────────────────────
        now = time.time()
        if capturing and dets and (now - last_cap) >= CAP_INTERVAL:
            det  = dets[0]
            crop = detector.crop_face(frame, det, margin=0.25)
            if crop is not None:
                emb = encoder.encode(crop)
                if emb is not None:
                    embeddings.append(emb)
                    if augment:
                        for aug in _augment(crop):
                            ae = encoder.encode(aug)
                            if ae is not None:
                                embeddings.append(ae)

                    # Save image
                    img_path = face_dir / f"{img_idx:04d}.jpg"
                    cv2.imwrite(str(img_path), crop)
                    img_idx += 1
                    captured += 1
                    last_cap  = now

                    # Flash feedback
                    ov = disp.copy()
                    cv2.rectangle(ov, (0, 0), (W, H), (80, 220, 80), -1)
                    cv2.addWeighted(ov, 0.08, disp, 0.92, 0, disp)

            if len(embeddings) >= target_count:
                print(f"\n[Enroll] Target tercapai! ({len(embeddings)} embeddings)")
                capturing = False

        cv2.imshow(f"Enroll — {identity_id}", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            capturing = not capturing
            state_str = "MULAI" if capturing else "JEDA"
            print(f"[Enroll] {state_str} capture  (sudah: {len(embeddings)} embeddings)")

        elif key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    if not embeddings:
        print("[Enroll] Tidak ada embedding yang diambil. Keluar tanpa menyimpan.")
        return

    # ── Merge & save ─────────────────────────────────────────────────
    db[identity_id] = embeddings
    meta[identity_id] = {"display_name": display_name, "role": role}

    _save(db, meta)

    total = sum(len(v) for v in db.values())
    print(f"\n[Enroll] Disimpan {len(embeddings)} embeddings untuk '{identity_id}'")
    print(f"[Enroll] Total DB : {total} embeddings, {len(db)} identitas")
    print(f"[Enroll] File     : {ENCODINGS_PATH}")
    print(f"[Enroll] Metadata : {IDENTITIES_PATH}")
    print(f"\n[Enroll] Selesai! Jalankan main.py untuk verifikasi.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Enroll wajah baru dari webcam ke face recognition database."
    )
    ap.add_argument("--identity", required=True,
                    help="ID unik user, harus sama dengan nama folder di data/faces/")
    ap.add_argument("--display",  default="",
                    help="Nama yang ditampilkan di HUD (default: ambil dari IDENTITY_CONFIG)")
    ap.add_argument("--role",     default="",
                    help="Role yang ditampilkan di HUD (default: ambil dari IDENTITY_CONFIG)")
    ap.add_argument("--count",    type=int, default=60,
                    help="Jumlah sample target (default: 60)")
    ap.add_argument("--camera",   type=int, default=0,
                    help="Index kamera (default: 0)")
    ap.add_argument("--no-augment", action="store_true",
                    help="Matikan augmentasi data (flip, brightness, blur)")
    args = ap.parse_args()

    # Resolve display name & role
    cfg = IDENTITY_CONFIG.get(args.identity, {})
    display = args.display or cfg.get("display_name", args.identity)
    role    = args.role    or cfg.get("role", "Member")

    enroll(
        identity_id  = args.identity,
        display_name = display,
        role         = role,
        target_count = args.count,
        camera_idx   = args.camera,
        augment      = not args.no_augment,
    )


if __name__ == "__main__":
    main()
