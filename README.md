# 🛸 Unified AI Face + Cinematic Hand FX System — v3

Sistem real-time Python yang menggabungkan **face recognition**, **hand tracking**, 
**gesture control**, dan beragam **mini-game interaktif** yang dikendalikan sepenuhnya 
oleh gerakan tangan menggunakan kamera webcam biasa.

---

## ✨ Fitur Utama

### 🎯 Face Recognition Pipeline
| Fitur | Detail |
|---|---|
| Multi-face | Sampai **5 wajah** secara bersamaan |
| Backend detektor | OpenCV DNN SSD → MediaPipe → Haar Cascade (fallback otomatis) |
| Face ID–style UI | Animasi scanning oval → kartu nama verified |
| Identity locking | Setelah 5 frame konsisten → verified, enkoding berhenti |
| Wajah tidak dikenal | UI merah "Unknown User / Guest" |
| Tambah wajah baru | Cukup tambah foto di `data/faces/<id>/` + tekan R |

### ✋ Hand Tracking & Gesture Engine
| Gesture | Simbol | Aksi |
|---|---|---|
| POINT | ☝ | Hover cursor |
| PINCH | 🤌 | Click / Fire |
| OPEN_PALM | ✋ | Reset / Brake |
| ILY | 🤟 | Special FX / EMP Blast |
| FIST | ✊ | Boost / Activate Shield |
| OTHER | — | Idle |

- **MediaPipe HandLandmarker** (Tasks API atau Solutions fallback)
- Majority-vote buffer 10 frame → gesture stabil, anti-flicker
- Cooldown per-gesture mencegah spam

---

## 🎮 Mode & Mini-Game

### ✈️ Flight Simulator — Dual-Hand Yoke (`F`)
Kontrol pesawat dengan **dua tangan** — setiap tangan punya satu tugas yang jelas.

| Tangan | Fungsi | Cara |
|---|---|---|
| **Tangan Kiri** | Throttle (gas) | Angkat tinggi = gas penuh, turunkan = idle |
| **Tangan Kanan** | Pitch (hidung) | Angkat = hidung naik, turunkan = menukik |
| **Tangan Kanan** | Roll (bank) | Miringkan telapak = bank kiri/kanan |

**Cara tercepat menguasai:**
1. Tangan kiri setinggi pinggang = throttle medium (~50%)
2. Tangan kanan di depan dada = posisi netral
3. Miringkan tangan kanan ke kanan untuk belok kanan
4. Naikkan/turunkan tangan kanan untuk pitch

Fitur simulator:
- Auto-fire laser saat throttle > 5%
- Enemy fighters yang mengejar dan menembak
- Mini artificial horizon indicator
- Sistem lives & wave escalation
- Screen-shake, invincibility blink, particle explosion

### ☄️ Asteroid Dodge (`A`) ← BARU
Game survival — hindari asteroid yang jatuh dari atas!

| Gerakan/Gesture | Aksi |
|---|---|
| Gerakkan palm kiri/kanan | Gerakkan kapal |
| ✊ FIST | Aktifkan **Shield** (3 detik, cooldown 6 detik) |
| 🤟 ILY | **EMP Blast** — hancurkan semua asteroid di layar (cooldown 8 detik) |
| 🤌 PINCH | **Slow-Motion** — perlambat waktu 2 detik (cooldown 5 detik) |
| ✋ OPEN_PALM | Brake — kurangi kecepatan |

**Fitur visual:**
- Parallax star field 3 layer
- Asteroid berputar dengan bentuk tidak beraturan
- Shield bubble dengan charge meter
- EMP shockwave ring animasi
- Score multiplier (combo streak hingga x4)
- Particle explosion warna-warni
- Lives system + wave escalation setiap 15 detik

### 🐍 Snake Game (`N`)
Kontrol ular dengan arah jari telunjuk. PINCH = speed boost. OPEN_PALM = pause.

### 🪣 Catch Game (`G`)
Tangkap item jatuh dengan ujung jari. Combo, magnet, slow-mo power-up.

### 🖌️ Air Canvas (`C`)
Gambar di udara dengan jari. Ubah warna dengan gesture. `X` = clear canvas.

### 💘 Love Meter (Default)
Meter emosi 0–100 dikendalikan gesture. ILY = +20, PINCH = +5, OPEN_PALM = reset.

---

## 🚀 Cara Instalasi

### Prasyarat
- Python 3.9 – 3.11
- Webcam

### Linux / macOS
```bash
bash setup.sh
```

### Windows
```bat
setup.bat
```

### Manual
```bash
pip install -r requirements.txt
python scripts/build_dataset.py
```

---

## ▶️ Cara Menjalankan

```bash
# Default (kamera 0)
python main.py

# Kamera alternatif
python main.py --camera 1

# Mode debug (tampilkan info teknis)
python main.py --debug

# Atur threshold pengenalan wajah
python main.py --threshold 0.68
```

---

## ⌨️ Keyboard Controls Lengkap

| Tombol | Fungsi |
|---|---|
| `Q` / `ESC` | Keluar |
| `S` | Simpan screenshot |
| `R` | Rebuild face embeddings |
| `D` | Toggle debug overlay |
| `M` | Toggle mirror (flip kamera) |
| `H` | Toggle kualitas Hand FX (high ↔ low) |
| `C` | Toggle **Air Canvas** mode |
| `G` | Toggle **Catch Game** |
| `N` | Toggle **Snake Game** |
| `F` | Toggle **Flight Simulator** (dual-hand yoke) |
| `A` | Toggle **Asteroid Dodge** ← BARU |
| `X` | Clear canvas (saat Air Canvas aktif) |
| `U` | Undo stroke terakhir (saat Air Canvas aktif) |

> Setiap kali beralih mode, game direset otomatis (high score disimpan).

---

## 📁 Struktur Proyek

```
face+cinematic_hand/
│
├── main.py                       ← Entry point utama
│
├── hand_flight_controller.py     ← BARU: Single-hand aircraft controller
│
├── virtual_yoke.py               ← Dual-hand yoke controller
├── flight_sim.py                 ← Aerial Shooter Flight Simulator
│
├── core/                         ← Face recognition pipeline
│   ├── face_detector.py          ← Multi-backend (DNN/MediaPipe/Haar)
│   ├── encoder.py                ← 128-d face embeddings
│   ├── matcher.py                ← Cosine-similarity identity matching
│   ├── tracker.py                ← SORT multi-object tracker
│   ├── smoother.py               ← EMABox, IdentityBuffer
│   └── identity_cache.py         ← Per-track identity + FaceUIState
│
├── hand/                         ← Hand tracking + gesture + FX
│   ├── tracker.py                ← MediaPipe HandLandmarker
│   ├── gesture.py                ← Gesture classifier (majority-vote)
│   ├── fx.py                     ← Trail, particles, pinch flash
│   ├── cursor.py                 ← Smoothed cursor
│   └── air_canvas.py             ← Air drawing canvas
│
├── game/                         ← Mini-game collection
│   ├── love_meter.py             ← Love Meter (default mode)
│   ├── catch_game.py             ← Catch Game
│   ├── snake_game.py             ← Neon Snake
│   └── asteroid_dodge.py         ← BARU: Asteroid Dodge
│
├── ui/                           ← Rendering overlay
│   ├── multi_face_overlay.py     ← Face ID-style UI
│   └── hud.py                    ← FPS, gesture, status HUD
│
├── models/                       ← Model weights
│   ├── deploy.prototxt
│   └── res10_300x300_ssd.caffemodel
│
├── data/
│   ├── faces/                    ← Training images per orang
│   ├── identities.json           ← Nama & role display
│   └── encodings.pkl             ← Di-build oleh setup
│
├── scripts/
│   └── build_dataset.py          ← Encode semua foto → encodings.pkl
│
├── hand_landmarker.task          ← MediaPipe model file
├── requirements.txt
├── setup.sh
└── setup.bat
```

---

## 👤 Menambahkan Wajah Baru

1. Buat folder `data/faces/<id_unik>/` dan masukkan 4–10 foto jelas.
2. Edit `data/identities.json`:
   ```json
   {
     "id_unik": {
       "display_name": "Nama Lengkap",
       "role": "Jabatan / Role"
     }
   }
   ```
3. Tekan **R** saat sistem berjalan, atau jalankan ulang `bash setup.sh`.

---

## ⚡ Tips Performa

- **Target:** 30 FPS (auto-downscale jika FPS < 20)
- Setelah semua wajah verified, face detection di-skip setiap frame genap.
- Particle pool dibatasi 600 objek.
- Mode `H` → low particle mode untuk CPU lemah.
- Pastikan pencahayaan cukup untuk akurasi gesture.

---

## 🗺️ Alur Sistem

```
Webcam
  ↓
FaceDetector (max 5 wajah)  ←→  SORTTracker  ←→  IdentityCache
  ↓
MultiFaceOverlay (SCANNING → VERIFIED UI)
  ↓
HandTracker (MediaPipe, selalu aktif)
  ↓
GestureEngine (majority-vote, cooldown)
  ↓
HandFX (trail, partikel, pinch flash, ILY burst)
  ↓
MODE SELECTOR:
  C → AirCanvas
  G → CatchGame
  N → SnakeGame
  F → FlightSim (dual-hand yoke)
  B → FlightSim (single-hand)  ← BARU
  A → AsteroidDodge            ← BARU
  default → LoveMeter
  ↓
HUD (FPS, gesture, face/hand count)
  ↓
cv2.imshow
```

---

## 📋 Requirements

```
opencv-python>=4.8
numpy>=1.24
mediapipe>=0.10
face_recognition
dlib
```

---

## 📝 Changelog v3

- **BARU** `hand_flight_controller.py` — single-hand aircraft controller
- **BARU** `game/asteroid_dodge.py` — Asteroid Dodge mini-game dengan shield, EMP, slowmo
- **UPDATE** `main.py` — tambah mode `B` (single-hand flight) dan `A` (asteroid dodge)
- **UPDATE** README — dokumentasi lengkap semua fitur dan kontrol
