"""
main.py — Unified AI Face + Cinematic Hand FX System  [PRO v3]
===============================================================

Global Controls:
  Q / ESC        quit
  Shift+S        screenshot (S alone used for Snake steering)
  R              rebuild face embeddings
  D              toggle debug info
  M              toggle mirror (flip camera)
  H              toggle hand FX quality (high <-> low)
  TAB            open/close mode selection menu

Mode Keys:
  C              Air Canvas (draw with finger)
  G              Catch Game
  N              Snake Game  ← keyboard steerable
  F              Flight Simulator (dual-hand yoke)
  B              Flight Simulator — single-hand mode
  A              Asteroid Dodge

Snake Game Controls (when N mode active):
  Arrow Keys     steer (cross-platform)
  W A S D        steer (WASD — note: S only works here in snake mode)
  Space          boost speed (2.5 s)
  P              pause / resume
  ─── OR use hand gestures ───
  ☝  POINT       steer with finger direction
  🤌 PINCH        boost
  🖐 OPEN_PALM    pause

Canvas Controls (when C mode active):
  U              undo last stroke
  X              clear canvas
  [ / ]          brush size decrease / increase
"""

import os, sys, time, math, argparse
import cv2, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.face_detector  import FaceDetector
from core.encoder        import FaceEncoder
from core.matcher        import IdentityMatcher
from core.tracker        import SORTTracker
from core.identity_cache import IdentityCache, FaceUIState

from hand.tracker  import HandTracker, HAND_CONNECTIONS, FINGERTIP_INDICES, WRIST
from hand.gesture  import GestureEngine, Gesture
from hand.fx       import HandFX
from hand.cursor   import CursorController

from ui.multi_face_overlay import MultiFaceOverlay
from ui.hud                import HUD
from game.love_meter       import LoveMeter
from hand.air_canvas       import AirCanvas
from game.catch_game       import CatchGame
from game.snake_game       import SnakeGame, snake_key_event

# ── NEW SYSTEMS ────────────────────────────────────────────────────────────────
from virtual_yoke import VirtualYoke, YokeData
from flight_sim   import FlightSimulator

# ── BARU: Single-hand controller + Asteroid Dodge ─────────────────────────────
from hand_flight_controller import HandFlightController, HandControlData
from game.asteroid_dodge    import AsteroidDodge


try:
    from core.multitask_model import MultitaskModel
    _HAS_MULTITASK = True
except Exception:
    _HAS_MULTITASK = False

TARGET_FPS   = 30
MIN_FPS      = 20
FRAME_BUDGET = 1.0 / TARGET_FPS

# ── UPGRADED: max 5 faces ─────────────────────────────────────────────────────
MAX_FACES    = 5

FACE_ENCODE_SKIP = 3


# ══════════════════════════════════════════════════════════════════════════════
# Face helpers (NEW: filter + status)
# ══════════════════════════════════════════════════════════════════════════════

def filter_faces(detections, max_faces: int = 5):
    """
    Keep at most *max_faces* detections, preferring the largest bounding boxes.
    Works on any list of objects with a .bbox attribute (x1,y1,x2,y2).
    """
    if len(detections) <= max_faces:
        return detections

    def _area(d):
        if hasattr(d, 'bbox'):
            x1, y1, x2, y2 = d.bbox
        else:
            x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
        return max(0, x2 - x1) * max(0, y2 - y1)

    return sorted(detections, key=_area, reverse=True)[:max_faces]


def get_face_status(face_count: int) -> tuple:
    """
    Returns (text, colour_BGR) for the face-status overlay.
      0 faces  → grey
      1-4      → green
      5        → yellow  (near limit)
      >5       → red     (shouldn't happen after filter)
    """
    if face_count == 0:
        return "NO FACE DETECTED", (120, 120, 120)
    elif face_count < 5:
        return f"FACES: {face_count} / 5", (80, 220, 80)
    elif face_count == 5:
        return "FACES: 5 / 5", (60, 200, 220)
    else:
        return "LIMIT REACHED (5 MAX)", (60, 60, 220)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1: Face Aura FX  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════
class FaceAuraFX:
    """Cinematic animated ring around each tracked face based on state."""

    def __init__(self):
        self._phase = {}

    def update_and_draw(self, frame, active_faces, dt):
        for face in active_faces:
            tid   = face["track_id"]
            state = face["state"]
            bbox  = face["bbox"]

            if tid not in self._phase:
                self._phase[tid] = 0.0
            self._phase[tid] += dt
            ph = self._phase[tid]

            x1,y1,x2,y2 = bbox
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            rx = max(20, (x2-x1)//2 + 20)
            ry = max(24, (y2-y1)//2 + 26)

            if state == FaceUIState.SCANNING:
                self._scanning(frame, cx, cy, rx, ry, ph)
            elif state == FaceUIState.UNKNOWN:
                self._unknown(frame, cx, cy, rx, ry, ph)

    def _scanning(self, frame, cx, cy, rx, ry, ph):
        ov = frame.copy()
        n  = 20
        for i in range(n):
            if i % 4 == 0:
                continue
            a1 = math.radians(360/n*i + ph*90)
            a2 = math.radians(360/n*(i+1) + ph*90)
            s  = int(math.degrees(a1)) % 360
            e  = int(math.degrees(a2)) % 360
            b  = 0.5 + 0.3*math.sin(ph*3 + i)
            cv2.ellipse(ov,(cx,cy),(rx,ry),0,s,e,
                        (int(180*b),int(230*b),int(255*b)),2,cv2.LINE_AA)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    def _verified(self, frame, cx, cy, rx, ry, ph):
        pulse = 1.0 + 0.045*math.sin(ph*4.0)
        prx, pry = int(rx*pulse), int(ry*pulse)
        ov = frame.copy()
        for thick, shrink, base_a in [(20,8,0.06),(12,4,0.13),(4,0,0.40)]:
            a = base_a*(0.85+0.15*math.sin(ph*3))
            cv2.ellipse(ov,(cx,cy),(max(1,prx-shrink),max(1,pry-shrink)),
                        0,0,360,(int(30*a),int(190*a),int(255*a)),thick,cv2.LINE_AA)
        cv2.addWeighted(ov, 1.0, frame, 0.0, 0, frame)
        for i in range(8):
            ang = ph*1.3 + 2*math.pi/8*i
            ex  = int(cx + prx*1.1*math.cos(ang))
            ey  = int(cy + pry*1.1*math.sin(ang))
            br  = int(160 + 95*math.sin(ph*4+i))
            cv2.circle(frame,(ex,ey),3,(br//3,br,br),-1,cv2.LINE_AA)
            cv2.circle(frame,(ex,ey),5,(br//6,br//2,br//2),1,cv2.LINE_AA)

    def _unknown(self, frame, cx, cy, rx, ry, ph):
        blink = abs(math.sin(ph*5.5))
        a     = 0.25 + 0.55*blink
        ov = frame.copy()
        cv2.ellipse(ov,(cx,cy),(rx,ry),0,0,360,
                    (int(40*a),int(40*a),int(255*a)),3,cv2.LINE_AA)
        cv2.ellipse(ov,(cx,cy),(rx+6,ry+6),0,0,360,
                    (int(15*a),int(15*a),int(180*a)),1,cv2.LINE_AA)
        cv2.addWeighted(ov, 0.9, frame, 0.1, 0, frame)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2: Neon Hand Skeleton  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════
_FINGER_HUE = {
    **dict.fromkeys([1,2,3,4],   22),
    **dict.fromkeys([5,6,7,8],   80),
    **dict.fromkeys([9,10,11,12],130),
    **dict.fromkeys([13,14,15,16],160),
    **dict.fromkeys([17,18,19,20],270),
    0: 30,
}

def _hsv_bgr(hue_deg, s=220, v=255):
    h = int(hue_deg/2) % 180
    return cv2.cvtColor(np.uint8([[[h,s,v]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()

def draw_neon_hand(frame, hr, W, H, gesture_label, ph):
    lm = hr.landmarks
    def px(i):
        return int(lm[i].x*W), int(lm[i].y*H)

    for a,b in HAND_CONNECTIONS:
        pa, pb = px(a), px(b)
        col    = _hsv_bgr(_FINGER_HUE.get(b, 90))
        glow   = [c//5 for c in col]
        cv2.line(frame, pa, pb, glow, 8, cv2.LINE_AA)
        cv2.line(frame, pa, pb, col,  2, cv2.LINE_AA)

    for i in range(21):
        pt  = px(i)
        col = _hsv_bgr(_FINGER_HUE.get(i,90))
        r   = 7 if i in FINGERTIP_INDICES else 3
        cv2.circle(frame, pt, r+4, [c//4 for c in col], -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r,   col,                 -1, cv2.LINE_AA)
        if i in FINGERTIP_INDICES:
            cv2.circle(frame, pt, r+8, [c//8 for c in col], 1, cv2.LINE_AA)

    wp = px(WRIST)
    wb = int(180 + 75*math.sin(ph*4))
    cv2.circle(frame, wp, 11, (wb//4, wb//4, wb), -1, cv2.LINE_AA)
    cv2.circle(frame, wp, 16, (wb//8, wb//8, wb//2), 1, cv2.LINE_AA)

    if gesture_label and gesture_label not in ("OTHER",""):
        BADGE_HUE = {"PINCH":200,"POINT":80,"ILY":140,"OPEN_PALM":40,"FIST":0}
        col = _hsv_bgr(BADGE_HUE.get(gesture_label, 90))
        bx, by = wp[0]-52, wp[1]-36
        cv2.putText(frame, gesture_label,(bx+1,by+1),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(frame, gesture_label,(bx,by),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,col,1,cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Face Status UI
# ══════════════════════════════════════════════════════════════════════════════
class FaceStatusUI:
    """
    Semi-transparent glow banner showing face count.
    Uses a 3-frame median buffer to prevent single-frame flicker.
    """
    _BUFFER_FRAMES = 3

    def __init__(self):
        self._count_buf  = []
        self._displayed  = 0

    def update(self, raw_count: int) -> int:
        self._count_buf.append(raw_count)
        if len(self._count_buf) > self._BUFFER_FRAMES:
            self._count_buf.pop(0)
        sorted_buf = sorted(self._count_buf)
        self._displayed = sorted_buf[len(sorted_buf) // 2]
        return self._displayed

    def draw(self, frame, count: int):
        text, col = get_face_status(count)
        H, W = frame.shape[:2]

        if count == 0:
            glow = (50, 50, 50)
        elif count < 5:
            glow = (0, 80, 0)
        elif count == 5:
            glow = (0, 80, 80)
        else:
            glow = (0, 0, 100)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

        margin = 10
        bx0 = W - tw - 20 - margin
        by0 = margin
        bw  = tw + 20
        bh  = th + 14

        bg_roi = frame[by0:by0+bh, bx0:bx0+bw]
        if bg_roi.size > 0:
            dark = np.full_like(bg_roi, (8, 10, 16))
            cv2.addWeighted(dark, 0.78, bg_roi, 0.22, 0, bg_roi)
            frame[by0:by0+bh, bx0:bx0+bw] = bg_roi

        cv2.rectangle(frame, (bx0, by0), (bx0+bw, by0+bh), glow, 1)

        tx = bx0 + 10
        ty = by0 + th + 4
        cv2.putText(frame, text, (tx+1, ty+1), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
        cv2.putText(frame, text, (tx,   ty  ), font, scale, col,     thick,   cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
class UnifiedSystem:

    def __init__(self, args):
        print("\n[System] Starting Unified AI Face + Hand FX System By @f1qxzz\n")

        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        ok, _ = self.cap.read()
        if not ok:
            raise RuntimeError("Cannot open camera.")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.mirror = getattr(args, "mirror", False)
        print(f"[Camera]  {self.W}x{self.H} @ {TARGET_FPS}fps  mirror={self.mirror}  (M=toggle)")

        self.detector = FaceDetector(model_selection=1,
                                     min_detection_confidence=0.55,
                                     max_faces=MAX_FACES)
        self.encoder  = FaceEncoder(model_path=str(ROOT/"models"/"face_encoder.h5"))
        self.matcher  = IdentityMatcher(
            encodings_path  = str(ROOT/"data"/"encodings.pkl"),
            identities_path = str(ROOT/"data"/"identities.json"),
            threshold_accept = args.threshold,
        )
        if not self.matcher._db:
            print("[Face]    WARNING: No encodings — detection-only, all faces = 'Unknown'")
        self.sort     = SORTTracker(max_age=10, min_hits=2)
        self.id_cache = IdentityCache()

        self.multitask = None
        if _HAS_MULTITASK:
            try: self.multitask = MultitaskModel()
            except Exception: pass

        self.face_aura      = FaceAuraFX()
        self.face_status_ui = FaceStatusUI()          # NEW

        self.hand_tracker   = HandTracker(max_hands=2, min_conf=0.45)
        self.gesture_engine = GestureEngine()
        self.cursor         = CursorController()
        self.hand_fx        = HandFX(low_particle_mode=False)
        self.face_ui        = MultiFaceOverlay(self.W, self.H)
        self.hud            = HUD(self.W, self.H)
        self.love_meter     = LoveMeter()
        self.air_canvas     = AirCanvas(self.W, self.H)
        self.catch_game     = CatchGame(self.W, self.H)
        self.snake_game     = SnakeGame(self.W, self.H)

        # ── NEW ─────────────────────────────────────────────────────────
        self.yoke       = VirtualYoke(self.W, self.H)
        self.flight_sim = FlightSimulator(self.W, self.H)
        self._yoke_data = YokeData()

        # ── BARU v3: Single-hand controller + Asteroid Dodge ─────────────
        self.hand_ctrl       = HandFlightController(self.W, self.H)
        self._hand_ctrl_data = HandControlData()
        self.asteroid_game   = AsteroidDodge(self.W, self.H)

        # Mode flags
        self._canvas_mode     = False
        self._game_mode       = False
        self._snake_mode      = False
        self._flight_mode     = False   # dual-hand yoke
        self._hand_flight_mode = False  # NEW: single-hand mode
        self._asteroid_mode   = False   # NEW: asteroid dodge

        # ── Runtime state ────────────────────────────────────────────────
        self.debug            = getattr(args, "debug", False)
        self._prev_frame_t    = time.time()
        self._global_ph       = 0.0
        self._frame_count     = 0
        self._fps_buf         = []
        self._fps_display     = 0.0
        self._low_res_mode    = False
        self._last_hands      = []
        self._last_gesture    = "OTHER"
        self._last_landmarks  = None
        self.screenshot_n     = 0

        print("[Camera]  OK")
        print("[Face]    Running  (max 5) + AuraFX + StatusUI")
        print("[Tracker] Stable")
        print("[Hand]    Active   + NeonSkeleton + AirCanvas + CatchGame + FlightYoke + HandCtrl + AsteroidDodge")

    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow("Unified System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Unified System", self.W, self.H)

        while True:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01); continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            dt = min(t0 - self._prev_frame_t, 0.1)
            self._prev_frame_t = t0
            self._global_ph   += dt
            self._frame_count += 1

            active_faces = self._run_face_pipeline(frame)
            self.face_aura.update_and_draw(frame, active_faces, dt)

            # Face status UI (flicker-smoothed)
            stable_count = self.face_status_ui.update(len(active_faces))
            self.face_status_ui.draw(frame, stable_count)

            gesture, cursor_pos = self._run_hand_pipeline(frame, dt)
            self.love_meter.update(dt)

            # ── Mode-specific overlays ────────────────────────────────────
            if self._canvas_mode:
                self.air_canvas.render(frame)
            elif self._game_mode:
                self.catch_game.render(frame)
            elif self._snake_mode:
                self.snake_game.render(frame)
            elif self._flight_mode:
                self.flight_sim.render(frame)
                self.yoke.draw(frame, self._yoke_data)
            elif self._hand_flight_mode:
                self.flight_sim.render(frame)
                self.hand_ctrl.draw(frame, self._hand_ctrl_data)
            elif self._asteroid_mode:
                self.asteroid_game.render(frame)
                self.hand_ctrl.draw(frame, self._hand_ctrl_data)
            else:
                self.love_meter.render(frame, 12, self.H-130)

            self.face_ui.render(frame, active_faces)
            self.hud.render(
                frame,
                fps            = self._fps_display,
                gesture        = gesture.value if gesture else "OTHER",
                hands_detected = len(self._last_hands),
                faces_detected = len(active_faces),
                faces_verified = sum(1 for f in active_faces
                                     if f["state"]==FaceUIState.VERIFIED),
                mirror         = self.mirror,
                canvas_mode    = self._canvas_mode,
                game_mode      = self._game_mode or self._snake_mode or self._flight_mode or self._hand_flight_mode or self._asteroid_mode,
            )
            if self.debug:
                self._draw_debug(frame)

            elapsed = time.time() - t0
            self._update_fps(elapsed)
            self._fps_throttle(elapsed)

            cv2.imshow("AI Face + Hand System @f1qxzz", frame)

            # ── Read key (raw = unmasked, for arrow-key detection) ────────
            raw = cv2.waitKey(1)
            key = raw & 0xFF

            # ── Quit (always first) ───────────────────────────────────────
            if key in (ord('q'), 27):
                break

            # ── Snake mode: route keys through snake_key_event exclusively ─
            if self._snake_mode:
                if snake_key_event(key):
                    pass   # WASD / Arrow consumed
                elif key == 32:   # Space → boost
                    self.snake_game._boost_end = time.time() + 2.5
                elif key in (ord('p'), ord('P')):   # P → pause
                    self.snake_game._paused = not self.snake_game._paused
                elif key == ord('n'):   # N → exit snake mode
                    self._snake_mode = False
                    print("[Mode] SnakeGame → False")
                # All other keys ignored while in snake mode to prevent conflicts
                continue   # ← skip global key block entirely

            # ── Global / mode keys (only when NOT in snake mode) ──────────
            if key == ord('s'):
                self._save_screenshot(frame)
            elif key == ord('r'):
                self._rebuild_embeddings()
            elif key == ord('d'):
                self.debug = not self.debug
            elif key == ord('m'):
                self.mirror = not self.mirror
                print(f"[Camera] Mirror → {self.mirror}")
            elif key == ord('h'):
                low = not self.hand_fx._low_mode
                self.hand_fx.set_low_mode(low)
                print(f"[HandFX] Low-mode → {low}")
            elif key == ord('c'):
                self._canvas_mode = not self._canvas_mode
                if self._canvas_mode:
                    self._game_mode = self._snake_mode = False
                    self._flight_mode = self._hand_flight_mode = self._asteroid_mode = False
                print(f"[Mode] AirCanvas → {self._canvas_mode}")
            elif key == ord('g'):
                self._game_mode = not self._game_mode
                if self._game_mode:
                    self._canvas_mode = self._snake_mode = False
                    self._flight_mode = self._hand_flight_mode = self._asteroid_mode = False
                    self.catch_game = CatchGame(self.W, self.H)
                print(f"[Mode] CatchGame → {self._game_mode}")
            elif key == ord('n'):
                self._snake_mode = not self._snake_mode
                if self._snake_mode:
                    self._canvas_mode = self._game_mode = False
                    self._flight_mode = self._hand_flight_mode = self._asteroid_mode = False
                    hs = self.snake_game.high_score
                    self.snake_game = SnakeGame(self.W, self.H)
                    self.snake_game.high_score = hs
                    print("[Snake] WASD / Arrow Keys = Steer  |  Space = Boost  |  P = Pause  |  N = Exit")
                print(f"[Mode] SnakeGame → {self._snake_mode}")
            elif key == ord('f'):
                if self._flight_mode and self.flight_sim._game_over:
                    hs = self.flight_sim.high_score
                    self.flight_sim = FlightSimulator(self.W, self.H)
                    self.flight_sim.high_score = hs
                else:
                    self._flight_mode = not self._flight_mode
                    if self._flight_mode:
                        self._canvas_mode = self._game_mode = self._snake_mode = False
                        self._hand_flight_mode = self._asteroid_mode = False
                        hs = self.flight_sim.high_score
                        self.flight_sim = FlightSimulator(self.W, self.H)
                        self.flight_sim.high_score = hs
                print(f"[Mode] FlightSim → {self._flight_mode}")
            elif key == ord('a'):
                if self._asteroid_mode and self.asteroid_game._game_over:
                    hs = self.asteroid_game.high_score
                    self.asteroid_game = AsteroidDodge(self.W, self.H)
                    self.asteroid_game.high_score = hs
                else:
                    self._asteroid_mode = not self._asteroid_mode
                    if self._asteroid_mode:
                        self._canvas_mode = self._game_mode = self._snake_mode = False
                        self._flight_mode = self._hand_flight_mode = False
                        hs = self.asteroid_game.high_score
                        self.asteroid_game = AsteroidDodge(self.W, self.H)
                        self.asteroid_game.high_score = hs
                print(f"[Mode] AsteroidDodge → {self._asteroid_mode}")
            elif key == ord('b'):
                if self._hand_flight_mode and self.flight_sim._game_over:
                    hs = self.flight_sim.high_score
                    self.flight_sim = FlightSimulator(self.W, self.H)
                    self.flight_sim.high_score = hs
                else:
                    self._hand_flight_mode = not self._hand_flight_mode
                    if self._hand_flight_mode:
                        self._canvas_mode = self._game_mode = self._snake_mode = False
                        self._flight_mode = self._asteroid_mode = False
                        hs = self.flight_sim.high_score
                        self.flight_sim = FlightSimulator(self.W, self.H)
                        self.flight_sim.high_score = hs
                print(f"[Mode] SingleHandFlight → {self._hand_flight_mode}")
            elif key == ord('u'):
                if self._canvas_mode:
                    self.air_canvas.undo()
            elif key == ord('x'):
                if self._canvas_mode:
                    self.air_canvas._clear_canvas()
                    print("[Canvas] Cleared")
            elif key == 9:   # TAB
                self.hud.toggle_menu()
            elif key == ord('['):
                if self._canvas_mode:
                    self.air_canvas.change_brush_size(-1)
            elif key == ord(']'):
                if self._canvas_mode:
                    self.air_canvas.change_brush_size(1)

        self._cleanup()

    # ── Face pipeline ─────────────────────────────────────────────────────────
    def _run_face_pipeline(self, frame):
        all_verified = (
            all(self.id_cache[t].verified for t in self.id_cache.active_ids()
                if t in self.id_cache)
            and len(self.id_cache) > 0
        )
        if all_verified and self._frame_count % 2 == 0:
            return self._build_face_data_from_cache()

        detections = self.detector.detect(frame)
        # NEW: filter to max 5, largest-first
        detections = filter_faces(detections, max_faces=MAX_FACES)

        det_array  = _dets_to_array(detections)
        tracks     = self.sort.update(det_array)
        self.id_cache.evict_stale()

        active_faces = []
        for row in tracks:
            x1,y1,x2,y2,tid = (int(v) for v in row)
            tid   = int(tid)
            state = self.id_cache.get_or_create(tid)
            bbox  = state.box_smoother.update((x1,y1,x2,y2))

            if state.needs_encode():
                from core.face_detector import FaceDetection
                obj  = FaceDetection(bbox=bbox,confidence=1.0,landmarks=None,
                                     center=((bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2))
                crop = self.detector.crop_face(frame, obj)
                if crop is not None:
                    emb = self.encoder.encode(crop)
                    if emb is not None:
                        result = self.matcher.match(emb)
                        state.push_result(result)
                        state.set_encoded()
                        if self.multitask and not state.gender:
                            try:
                                attr = self.multitask.predict(crop)
                                state.gender  = attr.get("gender","")
                                state.emotion = attr.get("emotion","")
                            except Exception: pass

            if state.verified:
                self.hand_fx.set_low_mode(True)

            dr = state.display_result
            ui_state = (FaceUIState.VERIFIED  if state.verified and dr.is_known
                        else FaceUIState.UNKNOWN if dr.identity_id=="unknown"
                             and len(self.id_cache)>0
                        else FaceUIState.SCANNING)

            active_faces.append({
                "track_id":    tid,   "bbox":        bbox,
                "state":       ui_state,
                "display_name": dr.display_name,
                "role":        dr.role,
                "similarity":  state.sim_smoother._val,
            })
        return active_faces

    def _build_face_data_from_cache(self):
        result = []
        for tid in self.id_cache.active_ids():
            if tid not in self.id_cache: continue
            state = self.id_cache[tid]
            dr    = state.display_result
            ui_st = (FaceUIState.VERIFIED if state.verified and dr.is_known
                     else FaceUIState.UNKNOWN)
            bbox = state.box_smoother._state
            if bbox is None: continue
            bbox = tuple(int(v) for v in bbox)
            result.append({"track_id":tid,"bbox":bbox,"state":ui_st,
                           "display_name":dr.display_name,"role":dr.role,
                           "similarity":state.sim_smoother._val})
        return result

    # ── Hand pipeline ─────────────────────────────────────────────────────────
    def _run_hand_pipeline(self, frame, dt):
        hands = self.hand_tracker.detect(frame)
        self._last_hands = hands
        gesture = Gesture.OTHER
        cursor_pos = None

        # Dual-hand yoke (always update, self-deactivates when < 2 hands)
        if self._flight_mode:
            self._yoke_data = self.yoke.update(hands)
            self.flight_sim.update(self._yoke_data, dt)

        # BARU: Single-hand controller for flight and asteroid
        if self._hand_flight_mode or self._asteroid_mode:
            self._hand_ctrl_data = self.hand_ctrl.update(hands)
            if self._hand_flight_mode:
                yd = YokeData(
                    roll=self._hand_ctrl_data.roll,
                    pitch=self._hand_ctrl_data.pitch,
                    throttle=self._hand_ctrl_data.throttle,
                    active=self._hand_ctrl_data.active,
                )
                self.flight_sim.update(yd, dt)
            elif self._asteroid_mode:
                self.asteroid_game.update(self._hand_ctrl_data, dt)

        if hands:
            hr             = hands[0]
            gesture, click = self.gesture_engine.update(hr)
            self._last_gesture = gesture.value
            self.cursor.update(*hr.index_tip)
            cursor_pos = self.cursor.position

            for i, h in enumerate(hands):
                lbl = gesture.value if i == 0 else ""
                draw_neon_hand(frame, h, self.W, self.H, lbl,
                               self._global_ph + i*1.7)

            self.hand_fx.update(hands, frame, self.W, self.H)

            if self._canvas_mode:
                self.air_canvas.update_from_landmarks(
                    hr.landmarks, self.W, self.H, gesture.value, dt)

            elif self._game_mode:
                self.catch_game.update(gesture.value, hr.index_tip, dt)
                if self.catch_game._restart_pending:
                    self.catch_game = CatchGame(self.W, self.H)

            elif self._snake_mode:
                self._last_landmarks = hr.landmarks
                self.snake_game.update(gesture.value, hr.landmarks, self.W, self.H, dt)
                if self.snake_game._restart_pending:
                    hs = self.snake_game.high_score
                    self.snake_game = SnakeGame(self.W, self.H)
                    self.snake_game.high_score = hs

            elif self._flight_mode or self._hand_flight_mode or self._asteroid_mode:
                pass   # handled above

            else:
                if gesture == Gesture.PINCH and click:
                    self.love_meter.on_pinch()
                    self.hand_fx.trigger_pinch(0, *hr.index_tip)
                elif gesture == Gesture.OPEN_PALM:
                    self.love_meter.on_open_palm()
                elif gesture == Gesture.ILY:
                    if self.hand_fx.trigger_ily(*hr.index_tip):
                        self.love_meter.on_ily()

            self.cursor.draw_cursor(frame, gesture.value)
        else:
            self._last_gesture = "OTHER"
            self.gesture_engine.reset()
            self.cursor.reset()
            if self._game_mode:
                self.catch_game.update("OTHER", None, dt)
            if self._canvas_mode:
                self.air_canvas.update_from_landmarks(None, self.W, self.H, "OTHER", dt)
            if self._snake_mode:
                self.snake_game.update("OTHER", None, self.W, self.H, dt)
                if self.snake_game._restart_pending:
                    hs = self.snake_game.high_score
                    self.snake_game = SnakeGame(self.W, self.H)
                    self.snake_game.high_score = hs

        return gesture, cursor_pos

    # ── FPS ───────────────────────────────────────────────────────────────────
    def _update_fps(self, elapsed):
        self._fps_buf.append(elapsed)
        if len(self._fps_buf) > 60: self._fps_buf.pop(0)
        avg = sum(self._fps_buf)/len(self._fps_buf)
        self._fps_display = 1.0/avg if avg > 0 else 0.0
        if self._fps_display < MIN_FPS - 2:
            self._low_res_mode = True
            self.hand_fx.set_low_mode(True)
        elif self._fps_display > TARGET_FPS + 8 and self._low_res_mode:
            self._low_res_mode = False

    def _fps_throttle(self, elapsed):
        r = FRAME_BUDGET - elapsed
        if r > 0.001:
            time.sleep(r * 0.95)

    # ── Misc ──────────────────────────────────────────────────────────────────
    def _draw_mode_banner(self, frame, text, color):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (280, 32), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _save_screenshot(self, frame):
        self.screenshot_n += 1
        p = f"screenshot_{self.screenshot_n:04d}.jpg"
        cv2.imwrite(p, frame)
        print(f"[Screenshot] → {p}")

    def _rebuild_embeddings(self):
        print("[System] Rebuilding embeddings…")
        os.system(f"python {ROOT/'scripts'/'build_dataset.py'}")
        self.matcher = IdentityMatcher(
            encodings_path  = str(ROOT/"data"/"encodings.pkl"),
            identities_path = str(ROOT/"data"/"identities.json"),
        )
        print("[System] Done.")

    def _draw_debug(self, frame):
        yd = self._yoke_data
        yoke_txt = (f"YOKE r={yd.roll:+.2f} p={yd.pitch:+.2f} "
                    f"t={yd.throttle:.2f} active={yd.active}")
        for i, ln in enumerate([
            f"Tracks : {len(self.id_cache)}",
            f"Pcls   : {len(self.hand_fx._particles)}",
            f"LowRes : {self._low_res_mode}",
            f"Mirror : {self.mirror}",
            f"Gesture: {self._last_gesture}",
            yoke_txt,
        ]):
            cv2.putText(frame, ln, (10, 80+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160,210,160),1,cv2.LINE_AA)

    def _cleanup(self):
        self.cap.release()
        self.detector.release()
        self.hand_tracker.release()
        cv2.destroyAllWindows()
        print("[System] Shutdown complete.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _dets_to_array(dets):
    if not dets:
        return np.empty((0,5), dtype=np.float32)
    return np.array([[*d.bbox, d.confidence] for d in dets], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Unified AI Face + Hand FX System v2")
    p.add_argument("--camera",    type=int,   default=0)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--debug",     action="store_true")
    p.add_argument("--mirror",    action="store_true", help="Start with mirror enabled")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    system = UnifiedSystem(args)
    system.run()