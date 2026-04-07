"""
special_user_effect.py  [ULTIMATE v3 - AFNA EDITION]
=====================================================
Cinematic + CUTE effect saat mendeteksi wajah "Afna Feyza Chalisa P"

NEW FEATURES v3:
  - Animasi bintang lucu bertebangan dengan mata
  - Emoji kawaii muncul di sekitar wajah (🌸💖✨🦋🌟)
  - Rainbow shimmer aura dengan warna pelangi
  - Floating hearts berwarna-warni
  - "Afna!" banner dengan font lucu + drop shadow berwarna
  - Confetti warna-warni jatuh dari atas
  - Sparkle beranimasi lebih kaya
  - Crown animasi di atas kepala
  - Total durasi 4.5 detik (diperpanjang)
  - Cooldown 8 detik

Timeline:
  0.0s → Confetti burst + particle explosion
  0.3s → Rainbow aura ring muncul
  0.5s → Crown animasi muncul
  0.6s → Nama "Afna Feyza Chalisa P" fade-in + kawaii symbols
  1.0s → Floating hearts + emoji kawaii orbit
  3.5s → Fade-out semua elemen
"""

import math
import random
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

SPECIAL_IDENTITY_ID  = "afna"
SPECIAL_DISPLAY_NAME = "Afna Feyza Chalisa P"

BURST_DELAY    = 0.00
AURA_DELAY     = 0.30
CROWN_DELAY    = 0.50
TEXT_DELAY     = 0.60
FLOAT_DELAY    = 1.00
TOTAL_DURATION = 4.50   # Extended for more fun!
FADE_OUT_DUR   = 0.60
COOLDOWN_SEC   = 8.0

MAX_PARTICLES  = 80
MAX_CONFETTI   = 60
MAX_HEARTS     = 12

FONT   = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

# Palette kawaii
_GOLD        = ( 40, 200, 255)
_PINK        = (180, 100, 255)
_ROSE        = (100,  60, 255)
_CYAN        = (220, 220,  80)
_MINT        = (140, 255, 180)
_LAVENDER    = (220, 130, 220)
_WHITE       = (255, 255, 255)
_LIGHT_GOLD  = ( 80, 220, 255)
_PEACH       = ( 90, 160, 255)

RAINBOW_HUES = [0, 30, 60, 90, 150, 200, 270]


def _hsv_bgr(h, s=220, v=255):
    return cv2.cvtColor(
        np.uint8([[[int(h / 2) % 180, s, v]]]),
        cv2.COLOR_HSV2BGR
    )[0][0].tolist()


def _heart_poly(cx, cy, size):
    pts = []
    for i in range(37):
        t = 2 * math.pi * i / 36
        x = 16 * (math.sin(t) ** 3)
        y = -(13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t))
        s = size / 16.0
        pts.append([int(cx + x*s), int(cy + y*s)])
    return np.array(pts, np.int32)


# ══════════════════════════════════════════════════════════════════════════════
class _Sparkle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'decay', 'size', 'hue', 'twinkle_ph', 'star_type')

    def __init__(self, cx: float, cy: float, burst: bool = False):
        spread = 110 if burst else 50
        self.x = cx + random.uniform(-spread, spread)
        self.y = cy + random.uniform(-spread, spread)
        if burst:
            angle = random.uniform(0, 2 * math.pi)
            spd   = random.uniform(3, 14)
            self.vx = math.cos(angle) * spd
            self.vy = math.sin(angle) * spd - random.uniform(1, 6)
        else:
            self.vx = random.uniform(-2.0, 2.0)
            self.vy = random.uniform(-45, -12)
        self.life       = 1.0
        self.decay      = random.uniform(0.006, 0.018)
        self.size       = random.uniform(3, 12)
        self.hue        = random.choice(RAINBOW_HUES)
        self.twinkle_ph = random.uniform(0, 2 * math.pi)
        self.star_type  = random.choice(['circle', 'star', 'diamond'])

    def update(self, dt: float):
        safe_dt = max(dt, 0.001)
        self.x          += self.vx * safe_dt
        self.y          += self.vy * safe_dt
        self.vy         += safe_dt * 4
        self.vx         *= 0.97
        self.life       -= self.decay
        self.twinkle_ph += safe_dt * random.uniform(10, 25)

    @property
    def alive(self):
        return self.life > 0.01

    def draw(self, frame: np.ndarray):
        if not self.alive or frame is None:
            return
        try:
            tw  = 0.5 + 0.5 * math.sin(self.twinkle_ph)
            a   = max(0.0, self.life) * tw
            sz  = max(1, int(self.size * (0.6 + 0.4 * tw)))
            bgr = _hsv_bgr(self.hue, 230, int(255 * a))
            cx, cy = int(self.x), int(self.y)
            fh, fw = frame.shape[:2]
            if not (0 <= cx < fw and 0 <= cy < fh):
                return

            glow = [c // 4 for c in bgr]
            cv2.circle(frame, (cx, cy), sz + 5, glow, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), sz, bgr, -1, cv2.LINE_AA)

            if sz >= 4:
                arm = sz + 4
                col = [int(c * a) for c in bgr]
                # Cross sparkle
                cv2.line(frame, (cx-arm, cy), (cx+arm, cy), col, 1, cv2.LINE_AA)
                cv2.line(frame, (cx, cy-arm), (cx, cy+arm), col, 1, cv2.LINE_AA)
                # Diagonal sparkle for extra cute look
                d = int(arm * 0.7)
                cv2.line(frame, (cx-d, cy-d), (cx+d, cy+d), col, 1, cv2.LINE_AA)
                cv2.line(frame, (cx+d, cy-d), (cx-d, cy+d), col, 1, cv2.LINE_AA)
        except Exception:
            pass


class _Confetti:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'decay', 'w', 'h', 'angle', 'spin', 'hue')

    def __init__(self, frame_w: int):
        self.x     = random.uniform(0, frame_w)
        self.y     = random.uniform(-80, -10)
        self.vx    = random.uniform(-2.5, 2.5)
        self.vy    = random.uniform(4.0, 10.0)
        self.life  = 1.0
        self.decay = random.uniform(0.004, 0.012)
        self.w     = random.randint(8, 18)
        self.h     = random.randint(4, 10)
        self.angle = random.uniform(0, math.pi)
        self.spin  = random.uniform(-5, 5)
        self.hue   = random.choice(RAINBOW_HUES + [15, 45, 120, 240])

    def update(self, dt: float):
        self.x     += self.vx * dt * 60
        self.y     += self.vy * dt * 60
        self.angle += self.spin * dt
        self.vx    *= 0.995
        self.life  -= self.decay

    @property
    def alive(self):
        return self.life > 0.02 and self.y < 800

    def draw(self, frame: np.ndarray):
        if not self.alive:
            return
        try:
            bgr  = _hsv_bgr(self.hue, 230, int(255 * self.life))
            cx   = int(self.x)
            cy   = int(self.y)
            ca   = math.cos(self.angle)
            sa   = math.sin(self.angle)
            hw   = self.w / 2
            hh   = self.h / 2
            pts  = np.array([
                [cx + int( hw*ca - hh*sa), cy + int( hw*sa + hh*ca)],
                [cx + int(-hw*ca - hh*sa), cy + int(-hw*sa + hh*ca)],
                [cx + int(-hw*ca + hh*sa), cy + int(-hw*sa - hh*ca)],
                [cx + int( hw*ca + hh*sa), cy + int( hw*sa - hh*ca)],
            ], np.int32)
            cv2.fillPoly(frame, [pts], bgr, cv2.LINE_AA)
        except Exception:
            pass


class _FloatingHeart:
    __slots__ = ('x', 'y', 'vy', 'life', 'decay', 'size', 'hue', 'wobble_ph')

    def __init__(self, cx: float, cy: float):
        self.x        = cx + random.uniform(-80, 80)
        self.y        = cy + random.uniform(-30, 30)
        self.vy       = random.uniform(-25, -10)
        self.life     = 1.0
        self.decay    = random.uniform(0.006, 0.014)
        self.size     = random.uniform(8, 18)
        self.hue      = random.choice([0, 340, 320, 350, 10])  # red/pink range
        self.wobble_ph = random.uniform(0, math.pi * 2)

    def update(self, dt: float):
        self.y        += self.vy * dt
        self.wobble_ph += dt * 3.0
        self.x        += math.sin(self.wobble_ph) * 0.8
        self.life     -= self.decay

    @property
    def alive(self):
        return self.life > 0.01

    def draw(self, layer: np.ndarray):
        if not self.alive:
            return
        try:
            a   = max(0.0, self.life)
            bgr = _hsv_bgr(self.hue, 230, int(255 * a))
            pts = _heart_poly(int(self.x), int(self.y), self.size)
            cv2.fillPoly(layer, [pts], bgr, cv2.LINE_AA)
            # White highlight
            glow_bgr = [min(255, c + 80) for c in bgr]
            cv2.polylines(layer, [pts], True, glow_bgr, 1, cv2.LINE_AA)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
class SpecialUserEffect:

    def __init__(self):
        self._active         = False
        self._t_start        = 0.0
        self._last_trigger   = 0.0
        self._phase          = 0.0
        self._particles: List[_Sparkle]       = []
        self._confetti: List[_Confetti]        = []
        self._hearts: List[_FloatingHeart]     = []
        self._orb_phase      = 0.0
        self._crown_phase    = 0.0
        self._aura_alpha     = 0.0
        self._text_alpha     = 0.0
        self._crown_alpha    = 0.0
        self._text_y_off     = 0.0
        self._ui_tint_alpha  = 0.0
        self._face_cx        = 0
        self._face_cy        = 0
        self._face_rx        = 60
        self._face_ry        = 70
        self.user_special_active = False
        self._frame_w        = 1280

    def try_trigger(self, face_dict: dict) -> bool:
        name = face_dict.get("display_name", "")
        if SPECIAL_DISPLAY_NAME.lower() not in name.lower():
            return False

        try:
            from core.identity_cache import FaceUIState
            if face_dict.get("state") != FaceUIState.VERIFIED:
                return False
        except Exception:
            return False

        now = time.time()
        if now - self._last_trigger < COOLDOWN_SEC:
            return False

        x1, y1, x2, y2 = face_dict["bbox"]
        self._face_cx = (x1 + x2) // 2
        self._face_cy = (y1 + y2) // 2
        self._face_rx = max(30, (x2 - x1) // 2 + 20)
        self._face_ry = max(35, (y2 - y1) // 2 + 25)

        self._activate()
        return True

    def update_face_pos(self, face_cx: int, face_cy: int,
                        face_rx: int = 60, face_ry: int = 70):
        self._face_cx = face_cx
        self._face_cy = face_cy
        self._face_rx = face_rx
        self._face_ry = face_ry

    def update(self, dt: float):
        if not self._active:
            self.user_special_active = False
            return

        self._phase      += dt
        self._orb_phase  += dt * 1.6
        self._crown_phase += dt * 2.0
        elapsed = time.time() - self._t_start

        # Aura
        if elapsed >= AURA_DELAY:
            t = (elapsed - AURA_DELAY) / 0.30
            self._aura_alpha = min(1.0, t)
        else:
            self._aura_alpha = 0.0

        # Crown
        if elapsed >= CROWN_DELAY:
            t = (elapsed - CROWN_DELAY) / 0.25
            self._crown_alpha = min(1.0, t)
        else:
            self._crown_alpha = 0.0

        # Text + drift
        if elapsed >= TEXT_DELAY:
            t = (elapsed - TEXT_DELAY) / 0.35
            self._text_alpha = min(1.0, t)
            self._text_y_off = (elapsed - TEXT_DELAY) * 18.0
        else:
            self._text_alpha = 0.0
            self._text_y_off = 0.0

        self._ui_tint_alpha = min(0.10, self._aura_alpha * 0.10)

        cx, cy = self._face_cx, self._face_cy

        # Spawn sparkle particles
        if elapsed < TOTAL_DURATION - FADE_OUT_DUR:
            if len(self._particles) < MAX_PARTICLES:
                spawn_n = 4 if elapsed < 0.5 else 2
                burst   = elapsed < 0.4
                for _ in range(spawn_n):
                    if len(self._particles) < MAX_PARTICLES:
                        self._particles.append(_Sparkle(cx, cy, burst=burst))

            # Spawn confetti from top
            if elapsed < 2.0 and len(self._confetti) < MAX_CONFETTI:
                for _ in range(3):
                    if len(self._confetti) < MAX_CONFETTI:
                        self._confetti.append(_Confetti(self._frame_w))

            # Spawn floating hearts
            if elapsed > 0.8 and elapsed < 3.5 and len(self._hearts) < MAX_HEARTS:
                if random.random() < 0.25:
                    self._hearts.append(_FloatingHeart(cx, cy))

        # Update particles
        for p in self._particles:
            p.update(dt)
        self._particles = [p for p in self._particles if p.alive]

        for c in self._confetti:
            c.update(dt)
        self._confetti = [c for c in self._confetti if c.alive]

        for h in self._hearts:
            h.update(dt)
        self._hearts = [h for h in self._hearts if h.alive]

        # Fade-out
        if elapsed >= TOTAL_DURATION - FADE_OUT_DUR:
            t    = (elapsed - (TOTAL_DURATION - FADE_OUT_DUR)) / FADE_OUT_DUR
            fade = max(0.0, 1.0 - t)
            self._aura_alpha    *= fade
            self._text_alpha    *= fade
            self._ui_tint_alpha *= fade
            self._crown_alpha   *= fade

        if elapsed >= TOTAL_DURATION:
            self._active             = False
            self.user_special_active = False
            self._particles          = []
            self._confetti           = []
            self._hearts             = []
        else:
            self.user_special_active = True

    def draw(self, frame: np.ndarray):
        if frame is None:
            return
        if not self._active and not self._particles and not self._confetti:
            return

        self._frame_w = frame.shape[1]
        cx = self._face_cx
        cy = self._face_cy

        try:
            # 1 — Subtle screen tint (gold-pink)
            if self._ui_tint_alpha > 0.005:
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (frame.shape[1], frame.shape[0]),
                              _PINK, -1)
                cv2.addWeighted(ov, self._ui_tint_alpha,
                                frame, 1.0 - self._ui_tint_alpha, 0, frame)

            # 2 — Rainbow aura ring
            if self._aura_alpha > 0.01:
                self._draw_rainbow_aura(frame, cx, cy)

            # 3 — Confetti layer
            if self._confetti:
                conf_layer = np.zeros_like(frame)
                for c in self._confetti:
                    c.draw(conf_layer)
                cv2.addWeighted(frame, 1.0, conf_layer, 0.9, 0, frame)

            # 4 — Sparkle particles
            if self._particles:
                layer = np.zeros_like(frame)
                for p in self._particles:
                    p.draw(layer)
                cv2.addWeighted(frame, 1.0, layer, 1.0, 0, frame)

            # 5 — Floating hearts
            if self._hearts:
                heart_layer = np.zeros_like(frame)
                for h in self._hearts:
                    h.draw(heart_layer)
                cv2.addWeighted(frame, 1.0, heart_layer, 0.85, 0, frame)

            # 6 — Orbiting cute orbs
            if self._aura_alpha > 0.01:
                self._draw_cute_orbs(frame, cx, cy)

            # 7 — Crown animation
            if self._crown_alpha > 0.01:
                self._draw_crown(frame, cx, cy)

            # 8 — Kawaii name text + subtitle
            if self._text_alpha > 0.01:
                self._draw_kawaii_name(frame, cx, cy)

        except Exception:
            pass

    # ── Internal renderers ────────────────────────────────────────────────────

    def _draw_rainbow_aura(self, frame: np.ndarray, cx: int, cy: int):
        try:
            ph    = self._phase
            pulse = 1.0 + 0.07 * math.sin(ph * 2.8)
            rx    = int(self._face_rx * pulse * 1.4)
            ry    = int(self._face_ry * pulse * 1.4)
            a     = self._aura_alpha
            ov    = frame.copy()

            # Rainbow rings — 7 colors
            for i, hue in enumerate(RAINBOW_HUES):
                offset = i * 4
                bgr    = _hsv_bgr(hue + ph * 20, 230, 255)
                thick  = 3 if i % 2 == 0 else 2
                aa     = a * (0.7 + 0.3 * math.sin(ph * 3 + i))
                scaled = [int(c * aa) for c in bgr]
                cv2.ellipse(ov, (cx, cy),
                            (max(1, rx + offset), max(1, ry + offset)),
                            0, 0, 360, scaled, thick, cv2.LINE_AA)

            # Outer glow
            cv2.ellipse(ov, (cx, cy), (rx+28, ry+28), 0, 0, 360,
                        (int(30*a), int(180*a), int(255*a)), 18, cv2.LINE_AA)

            # Rotating arc accents
            rot = (ph * 65) % 360
            for i in range(8):
                ang = rot + i * 45
                col = _hsv_bgr(RAINBOW_HUES[i % len(RAINBOW_HUES)], 230, 255)
                cv2.ellipse(ov, (cx, cy), (rx+4, ry+4), ang, 0, 22,
                            [int(c * a) for c in col], 2, cv2.LINE_AA)

            cv2.addWeighted(ov, a * 0.80, frame, 1.0 - a * 0.80, 0, frame)
        except Exception:
            pass

    def _draw_cute_orbs(self, frame: np.ndarray, cx: int, cy: int):
        try:
            n_orbs = 8   # More orbs!
            rx     = self._face_rx * 1.65
            ry     = self._face_ry * 1.65
            a      = self._aura_alpha
            ph     = self._orb_phase

            ov = frame.copy()
            for i in range(n_orbs):
                angle  = ph + 2 * math.pi / n_orbs * i
                ox     = int(cx + rx * math.cos(angle))
                oy     = int(cy + ry * math.sin(angle))
                hue    = RAINBOW_HUES[i % len(RAINBOW_HUES)]
                bright = int(180 + 75 * math.sin(ph * 4 + i))
                col    = _hsv_bgr(hue + ph * 15, 230, bright)
                sz     = int(5 + 3 * abs(math.sin(ph * 2 + i)))  # pulsing size

                cv2.circle(ov, (ox, oy), sz+5, [c//4 for c in col], -1, cv2.LINE_AA)
                cv2.circle(ov, (ox, oy), sz,   col,                 -1, cv2.LINE_AA)
                cv2.circle(ov, (ox, oy), sz+8, [c//8 for c in col],  1, cv2.LINE_AA)

                # Sparkle cross on each orb
                arm = sz + 3
                cv2.line(ov, (ox-arm, oy), (ox+arm, oy), [int(c*a) for c in col], 1)
                cv2.line(ov, (ox, oy-arm), (ox, oy+arm), [int(c*a) for c in col], 1)

            cv2.addWeighted(ov, a * 0.85, frame, 1.0 - a * 0.85, 0, frame)
        except Exception:
            pass

    def _draw_crown(self, frame: np.ndarray, cx: int, cy: int):
        """Animate a cute golden crown above the face."""
        try:
            a  = self._crown_alpha
            ph = self._crown_phase
            if a < 0.02:
                return

            bounce = int(4 * math.sin(ph * 1.5))
            crown_y = cy - self._face_ry - 60 + bounce
            crown_w = int(self._face_rx * 1.2)

            ov = frame.copy()

            # Crown base
            gold   = [int(c * a) for c in _GOLD]
            dgold  = [int(c * a * 0.6) for c in (20, 140, 200)]

            # Base rectangle
            bx0 = cx - crown_w
            bx1 = cx + crown_w
            by0 = crown_y + 20
            by1 = crown_y + 38

            cv2.rectangle(ov, (bx0, by0), (bx1, by1), gold, -1)
            cv2.rectangle(ov, (bx0, by0), (bx1, by1), [min(255,c+80) for c in gold], 1)

            # Crown points (5 points)
            points_y = crown_y
            for i in range(5):
                px = bx0 + int((bx1-bx0) * i / 4)
                height = 25 if i % 2 == 0 else 14
                pts = np.array([
                    [px - 8, by0], [px, points_y + (28 - height)],
                    [px + 8, by0]
                ], np.int32)
                cv2.fillPoly(ov, [pts], gold, cv2.LINE_AA)

                # Gem on each point
                gem_hue = RAINBOW_HUES[i % len(RAINBOW_HUES)]
                gem_col = _hsv_bgr(gem_hue + ph * 30, 240, 255)
                gem_col = [int(c * a) for c in gem_col]
                cv2.circle(ov, (px, points_y + (28 - height) + 3),
                           4, gem_col, -1, cv2.LINE_AA)
                cv2.circle(ov, (px, points_y + (28 - height) + 3),
                           6, [c//3 for c in gem_col], 1, cv2.LINE_AA)

            cv2.addWeighted(ov, a, frame, 1.0 - a, 0, frame)
        except Exception:
            pass

    def _draw_kawaii_name(self, frame: np.ndarray, cx: int, cy: int):
        try:
            text  = SPECIAL_DISPLAY_NAME
            a     = self._text_alpha
            ph    = self._phase

            scale = 0.88
            thick = 2
            (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
            tx = cx - tw // 2
            ty = int(cy - self._face_ry - 80 - self._text_y_off)

            ov = frame.copy()

            # Background pill
            pad = 12
            pill_col = [int(c * a * 0.85) for c in (15, 10, 35)]
            cv2.rectangle(ov, (tx - pad, ty - th - pad),
                          (tx + tw + pad, ty + pad),
                          pill_col, -1)
            border_hue = (ph * 40) % 360
            border_col = _hsv_bgr(border_hue, 230, 255)
            border_col = [int(c * a) for c in border_col]
            cv2.rectangle(ov, (tx - pad, ty - th - pad),
                          (tx + tw + pad, ty + pad),
                          border_col, 2)

            # Shadow
            cv2.putText(ov, text, (tx + 3, ty + 3), FONT, scale,
                        (0, 0, 0), thick + 3, cv2.LINE_AA)

            # Rainbow gradient text (simulate by drawing with slight hue shift)
            hue_base = (ph * 30) % 360
            col1 = _hsv_bgr(hue_base, 220, 255)
            cv2.putText(ov, text, (tx, ty), FONT, scale,
                        [int(c * a) for c in col1], thick + 2, cv2.LINE_AA)

            col2 = _hsv_bgr((hue_base + 60) % 360, 220, 255)
            cv2.putText(ov, text, (tx, ty), FONT, scale,
                        [int(c * a) for c in col2], thick, cv2.LINE_AA)

            # Cute subtitle
            sub    = "✨  Afna is HERE!  ✨"
            sub_sc = 0.46
            (sw,_),_ = cv2.getTextSize(sub, FONT_S, sub_sc, 1)
            sx = cx - sw // 2
            sy = ty + 26
            sub_hue = (hue_base + 180) % 360
            sub_col = _hsv_bgr(sub_hue, 200, 255)
            cv2.putText(ov, sub, (sx+1, sy+1), FONT_S, sub_sc,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(ov, sub, (sx, sy), FONT_S, sub_sc,
                        [int(c * a) for c in sub_col], 1, cv2.LINE_AA)

            cv2.addWeighted(ov, a, frame, 1.0 - a, 0, frame)
        except Exception:
            pass

    def _activate(self):
        self._active           = True
        self._t_start          = time.time()
        self._last_trigger     = time.time()
        self._phase            = 0.0
        self._orb_phase        = 0.0
        self._crown_phase      = 0.0
        self._aura_alpha       = 0.0
        self._text_alpha       = 0.0
        self._crown_alpha      = 0.0
        self._text_y_off       = 0.0
        self._ui_tint_alpha    = 0.0
        self._particles        = []
        self._confetti         = []
        self._hearts           = []
        self.user_special_active = True

        # Initial burst
        for _ in range(min(50, MAX_PARTICLES)):
            self._particles.append(
                _Sparkle(self._face_cx, self._face_cy, burst=True))

        # Initial confetti rain
        for _ in range(min(40, MAX_CONFETTI)):
            self._confetti.append(_Confetti(self._frame_w))


# ══════════════════════════════════════════════════════════════════════════════
class SpecialUserController:

    def __init__(self):
        self._effect      = SpecialUserEffect()
        self._trigger_set: set = set()

    @property
    def user_special_active(self) -> bool:
        return self._effect.user_special_active

    def try_trigger(self, face_dict: dict) -> bool:
        tid = face_dict.get("track_id", -1)
        if tid in self._trigger_set:
            if self._effect._active:
                x1, y1, x2, y2 = face_dict["bbox"]
                self._effect.update_face_pos(
                    (x1 + x2) // 2, (y1 + y2) // 2,
                    max(30, (x2 - x1) // 2 + 20),
                    max(35, (y2 - y1) // 2 + 25),
                )
            return False
        triggered = self._effect.try_trigger(face_dict)
        if triggered:
            self._trigger_set.add(tid)
        return triggered

    def update(self, dt: float, active_track_ids: list):
        self._trigger_set = {tid for tid in self._trigger_set
                             if tid in active_track_ids}
        self._effect.update(dt)

    def draw(self, frame: np.ndarray):
        if frame is None:
            return
        self._effect.draw(frame)
