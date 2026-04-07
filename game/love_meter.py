"""
game/love_meter.py  [v3 — FPS-optimised + Pro UI]

PERF FIXES:
  - Single shared overlay per frame (1 copy max), not 1 copy per particle
  - Heart/star drawn directly onto a particle layer (np.zeros), then cv2.add
  - Confetti uses direct fillPoly on particle layer — no per-item blend
  - Twinkles drawn direct — no per-twinkle copy
  - Big bubble hearts: one batch layer
  - Orbit + corner: drawn direct on frame after single ROI pass
  - Total frame.copy() calls per frame: MAX 2 (bloom + composite)

UI IMPROVEMENTS:
  - Gradient rainbow bar with shimmer
  - Pill-style love counter with icon
  - Animated segment markers on bar
  - Pro glassmorphism panel
"""

import math
import random
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np


FONT    = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX

PINCH_COOLDOWN = 0.5
MAX_VAL        = 100.0
BAR_W          = 280
BAR_H          = 16
ANIM_SPEED     = 6.0

# Max particles kept alive (caps FPS cost)
MAX_HEARTS      = 18
MAX_TWINKLES    = 12
MAX_CONFETTI    = 50
MAX_BIG_HEARTS  = 3


def _hsv_bgr(h_deg, s=220, v=255):
    h = int(h_deg / 2) % 180
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()


def _draw_heart_pts(cx, cy, size):
    """Pre-compute heart polygon points — reuse across draws."""
    pts = []
    for i in range(37):
        t = 2 * math.pi * i / 36
        x = 16 * (math.sin(t) ** 3)
        y = -(13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t))
        s = size / 16.0
        pts.append([int(cx + x*s), int(cy + y*s)])
    return np.array(pts, np.int32)


def _star_pts(cx, cy, size, angle=0.0):
    pts = []
    for i in range(10):
        r = size if i % 2 == 0 else size * 0.42
        a = math.pi * i / 5 - math.pi/2 + angle
        pts.append([int(cx + r*math.cos(a)), int(cy + r*math.sin(a))])
    return np.array(pts, np.int32)


# ── Lightweight particle structs ──────────────────────────────────────────────

class _FloatHeart:
    __slots__ = ('x','y','vy','size','alpha','hue','sway','phase')
    def __init__(self, W, H, size=None):
        self.x     = float(random.randint(W//8, 7*W//8))
        self.y     = float(H + random.randint(0, 40))
        self.vy    = random.uniform(1.0, 2.8)
        self.size  = size or random.randint(9, 28)
        self.alpha = random.uniform(0.7, 1.0)
        self.hue   = float(random.choice([340,350,0,10,320,300]))
        self.sway  = random.uniform(0.5, 1.8)
        self.phase = random.uniform(0, 6.28)

    def step(self):
        self.y -= self.vy
        self.x += math.sin(self.phase) * self.sway
        self.phase += 0.06
        self.hue = (self.hue + 0.5) % 360
        if self.y < 100:
            self.alpha = max(0.0, self.alpha - 0.018)

    @property
    def alive(self): return self.y > -self.size*2 and self.alpha > 0.05


class _Twinkle:
    __slots__ = ('x','y','max_size','phase','speed','hue')
    def __init__(self, W, H):
        self.x        = random.randint(20, W-20)
        self.y        = random.randint(20, H-20)
        self.max_size = random.randint(5, 18)
        self.phase    = 0.0
        self.speed    = random.uniform(0.10, 0.22)
        self.hue      = float(random.choice([0,10,330,350,300,60]))

    def step(self): self.phase += self.speed

    @property
    def alive(self): return self.phase < math.pi * 3.2

    @property
    def size(self): return self.max_size * abs(math.sin(self.phase))

    @property
    def alpha(self): return abs(math.sin(self.phase))


class _Confetti:
    __slots__ = ('x','y','vx','vy','angle','spin','w','h','color')
    def __init__(self, W, H):
        self.x     = float(random.randint(0, W))
        self.y     = float(random.randint(-50, 0))
        self.vx    = random.uniform(-1.2, 1.2)
        self.vy    = random.uniform(1.8, 3.5)
        self.angle = random.uniform(0, math.pi)
        self.spin  = random.uniform(-0.07, 0.07)
        self.w     = random.randint(6, 13)
        self.h     = random.randint(3, 6)
        hue = random.choice([0,10,30,300,320,340,270,60,200])
        self.color = _hsv_bgr(hue, 200, 255)

    def step(self, W, H):
        self.x += self.vx + math.sin(self.angle) * 0.4
        self.y += self.vy
        self.angle += self.spin
        if self.y > H + 10:
            self.y = float(random.randint(-30, -5))
            self.x = float(random.randint(0, W))


class _BubbleHeart:
    __slots__ = ('x','y','vy','vx','size','alpha','hue','phase')
    def __init__(self, W, H):
        self.x     = float(random.randint(W//4, 3*W//4))
        self.y     = float(H)
        self.vy    = random.uniform(0.4, 1.0)
        self.vx    = random.uniform(-0.3, 0.3)
        self.size  = random.randint(50, 100)
        self.alpha = random.uniform(0.08, 0.18)
        self.hue   = float(random.choice([340,350,0,320]))
        self.phase = random.uniform(0, 6.28)

    def step(self):
        self.y -= self.vy
        self.x += self.vx + math.sin(self.phase)*0.4
        self.phase += 0.018
        self.alpha = max(0, self.alpha - 0.00025)

    @property
    def alive(self): return self.y > -self.size*2 and self.alpha > 0.008


# ── Love Meter ────────────────────────────────────────────────────────────────

class LoveMeter:
    def __init__(self):
        self.value:       float = 0.0
        self.target:      float = 0.0
        self._last_pinch: float = 0.0
        self._perfect_at: Optional[float] = None
        self._glow_phase: float = 0.0
        self._bar_phase:  float = 0.0
        self._burst_ph:   float = 0.0
        self._text_ph:    float = 0.0

        self._hearts:     List[_FloatHeart]  = []
        self._big_hearts: List[_BubbleHeart] = []
        self._twinkles:   List[_Twinkle]     = []
        self._confetti:   List[_Confetti]    = []

        self._last_spawn: float = 0.0
        self._W = 1280
        self._H = 720

    # ── Inputs ───────────────────────────────────────────────────────────────

    def on_pinch(self):
        now = time.time()
        if now - self._last_pinch >= PINCH_COOLDOWN and self.target < MAX_VAL:
            self.target = min(MAX_VAL, self.target + 5)
            self._last_pinch = now
            self._check_perfect()

    ILY_COOLDOWN = 2.0

    def on_ily(self):
        now = time.time()
        if now - getattr(self, '_last_ily', 0.0) < self.ILY_COOLDOWN:
            return
        self._last_ily = now
        if self.target < MAX_VAL:
            self.target = min(MAX_VAL, self.target + 20)
            self._check_perfect()

    def on_open_palm(self):
        self.target = 0.0
        self._perfect_at = None
        self._hearts.clear()
        self._big_hearts.clear()
        self._twinkles.clear()
        self._confetti.clear()

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, dt: float):
        diff = self.target - self.value
        step = min(abs(diff), ANIM_SPEED * dt * 60)
        self.value += math.copysign(step, diff)
        self.value  = max(0.0, min(MAX_VAL, self.value))
        self._glow_phase += dt * 2.5
        self._bar_phase  += dt * 60

        if self._perfect_at is not None:
            self._burst_ph += dt * 1.8
            self._text_ph  += dt * 3.0

            now = time.time()
            if now - self._last_spawn > 0.35:
                self._last_spawn = now
                self._spawn_wave()

            # Step + prune with hard caps
            self._hearts     = [h for h in self._hearts     if h.alive][:MAX_HEARTS]
            self._big_hearts = [h for h in self._big_hearts if h.alive][:MAX_BIG_HEARTS]
            self._twinkles   = [t for t in self._twinkles   if t.alive][:MAX_TWINKLES]
            self._confetti   = self._confetti[:MAX_CONFETTI]

            for h in self._hearts:     h.step()
            for h in self._big_hearts: h.step()
            for t in self._twinkles:   t.step()
            for c in self._confetti:   c.step(self._W, self._H)

    def _spawn_wave(self):
        W, H = self._W, self._H
        if len(self._hearts) < MAX_HEARTS:
            for _ in range(random.randint(2, 4)):
                self._hearts.append(_FloatHeart(W, H))
        if len(self._big_hearts) < MAX_BIG_HEARTS and random.random() < 0.3:
            self._big_hearts.append(_BubbleHeart(W, H))
        if len(self._twinkles) < MAX_TWINKLES:
            for _ in range(random.randint(1, 3)):
                self._twinkles.append(_Twinkle(W, H))

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, frame: np.ndarray, x: int, y: int):
        H, W = frame.shape[:2]
        self._W, self._H = W, H

        self._draw_panel(frame, x, y)

        if self._perfect_at is not None:
            self._draw_sweet_optimised(frame)

    # ── Professional bar panel ────────────────────────────────────────────────

    def _draw_panel(self, frame, x, y):
        H, W = frame.shape[:2]
        pw = BAR_W + 100
        ph = 88

        # Glassmorphism panel
        roi = frame[y:y+ph, x:x+pw]
        bg  = np.full_like(roi, (12, 12, 22))
        cv2.addWeighted(bg, 0.78, roi, 0.22, 0, roi)
        frame[y:y+ph, x:x+pw] = roi

        # Border with accent top
        cv2.rectangle(frame, (x, y), (x+pw, y+ph), (45, 45, 75), 1)
        # Colored top accent line
        accent = self._bar_color_solid()
        if self._perfect_at is not None:
            hue_acc = int(self._bar_phase) % 360
            accent  = _hsv_bgr(hue_acc, 200, 255)
        cv2.line(frame, (x+1, y+1), (x+pw-1, y+1), accent, 2)

        # Title pill — improved
        title_col = [int(c*0.55) for c in accent]
        cv2.rectangle(frame, (x+6, y+6), (x+60, y+34), title_col, -1)
        cv2.rectangle(frame, (x+6, y+6), (x+60, y+34), [min(255,c+80) for c in title_col], 1)
        cv2.putText(frame, "LOVE", (x+10, y+18),
                    FONT_SM, 0.36, (220,220,240), 1, cv2.LINE_AA)
        cv2.putText(frame, "METER", (x+10, y+30),
                    FONT_SM, 0.36, (200,200,220), 1, cv2.LINE_AA)

        # Percentage (large)
        pct    = int(self.value)
        pct_tx = f"{pct}%"
        pct_col = accent if self._perfect_at is None else _hsv_bgr(
            (self._bar_phase * 2) % 360, 200, 255)
        scale  = 1.1 if self._perfect_at and abs(math.sin(self._text_ph)) > 0.5 else 0.95
        cv2.putText(frame, pct_tx, (x+54, y+34), FONT, scale, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(frame, pct_tx, (x+52, y+32), FONT, scale, pct_col, 2, cv2.LINE_AA)

        # Bar background
        bx = x + 10
        by = y + 44
        cv2.rectangle(frame, (bx, by), (bx+BAR_W, by+BAR_H), (25,25,35), -1)
        cv2.rectangle(frame, (bx, by), (bx+BAR_W, by+BAR_H), (50,50,80), 1)

        # Segment markers every 25%
        for pct_m in [25, 50, 75]:
            mx = bx + int(BAR_W * pct_m / 100)
            cv2.line(frame, (mx, by), (mx, by+BAR_H), (40,40,60), 1)

        # Fill
        fill_w = int(BAR_W * self.value / MAX_VAL)
        if fill_w > 1:
            if self._perfect_at is not None:
                # Rainbow gradient bar — draw column by column (fast)
                bar_roi = frame[by:by+BAR_H, bx:bx+fill_w]
                cols    = fill_w
                if cols > 0:
                    grad = np.zeros((BAR_H, cols, 3), dtype=np.uint8)
                    ph   = int(self._bar_phase) % 180
                    for col_i in range(cols):
                        hue = (ph + col_i * 180 // cols) % 180
                        bgr = cv2.cvtColor(np.uint8([[[hue, 210, 255]]]),
                                           cv2.COLOR_HSV2BGR)[0][0]
                        grad[:, col_i] = bgr
                    # Shimmer highlight top strip
                    shim = int(120 + 80 * math.sin(self._glow_phase * 2))
                    grad[:BAR_H//3, :] = np.clip(
                        grad[:BAR_H//3, :].astype(np.int16) + shim//4, 0, 255
                    ).astype(np.uint8)
                    frame[by:by+BAR_H, bx:bx+fill_w] = grad
            else:
                col = self._bar_color_solid()
                cv2.rectangle(frame, (bx, by), (bx+fill_w, by+BAR_H), col, -1)
                # Highlight top
                light = [min(255, c+60) for c in col]
                cv2.rectangle(frame, (bx, by), (bx+fill_w, by+BAR_H//3), light, -1)

        # Heart icon at fill end
        if fill_w > 4:
            hx = bx + fill_w
            hy = by + BAR_H//2
            pts = _draw_heart_pts(hx, hy, 7)
            col = accent if self._perfect_at is None else _hsv_bgr(
                (self._bar_phase) % 360, 220, 255)
            cv2.fillPoly(frame, [pts], col)

        # Hint text
        cv2.putText(frame, "PINCH +5  |  ILY +20  |  PALM RESET",
                    (bx, by+BAR_H+18), FONT_SM, 0.28, (70,70,90), 1, cv2.LINE_AA)

        # Perfect badge
        if self._perfect_at is not None:
            badge_col = _hsv_bgr((self._bar_phase*3) % 360, 200, 255)
            cv2.putText(frame, "PERFECT!", (x+pw-84, y+18),
                        FONT_SM, 0.45, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, "PERFECT!", (x+pw-85, y+17),
                        FONT_SM, 0.45, badge_col, 1, cv2.LINE_AA)

    # ── FPS-safe celebration ──────────────────────────────────────────────────

    def _draw_sweet_optimised(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        ph = self._burst_ph
        tp = self._text_ph

        # ── 1. Soft bloom: ONE addWeighted on full frame (single copy) ───────
        bloom = np.zeros((H, W, 3), dtype=np.uint8)
        # Radial ellipse — cheap, no loop
        cv2.ellipse(bloom, (W//2, H//2), (W//2, H//2), 0, 0, 360, (120, 60, 180), -1)
        cv2.addWeighted(bloom, 0.07 + 0.03*math.sin(ph), frame, 0.93, 0, frame)
        bloom_buf = bloom   # reuse buffer below

        # ── 2. Particle layer — ONE np.zeros, batch ALL particles onto it ────
        layer = np.zeros((H, W, 3), dtype=np.uint8)

        # Confetti (direct fillPoly, no copy)
        for c in self._confetti:
            cx, cy = int(c.x), int(c.y)
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            cos_a, sin_a = math.cos(c.angle), math.sin(c.angle)
            hw, hh = c.w//2, c.h//2
            pts = np.array([
                [int(cx + (-hw)*cos_a - (-hh)*sin_a), int(cy + (-hw)*sin_a + (-hh)*cos_a)],
                [int(cx + hw*cos_a  - (-hh)*sin_a),  int(cy + hw*sin_a  + (-hh)*cos_a)],
                [int(cx + hw*cos_a  - hh*sin_a),     int(cy + hw*sin_a  + hh*cos_a)],
                [int(cx + (-hw)*cos_a - hh*sin_a),   int(cy + (-hw)*sin_a + hh*cos_a)],
            ], np.int32)
            cv2.fillPoly(layer, [pts], c.color)

        # Big bubble hearts (transparent — batch on layer)
        for bh in self._big_hearts:
            col = _hsv_bgr(bh.hue, 100, 255)
            pts = _draw_heart_pts(int(bh.x), int(bh.y), bh.size)
            dim_col = [int(c * bh.alpha * 2) for c in col]
            cv2.fillPoly(layer, [pts], dim_col)

        # Twinkles (4-point star, direct to layer)
        for tw in self._twinkles:
            if tw.size < 1:
                continue
            tx, ty = int(tw.x), int(tw.y)
            if not (0 <= tx < W and 0 <= ty < H):
                continue
            col = _hsv_bgr(tw.hue, 160, 255)
            s   = int(tw.size)
            dim = [int(c * tw.alpha) for c in col]
            for ang in [0, math.pi/4]:
                for sgn in [1, -1]:
                    ex = int(tx + s * math.cos(ang) * sgn)
                    ey = int(ty + s * math.sin(ang) * sgn)
                    cv2.line(layer, (tx, ty), (ex, ey), dim, max(1, s//3), cv2.LINE_AA)
            cv2.circle(layer, (tx, ty), max(1, s//4), (200, 200, 200), -1, cv2.LINE_AA)

        # Floating hearts (direct to layer)
        for fh in self._hearts:
            col  = _hsv_bgr(fh.hue, 200, 255)
            pts  = _draw_heart_pts(int(fh.x), int(fh.y), fh.size)
            dim  = [int(c * fh.alpha) for c in col]
            cv2.fillPoly(layer, [pts], dim)
            # shine
            sx = int(fh.x - fh.size//4)
            sy = int(fh.y - fh.size//3)
            sr = max(1, fh.size//5)
            cv2.circle(layer, (sx, sy), sr,
                       [int(255*fh.alpha)]*3, -1, cv2.LINE_AA)

        # Add particle layer in ONE call
        cv2.add(frame, layer, frame)

        # ── 3. Orbit hearts (direct on frame, no copy) ────────────────────────
        orx, ory = W//4, int(H*0.12)
        ocx, ocy = W//2, H//2 - 8
        for i in range(6):
            ang = ph * 0.85 + (2*math.pi/6)*i
            hx  = int(ocx + orx * math.cos(ang))
            hy  = int(ocy + ory * math.sin(ang))
            sz  = int(11 + 4*math.sin(ph*2.5 + i*1.2))
            col = _hsv_bgr((330 + i*12) % 360, 200, 255)
            pts = _draw_heart_pts(hx, hy, sz)
            cv2.fillPoly(frame, [pts], col)
            cv2.circle(frame, (hx - sz//4, hy - sz//3),
                       max(1, sz//5), (255,255,255), -1, cv2.LINE_AA)

        # ── 4. Corner sparkles (direct, no copy) ─────────────────────────────
        corners = [(70,70),(W-70,70),(70,H-70),(W-70,H-70)]
        c_hues  = [340, 10, 300, 350]
        for i, ((cx, cy), hue) in enumerate(zip(corners, c_hues)):
            for j in range(5):
                ang = ph*1.1 + (2*math.pi/5)*j + i*0.5
                r   = 26 + 8*math.sin(ph*3+j)
                ex  = int(cx + r*math.cos(ang))
                ey  = int(cy + r*math.sin(ang))
                sz  = int(6 + 2*math.sin(ph*4+j))
                col = _hsv_bgr((hue+j*8)%360, 210, 255)
                if 0 <= ex < W and 0 <= ey < H:
                    pts = _draw_heart_pts(ex, ey, sz)
                    cv2.fillPoly(frame, [pts], col)
            # Center star
            spts = _star_pts(cx, cy, 9)
            cv2.fillPoly(frame, [spts], _hsv_bgr(hue, 180, 255))

        # ── 5. Top heart band (direct) ────────────────────────────────────────
        for i in range(9):
            bx = int(W / 10 * (i + 1))
            by = int(56 + 7*math.sin(ph*2 + i*0.7))
            sz = int(10 + 3*math.sin(ph*3+i))
            col = _hsv_bgr((i*35 + ph*25) % 360, 200, 255)
            pts = _draw_heart_pts(bx, by, sz)
            cv2.fillPoly(frame, [pts], col)
            cv2.circle(frame, (bx - sz//4, by - sz//3),
                       max(1, sz//5), (255,255,255), -1, cv2.LINE_AA)

        # ── 6. Bouncy wave text "So Sweet~" ──────────────────────────────────
        chars   = list("So Sweet~")
        char_w  = 38
        total_w = len(chars) * char_w
        sx      = W//2 - total_w//2
        base_y  = H//2 - 8

        for idx, ch in enumerate(chars):
            bounce = int(11 * math.sin(tp + idx*0.55))
            scale  = 0.95 + 0.10*math.sin(tp*1.3 + idx*0.4)
            cx_ch  = sx + idx*char_w
            cy_ch  = base_y + bounce
            hue    = (tp*35 + idx*28) % 360
            col    = _hsv_bgr(hue, 200, 255)
            cv2.putText(frame, ch, (cx_ch+2, cy_ch+2), FONT, scale, (0,0,0), 6, cv2.LINE_AA)
            cv2.putText(frame, ch, (cx_ch,   cy_ch),   FONT, scale, (255,255,255), 4, cv2.LINE_AA)
            cv2.putText(frame, ch, (cx_ch,   cy_ch),   FONT, scale, col, 2, cv2.LINE_AA)

        # ── 7. Subtitle ───────────────────────────────────────────────────────
        sub      = "You're a perfect match!"
        sub_b    = int(4 * math.sin(tp*1.8 + 1.0))
        (sw,_),_ = cv2.getTextSize(sub, FONT_SM, 0.70, 2)
        sub_x    = (W - sw)//2
        sub_y    = H//2 + 52 + sub_b
        sub_col  = _hsv_bgr((tp*45+180) % 360, 180, 255)
        cv2.putText(frame, sub, (sub_x+1, sub_y+1), FONT_SM, 0.70, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, sub, (sub_x,   sub_y),   FONT_SM, 0.70, sub_col, 2, cv2.LINE_AA)

        # Tiny flanking hearts
        for sign, off in [(-1, -16), (1, sw+16)]:
            hx2 = sub_x + off
            hy2 = sub_y - 7
            pts = _draw_heart_pts(hx2, hy2, 9)
            cv2.fillPoly(frame, [pts], _hsv_bgr((tp*55+20)%360, 220, 255))

        # ── 8. LOVE x100 pill badge ───────────────────────────────────────────
        badge_s = 0.58 + 0.07*abs(math.sin(tp*2))
        btxt    = "LOVE x100"
        (bw,_),_ = cv2.getTextSize(btxt, FONT_SM, badge_s, 2)
        bx = (W - bw)//2
        by2 = H//2 + 90
        pad = 10
        # pill bg direct (no copy — small ROI)
        roi = frame[by2-22:by2+7, bx-pad:bx+bw+pad]
        if roi.size > 0:
            pill_bg = np.full_like(roi, (40, 25, 120))
            cv2.addWeighted(pill_bg, 0.65, roi, 0.35, 0, roi)
            frame[by2-22:by2+7, bx-pad:bx+bw+pad] = roi
        cv2.rectangle(frame, (bx-pad, by2-22), (bx+bw+pad, by2+7), (160,100,240), 1)
        b_col = _hsv_bgr((tp*65)%360, 200, 255)
        cv2.putText(frame, btxt, (bx+1, by2+1), FONT_SM, badge_s, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, btxt, (bx,   by2),   FONT_SM, badge_s, b_col, 2, cv2.LINE_AA)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_perfect(self):
        if self.target >= MAX_VAL and self._perfect_at is None:
            self._perfect_at = time.time()
            W, H = self._W, self._H
            # Pre-spawn
            for _ in range(min(40, MAX_CONFETTI)):
                self._confetti.append(_Confetti(W, H))
            for _ in range(8):
                h = _FloatHeart(W, H, size=random.randint(14,36))
                h.y = float(random.randint(H//4, 3*H//4))
                self._hearts.append(h)
            for _ in range(min(3, MAX_BIG_HEARTS)):
                self._big_hearts.append(_BubbleHeart(W, H))
            for _ in range(10):
                self._twinkles.append(_Twinkle(W, H))
            self._last_spawn = time.time()

    def _bar_color_solid(self):
        _CR = (60,60,220); _CY = (50,200,230); _CG = (60,210,80)
        t = self.value / MAX_VAL
        r, g = (_CR, _CY) if t < 0.5 else (_CY, _CG)
        f = (t*2) if t < 0.5 else ((t-0.5)*2)
        return tuple(int(r[i] + (g[i]-r[i])*f) for i in range(3))
