"""
ui/hud.py  [v6 — CINEMATIC PRO HUD — Modern Dark Glass]
=========================================================
Complete rewrite for maximum professional look:
- Sleek dark glassmorphism panels
- Animated gradient accent lines
- Modern typography with shadow depth
- Smooth status pills with glow
- TAB → radial-style mode selector overlay
- All previous bugs fixed
"""

import cv2
import numpy as np
import math
import time

FONT   = cv2.FONT_HERSHEY_SIMPLEX
FONT_D = cv2.FONT_HERSHEY_DUPLEX

# Color palette (BGR)
C_BG      = (6,  8, 18)
C_BORDER  = (32, 36, 72)
C_GRAY    = (100, 108, 125)
C_WHITE   = (230, 235, 245)
C_GREEN   = (55,  215, 100)
C_YELLOW  = (35,  205, 235)
C_RED     = (55,   65, 225)
C_CYAN    = (230, 245,  75)
C_PURPLE  = (215,  85, 195)
C_BLUE    = (255, 165,  90)
C_ORANGE  = (45,  155, 255)

GESTURE_COLORS = {
    "POINT":     (75,  205,  75),
    "PINCH":     (195, 205,  75),
    "OPEN_PALM": (75,  145, 230),
    "ILY":       (205,  85, 205),
    "FIST":      (185,  75,  75),
    "OTHER":     (60,   65,  80),
}

GESTURE_LABELS = {
    "POINT":     "DRAW / HOVER",
    "PINCH":     "BOOST / CLICK",
    "OPEN_PALM": "RESET / CLEAR",
    "ILY":       "ILY  +20 LOVE",
    "FIST":      "ERASE",
    "OTHER":     "—",
}

MODE_DEFS = [
    # (key, label,   color_BGR,       desc)
    ("C", "Canvas",   (70, 215,  75),  "Draw"),
    ("G", "Catch",    (70, 175, 255),  "Catch"),
    ("N", "Snake",    (70, 235, 155),  "Snake"),
    ("F", "Flight",   (215, 175, 75),  "Fly"),
    ("B", "HandFly",  (195, 135, 255), "Solo"),
    ("A", "Asteroid", (70,  195, 255), "Dodge"),
]


# ── Low-level drawing helpers ─────────────────────────────────────────────────

def _hsv_bgr(h, s=220, v=255):
    return cv2.cvtColor(
        np.uint8([[[int(h / 2) % 180, s, v]]]),
        cv2.COLOR_HSV2BGR)[0][0].tolist()


def _clamp_roi(frame, x1, y1, x2, y2):
    H, W = frame.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(W, x2);  y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None, x1, y1
    return frame[y1:y2, x1:x2], x1, y1


def _glass(frame, x1, y1, x2, y2, alpha=0.82, tint=None, border_col=None):
    """Semi-transparent dark glass panel."""
    roi, ax1, ay1 = _clamp_roi(frame, x1, y1, x2, y2)
    if roi is None:
        return
    bg = np.full_like(roi, tint if tint else C_BG)
    cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0, roi)
    frame[ay1:ay1 + roi.shape[0], ax1:ax1 + roi.shape[1]] = roi
    bc = border_col if border_col else C_BORDER
    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 1)


def _glow_line(frame, p1, p2, col, thick=1):
    glow = [c // 5 for c in col]
    cv2.line(frame, p1, p2, glow, thick + 5, cv2.LINE_AA)
    cv2.line(frame, p1, p2, col,  thick,     cv2.LINE_AA)


def _txt(frame, s, x, y, scale, col, thick=1, font=FONT, shadow=True):
    if shadow:
        cv2.putText(frame, s, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), font, scale, col, thick, cv2.LINE_AA)


def _pill(frame, x, y, w, h, col, label, fg=(235, 240, 250), glow=False):
    cv2.rectangle(frame, (x, y), (x + w, y + h), col, -1)
    bright = tuple(min(255, c + 60) for c in col)
    cv2.rectangle(frame, (x, y), (x + w, y + h), bright, 1)
    if glow:
        outer = tuple(c // 3 for c in bright)
        cv2.rectangle(frame, (x - 1, y - 1), (x + w + 1, y + h + 1), outer, 1)
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.30, 1)
    cv2.putText(frame, label, (x + (w - tw) // 2, y + h - 4),
                FONT, 0.30, fg, 1, cv2.LINE_AA)


def _bracket(frame, x, y, size, col, flip_x=False, flip_y=False):
    sx = -1 if flip_x else 1
    sy = -1 if flip_y else 1
    pts = [(x, y + sy * size), (x, y), (x + sx * size, y)]
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], col, 1, cv2.LINE_AA)


# =============================================================================
class HUD:

    def __init__(self, W, H):
        self.W, self.H      = W, H
        self._phase         = 0.0
        self._prev_t        = time.time()
        self._fps_smooth    = 30.0
        self._menu_visible  = False
        self._menu_alpha    = 0.0
        self._menu_phase    = 0.0

    def toggle_menu(self):
        self._menu_visible = not self._menu_visible

    # ── Main render ───────────────────────────────────────────────────────────
    def render(self, frame,
               fps=30.0, gesture="OTHER",
               hands_detected=0, faces_detected=0, faces_verified=0,
               mirror=False, canvas_mode=False, game_mode=False):

        now = time.time()
        dt  = min(now - self._prev_t, 0.12)
        self._prev_t  = now
        self._phase  += dt
        self._menu_phase += dt
        self._fps_smooth += (fps - self._fps_smooth) * 0.10

        target = 1.0 if self._menu_visible else 0.0
        self._menu_alpha = min(1.0, max(0.0,
            self._menu_alpha + (target - self._menu_alpha) * min(dt * 9, 1.0)))

        self._top_bar(frame, gesture, hands_detected,
                      faces_detected, faces_verified, mirror)
        self._bottom_bar(frame, canvas_mode, game_mode)

        if not canvas_mode and not game_mode:
            self._right_panel(frame, gesture)

        self._screen_corners(frame)

        if self._menu_alpha > 0.02:
            self._mode_menu(frame)

    # ── Top bar ───────────────────────────────────────────────────────────────
    def _top_bar(self, frame, gesture, hands, faces, verified, mirror):
        W, ph = self.W, self._phase

        # Background panel
        _glass(frame, 0, 0, W, 56, alpha=0.90)

        # Animated accent line
        hue     = (ph * 22) % 360
        line_c  = _hsv_bgr(hue, 200, 195)
        cx      = W // 2
        acc_w   = int(W * 0.32 + W * 0.08 * math.sin(ph * 1.4))
        _glow_line(frame, (cx - acc_w // 2, 56), (cx + acc_w // 2, 56), line_c, 2)

        # ── FPS block ─────────────────────────────────────────────────────────
        fps_v = self._fps_smooth
        fps_c = C_GREEN if fps_v >= 27 else C_YELLOW if fps_v >= 18 else C_RED
        _glass(frame, 6, 5, 138, 51, alpha=0.65)

        _txt(frame, "FPS", 14, 22, 0.32, (80, 90, 120))
        _txt(frame, f"{fps_v:5.1f}", 45, 40, 0.80, fps_c, 2)

        bw = int(72 * min(fps_v / 30.0, 1.0))
        cv2.rectangle(frame, (46, 44), (118, 48), (14, 16, 30), -1)
        if bw > 0:
            cv2.rectangle(frame, (46, 44), (46 + bw, 48), fps_c, -1)

        # ── Center gesture indicator ──────────────────────────────────────────
        g_c   = GESTURE_COLORS.get(gesture, C_GRAY)
        g_lbl = GESTURE_LABELS.get(gesture, gesture)
        pulse = 0.80 + 0.20 * math.sin(ph * 5.5)
        g_cp  = tuple(int(c * pulse) for c in g_c)

        (tw, th), _ = cv2.getTextSize(g_lbl, FONT, 0.58, 2)
        mid  = W // 2
        pw2  = tw // 2 + 28
        _glass(frame, mid - pw2, 5, mid + pw2, 51, alpha=0.60)
        cv2.line(frame, (mid - pw2 + 2, 6), (mid + pw2 - 2, 6), g_c, 2)

        # Dot icon
        cv2.circle(frame, (mid - pw2 + 14, 28),
                   10, [c // 4 for c in g_c], -1, cv2.LINE_AA)
        cv2.circle(frame, (mid - pw2 + 14, 28),
                   10, g_c, 1, cv2.LINE_AA)

        _txt(frame, g_lbl, mid - tw // 2, 36, 0.58, g_cp, 2)

        # ── Right status ──────────────────────────────────────────────────────
        rx = W - 12

        mir_c = C_BLUE if mirror else (28, 32, 52)
        _pill(frame, rx - 72, 8, 64, 18, mir_c, "MIRROR", glow=mirror)

        fc_c = C_GREEN if verified > 0 else C_GRAY
        _txt(frame, f"FACE: {faces}", rx - 152, 29, 0.36, fc_c)
        _txt(frame, f"HAND: {hands}", rx - 152, 45, 0.36, C_GRAY)

        if verified > 0:
            vt = f"✓ {verified} VERIFIED"
            (vw, _), _ = cv2.getTextSize(vt, FONT, 0.30, 1)
            _pill(frame, rx - vw - 18, 30, vw + 14, 16,
                  (12, 52, 22), vt, (70, 220, 88), glow=True)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    def _bottom_bar(self, frame, canvas_mode, game_mode):
        W, H, ph = self.W, self.H, self._phase

        _glass(frame, 0, H - 40, W, H, alpha=0.88)
        hue   = (ph * 18 + 120) % 360
        lcol  = _hsv_bgr(hue, 175, 175)
        cv2.line(frame, (0, H - 40), (W, H - 40), C_BORDER, 1)

        # Mode label
        if canvas_mode:
            m_txt, m_col = "[ AIR CANVAS — PRO ]", (70, 225, 100)
        elif game_mode:
            m_txt, m_col = "[ GAME MODE ]",         (105, 155, 255)
        else:
            m_txt, m_col = "[ LOVE METER ]",        (155, 105, 230)

        m_pulse = tuple(int(c * (0.78 + 0.22 * math.sin(ph * 3))) for c in m_col)
        _txt(frame, m_txt, 14, H - 13, 0.44, m_pulse)

        # Animated dot
        cv2.circle(frame, (8, H - 18), 3,
                   _hsv_bgr((ph * 40) % 360, 230, 255), -1, cv2.LINE_AA)

        # Key hints
        keys = "TAB=Menu  C=Canvas  G=Catch  N=Snake  F=Flight  A=Asteroid  M=Mirror  Q=Quit"
        (kw, _), _ = cv2.getTextSize(keys, FONT, 0.27, 1)
        cv2.putText(frame, keys, (W - kw - 10, H - 11),
                    FONT, 0.27, (46, 50, 74), 1, cv2.LINE_AA)

    # ── Right panel (gesture guide) ───────────────────────────────────────────
    def _right_panel(self, frame, gesture):
        W, H, ph = self.W, self.H, self._phase
        px  = W - 178
        py  = 62
        pw  = 168
        pht = 155

        _glass(frame, px, py, px + pw, py + pht, alpha=0.74)

        # Colour side bar
        for i in range(pht):
            t   = i / pht
            hue = (t * 115 + ph * 28) % 360
            c   = _hsv_bgr(hue, 175, 195)
            frame[py + i, px:px + 3] = c

        _txt(frame, "GESTURES", px + 8, py + 18, 0.33, (65, 75, 106))
        cv2.line(frame, (px + 6, py + 24), (px + pw - 6, py + 24), C_BORDER, 1)

        shortcuts = [
            ("POINT",     "Draw / Hover"),
            ("PINCH",     "Boost +Love"),
            ("ILY",       "+20 Love"),
            ("FIST",      "Erase"),
            ("OPEN_PALM", "Reset / Clear"),
        ]
        for i, (key, desc) in enumerate(shortcuts):
            yy     = py + 40 + i * 22
            col    = GESTURE_COLORS.get(key, C_GRAY)
            active = gesture == key

            if active:
                roi, _, _ = _clamp_roi(frame, px + 4, yy - 14, px + pw - 4, yy + 7)
                if roi is not None:
                    hl = np.full_like(roi, [c // 5 for c in col])
                    cv2.addWeighted(hl, 0.60, roi, 0.40, 0, roi)
                    frame[yy - 14:yy + 7, px + 4:px + pw - 4] = roi
                cv2.rectangle(frame, (px + 4, yy - 14), (px + pw - 4, yy + 7), col, 1)

            dot_c = col if active else (38, 42, 60)
            cv2.circle(frame, (px + 12, yy - 3), 3, dot_c, -1, cv2.LINE_AA)
            _txt(frame, key,  px + 22, yy, 0.29, col if active else (55, 62, 85))
            _txt(frame, desc, px + 86, yy, 0.28,
                 (165, 173, 192) if active else (55, 62, 85))

    # ── Screen corners ────────────────────────────────────────────────────────
    def _screen_corners(self, frame):
        W, H, ph = self.W, self.H, self._phase
        col = _hsv_bgr((ph * 18) % 360, 175, 148)
        sz  = 22
        for (x, y, fx, fy) in [(2, 58, False, False), (W - 2, 58, True, False),
                                 (2, H - 42, False, True), (W - 2, H - 42, True, True)]:
            _bracket(frame, x, y, sz, col, flip_x=fx, flip_y=fy)

    # ── Mode menu overlay (TAB) ───────────────────────────────────────────────
    def _mode_menu(self, frame):
        H, W = frame.shape[:2]
        a    = self._menu_alpha
        ph   = self._menu_phase

        # Dim overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (4, 5, 12), -1)
        cv2.addWeighted(ov, a * 0.60, frame, 1 - a * 0.60, 0, frame)

        cx = W // 2
        cy = H // 2

        # Title
        title = "SELECT MODE"
        (tw, _), _ = cv2.getTextSize(title, FONT_D, 0.75, 2)
        tc = int(255 * a)
        cv2.putText(frame, title, (cx - tw // 2 + 2, cy - 138),
                    FONT_D, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, title, (cx - tw // 2, cy - 140),
                    FONT_D, 0.75, (tc, tc, tc), 2, cv2.LINE_AA)

        sub = "Press key to activate   |   TAB to close"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.35, 1)
        sc = int(140 * a)
        cv2.putText(frame, sub, (cx - sw // 2, cy - 116), FONT, 0.35, (sc, sc, sc), 1, cv2.LINE_AA)

        # Mode cards
        n       = len(MODE_DEFS)
        cw, ch  = 94, 76
        gap     = 8
        total_w = n * cw + (n - 1) * gap
        sx      = cx - total_w // 2
        card_y  = cy - 50

        for i, (key, label, col, desc) in enumerate(MODE_DEFS):
            bx = sx + i * (cw + gap)
            by = card_y

            pulse = 1.0 + 0.045 * math.sin(ph * 2.8 + i * 1.05)
            ac    = tuple(int(c * pulse * a) for c in col)

            # Card glass background
            roi, _, _ = _clamp_roi(frame, bx, by, bx + cw, by + ch)
            if roi is not None:
                dark = np.full_like(roi, (10, 12, 24))
                cv2.addWeighted(dark, 0.92 * a, roi, 1 - 0.92 * a, 0, roi)
                frame[by:by + roi.shape[0], bx:bx + roi.shape[1]] = roi

            bc = tuple(int(c * a) for c in col)
            cv2.rectangle(frame, (bx, by), (bx + cw, by + ch), bc, 1)
            cv2.line(frame, (bx + 1, by), (bx + cw - 1, by), ac, 2)

            # Key badge
            kb_bg = tuple(int(c * 0.25 * a) for c in col)
            cv2.rectangle(frame,
                          (bx + cw // 2 - 13, by + 8),
                          (bx + cw // 2 + 13, by + 30), kb_bg, -1)
            cv2.rectangle(frame,
                          (bx + cw // 2 - 13, by + 8),
                          (bx + cw // 2 + 13, by + 30), bc, 1)
            (kw, _), _ = cv2.getTextSize(key, FONT_D, 0.62, 2)
            cv2.putText(frame, key, (bx + cw // 2 - kw // 2, by + 26),
                        FONT_D, 0.62, ac, 2, cv2.LINE_AA)

            # Label + desc
            (lw, _), _ = cv2.getTextSize(label, FONT, 0.38, 1)
            lc = tuple(int(c * a) for c in (195, 202, 218))
            cv2.putText(frame, label, (bx + (cw - lw) // 2, by + ch - 18),
                        FONT, 0.38, lc, 1, cv2.LINE_AA)
            (dw, _), _ = cv2.getTextSize(desc, FONT, 0.26, 1)
            dc = tuple(int(c * a * 0.7) for c in col)
            cv2.putText(frame, desc, (bx + (cw - dw) // 2, by + ch - 6),
                        FONT, 0.26, dc, 1, cv2.LINE_AA)

        # Bottom hint
        hint = "M=Mirror   D=Debug   S=Screenshot   Q=Quit   U=Undo (Canvas)"
        (hw, _), _ = cv2.getTextSize(hint, FONT, 0.30, 1)
        hc = int(110 * a)
        cv2.putText(frame, hint, (cx - hw // 2, cy + 54),
                    FONT, 0.30, (hc, hc, hc), 1, cv2.LINE_AA)
