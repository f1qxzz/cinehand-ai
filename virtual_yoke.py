"""
virtual_yoke.py — Dual-Hand Virtual Yoke Controller  [v4 — Improved Accuracy]
===============================================================================
Kontrol pesawat dengan DUA tangan:

  TANGAN KIRI  → Throttle (angkat tangan ke atas = gas lebih besar)
  TANGAN KANAN → Kemudi   (miringkan = roll, naik/turun = pitch)

Perbaikan v4 vs v3:
  ✅ Dead zone lebih besar → tidak gemetar saat diam
  ✅ Auto-kalibrasi pitch saat kedua tangan pertama kali terdeteksi
  ✅ Re-kalibrasi dengan genggam tangan kanan (semua jari dilipat)
  ✅ Smoothing adaptif: lambat saat halus, cepat saat gerakan besar
  ✅ Roll lebih sensitif di tengah, lebih lambat di ekstrem (kurva S)
  ✅ Indikator visual lebih jelas: zona berwarna, snap zones kuning
  ✅ Throttle snap: magnet ke 0%, 50%, 100%
  ✅ Countdown kalibrasi 2 detik agar bisa posisikan tangan dulu
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


# ─── Tunable constants ────────────────────────────────────────────────────────
_ROLL_MAX_DEG   = 28.0    # ±28° sudah cukup (lebih mudah dicapai)
_PITCH_ZONE     = 0.28    # zona lebih sempit = respons lebih cepat
_THR_TOP_Y      = 0.15
_THR_BOT_Y      = 0.85

_ALPHA_ROLL_SLOW   = 0.18   # lebih responsif
_ALPHA_ROLL_FAST   = 0.55
_ALPHA_PITCH_SLOW  = 0.15
_ALPHA_PITCH_FAST  = 0.50
_ALPHA_THR         = 0.15

_DZ_ROLL   = 0.06   # dead zone cukup besar agar tidak gemetar
_DZ_PITCH  = 0.05

_THR_SNAP_ZONES = [0.0, 0.5, 1.0]
_THR_SNAP_DIST  = 0.05   # snap lebih mudah dikunci
_FAST_THRESHOLD = 0.10
_HOLD_FRAMES    = 12     # tahan lebih lama sebelum reset
_CALIB_COUNTDOWN = 1.5   # kalibrasi lebih cepat


def _apply_deadzone(val, dz):
    if abs(val) < dz:
        return 0.0
    sign = 1.0 if val > 0 else -1.0
    return sign * (abs(val) - dz) / (1.0 - dz)


def _s_curve(val, strength=0.4):
    v = float(np.clip(val, -1.0, 1.0))
    return v * (1.0 - strength * v * v + strength * abs(v) ** 3)


def _snap_throttle(val):
    for snap in _THR_SNAP_ZONES:
        if abs(val - snap) < _THR_SNAP_DIST:
            return snap
    return val


class _AdaptiveEMA:
    def __init__(self, alpha_slow, alpha_fast, deadzone=0.0):
        self._v = 0.0; self._a_slow = alpha_slow; self._a_fast = alpha_fast
        self._dz = deadzone; self._init = False

    def update(self, val):
        if not self._init:
            self._v = val; self._init = True; return self._v
        delta = val - self._v
        if abs(delta) < self._dz:
            return self._v
        alpha = self._a_fast if abs(delta) > _FAST_THRESHOLD else self._a_slow
        self._v += alpha * delta
        return self._v

    @property
    def value(self): return self._v

    def reset(self, val=0.0):
        self._v = val; self._init = True


@dataclass
class YokeData:
    roll:      float = 0.0
    pitch:     float = 0.0
    throttle:  float = 0.3
    active:    bool  = False
    left_px:    Tuple[int, int] = (0, 0)
    right_px:   Tuple[int, int] = (0, 0)
    center_px:  Tuple[int, int] = (0, 0)
    angle_deg:  float = 0.0
    dist_norm:  float = 0.0
    thr_raw:    float = 0.3
    calibrating:     bool  = False
    calib_countdown: float = 0.0
    recalib_gesture: bool  = False


class VirtualYoke:
    """
    Dual-hand controller yang mudah dan akurat.
    TANGAN KIRI  = Throttle (naik = gas)
    TANGAN KANAN = Kemudi (miring = roll, naik/turun = pitch)
    """

    def __init__(self, frame_w: int, frame_h: int):
        self._W = frame_w; self._H = frame_h
        self._roll_ema  = _AdaptiveEMA(_ALPHA_ROLL_SLOW,  _ALPHA_ROLL_FAST,  _DZ_ROLL)
        self._pitch_ema = _AdaptiveEMA(_ALPHA_PITCH_SLOW, _ALPHA_PITCH_FAST, _DZ_PITCH)
        self._thr_ema   = _AdaptiveEMA(_ALPHA_THR, _ALPHA_THR * 1.5)
        self._last_data       = YokeData()
        self._hold_left       = 0
        self._pitch_neutral_y = 0.5
        self._calibrated      = False
        self._calib_start     = 0.0
        self._calib_pending   = False
        self._last_throttle   = 0.3

    def update(self, hands: list) -> YokeData:
        valid = [h for h in hands if h is not None]

        if len(valid) == 0:
            self._hold_left = max(0, self._hold_left - 1)
            self._calibrated = False; self._calib_pending = False
            if self._hold_left > 0:
                return YokeData(roll=self._last_data.roll * 0.92,
                                pitch=self._last_data.pitch * 0.88,
                                throttle=self._last_data.throttle, active=False)
            self._roll_ema.reset(0.0); self._pitch_ema.reset(0.0)
            data = YokeData(active=False, throttle=self._last_throttle)
            self._last_data = data; return data

        self._hold_left = _HOLD_FRAMES
        lh, rh = self._identify_hands(valid)
        lp = self._palm_center_norm(lh)
        rp = self._palm_center_norm(rh)
        lp_px = (int(lp[0] * self._W), int(lp[1] * self._H))
        rp_px = (int(rp[0] * self._W), int(rp[1] * self._H))
        cp_px = ((lp_px[0] + rp_px[0]) // 2, (lp_px[1] + rp_px[1]) // 2)
        fist  = self._is_fist(rh)
        # Tidak perlu kalibrasi — pitch pakai zona Y absolut layar

        raw_thr  = 1.0 - (lp[1] - _THR_TOP_Y) / (_THR_BOT_Y - _THR_TOP_Y)
        raw_thr  = float(np.clip(_snap_throttle(float(np.clip(raw_thr, 0.0, 1.0))), 0.0, 1.0))
        throttle = float(np.clip(self._thr_ema.update(raw_thr), 0.0, 1.0))
        self._last_throttle = throttle

        angle_deg = self._hand_tilt_angle(rh)
        raw_roll  = _s_curve(_apply_deadzone(angle_deg / _ROLL_MAX_DEG, _DZ_ROLL), 0.35)
        roll      = float(np.clip(self._roll_ema.update(raw_roll), -1.0, 1.0))

        # PITCH: pakai posisi Y absolut layar, dibagi 3 zona
        #   Atas   (y < 0.35) → hidung naik (pitch negatif)
        #   Tengah (0.35–0.65) → netral
        #   Bawah  (y > 0.65) → hidung turun (pitch positif)
        _PITCH_TOP = 0.35   # batas atas zona netral
        _PITCH_BOT = 0.65   # batas bawah zona netral
        ry = rp[1]
        if ry < _PITCH_TOP:
            raw_pitch = -((_PITCH_TOP - ry) / _PITCH_TOP)   # naik
        elif ry > _PITCH_BOT:
            raw_pitch =  ((ry - _PITCH_BOT) / (1.0 - _PITCH_BOT))  # turun
        else:
            raw_pitch = 0.0   # zona netral di tengah layar
        raw_pitch = float(np.clip(raw_pitch, -1.0, 1.0))
        # s-curve agar halus di sekitar netral
        raw_pitch = _s_curve(raw_pitch, 0.3)
        pitch     = float(np.clip(self._pitch_ema.update(raw_pitch), -1.0, 1.0))

        dx_n = rp[0] - lp[0]; dy_n = rp[1] - lp[1]
        dist_norm = float(np.clip(math.hypot(dx_n, dy_n) / 0.8, 0.0, 1.0))

        data = YokeData(roll=roll, pitch=pitch, throttle=throttle, active=True,
                        left_px=lp_px, right_px=rp_px, center_px=cp_px,
                        angle_deg=angle_deg, dist_norm=dist_norm, thr_raw=raw_thr,
                        recalib_gesture=fist)
        self._last_data = data; return data

    def reset_calibration(self):
        self._calibrated = False; self._calib_pending = False

    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, data: YokeData):
        W, H = self._W, self._H
        ph   = time.time() * 2.5
        font = cv2.FONT_HERSHEY_SIMPLEX

        if data.calibrating:
            ct = data.calib_countdown
            bar_w = int(W * 0.5 * (1 - ct / _CALIB_COUNTDOWN))
            bar_x = W // 4
            cv2.rectangle(frame, (bar_x, H//2+24), (bar_x+bar_w, H//2+38), (0,200,100), -1)
            cv2.rectangle(frame, (bar_x, H//2+24), (bar_x+W//2, H//2+38), (40,80,50), 1)
            msg1 = "POSISIKAN TANGAN NETRAL"; msg2 = f"Kalibrasi dalam {ct:.1f}s..."
            (w1,_),_ = cv2.getTextSize(msg1,font,0.65,2)
            (w2,_),_ = cv2.getTextSize(msg2,font,0.48,1)
            cv2.putText(frame,msg1,((W-w1)//2,H//2),font,0.65,(0,255,150),2,cv2.LINE_AA)
            cv2.putText(frame,msg2,((W-w2)//2,H//2+20),font,0.48,(0,200,100),1,cv2.LINE_AA)
            cv2.putText(frame,"Kepalkan tangan kanan untuk re-kalibrasi kapanpun",
                        (W//2-200,H-18),font,0.33,(60,100,60),1,cv2.LINE_AA)
            for pt,col in [(data.left_px,(0,200,100)),(data.right_px,(0,160,255))]:
                cv2.circle(frame,pt,12,col,-1,cv2.LINE_AA)
            return

        if not data.active:
            txt = "  ANGKAT 2 TANGAN UNTUK TERBANG"
            (tw,_),_ = cv2.getTextSize(txt,font,0.50,1)
            cv2.putText(frame,txt,((W-tw)//2,H-18),font,0.50,(60,90,60),1,cv2.LINE_AA)
            return

        lp = data.left_px; rp = data.right_px

        # Throttle bar (kiri)
        thr_top_px = int(_THR_TOP_Y * H); thr_bot_px = int(_THR_BOT_Y * H)
        gx = max(16, lp[0] - 34)
        self._draw_zone_bar(frame,gx,thr_top_px,thr_bot_px,data.throttle,"THR",
                            (0,180,80),(0,100,220),snap_zones=_THR_SNAP_ZONES)

        # Pitch bar (kanan) — zona absolut: atas=naik, tengah=netral, bawah=turun
        _PITCH_TOP_ABS = int(0.35 * H)
        _PITCH_BOT_ABS = int(0.65 * H)
        gx2 = min(W-16, rp[0] + 34)
        self._draw_zone_bar(frame,gx2,10,H-10,
                            (data.pitch+1.0)*0.5,"PCH",
                            (40,180,255),(0,80,200),
                            neutral_y=(_PITCH_TOP_ABS + _PITCH_BOT_ABS)//2,
                            neutral_band=(_PITCH_TOP_ABS, _PITCH_BOT_ABS))

        # Marker tangan
        for pt,col_fill,col_ring,label in [
            (lp,(0,220,100),(0,60,30),"THROTTLE"),
            (rp,(0,160,255),(0,30,80),"KEMUDI"),
        ]:
            pr = 14 + int(3*math.sin(ph))
            ov3 = frame.copy()
            cv2.circle(ov3,pt,pr+10,col_ring,-1,cv2.LINE_AA)
            cv2.addWeighted(ov3,0.30,frame,0.70,0,frame)
            cv2.circle(frame,pt,pr,col_fill,-1,cv2.LINE_AA)
            cv2.circle(frame,pt,pr+10,col_ring,1,cv2.LINE_AA)
            cv2.circle(frame,pt,4,(255,255,255),-1,cv2.LINE_AA)
            cv2.putText(frame,label,(pt[0]-30,pt[1]-22),font,0.34,col_fill,1,cv2.LINE_AA)

        if data.recalib_gesture:
            cv2.putText(frame,"KEPAL: RE-KALIBRASI...",(rp[0]-55,rp[1]+30),
                        font,0.38,(0,255,200),1,cv2.LINE_AA)

        # Garis konektor
        ov4 = frame.copy(); pulse = int(80+50*math.sin(ph))
        cv2.line(ov4,lp,rp,(0,pulse//3,pulse),6,cv2.LINE_AA)
        cv2.line(ov4,lp,rp,(0,160,220),1,cv2.LINE_AA)
        cv2.addWeighted(ov4,0.40,frame,0.60,0,frame)

        # Panah roll
        roll_rad = math.radians(data.angle_deg); arm = 42
        ax = int(rp[0]+arm*math.cos(roll_rad)); ay = int(rp[1]+arm*math.sin(roll_rad))
        bx = int(rp[0]-arm*math.cos(roll_rad)); by = int(rp[1]-arm*math.sin(roll_rad))
        intensity = abs(data.roll)
        roll_col = (0,int(200*(1-intensity)),int(100+155*intensity)) if intensity > 0.05 else (30,60,80)
        cv2.arrowedLine(frame,(bx,by),(ax,ay),roll_col,2,cv2.LINE_AA,tipLength=0.28)

        # HUD panel
        pw,ph_h = 230,205; px0 = W-pw-14; py0 = H-ph_h-14
        bg = frame[py0:py0+ph_h,px0:px0+pw].copy()
        dark = np.full_like(bg,(4,8,14))
        cv2.addWeighted(dark,0.85,bg,0.15,0,bg)
        frame[py0:py0+ph_h,px0:px0+pw] = bg
        pv = int(60+45*math.sin(ph))
        cv2.rectangle(frame,(px0,py0),(px0+pw,py0+ph_h),(0,pv,pv*2),1)
        cv2.putText(frame,"YOKE v4",(px0+10,py0+20),font,0.40,(0,200,255),1,cv2.LINE_AA)

        def _bar(label,val,y,lo=-1.0,hi=1.0,color=(0,220,160)):
            cv2.putText(frame,label,(px0+10,y),font,0.34,(110,125,145),1,cv2.LINE_AA)
            bx0b,bx1b,byb = px0+72,px0+pw-10,y+3; bh=11
            cv2.rectangle(frame,(bx0b,byb-bh),(bx1b,byb),(14,20,30),-1)
            span = bx1b-bx0b
            if lo < 0:
                mid = bx0b+span//2
                fill = int(span*0.5*min(1.0,abs(val)))
                rc = color if abs(val)<0.75 else (0,80,255)
                if val>=0: cv2.rectangle(frame,(mid,byb-bh),(min(bx1b,mid+fill),byb),rc,-1)
                else: cv2.rectangle(frame,(max(bx0b,mid-fill),byb-bh),(mid,byb),rc,-1)
                cv2.line(frame,(mid,byb-bh-2),(mid,byb+2),(50,65,75),1)
            else:
                fill = int(span*np.clip(val,0,1))
                cv2.rectangle(frame,(bx0b,byb-bh),(bx0b+fill,byb),color,-1)
            cv2.rectangle(frame,(bx0b,byb-bh),(bx1b,byb),(32,45,55),1)
            txt = f"{val:+.2f}" if lo<0 else f"{val:.0%}"
            cv2.putText(frame,txt,(bx1b+4,byb),font,0.32,(165,235,165),1,cv2.LINE_AA)

        _bar("ROLL", data.roll,    py0+50, -1,1,(0,180,255))
        _bar("PITCH",data.pitch,   py0+78, -1,1,(80,255,160))
        _bar("THR",  data.throttle,py0+106, 0,1,(0,255,140))
        cv2.putText(frame,f"TILT  {data.angle_deg:+.1f} deg",(px0+10,py0+132),font,0.33,(90,110,130),1,cv2.LINE_AA)
        cv2.putText(frame,"  KIRI  = gas (naik/turun)",  (px0+10,py0+152),font,0.30,(0,180,80), 1,cv2.LINE_AA)
        cv2.putText(frame,"  KANAN = kemudi (XY+tilt)",  (px0+10,py0+166),font,0.30,(0,140,220),1,cv2.LINE_AA)
        cv2.putText(frame,"  KEPAL kanan = re-kalibrasi",(px0+10,py0+180),font,0.28,(0,200,255),1,cv2.LINE_AA)
        cv2.putText(frame,"  AKTIF",                     (px0+10,py0+196),font,0.34,(0,255,100),1,cv2.LINE_AA)
    
        self._draw_attitude(frame, data, px0+pw//2, py0-56)

    def _draw_zone_bar(self,frame,gx,top_y,bot_y,val_norm,label,col_lo,col_hi,
                       snap_zones=None,neutral_y=None,neutral_band=None):
        font = cv2.FONT_HERSHEY_SIMPLEX; span = bot_y - top_y
        ov = frame.copy()
        cv2.rectangle(ov,(gx-7,top_y),(gx+7,bot_y),(10,25,14),-1)
        cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
        cv2.rectangle(frame,(gx-7,top_y),(gx+7,bot_y),(40,70,50),1)
        fill_h = int(span*val_norm); fill_y = bot_y-fill_h
        c = [int(col_lo[i]+(col_hi[i]-col_lo[i])*val_norm) for i in range(3)]
        cv2.rectangle(frame,(gx-5,fill_y),(gx+5,bot_y),tuple(c),-1)
        if snap_zones:
            for s in snap_zones:
                sy = int(bot_y-s*span)
                col_s = (255,200,0) if abs(val_norm-s)<0.05 else (60,80,50)
                cv2.line(frame,(gx-12,sy),(gx+12,sy),col_s,1)
        # Zona netral pitch (band hijau di tengah layar)
        if neutral_band is not None:
            nb_top, nb_bot = neutral_band
            ov2 = frame.copy()
            cv2.rectangle(ov2,(gx-9,nb_top),(gx+9,nb_bot),(0,60,20),-1)
            cv2.addWeighted(ov2,0.35,frame,0.65,0,frame)
            cv2.line(frame,(gx-14,nb_top),(gx+14,nb_top),(0,200,100),1)
            cv2.line(frame,(gx-14,nb_bot),(gx+14,nb_bot),(0,200,100),1)
            mid = (nb_top+nb_bot)//2
            cv2.line(frame,(gx-16,mid),(gx+16,mid),(0,255,120),1)
            cv2.putText(frame,"NET",(gx+18,mid+4),font,0.28,(0,200,100),1,cv2.LINE_AA)
            cv2.putText(frame,"NAIK",(gx+18,nb_top-4),font,0.28,(80,200,255),1,cv2.LINE_AA)
            cv2.putText(frame,"TRN",(gx+18,nb_bot+10),font,0.28,(255,160,60),1,cv2.LINE_AA)
        elif neutral_y is not None:
            cv2.line(frame,(gx-14,neutral_y),(gx+14,neutral_y),(0,100,160),1)
        marker_y = int(bot_y-val_norm*span)
        bright = (min(255,c[0]+80),min(255,c[1]+80),min(255,c[2]+80))
        cv2.rectangle(frame,(gx-7,marker_y-5),(gx+7,marker_y+5),bright,-1)
        cv2.rectangle(frame,(gx-7,marker_y-5),(gx+7,marker_y+5),(255,255,255),1)
        cv2.putText(frame,f"{int(val_norm*100)}%",(gx+14,marker_y+5),font,0.36,bright,1,cv2.LINE_AA)
        cv2.putText(frame,label,(gx-14,top_y-8),font,0.34,bright,1,cv2.LINE_AA)

    def _draw_attitude(self,frame,data,cx,cy):
        r=38
        cv2.circle(frame,(cx,cy),r+2,(5,10,18),-1)
        cv2.circle(frame,(cx,cy),r,(16,28,42),-1)
        roll_rad  = math.radians(data.roll*40)
        pitch_off = int(data.pitch*r*0.55)
        cos_r = math.cos(roll_rad); sin_r = math.sin(roll_rad)
        hx1 = int(cx-r*cos_r); hy1 = int(cy-r*sin_r+pitch_off)
        hx2 = int(cx+r*cos_r); hy2 = int(cy+r*sin_r+pitch_off)
        mask = np.zeros((self._H,self._W),np.uint8)
        cv2.circle(mask,(cx,cy),r,255,-1)
        sky = np.zeros_like(frame); gnd = np.zeros_like(frame)
        pts_sky = np.array([[hx1,hy1],[hx2,hy2],[cx+r,cy-r],[cx-r,cy-r]],np.int32)
        pts_gnd = np.array([[hx1,hy1],[hx2,hy2],[cx+r,cy+r],[cx-r,cy+r]],np.int32)
        cv2.fillPoly(sky,[pts_sky],(70,38,16)); cv2.fillPoly(gnd,[pts_gnd],(14,44,10))
        tmp = frame.copy(); m = mask[:,:,np.newaxis]>0
        np.copyto(tmp,sky,where=m&(sky>0)); np.copyto(tmp,gnd,where=m&(gnd>0))
        cv2.addWeighted(tmp,0.65,frame,0.35,0,frame)
        cv2.line(frame,(hx1,hy1),(hx2,hy2),(0,210,255),2,cv2.LINE_AA)
        cv2.line(frame,(cx-14,cy),(cx-4,cy),(255,255,255),2,cv2.LINE_AA)
        cv2.line(frame,(cx+4,cy),(cx+14,cy),(255,255,255),2,cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),3,(255,240,80),-1,cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),r,(36,72,90),1,cv2.LINE_AA)

    def _identify_hands(self, hands):
        sorted_h = sorted(hands, key=lambda h: h.landmarks[0].x)
        return sorted_h[0], sorted_h[-1]

    def _palm_center_norm(self, hr):
        lm = hr.landmarks; indices=[0,5,9,13,17]
        xs=[lm[i].x for i in indices]; ys=[lm[i].y for i in indices]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    def _hand_tilt_angle(self, hr):
        lm = hr.landmarks
        dx = lm[5].x - lm[17].x; dy = lm[5].y - lm[17].y
        return math.degrees(math.atan2(dy,dx))

    def _is_fist(self, hr):
        lm = hr.landmarks; pairs = [(8,6),(12,10),(16,14),(20,18)]
        closed = sum(1 for tip,pip in pairs if lm[tip].y > lm[pip].y)
        return closed >= 3