"""
hand/air_canvas.py  [v5 — PRO DRAWING STUDIO]
===============================================
Air Canvas: draw on screen with index fingertip.

IMPROVEMENTS in v5:
  - Pro drawing toolbar on LEFT side (vertical, icon-based)
  - Fixed line continuity: MAX_GAP increased, adaptive gap detection
  - Stroke interpolation: fills gaps between frames
  - Pressure simulation based on speed
  - Canvas overlay with grid option
  - Better color picker with recent colors
  - Smart canvas composite (no draw flicker)
  - Stroke smoothing with bezier-like interpolation

Controls:
  ☝  POINT      Draw
  ✊  FIST       Erase (circle preview)
  🖐  OPEN_PALM  Clear canvas (with fade)
  🤟  ILY        Cycle brush style
  🤏  PINCH      Pick color / use tools
  U              Undo last stroke
  [ / ]          Decrease / Increase brush size
"""

import math, time, random
from collections import deque
from typing import Optional, Tuple, List
import cv2, numpy as np

PALETTES = [
    ("Red",     (40,  40, 220)),
    ("Orange",  (20, 160, 255)),
    ("Yellow",  (20, 210, 255)),
    ("Green",   (40, 230,  80)),
    ("Cyan",    (255,230,  50)),
    ("Blue",    (255,160,  80)),
    ("Violet",  (220, 60, 200)),
    ("White",   (240,240, 240)),
]

BRUSH_NAMES  = ["Pen", "Neon", "Chalk", "Water", "Spray", "Marker"]
ERASER_R     = 44
MIN_B, MAX_B = 2, 30
MAX_GAP      = 90    # FIXED: bigger gap allowed before breaking stroke
INTERP_GAP   = 220   # still interpolate up to this distance
SMOOTH_WIN   = 5
UNDO_MAX     = 20

_WHEEL_R    = 60
_PANEL_W    = 210
_PANEL_H    = 300
_PANEL_Y    = 62

_TB_W       = 52
_TB_PAD     = 6
_TB_BTN_H   = 44
_TB_X       = 6
_TB_Y       = 62


def _hsv_bgr(h, s=230, v=255):
    return cv2.cvtColor(np.uint8([[[int(h/2)%180,s,v]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

def _clamp(v,lo,hi): return max(lo,min(hi,v))
def _dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

def _make_color_wheel(r):
    d = r*2
    img = np.zeros((d,d,3),np.uint8)
    for y in range(d):
        for x in range(d):
            dx=x-r; dy=y-r
            dist=math.hypot(dx,dy)
            if dist>r: continue
            hue=int(math.degrees(math.atan2(dy,dx)))%360
            sat=int(dist/r*255)
            img[y,x]=cv2.cvtColor(np.uint8([[[int(hue/2)%180,sat,255]]]),
                                   cv2.COLOR_HSV2BGR)[0][0]
    return img


class AirCanvas:
    def __init__(self, W, H):
        self.W,self.H = W,H
        self._canvas = np.zeros((H,W,3),np.uint8)
        self._undo_stack = []

        self._color     = (40,200,255)
        self._brush_idx = 1
        self._brush_size= 8

        self._smooth_buf = deque(maxlen=SMOOTH_WIN)
        self._stroke_pts = []

        self._drawing   = False
        self._last_pt   = None
        self._prev_pt   = None
        self._erasing   = False
        self._erase_pt  = None
        self._ily_prev  = False
        self._clear_alpha = 0.0
        self._last_spd  = 0.0
        self._cursor_pt = None

        self._recent_colors = [list(c) for _,c in PALETTES[:4]]

        self._px0 = W - _PANEL_W - 10
        self._py0 = _PANEL_Y
        self._wheel_cx = self._px0 + _PANEL_W//2
        self._wheel_cy = self._py0 + 22 + _WHEEL_R + 4
        self._wheel_img = _make_color_wheel(_WHEEL_R)
        self._hover_color = None
        self._hover_pt = None
        self._show_picker = True

        self._sat   = 1.0
        self._val   = 1.0
        self._hue   = 200.0

        self._show_toolbar = True
        self._toolbar_btns = self._build_toolbar()
        self._tool_hover   = -1
        self._show_grid    = False

        self._notif    = ""
        self._notif_t  = 0.0
        self._notif_col= (255,255,255)

        self._last_frame_t = time.time()
        self._frame_dt     = 0.033

        self._sync_color_from_hsv()

    def _build_toolbar(self):
        btns = []
        items = [
            ("Pen",   "P",  "brush_0"),
            ("Neon",  "N",  "brush_1"),
            ("Chalk", "Ch", "brush_2"),
            ("Water", "W",  "brush_3"),
            ("Spray", "Sp", "brush_4"),
            ("Mark",  "M",  "brush_5"),
            (None,    None, "sep"),
            ("Erase", "E",  "eraser"),
            ("Clear", "X",  "clear"),
            ("Undo",  "U",  "undo"),
            (None,    None, "sep"),
            ("Grid",  "G",  "grid"),
            ("Color", "C",  "picker"),
        ]
        y = _TB_Y + _TB_PAD + 14
        for label, icon, action in items:
            if action == "sep":
                btns.append({"type":"sep", "y":y+2}); y += 10
            else:
                btns.append({
                    "type": "btn",
                    "icon": icon,
                    "label": label,
                    "action": action,
                    "x": _TB_X,
                    "y": y,
                    "w": _TB_W,
                    "h": _TB_BTN_H,
                })
                y += _TB_BTN_H + 3
        self._toolbar_h = y - _TB_Y + _TB_PAD
        return btns

    @property
    def color(self): return self._color
    @property
    def brush_name(self): return BRUSH_NAMES[self._brush_idx]

    def _sync_color_from_hsv(self):
        h = int(self._hue/2)%180
        s = int(self._sat*255)
        v = int(self._val*255)
        self._color = cv2.cvtColor(np.uint8([[[h,s,v]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()

    def _add_recent_color(self, col):
        col_list = list(col)
        if col_list not in self._recent_colors:
            self._recent_colors.insert(0, col_list)
            self._recent_colors = self._recent_colors[:6]

    def undo(self):
        if self._undo_stack:
            self._canvas = self._undo_stack.pop()
            self._notify("Undo ↩", (200,200,255))

    def change_brush_size(self, delta):
        self._brush_size = _clamp(self._brush_size + delta, MIN_B, MAX_B)
        self._notify(f"Size: {self._brush_size}", (200,200,200))

    def update_from_landmarks(self, landmarks, W, H, gesture_val, dt):
        now = time.time()
        self._frame_dt = min(now - self._last_frame_t, 0.1)
        self._last_frame_t = now

        if landmarks is None:
            self._drawing=False; self._erasing=False
            self._last_pt=None; self._prev_pt=None
            self._smooth_buf.clear(); self._stroke_pts.clear()
            self._hover_pt=None; self._hover_color=None
            return

        lm=landmarks
        def px(i): return int(_clamp(lm[i].x,0,1)*W), int(_clamp(lm[i].y,0,1)*H)

        tip=px(8)
        pinch_d=math.hypot(lm[4].x-lm[8].x, lm[4].y-lm[8].y)

        index_up  = lm[8].y  < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up   = lm[16].y < lm[14].y
        pinky_up  = lm[20].y < lm[18].y
        n_up      = sum([index_up,middle_up,ring_up,pinky_up])
        is_fist   = n_up==0
        is_point  = index_up and not middle_up and not ring_up and not pinky_up
        is_palm   = n_up>=4
        is_ily    = index_up and pinky_up and not middle_up and not ring_up
        is_pinch  = pinch_d<0.06

        self._cursor_pt=tip

        if self._clear_alpha>0:
            self._clear_alpha=max(0.0,self._clear_alpha-dt*3.0)

        # Toolbar check
        on_toolbar = (_TB_X <= tip[0] <= _TB_X+_TB_W+4 and
                      _TB_Y <= tip[1] <= _TB_Y+self._toolbar_h)
        self._tool_hover = -1
        if on_toolbar:
            for i, btn in enumerate(self._toolbar_btns):
                if btn["type"] == "sep": continue
                bx,by,bw,bh = btn["x"],btn["y"],btn["w"],btn["h"]
                if bx<=tip[0]<=bx+bw and by<=tip[1]<=by+bh:
                    self._tool_hover = i
                    if is_pinch:
                        self._trigger_tool(btn["action"])
                        self._drawing=False; self._last_pt=None
                        return
            if not is_fist:
                self._drawing=False; self._last_pt=None; return

        # Color picker check
        on_panel = (self._show_picker and
                    self._px0 <= tip[0] <= self._px0+_PANEL_W and
                    self._py0 <= tip[1] <= self._py0+_PANEL_H)

        dx_w=tip[0]-self._wheel_cx; dy_w=tip[1]-self._wheel_cy
        dist_w=math.hypot(dx_w,dy_w)
        on_wheel = on_panel and dist_w <= _WHEEL_R

        sat_y0=self._wheel_cy+_WHEEL_R+16
        val_y0=sat_y0+26
        sl_x0=self._px0+10; sl_x1=self._px0+_PANEL_W-10
        on_sat_slider=(on_panel and sl_x0<=tip[0]<=sl_x1 and sat_y0-8<=tip[1]<=sat_y0+18)
        on_val_slider=(on_panel and sl_x0<=tip[0]<=sl_x1 and val_y0-8<=tip[1]<=val_y0+18)

        swatch_y0=val_y0+30; sw,sh,gap=20,14,4
        on_preset=None
        if on_panel and swatch_y0<=tip[1]<=swatch_y0+sh+gap+sh+8:
            for i,(name,c) in enumerate(PALETTES):
                row=i//4; ci=i%4
                bx=self._px0+8+ci*(sw+gap); by=swatch_y0+3+row*(sh+gap)
                if bx<=tip[0]<=bx+sw and by<=tip[1]<=by+sh:
                    on_preset=i; break

        if on_wheel:
            self._hue=math.degrees(math.atan2(dy_w,dx_w))%360
            self._hover_pt=tip
            h=int(self._hue/2)%180; s_v=int(self._sat*255); v_v=int(self._val*255)
            self._hover_color=cv2.cvtColor(np.uint8([[[h,s_v,v_v]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
        else:
            self._hover_pt=None; self._hover_color=None

        if is_pinch or is_point:
            if on_sat_slider:
                self._sat=_clamp((tip[0]-sl_x0)/(sl_x1-sl_x0),0.0,1.0)
                self._sync_color_from_hsv()
            elif on_val_slider:
                self._val=_clamp((tip[0]-sl_x0)/(sl_x1-sl_x0),0.0,1.0)
                self._sync_color_from_hsv()

        if is_pinch and on_panel:
            self._drawing=False; self._last_pt=None
            if on_wheel:
                self._sync_color_from_hsv()
                self._add_recent_color(self._color)
                self._notify("Color picked!", self._color)
            elif on_preset is not None:
                c=PALETTES[on_preset][1]
                self._color=list(c)
                bgr=np.uint8([[[c[0],c[1],c[2]]]])
                hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)[0][0]
                self._hue=float(hsv[0])*2.0; self._sat=float(hsv[1])/255.0; self._val=float(hsv[2])/255.0
                self._add_recent_color(c)
                self._notify(f"{PALETTES[on_preset][0]}", list(c))
            return

        if on_panel and not is_fist:
            self._drawing=False; self._last_pt=None; return

        # Mode handling
        if is_palm and gesture_val=="OPEN_PALM":
            if not self._drawing:
                self._save_undo(); self._canvas[:]=0
                self._clear_alpha=1.0; self._notify("Canvas Cleared!",(180,255,180))
                self._drawing=False; self._last_pt=None; self._stroke_pts.clear()

        elif is_ily:
            if not self._ily_prev:
                self._brush_idx=(self._brush_idx+1)%len(BRUSH_NAMES)
                self._notify(f"Brush: {self.brush_name}",(255,200,100))
            self._ily_prev=True; self._drawing=False; self._last_pt=None

        elif is_fist:
            self._ily_prev=False; self._drawing=False
            self._last_pt=None; self._prev_pt=None; self._stroke_pts.clear()
            self._erasing=True; self._erase_pt=tip
            cv2.circle(self._canvas,tip,ERASER_R,(0,0,0),-1)

        elif is_point:
            self._ily_prev=False; self._erasing=False
            self._smooth_buf.append(tip)
            sx=int(sum(p[0] for p in self._smooth_buf)/len(self._smooth_buf))
            sy=int(sum(p[1] for p in self._smooth_buf)/len(self._smooth_buf))
            stip=(sx,sy)

            if self._last_pt is not None:
                d=_dist(stip,self._last_pt)
                if d < MAX_GAP:
                    # Normal drawing - tight gap, smooth line
                    spd=min(d/max(self._frame_dt,0.001),1000)
                    self._last_spd=self._last_spd*0.6+spd*0.4
                    pressure=_clamp(1.0-self._last_spd/700,0.35,1.0)
                    b=max(1, int(self._brush_size*pressure))
                    self._paint(self._last_pt, stip, list(self._color), b)

                elif d < INTERP_GAP:
                    # Medium gap: interpolate to fill - FIXES LINE BREAKS
                    steps = max(2, int(d / 15))
                    p_prev = self._last_pt
                    for k in range(1, steps+1):
                        t = k / steps
                        ix = int(self._last_pt[0] + (stip[0]-self._last_pt[0])*t)
                        iy = int(self._last_pt[1] + (stip[1]-self._last_pt[1])*t)
                        ipt = (ix, iy)
                        spd=min(d/max(self._frame_dt,0.001),1000)
                        self._last_spd=self._last_spd*0.6+spd*0.4
                        pressure=_clamp(1.0-self._last_spd/700,0.35,1.0)
                        b=max(1, int(self._brush_size*pressure))
                        self._paint(p_prev, ipt, list(self._color), b)
                        p_prev = ipt
                # else gap too large, lift pen

            if not self._drawing:
                self._save_undo(); self._drawing=True
            self._prev_pt = self._last_pt
            self._last_pt = stip
            self._stroke_pts.append(stip)

        else:
            self._ily_prev=False; self._drawing=False
            self._last_pt=None; self._prev_pt=None
            self._erasing=False; self._smooth_buf.clear()
            self._stroke_pts.clear()

    def _trigger_tool(self, action):
        if action.startswith("brush_"):
            idx = int(action.split("_")[1])
            self._brush_idx = idx
            self._notify(f"Brush: {BRUSH_NAMES[idx]}", (255,200,100))
        elif action == "clear":
            self._save_undo(); self._canvas[:]=0
            self._clear_alpha=1.0; self._notify("Canvas Cleared!",(180,255,180))
        elif action == "undo":
            self.undo()
        elif action == "grid":
            self._show_grid = not self._show_grid
            self._notify(f"Grid {'ON' if self._show_grid else 'OFF'}", (180,220,180))
        elif action == "picker":
            self._show_picker = not self._show_picker
            self._notify(f"Color Picker {'ON' if self._show_picker else 'OFF'}", (200,180,255))
        elif action == "eraser":
            self._notify("Make FIST gesture to erase", (180,180,255))

    def _paint(self, p0, p1, col, b):
        b = max(1, b)
        style = self._brush_idx

        if style==0:  # Pen
            cv2.line(self._canvas,p0,p1,col,b,cv2.LINE_AA)
            bright=[min(255,c+60) for c in col]
            cv2.line(self._canvas,p0,p1,bright,max(1,b//3),cv2.LINE_AA)

        elif style==1:  # Neon
            cv2.line(self._canvas,p0,p1,[c//6 for c in col],b+16,cv2.LINE_AA)
            cv2.line(self._canvas,p0,p1,[c//3 for c in col],b+8,cv2.LINE_AA)
            cv2.line(self._canvas,p0,p1,col,b,cv2.LINE_AA)
            cv2.line(self._canvas,p0,p1,[min(255,c+140) for c in col],max(1,b//2),cv2.LINE_AA)

        elif style==2:  # Chalk
            steps=max(1,int(_dist(p0,p1)/4))
            for k in range(steps+1):
                t=k/max(steps,1)
                mx=int(p0[0]+(p1[0]-p0[0])*t)+random.randint(-4,4)
                my=int(p0[1]+(p1[1]-p0[1])*t)+random.randint(-4,4)
                a=random.uniform(0.35,1.0)
                cv2.circle(self._canvas,(mx,my),max(1,b//2+random.randint(0,4)),[int(c*a) for c in col],-1,cv2.LINE_AA)

        elif style==3:  # Water
            for _ in range(4):
                ox,oy=random.randint(-b,b),random.randint(-b,b)
                pa=(p0[0]+ox,p0[1]+oy); pb=(p1[0]+ox,p1[1]+oy)
                dim=[max(0,c-random.randint(10,50)) for c in col]
                rx1=max(0,min(pa[0],pb[0])-b-2); ry1=max(0,min(pa[1],pb[1])-b-2)
                rx2=min(self.W,max(pa[0],pb[0])+b+2); ry2=min(self.H,max(pa[1],pb[1])+b+2)
                if rx2>rx1 and ry2>ry1:
                    tmp=self._canvas[ry1:ry2,rx1:rx2].copy()
                    cv2.line(tmp,(pa[0]-rx1,pa[1]-ry1),(pb[0]-rx1,pb[1]-ry1),dim,b+3,cv2.LINE_AA)
                    cv2.addWeighted(tmp,0.30,self._canvas[ry1:ry2,rx1:rx2],0.70,0,self._canvas[ry1:ry2,rx1:rx2])

        elif style==4:  # Spray
            steps=max(1,int(_dist(p0,p1)/3))
            for k in range(steps+1):
                t=k/max(steps,1)
                mx=int(p0[0]+(p1[0]-p0[0])*t); my=int(p0[1]+(p1[1]-p0[1])*t)
                for _ in range(10):
                    ang=random.uniform(0,6.28); r=random.uniform(0,b*2.5)
                    sx=int(mx+math.cos(ang)*r); sy=int(my+math.sin(ang)*r)
                    if 0<=sx<self.W and 0<=sy<self.H:
                        a=max(0.1,1.0-r/(b*2.5))
                        cv2.circle(self._canvas,(sx,sy),1,[int(c*a) for c in col],-1)

        elif style==5:  # Marker
            ov_b = b + 4
            rx1=max(0,min(p0[0],p1[0])-ov_b-2); ry1=max(0,min(p0[1],p1[1])-ov_b-2)
            rx2=min(self.W,max(p0[0],p1[0])+ov_b+2); ry2=min(self.H,max(p0[1],p1[1])+ov_b+2)
            if rx2>rx1 and ry2>ry1:
                tmp=self._canvas[ry1:ry2,rx1:rx2].copy()
                cv2.line(tmp,(p0[0]-rx1,p0[1]-ry1),(p1[0]-rx1,p1[1]-ry1),col,ov_b,cv2.LINE_AA)
                cv2.addWeighted(tmp,0.55,self._canvas[ry1:ry2,rx1:rx2],0.45,0,self._canvas[ry1:ry2,rx1:rx2])

    def _save_undo(self):
        if len(self._undo_stack)>=UNDO_MAX: self._undo_stack.pop(0)
        self._undo_stack.append(self._canvas.copy())

    def _notify(self, text, color=(255,255,255)):
        self._notif=text; self._notif_t=time.time()
        self._notif_col=tuple(color) if not isinstance(color,tuple) else color

    def render(self, frame):
        H,W=frame.shape[:2]

        if self._show_grid:
            spacing=40
            ov=frame.copy()
            for x in range(0,W,spacing): cv2.line(ov,(x,0),(x,H),(55,65,85),1)
            for y in range(0,H,spacing): cv2.line(ov,(0,y),(W,y),(55,65,85),1)
            cv2.addWeighted(ov,0.08,frame,0.92,0,frame)

        # Composite canvas
        gray=cv2.cvtColor(self._canvas,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,6,255,cv2.THRESH_BINARY)
        if mask.any():
            blended=cv2.addWeighted(frame,0.10,self._canvas,0.90,0)
            frame[mask>0]=blended[mask>0]

        # Clear flash
        if self._clear_alpha>0.02:
            ov=np.full((H,W,3),(200,220,255),np.uint8)
            cv2.addWeighted(ov,self._clear_alpha*0.45,frame,1-self._clear_alpha*0.45,0,frame)

        # Cursor
        if self._erasing and self._erase_pt:
            cx,cy=self._erase_pt
            cv2.circle(frame,(cx,cy),ERASER_R,(25,28,45),-1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),ERASER_R,(100,110,255),2,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),ERASER_R+4,[40,45,160],1,cv2.LINE_AA)
            cv2.putText(frame,"ERASE",(cx-22,cy-ERASER_R-10),cv2.FONT_HERSHEY_SIMPLEX,0.40,(100,100,255),1,cv2.LINE_AA)
        elif self._cursor_pt:
            cx,cy=self._cursor_pt
            col=list(self.color); r=self._brush_size+6
            cv2.circle(frame,(cx,cy),r+4,[c//5 for c in col],2,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),r,col,2,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),3,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),3,col,1,cv2.LINE_AA)

        if self._show_toolbar:
            self._draw_toolbar(frame)
        if self._show_picker:
            self._draw_color_picker(frame)
        self._draw_bottom_hud(frame)

        elapsed=time.time()-self._notif_t
        if self._notif and elapsed<2.0:
            a=max(0.0,1.0-elapsed/2.0)
            (tw,_),_=cv2.getTextSize(self._notif,cv2.FONT_HERSHEY_DUPLEX,0.85,2)
            mx=(W-tw)//2; my=H//2-90
            ov=frame.copy()
            cv2.rectangle(ov,(mx-14,my-30),(mx+tw+14,my+10),(8,10,22),-1)
            cv2.rectangle(ov,(mx-14,my-30),(mx+tw+14,my+10),list(self._notif_col),1)
            cv2.putText(ov,self._notif,(mx,my),cv2.FONT_HERSHEY_DUPLEX,0.85,self._notif_col,2,cv2.LINE_AA)
            cv2.addWeighted(ov,a,frame,1-a,0,frame)

    def _draw_toolbar(self, frame):
        H,W=frame.shape[:2]
        tbx=_TB_X; tby=_TB_Y
        tbw=_TB_W+4; tbh=self._toolbar_h

        roi=frame[tby:tby+tbh, tbx:tbx+tbw]
        if roi.size==0: return
        dark=np.full_like(roi,(8,10,22))
        cv2.addWeighted(dark,0.88,roi,0.12,0,roi)
        frame[tby:tby+tbh, tbx:tbx+tbw]=roi

        cv2.rectangle(frame,(tbx,tby),(tbx+tbw,tby+tbh),(35,40,72),1)
        active_col=list(self._color)
        cv2.line(frame,(tbx+1,tby),(tbx+tbw-1,tby),active_col,2)

        # Header
        cv2.putText(frame,"TOOLS",(tbx+7,tby+12),cv2.FONT_HERSHEY_SIMPLEX,0.28,(70,80,115),1,cv2.LINE_AA)

        for i,btn in enumerate(self._toolbar_btns):
            if btn["type"]=="sep":
                sy=btn["y"]
                cv2.line(frame,(tbx+4,sy),(tbx+tbw-4,sy),(32,38,65),1)
                continue

            bx,by,bw,bh=btn["x"],btn["y"],btn["w"],btn["h"]
            is_hover=(i==self._tool_hover)
            is_active=(btn["action"].startswith("brush_") and
                       int(btn["action"].split("_")[1])==self._brush_idx)

            if is_active:
                bg=[c//4 for c in active_col]
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),bg,-1)
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),active_col,1)
            elif is_hover:
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),(22,28,50),-1)
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),(55,65,100),1)

            if btn["action"]=="grid" and self._show_grid:
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),(15,45,25),-1)
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),(55,170,75),1)
            if btn["action"]=="picker" and self._show_picker:
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),(28,18,48),-1)
                cv2.rectangle(frame,(bx+1,by),(bx+bw-1,by+bh),(150,75,210),1)

            icon_col=active_col if is_active else (175,185,210) if not is_hover else (220,225,240)
            icon_txt=btn["icon"]
            (iw,_),_=cv2.getTextSize(icon_txt,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
            cv2.putText(frame,icon_txt,(bx+(bw-iw)//2,by+19),cv2.FONT_HERSHEY_SIMPLEX,0.52,icon_col,1,cv2.LINE_AA)

            lbl=btn.get("label","")
            if lbl:
                (lw,_),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.23,1)
                cv2.putText(frame,lbl,(bx+(bw-lw)//2,by+bh-5),cv2.FONT_HERSHEY_SIMPLEX,0.23,icon_col,1,cv2.LINE_AA)

        size_y=tby+tbh+4
        cv2.putText(frame,f"sz:{self._brush_size}",(tbx+4,size_y+10),cv2.FONT_HERSHEY_SIMPLEX,0.28,(70,80,115),1,cv2.LINE_AA)
        if self._undo_stack:
            cv2.putText(frame,f"u:{len(self._undo_stack)}",(tbx+4,size_y+22),cv2.FONT_HERSHEY_SIMPLEX,0.25,(60,80,140),1,cv2.LINE_AA)

    def _draw_color_picker(self, frame):
        H,W=frame.shape[:2]
        px0,py0=self._px0,self._py0
        pw,ph=_PANEL_W,_PANEL_H

        roi=frame[py0:py0+ph,px0:px0+pw]
        if roi.size==0: return
        dark=np.full_like(roi,(8,10,22))
        cv2.addWeighted(dark,0.85,roi,0.15,0,roi)
        frame[py0:py0+ph,px0:px0+pw]=roi

        col=list(self._color)
        cv2.rectangle(frame,(px0,py0),(px0+pw,py0+ph),(35,40,70),1)
        cv2.line(frame,(px0+1,py0),(px0+pw-1,py0),col,2)

        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"COLOR",(px0+8,py0+13),font,0.32,(70,80,115),1,cv2.LINE_AA)
        cv2.putText(frame,"PICKER",(px0+8,py0+25),font,0.32,(70,80,115),1,cv2.LINE_AA)

        wcx=self._wheel_cx; wcy=self._wheel_cy; r=_WHEEL_R
        wy0=wcy-r; wx0=wcx-r; wd=r*2
        if 0<=wy0 and wy0+wd<=H and 0<=wx0 and wx0+wd<=W:
            wheel_copy=self._wheel_img.copy()
            if self._val<1.0:
                cv2.multiply(wheel_copy,np.array([self._val,self._val,self._val]),wheel_copy)
            frame[wy0:wy0+wd,wx0:wx0+wd]=cv2.addWeighted(wheel_copy,0.88,frame[wy0:wy0+wd,wx0:wx0+wd],0.12,0)

        cv2.circle(frame,(wcx,wcy),r+2,[c//5 for c in col],1,cv2.LINE_AA)
        cv2.circle(frame,(wcx,wcy),r,(48,52,88),1,cv2.LINE_AA)

        hue_r=math.radians(self._hue)
        sel_d=int(self._sat*r)
        sel_x=wcx+int(math.cos(hue_r)*sel_d)
        sel_y=wcy+int(math.sin(hue_r)*sel_d)
        cv2.circle(frame,(sel_x,sel_y),9,(255,255,255),2,cv2.LINE_AA)
        cv2.circle(frame,(sel_x,sel_y),7,col,-1,cv2.LINE_AA)
        cv2.circle(frame,(sel_x,sel_y),3,(255,255,255),-1,cv2.LINE_AA)

        if self._hover_pt and self._hover_color:
            dx=self._hover_pt[0]-wcx; dy=self._hover_pt[1]-wcy
            if math.hypot(dx,dy)<=r:
                cv2.circle(frame,(self._hover_pt[0],self._hover_pt[1]),11,self._hover_color,2,cv2.LINE_AA)
                cv2.circle(frame,(self._hover_pt[0],self._hover_pt[1]),3,(255,255,255),-1,cv2.LINE_AA)

        sat_y=wcy+r+16
        sl_x0=px0+10; sl_x1=px0+pw-10
        cv2.putText(frame,"SAT",(sl_x0,sat_y-3),font,0.28,(65,75,105),1,cv2.LINE_AA)
        for xi in range(sl_x0,sl_x1):
            t=(xi-sl_x0)/(sl_x1-sl_x0)
            h2=int(self._hue/2)%180; s2=int(t*255); v2=int(self._val*255)
            c2=cv2.cvtColor(np.uint8([[[h2,s2,v2]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
            cv2.line(frame,(xi,sat_y),(xi,sat_y+10),c2,1)
        cv2.rectangle(frame,(sl_x0,sat_y),(sl_x1,sat_y+10),(48,52,88),1)
        tx=sl_x0+int(self._sat*(sl_x1-sl_x0))
        cv2.circle(frame,(tx,sat_y+5),7,(255,255,255),-1,cv2.LINE_AA)
        cv2.circle(frame,(tx,sat_y+5),5,col,-1,cv2.LINE_AA)

        val_y=sat_y+26
        cv2.putText(frame,"BRI",(sl_x0,val_y-3),font,0.28,(65,75,105),1,cv2.LINE_AA)
        for xi in range(sl_x0,sl_x1):
            t=(xi-sl_x0)/(sl_x1-sl_x0)
            h2=int(self._hue/2)%180; s2=int(self._sat*255); v2=int(t*255)
            c2=cv2.cvtColor(np.uint8([[[h2,s2,v2]]]),cv2.COLOR_HSV2BGR)[0][0].tolist()
            cv2.line(frame,(xi,val_y),(xi,val_y+10),c2,1)
        cv2.rectangle(frame,(sl_x0,val_y),(sl_x1,val_y+10),(48,52,88),1)
        tx2=sl_x0+int(self._val*(sl_x1-sl_x0))
        cv2.circle(frame,(tx2,val_y+5),7,(255,255,255),-1,cv2.LINE_AA)
        cv2.circle(frame,(tx2,val_y+5),5,col,-1,cv2.LINE_AA)

        sw_y=val_y+18; sw_h=20; sw_w=pw-22
        cv2.rectangle(frame,(sl_x0,sw_y),(sl_x0+sw_w,sw_y+sw_h),col,-1)
        cv2.rectangle(frame,(sl_x0,sw_y),(sl_x0+sw_w,sw_y+sw_h),(65,70,100),1)
        hex_r=col[2]; hex_g=col[1]; hex_b=col[0]
        hex_txt=f"#{hex_r:02X}{hex_g:02X}{hex_b:02X}"
        bright=sum(col)>400
        tc=(0,0,0) if bright else (220,220,220)
        cv2.putText(frame,hex_txt,(sl_x0+6,sw_y+13),font,0.33,tc,1,cv2.LINE_AA)

        swatch_y=sw_y+sw_h+6
        sw2,sh2,gap2=20,14,4
        cv2.putText(frame,"QUICK",(sl_x0,swatch_y-2),font,0.26,(55,62,92),1,cv2.LINE_AA)
        swatch_y+=3
        for i,(name,c) in enumerate(PALETTES):
            row=i//4; ci=i%4
            bx=px0+8+ci*(sw2+gap2); by=swatch_y+row*(sh2+gap2)
            cv2.rectangle(frame,(bx,by),(bx+sw2,by+sh2),list(c),-1)
            if list(c)==list(self._color):
                cv2.rectangle(frame,(bx-2,by-2),(bx+sw2+2,by+sh2+2),(255,255,255),2)
            else:
                cv2.rectangle(frame,(bx,by),(bx+sw2,by+sh2),(28,32,50),1)

        brush_y=swatch_y+(sh2+gap2)*2+8
        tab_w=(pw-4)//len(BRUSH_NAMES)
        for i,name in enumerate(BRUSH_NAMES):
            bx=px0+2+i*tab_w
            active=(i==self._brush_idx)
            bg_c=[c//4 for c in col] if active else (20,22,36)
            cv2.rectangle(frame,(bx,brush_y),(bx+tab_w-1,brush_y+16),bg_c,-1)
            tc2=col if active else (60,65,90)
            short=name[:3]
            cv2.putText(frame,short,(bx+2,brush_y+12),font,0.27,tc2,1,cv2.LINE_AA)
        cv2.line(frame,(px0,brush_y+18),(px0+pw,brush_y+18),(32,38,65),1)

    def _draw_bottom_hud(self, frame):
        H,W=frame.shape[:2]
        guides="☝Draw  ✊Erase  🖐Clear  🤟Brush  🤏PickColor  U=Undo  Pinch toolbar=Tools"
        cv2.putText(frame,guides,(72,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.27,(55,62,92),1,cv2.LINE_AA)

    def _clear_canvas(self):
        self._save_undo(); self._canvas[:]=0; self._clear_alpha=1.0
