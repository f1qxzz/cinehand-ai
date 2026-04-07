"""
game/snake_game.py  [v4 — WASD + FPS Optimized]
=================================================
Neon Snake — WASD keyboard + hand gesture control.

Keyboard:
  W A S D       → Steer (primary)
  Arrow Keys    → Steer (alternative)
  Space         → Boost (2.5 s)
  P             → Pause / Resume

Gesture:
  ☝  POINT      → Steer via finger direction
  🤌 PINCH       → Boost
  🖐 OPEN_PALM   → Pause

FPS optimizations:
  - Single np.zeros layer per frame (NO frame.copy)
  - cv2.add instead of addWeighted wherever possible
  - Capped trail / popup counts
  - Lightweight particle step (no sqrt in hot path)
  - Batch ROI blend for score panel (one addWeighted)
  - Skip expensive grid draw every other frame
"""

import math, random, time
from typing import List, Tuple
import cv2, numpy as np

FONT    = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX

CELL       = 28
SPEED_N    = 8
SPEED_B    = 18
BOOST_DUR  = 2.5
MAX_TRAIL  = 40       # ← capped for FPS
MAX_POPUPS = 8        # ← capped

# ── WASD shared state (written by main.py key handler) ───────────────────────
_PENDING_DIR: List[Tuple[int,int]] = []

def snake_key_event(key: int) -> bool:
    """
    Call this from the main loop with the raw cv2 key value.
    Returns True if the key was consumed by the snake controller.
    WASD + arrow keys.  Space = boost toggle.  P = pause toggle.
    """
    global _PENDING_DIR
    _DIR_MAP = {
        ord('w'): (0,-1),  ord('W'): (0,-1),
        ord('s'): (0, 1),  ord('S'): (0, 1),
        ord('a'): (-1,0),  ord('A'): (-1,0),
        ord('d'): (1, 0),  ord('D'): (1, 0),
        # Arrow keys (Linux raw values after & 0xFF)
        82: (0,-1),  72: (0,-1),   # Up
        84: (0, 1),  80: (0, 1),   # Down
        81: (-1,0),  75: (-1,0),   # Left
        83: (1, 0),  77: (1, 0),   # Right
    }
    if key in _DIR_MAP:
        if len(_PENDING_DIR) < 4:
            _PENDING_DIR.append(_DIR_MAP[key])
        return True
    return False


def _hsv_bgr(h, s=220, v=255):
    return cv2.cvtColor(
        np.uint8([[[int(h/2) % 180, s, v]]]),
        cv2.COLOR_HSV2BGR)[0][0].tolist()

def _heart_pts(cx, cy, size):
    pts = []
    for i in range(37):
        t = 2*math.pi*i/36
        x = 16*(math.sin(t)**3)
        y = -(13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t))
        s = size/16.0
        pts.append([int(cx+x*s), int(cy+y*s)])
    return np.array(pts, np.int32)


class _Trail:
    """Ultra-lightweight trail entry (no sqrt, no velocity)."""
    __slots__ = ('x','y','life','color')
    def __init__(self, x, y, col):
        self.x, self.y = x, y
        self.life  = 1.0
        self.color = col
    def step(self):
        self.life = max(0.0, self.life - 0.06)


class _Popup:
    __slots__ = ('x','y','text','col','life')
    def __init__(self, x, y, text, col):
        self.x, self.y = float(x), float(y)
        self.text = text; self.col = col; self.life = 1.0
    def step(self, dt):
        self.life = max(0.0, self.life - dt*1.6)
        self.y   -= dt*52


# =============================================================================
class SnakeGame:

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fx1, self.fy1 = 10, 62
        self.fx2, self.fy2 = W-10, H-50
        self.gw = (self.fx2-self.fx1)//CELL
        self.gh = (self.fy2-self.fy1)//CELL
        self._frame = 0
        self._reset()
        self.high_score = 0

    def _reset(self):
        cx, cy = self.gw//2, self.gh//2
        self._body: List[Tuple[int,int]] = [(cx,cy),(cx-1,cy),(cx-2,cy)]
        self._dir       = (1, 0)
        self._next_dir  = (1, 0)
        self._score     = 0
        self._chain     = 0
        self._level     = 1
        self._move_t    = 0.0
        self._phase     = 0.0
        self._boost_end = 0.0
        self._paused    = False
        self._dead      = False
        self._dead_t    = 0.0
        self._foods: List[dict]    = []
        self._trail: List[_Trail]  = []
        self._popups: List[_Popup] = []
        self._flash_t   = 0.0
        self._prev_g    = None
        self._grow_q    = 0
        self._hue       = 120.0
        self._lvlup_t   = 0.0
        self._danger    = 0.0
        self._spawn_food(); self._spawn_food()
        self._restart_pending = False
        _PENDING_DIR.clear()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _cell_px(self, gx, gy):
        return self.fx1+gx*CELL+CELL//2, self.fy1+gy*CELL+CELL//2

    def _spawn_food(self):
        body_set = set(self._body)
        food_set = {(f['gx'],f['gy']) for f in self._foods}
        for _ in range(300):
            gx = random.randint(1, self.gw-2)
            gy = random.randint(1, self.gh-2)
            if (gx,gy) not in body_set and (gx,gy) not in food_set:
                self._foods.append({
                    'gx':gx,'gy':gy,
                    'special': random.random()<0.18,
                    'hue': random.uniform(0,360),
                    'phase': random.uniform(0,6.28),
                })
                return

    def _try_enqueue(self, nd):
        """Enqueue direction only if not a 180° reversal."""
        if nd[0] != -self._dir[0] or nd[1] != -self._dir[1]:
            self._next_dir = nd

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self, gesture_val, landmarks, W, H, dt):
        self._phase += dt*3.0
        self._frame += 1

        # ── Drain WASD queue FIRST ────────────────────────────────────────
        while _PENDING_DIR:
            self._try_enqueue(_PENDING_DIR.pop(0))

        if self._dead:
            if time.time()-self._dead_t > 2.8:
                self._restart_pending = True
            return

        g = gesture_val or "OTHER"

        # Pause / boost via gesture
        if g == "OPEN_PALM" and self._prev_g != "OPEN_PALM":
            self._paused = not self._paused
        if g == "PINCH" and self._prev_g != "PINCH":
            self._boost_end = time.time()+BOOST_DUR
        self._prev_g = g

        # Popups always animate
        for p in self._popups: p.step(dt)
        self._popups = [p for p in self._popups if p.life>0.05][:MAX_POPUPS]

        if self._paused:
            return

        # Gesture steer (only if no keyboard used recently)
        if landmarks is not None:
            lm = landmarks
            dx, dy = lm[8].x-lm[0].x, lm[8].y-lm[0].y
            if abs(dx)+abs(dy) > 0.06:
                nd = (1,0) if abs(dx)>abs(dy) and dx>0 else \
                     (-1,0) if abs(dx)>abs(dy) else \
                     (0,1)  if dy>0 else (0,-1)
                self._try_enqueue(nd)

        # Move ticks
        speed = SPEED_B if time.time()<self._boost_end else SPEED_N
        speed *= (1+(self._level-1)*0.07)
        tick  = 1.0/speed
        self._move_t += dt
        moved = False
        while self._move_t >= tick:
            self._move_t -= tick
            self._tick()
            moved = True

        # Trail (lightweight — no sqrt)
        if moved and self._body:
            hx, hy = self._cell_px(*self._body[0])
            col = _hsv_bgr(self._hue, 220, 255)
            self._trail.append(_Trail(hx, hy, col))
            if len(self._trail) > MAX_TRAIL:
                self._trail.pop(0)
        for p in self._trail: p.step()
        self._trail = [p for p in self._trail if p.life>0.04]

        # Food phase
        for f in self._foods:
            f['phase'] += dt*3.0

        # Danger proximity
        if self._body:
            hx2, hy2 = self._body[0]
            near = (hx2<=1 or hx2>=self.gw-2 or hy2<=1 or hy2>=self.gh-2)
            self._danger = min(1.0, self._danger+dt*5) if near else max(0.0, self._danger-dt*4)

    def _tick(self):
        self._dir = self._next_dir
        hx, hy   = self._body[0]
        nhx = hx+self._dir[0]; nhy = hy+self._dir[1]

        if nhx<0 or nhx>=self.gw or nhy<0 or nhy>=self.gh:
            self._die(); return
        if (nhx,nhy) in set(self._body[:-1]):
            self._die(); return

        self._body.insert(0,(nhx,nhy))

        eaten = next((f for f in self._foods if f['gx']==nhx and f['gy']==nhy), None)
        if eaten:
            self._foods.remove(eaten)
            self._chain += 1
            chain_bonus  = (self._chain-1)*5
            pts = (30 if eaten['special'] else 10)+chain_bonus
            pts *= self._level
            self._score += pts
            self._grow_q += 3 if eaten['special'] else 1
            self._flash_t = time.time()
            self._hue     = (self._hue+40)%360
            fx, fy = self._cell_px(eaten['gx'], eaten['gy'])
            col = _hsv_bgr(50,255,255) if eaten['special'] else _hsv_bgr(120,200,255)
            self._popups.append(_Popup(fx, fy, f"{'★' if eaten['special'] else '+'}{pts}", col))
            old_lv = self._level
            self._spawn_food()
            if len(self._foods)<2: self._spawn_food()
            self._level = 1+self._score//200
            if self._level>old_lv: self._lvlup_t = time.time()
        else:
            if self._grow_q>0: self._grow_q -= 1
            else:
                self._body.pop()
                self._chain = 0

        self.high_score = max(self.high_score, self._score)

    def _die(self):
        self._dead = True; self._dead_t = time.time(); self._chain = 0
        self.high_score = max(self.high_score, self._score)

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self, frame):
        H, W  = frame.shape[:2]
        now   = time.time()
        ph    = self._phase
        self._frame += 1

        # ── Playfield tint (single ROI blend, NO full frame.copy) ────────
        roi = frame[self.fy1:self.fy2, self.fx1:self.fx2]
        bg  = np.full_like(roi, (8,10,20))
        cv2.addWeighted(bg, 0.26, roi, 0.74, 0, roi)
        frame[self.fy1:self.fy2, self.fx1:self.fx2] = roi

        # Grid lines — skip every other frame for FPS
        if self._frame % 2 == 0:
            for gx in range(0, self.gw+1):
                x = self.fx1+gx*CELL
                cv2.line(frame,(x,self.fy1),(x,self.fy2),(20,22,36),1)
            for gy in range(0, self.gh+1):
                y = self.fy1+gy*CELL
                cv2.line(frame,(self.fx1,y),(self.fx2,y),(20,22,36),1)

        # Danger border
        if self._danger>0.05:
            dc = (int(30*self._danger),int(30*self._danger),int(220*self._danger))
            cv2.rectangle(frame,(self.fx1,self.fy1),(self.fx2,self.fy2),dc,max(1,int(self._danger*4)))
        else:
            cv2.rectangle(frame,(self.fx1,self.fy1),(self.fx2,self.fy2),(50,55,100),2)

        # ── Single draw layer (np.zeros → cv2.add, no copy) ──────────────
        layer = np.zeros((H,W,3),np.uint8)

        # Trail
        for p in self._trail:
            a  = p.life
            sz = max(1, int(CELL//2*a*0.5))
            c  = [int(cc*a*0.35) for cc in p.color]
            cv2.circle(layer,(p.x,p.y),sz+2,c,-1)

        # Body
        n = len(self._body)
        for i,(gx,gy) in enumerate(reversed(self._body)):
            px_,py_ = self._cell_px(gx,gy)
            t   = i/max(n,1)
            col = _hsv_bgr((self._hue+t*80)%360, int(180+40*t), int(200+55*(1-t)))
            s   = CELL//2-1
            glow = [min(255,c//3) for c in col]
            cv2.circle(layer,(px_,py_),s+5,glow,-1)
            cv2.rectangle(layer,(px_-s,py_-s),(px_+s,py_+s),col,-1)
            cv2.circle(layer,(px_,py_),s,col,-1)

        # Head + eyes
        if self._body:
            hx,hy    = self._cell_px(*self._body[0])
            head_col = _hsv_bgr(self._hue,255,255)
            s        = CELL//2
            cv2.circle(layer,(hx,hy),s+8,[c//5 for c in head_col],-1)
            cv2.circle(layer,(hx,hy),s+4,[c//3 for c in head_col],-1)
            cv2.circle(layer,(hx,hy),s,head_col,-1)
            dx,dy = self._dir
            for sign in (+1,-1):
                ex = hx+dx*5-dy*5*sign; ey = hy+dy*5+dx*5*sign
                cv2.circle(layer,(int(ex),int(ey)),3,(255,255,255),-1)
                cv2.circle(layer,(int(ex+dx),int(ey+dy)),1,(0,0,0),-1)

        # Food
        for f in self._foods:
            fx,fy  = self._cell_px(f['gx'],f['gy'])
            fph    = f['phase']; sp = f['special']
            pulse  = CELL//2-2+int(3*math.sin(fph))
            col    = _hsv_bgr((f['hue']+fph*20)%360, 200 if not sp else 255, 255)
            cv2.circle(layer,(fx,fy),pulse+9,[c//6 for c in col],-1)
            cv2.circle(layer,(fx,fy),pulse+5,[c//3 for c in col],-1)
            if sp:
                pts = []
                for i in range(10):
                    r = (pulse+4) if i%2==0 else (pulse//2+1)
                    a = math.pi*i/5-math.pi/2+fph*0.5
                    pts.append([int(fx+r*math.cos(a)),int(fy+r*math.sin(a))])
                cv2.fillPoly(layer,[np.array(pts,np.int32)],col)
            else:
                cv2.fillPoly(layer,[_heart_pts(fx,fy,pulse)],col)
                cv2.circle(layer,(fx-pulse//4,fy-pulse//3),max(1,pulse//4),(255,255,255),-1)

        cv2.add(frame, layer, frame)  # no copy!

        # ── Score flash (direct blend, tiny ROI) ─────────────────────────
        if now-self._flash_t < 0.20:
            a = 0.28*(1-(now-self._flash_t)/0.20)
            ov = np.full((H,W,3),(40,40,0),np.uint8)
            cv2.addWeighted(ov,a,frame,1-a,0,frame)

        # ── Level-up (cheap text only) ────────────────────────────────────
        if now-self._lvlup_t < 0.80:
            t   = (now-self._lvlup_t)/0.80
            col = _hsv_bgr((ph*80)%360,255,255)
            txt = f"LEVEL {self._level}!"
            (tw,_),_ = cv2.getTextSize(txt,FONT,0.9,3)
            sc  = 0.5+0.5*(1-t)*math.sin(t*math.pi)
            ac  = max(0.0,1-t*1.6)
            c   = tuple(int(cc*ac) for cc in col)
            cv2.putText(frame,txt,(W//2-tw//2+2,H//2-60),FONT,sc,(0,0,0),6,cv2.LINE_AA)
            cv2.putText(frame,txt,(W//2-tw//2,H//2-62),FONT,sc,c,3,cv2.LINE_AA)

        # ── Floating score popups ─────────────────────────────────────────
        for p in self._popups:
            c = tuple(int(cc*p.life) for cc in p.col)
            (tw,_),_ = cv2.getTextSize(p.text,FONT,0.45,2)
            x = int(p.x)-tw//2; y = int(p.y)
            cv2.putText(frame,p.text,(x+1,y+1),FONT,0.45,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(frame,p.text,(x,y),FONT,0.45,c,2,cv2.LINE_AA)

        # ── Boost bar ─────────────────────────────────────────────────────
        if now<self._boost_end:
            rem   = self._boost_end-now
            col   = _hsv_bgr((ph*80)%360,220,255)
            bar_w = int((rem/BOOST_DUR)*100)
            cv2.rectangle(frame,(self.fx1+8,self.fy1+30),(self.fx1+108,self.fy1+38),(18,20,36),-1)
            cv2.rectangle(frame,(self.fx1+8,self.fy1+30),(self.fx1+8+bar_w,self.fy1+38),col,-1)
            cv2.putText(frame,f"BOOST {rem:.1f}s",(self.fx1+8,self.fy1+26),FONT_SM,0.42,col,1,cv2.LINE_AA)

        # ── Pause overlay ─────────────────────────────────────────────────
        if self._paused:
            ov = np.full((H,W,3),(8,8,18),np.uint8)
            cv2.addWeighted(ov,0.38,frame,0.62,0,frame)
            txt = "PAUSED"
            (tw,_),_ = cv2.getTextSize(txt,FONT,1.3,3)
            cv2.putText(frame,txt,(W//2-tw//2+2,H//2+2),FONT,1.3,(0,0,0),6,cv2.LINE_AA)
            cv2.putText(frame,txt,(W//2-tw//2,H//2),FONT,1.3,(160,180,255),3,cv2.LINE_AA)
            sub = "P key / Open Palm to resume"
            (sw,_),_ = cv2.getTextSize(sub,FONT_SM,0.46,1)
            cv2.putText(frame,sub,(W//2-sw//2,H//2+44),FONT_SM,0.46,(80,90,140),1,cv2.LINE_AA)

        # ── Game-over ─────────────────────────────────────────────────────
        if self._dead:
            age = now-self._dead_t; a = min(1.0,age/0.6)
            ov  = np.full((H,W,3),(22,0,0),np.uint8)
            cv2.addWeighted(ov,a*0.45,frame,1-a*0.45,0,frame)
            col = _hsv_bgr((age*60)%360,200,255)
            txt = "GAME OVER"
            (tw,_),_ = cv2.getTextSize(txt,FONT,1.4,4)
            cv2.putText(frame,txt,(W//2-tw//2+2,H//2-18),FONT,1.4,(0,0,0),8,cv2.LINE_AA)
            cv2.putText(frame,txt,(W//2-tw//2,H//2-20),FONT,1.4,col,3,cv2.LINE_AA)
            sc_txt = f"Score: {self._score}   Best: {self.high_score}"
            (sw,_),_ = cv2.getTextSize(sc_txt,FONT_SM,0.65,2)
            cv2.putText(frame,sc_txt,(W//2-sw//2,H//2+24),FONT_SM,0.65,(210,215,230),1,cv2.LINE_AA)
            cv2.putText(frame,"Restarting...",(W//2-60,H//2+60),FONT_SM,0.44,(80,85,110),1,cv2.LINE_AA)

        self._draw_hud(frame, now)

    def _draw_hud(self, frame, now):
        W, H   = self.W, self.H
        ph     = self._phase
        lv_col = _hsv_bgr((self._level*35+ph*20)%360,220,255)

        # Score panel (single ROI blend)
        pw,pht = 340,72; px_ = (W-pw)//2; pt = 60
        roi = frame[pt:pt+pht, px_:px_+pw]
        bg  = np.full_like(roi,(4,6,14))
        cv2.addWeighted(bg,0.88,roi,0.12,0,roi)
        frame[pt:pt+pht, px_:px_+pw] = roi

        cv2.line(frame,(px_+1,pt+1),(px_+pw-1,pt+1),lv_col,3)
        cv2.rectangle(frame,(px_,pt),(px_+pw,pt+pht),[c//3 for c in lv_col],1)
        for cx2,cy2 in [(px_,pt),(px_+pw,pt),(px_,pt+pht),(px_+pw,pt+pht)]:
            sx=1 if cx2==px_ else -1; sy=1 if cy2==pt else -1
            cv2.line(frame,(cx2,cy2),(cx2+sx*10,cy2),lv_col,1)
            cv2.line(frame,(cx2,cy2),(cx2,cy2+sy*10),lv_col,1)

        cv2.putText(frame,"NEON SNAKE",(px_+8,pt+14),FONT_SM,0.32,[c//2 for c in lv_col],1,cv2.LINE_AA)
        sc_col = (255,255,255) if now-self._flash_t>0.20 else _hsv_bgr(50,255,255)
        cv2.putText(frame,f"{self._score:07d}",(px_+14,pt+45),FONT,1.0,(0,0,0),6,cv2.LINE_AA)
        cv2.putText(frame,f"{self._score:07d}",(px_+12,pt+43),FONT,1.0,sc_col,2,cv2.LINE_AA)

        cv2.rectangle(frame,(px_+8,pt+49),(px_+62,pt+66),[c//4 for c in lv_col],-1)
        cv2.rectangle(frame,(px_+8,pt+49),(px_+62,pt+66),lv_col,1)
        cv2.putText(frame,f"LV {self._level}",(px_+12,pt+62),FONT_SM,0.36,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(frame,f"BEST {self.high_score:07d}",(px_+pw-165,pt+62),FONT_SM,0.36,(90,100,155),1,cv2.LINE_AA)

        if self._chain>1:
            c_col = _hsv_bgr((self._chain*28)%360,230,255)
            cv2.putText(frame,f"x{self._chain} CHAIN",(px_+pw+10,pt+32),FONT_SM,0.44,c_col,1,cv2.LINE_AA)

        len_col = _hsv_bgr((ph*30)%360,200,220)
        cv2.putText(frame,f"LEN:{len(self._body)}",(self.fx1+8,self.fy2-8),FONT_SM,0.40,len_col,1,cv2.LINE_AA)

        # Control hints (static text, drawn directly)
        cv2.putText(frame,
            "WASD / Arrows = Steer   Space = Boost   P = Pause   |   ☝ = Steer   🤌 = Boost   🖐 = Pause",
            (W//2 - 380, H-12), FONT_SM, 0.26, (50,55,75), 1, cv2.LINE_AA)
