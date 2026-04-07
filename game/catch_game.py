"""
game/catch_game.py  [v4 — FPS Optimized]
=========================================
Catch falling items with your fingertip.

FPS optimizations:
  - Single np.zeros batch layer per frame (no per-item copy/blend)
  - Particle cap MAX_BURSTS=60
  - Item trail capped at 5 points
  - Fly-texts: direct putText + alpha on tiny ROI only
  - Bomb/slowmo/double effects: single addWeighted per frame max
  - No frame.copy() in hot path
"""

import math, random, time
from typing import List, Optional, Tuple
import cv2, numpy as np

FONT    = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX

CATCH_R       = 48
MAGNET_R      = 170
SPAWN_INT     = 1.1
SLOWMO_DUR    = 2.5
DOUBLE_DUR    = 3.5
COMBO_WIN     = 2.0
MAX_ITEMS     = 20
MAX_BURSTS    = 60      # ← capped
MAX_TRAIL_LEN = 5       # ← capped
GAME_DURATION = 60.0


def _hsv_bgr(h, s=220, v=255):
    return cv2.cvtColor(np.uint8([[[int(h/2)%180,s,v]]]),
                        cv2.COLOR_HSV2BGR)[0][0].tolist()

def _heart_pts(cx, cy, size):
    pts = []
    for i in range(37):
        t = 2*math.pi*i/36
        x = 16*(math.sin(t)**3)
        y = -(13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t))
        s = size/16.0
        pts.append([int(cx+x*s),int(cy+y*s)])
    return np.array(pts,np.int32)

def _star_pts(cx, cy, size, angle=0.0):
    pts=[]
    for i in range(10):
        r = size if i%2==0 else size*0.42
        a = math.pi*i/5-math.pi/2+angle
        pts.append([int(cx+r*math.cos(a)),int(cy+r*math.sin(a))])
    return np.array(pts,np.int32)


class _Burst:
    __slots__=('x','y','vx','vy','life','color','size')
    def __init__(self,x,y,col):
        a=random.uniform(0,6.28); spd=random.uniform(3,10)
        self.x=float(x);self.y=float(y)
        self.vx=math.cos(a)*spd;self.vy=math.sin(a)*spd-random.uniform(1,3)
        self.life=1.0;self.color=col;self.size=random.randint(3,7)
    def step(self):
        self.x+=self.vx;self.y+=self.vy
        self.vy+=0.20;self.vx*=0.95;self.life-=0.032
    @property
    def alive(self): return self.life>0.04


class _Item:
    __slots__=('x','y','vx','vy','kind','size','color','hue','angle','spin','trail')
    def __init__(self,x,y,vx,vy,kind,size,hue,spin):
        self.x=float(x);self.y=float(y)
        self.vx=vx;self.vy=vy;self.kind=kind;self.size=size
        self.hue=float(hue);self.spin=spin;self.angle=0.0
        s=220 if kind!='bomb' else 160; v=255 if kind!='bomb' else 180
        self.color=_hsv_bgr(hue,s,v)
        self.trail=[]
    def step(self,dt,mul=1.0,slow=False):
        m=0.3 if slow else mul
        self.x+=self.vx*m;self.y+=self.vy*m;self.angle+=self.spin
        self.hue=(self.hue+1.2)%360
        s=220 if self.kind!='bomb' else 160;v=255 if self.kind!='bomb' else 180
        self.color=_hsv_bgr(self.hue,s,v)
        if self.x<self.size:      self.x=self.size;      self.vx*=-0.7
        if self.x>1280-self.size: self.x=1280-self.size; self.vx*=-0.7
        self.trail.append((int(self.x),int(self.y)))
        if len(self.trail)>MAX_TRAIL_LEN: self.trail.pop(0)


class _FlyText:
    __slots__=('x','y','text','color','alpha','vy','scale')
    def __init__(self,x,y,text,color,scale=0.75):
        self.x=int(x);self.y=float(y);self.text=text;self.color=color
        self.alpha=1.0;self.vy=2.0;self.scale=scale
    def step(self): self.y-=self.vy;self.alpha=max(0.0,self.alpha-0.022)
    @property
    def alive(self): return self.alpha>0.04


class CatchGame:
    def __init__(self,W,H):
        self.W=W;self.H=H
        self.score=0;self.high_score=0;self.level=1
        self._items:  List[_Item]    = []
        self._bursts: List[_Burst]   = []
        self._texts:  List[_FlyText] = []
        self._last_spawn  = time.time()
        self._combo=0;self._last_catch=0.0
        self._slowmo_end=0.0;self._double_end=0.0
        self._magnet=False;self._phase=0.0
        self._score_flash=0.0;self._bomb_flash=0.0
        self._prev_g=None;self._fist_used=False
        self._lvl_banner="";self._lvl_banner_t=0.0
        self._index_tip=None
        self._restart_pending=False
        self._game_over=False;self._game_over_t=0.0
        self._game_start_t=time.time();self._time_left=GAME_DURATION

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self,gesture_val,index_tip,dt):
        if self._restart_pending: return
        now=time.time()
        g=gesture_val or "OTHER"
        self._phase+=dt*2.5;self._index_tip=index_tip

        if self._game_over:
            elapsed_go=now-self._game_over_t
            if g=="FIST" and self._prev_g!="FIST": self._restart_pending=True
            elif elapsed_go>4.0: self._restart_pending=True
            self._prev_g=g;return

        self._time_left=max(0.0,GAME_DURATION-(now-self._game_start_t))
        if self._time_left<=0.0:
            self._trigger_game_over();return

        slow=now<self._slowmo_end;dbl=now<self._double_end

        # Gesture triggers
        if g=="OPEN_PALM" and self._prev_g!="OPEN_PALM":
            self._slowmo_end=now+SLOWMO_DUR
            self._fly(self.W//2-60,self.H//2,"SLOW-MO!",(100,255,150))
        if g=="ILY" and self._prev_g!="ILY":
            self._double_end=now+DOUBLE_DUR
            self._fly(self.W//2-50,self.H//2,"x2 POINTS!",(255,150,255))
        if g=="FIST" and self._prev_g!="FIST" and not self._fist_used:
            self._fist_used=True
            bonus=len(self._items)*8;self.score+=bonus
            self._fly(self.W//2-60,self.H//2-30,f"BOOM! +{bonus}",(80,80,255),1.0)
            for item in self._items:
                for _ in range(8):          # ← reduced from 12
                    self._bursts.append(_Burst(item.x,item.y,item.color))
            self._items=[];self._bomb_flash=now
        self._fist_used=(g=="FIST");self._magnet=(g=="PINCH");self._prev_g=g

        # Magnet
        if self._magnet and index_tip:
            for item in self._items:
                if item.kind=="bomb": continue
                dx=index_tip[0]-item.x;dy=index_tip[1]-item.y
                d=math.hypot(dx,dy)
                if 0<d<MAGNET_R:
                    f=4.5*(1-d/MAGNET_R)
                    item.vx+=dx/d*f;item.vy+=dy/d*f*0.5

        # Spawn
        lvl_m=1.0+(self.level-1)*0.13
        interval=max(0.38,SPAWN_INT/lvl_m)
        if now-self._last_spawn>interval and len(self._items)<MAX_ITEMS:
            self._last_spawn=now;self._spawn(lvl_m)

        # Move
        for item in self._items: item.step(dt,lvl_m,slow)

        # Catch
        if index_tip:
            ix,iy=index_tip;survived=[]
            for item in self._items:
                if math.hypot(ix-item.x,iy-item.y)<CATCH_R:
                    self._on_catch(item,ix,iy,dbl,now)
                else:
                    survived.append(item)
            self._items=survived

        self._items=[i for i in self._items if i.y<self.H+60]
        if self._combo>0 and now-self._last_catch>COMBO_WIN: self._combo=0

        nl=1+self.score//220
        if nl>self.level:
            self.level=nl;self._lvl_banner=f"LEVEL {self.level}!";self._lvl_banner_t=now
        self.high_score=max(self.high_score,self.score)

        # Particles — capped
        self._bursts=self._bursts[-MAX_BURSTS:]
        for b in self._bursts: b.step()
        self._bursts=[b for b in self._bursts if b.alive]
        for t in self._texts: t.step()
        self._texts=[t for t in self._texts if t.alive]

    def _trigger_game_over(self):
        self._game_over=True;self._game_over_t=time.time()
        self.high_score=max(self.high_score,self.score)

    def _on_catch(self,item,ix,iy,dbl,now):
        self._last_catch=now;col=item.color
        n=12 if item.kind!='bomb' else 6     # ← reduced
        for _ in range(n): self._bursts.append(_Burst(item.x,item.y,col))
        if item.kind=="bomb":
            self.score=max(0,self.score-25);self._combo=0
            self._bomb_flash=now;self._fly(ix-30,iy,"-25 💥",(60,60,255));return
        self._combo+=1
        pts=12 if item.kind=="heart" else 18
        cm={1:1,2:2,3:3,4:5}.get(min(self._combo,4),5)
        if dbl: pts*=2
        total=pts*cm;self.score+=total
        suf="" if cm==1 else f" x{cm}"
        self._fly(ix-20,iy,f"+{total}{suf}",col);self._score_flash=now

    def _spawn(self,lvl_m):
        bc=min(0.20,0.04+self.level*0.016)
        kind="bomb" if random.random()<bc else random.choice(["heart","heart","star"])
        x=random.randint(60,self.W-60);hue=random.uniform(0,360)
        size=random.randint(18,26)
        vy=random.uniform(2.2,4.2)*(1+(self.level-1)*0.09)
        vx=random.uniform(-1.2,1.2);spin=random.uniform(-0.06,0.06)
        self._items.append(_Item(x,-30,vx,vy,kind,size,hue,spin))
        if self.level>=3 and random.random()<0.45:
            x2=random.randint(60,self.W-60);hue2=(hue+130)%360;k2=random.choice(["heart","star"])
            self._items.append(_Item(x2,-30,-vx,vy*0.88,k2,size-4,hue2,-spin))

    def _fly(self,x,y,text,color,scale=0.75):
        self._texts.append(_FlyText(x,y,text,color,scale))

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self,frame):
        H,W=frame.shape[:2]; now=time.time(); ph=self._phase

        # Effects (one addWeighted each, only when active)
        if now-self._bomb_flash<0.40:
            a=0.26*(1-(now-self._bomb_flash)/0.4)
            ov=np.full((H,W,3),(40,40,210),np.uint8)
            cv2.addWeighted(ov,a,frame,1-a,0,frame)

        if now<self._slowmo_end:
            rem=self._slowmo_end-now
            ov=np.full((H,W,3),(20,70,20),np.uint8)
            cv2.addWeighted(ov,0.14,frame,0.86,0,frame)
            cv2.putText(frame,f"SLOW-MO  {rem:.1f}s",(W//2-70,82),FONT_SM,0.55,(80,255,120),2,cv2.LINE_AA)

        if now<self._double_end:
            rem=self._double_end-now
            ov=np.full((H,W,3),(70,18,70),np.uint8)
            cv2.addWeighted(ov,0.09,frame,0.91,0,frame)
            cv2.putText(frame,f"x2 POINTS  {rem:.1f}s",(W//2-72,100),FONT_SM,0.50,(255,140,255),2,cv2.LINE_AA)

        # Magnet ring (direct draw, no copy)
        if self._magnet and self._index_tip:
            mx,my=self._index_tip;col=_hsv_bgr((ph*40)%360,200,255)
            cv2.circle(frame,(mx,my),MAGNET_R,col,1,cv2.LINE_AA)
            cv2.circle(frame,(mx,my),MAGNET_R//2,[c//2 for c in col],1,cv2.LINE_AA)

        # ── Batch layer: items + particles (single np.zeros, no copy) ────
        layer=np.zeros((H,W,3),np.uint8)

        # Trails
        for item in self._items:
            for k,pt in enumerate(item.trail):
                a=k/max(len(item.trail),1)
                c=[int(cc*a*0.45) for cc in item.color]
                cv2.circle(layer,pt,max(1,int(item.size*a*0.35)),c,-1,cv2.LINE_AA)

        # Items
        for item in self._items:
            cx,cy=int(item.x),int(item.y);s=item.size;col=list(item.color)
            if item.kind=="heart":
                cv2.fillPoly(layer,[_heart_pts(cx,cy,s+6)],[c//3 for c in col])
                cv2.fillPoly(layer,[_heart_pts(cx,cy,s)],col)
                cv2.circle(layer,(cx-s//4,cy-s//3),max(1,s//5),(255,255,255),-1,cv2.LINE_AA)
            elif item.kind=="star":
                cv2.fillPoly(layer,[_star_pts(cx,cy,s+5,item.angle)],[c//3 for c in col])
                cv2.fillPoly(layer,[_star_pts(cx,cy,s,item.angle)],col)
                cv2.circle(layer,(cx,cy),s//4,(255,255,255),-1,cv2.LINE_AA)
            elif item.kind=="bomb":
                cv2.circle(layer,(cx,cy),s+3,[int(c*0.3) for c in col],-1,cv2.LINE_AA)
                cv2.circle(layer,(cx,cy),s,(40,40,40),-1,cv2.LINE_AA)
                cv2.circle(layer,(cx,cy),s,[max(0,c-40) for c in col],3,cv2.LINE_AA)
                fx_=cx+s//2;fy_=cy-s
                cv2.line(layer,(cx,cy-s),(fx_,fy_-s//2),(80,200,255),2,cv2.LINE_AA)
                cv2.circle(layer,(fx_,fy_-s//2),3,(50,150,255),-1,cv2.LINE_AA)
                cv2.putText(layer,"!",(cx-4,cy+5),FONT_SM,0.38,(255,255,255),1,cv2.LINE_AA)

        # Particles
        for b in self._bursts:
            cx,cy=int(b.x),int(b.y)
            if 0<=cx<W and 0<=cy<H:
                a=max(0,b.life);c=[int(cc*a) for cc in b.color]
                sz=max(1,int(b.size*a))
                cv2.circle(layer,(cx,cy),sz+1,[cc//3 for cc in c],-1)
                cv2.circle(layer,(cx,cy),sz,c,-1)

        cv2.add(frame,layer,frame)  # no copy!

        # Fly texts (tiny ROI alpha blend — cheap)
        for ft in self._texts:
            txt=ft.text;x=ft.x;y=int(ft.y)
            (tw,th),_=cv2.getTextSize(txt,FONT,ft.scale,2)
            x1=max(0,x-4);y1=max(0,y-th-4);x2=min(W,x+tw+4);y2=min(H,y+8)
            if x2>x1 and y2>y1:
                roi=frame[y1:y2,x1:x2]
                ov=roi.copy()
                cv2.putText(ov,txt,(x-x1,y-y1+th-2),FONT,ft.scale,(0,0,0),4,cv2.LINE_AA)
                cv2.putText(ov,txt,(x-x1,y-y1+th-2),FONT,ft.scale,ft.color,2,cv2.LINE_AA)
                cv2.addWeighted(ov,ft.alpha,roi,1-ft.alpha,0,roi)
                frame[y1:y2,x1:x2]=roi

        self._draw_hud(frame,now)
        if self._game_over: self._draw_game_over(frame,now)

    def _draw_hud(self,frame,now):
        W,H=self.W,self.H;ph=self._phase
        pw,ph_=320,68;px_=(W-pw)//2;pt=58
        roi=frame[pt:pt+ph_,px_:px_+pw];bg=np.full_like(roi,(8,8,18))
        cv2.addWeighted(bg,0.84,roi,0.16,0,roi);frame[pt:pt+ph_,px_:px_+pw]=roi

        if now<self._double_end:   ac=_hsv_bgr((ph*60)%360,230,255)
        elif now<self._slowmo_end: ac=(60,240,90)
        else:                      ac=_hsv_bgr((ph*15+200)%360,180,200)

        cv2.line(frame,(px_+1,pt+1),(px_+pw-1,pt+1),ac,3)
        cv2.rectangle(frame,(px_,pt),(px_+pw,pt+ph_),[c//3 for c in ac],1)
        sz=8
        for cx2,cy2 in [(px_,pt),(px_+pw,pt),(px_,pt+ph_),(px_+pw,pt+ph_)]:
            sx=1 if cx2==px_ else -1;sy=1 if cy2==pt else -1
            cv2.line(frame,(cx2,cy2),(cx2+sx*sz,cy2),ac,1)
            cv2.line(frame,(cx2,cy2),(cx2,cy2+sy*sz),ac,1)

        cv2.putText(frame,"CATCH GAME",(px_+8,pt+13),FONT_SM,0.30,[c//2 for c in ac],1,cv2.LINE_AA)
        sc_col=(255,255,255) if now-self._score_flash>0.25 else _hsv_bgr(50,255,255)
        cv2.putText(frame,f"{self.score:06d}",(px_+14,pt+42),FONT,0.95,(0,0,0),6,cv2.LINE_AA)
        cv2.putText(frame,f"{self.score:06d}",(px_+12,pt+40),FONT,0.95,sc_col,2,cv2.LINE_AA)
        lv_col=_hsv_bgr((self.level*30)%360,200,255)
        cv2.rectangle(frame,(px_+8,pt+46),(px_+58,pt+62),[c//3 for c in lv_col],-1)
        cv2.rectangle(frame,(px_+8,pt+46),(px_+58,pt+62),lv_col,1)
        cv2.putText(frame,f"LV {self.level}",(px_+12,pt+59),FONT_SM,0.34,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(frame,f"BEST {self.high_score:06d}",(px_+pw-155,pt+59),FONT_SM,0.36,(120,120,170),1,cv2.LINE_AA)

        # Timer
        if not self._game_over:
            tl=self._time_left;urgent=tl<=10
            t_col=(80,255,80) if tl>20 else (80,200,255) if tl>10 else (60,80,255)
            scale=0.85+(0.15*abs(math.sin(now*8)) if urgent else 0)
            t_str=f"{int(tl):02d}s"
            (tw,_),_=cv2.getTextSize(t_str,FONT,scale,2)
            tx=W-tw-22
            cv2.putText(frame,t_str,(tx+2,52),FONT,scale,(0,0,0),5,cv2.LINE_AA)
            cv2.putText(frame,t_str,(tx,50),FONT,scale,t_col,2,cv2.LINE_AA)
            cv2.putText(frame,"TIME",(tx,68),FONT_SM,0.30,(80,80,100),1,cv2.LINE_AA)
            # arc
            cx_t=W-36;cy_t=36;ratio=tl/GAME_DURATION;ea=int(360*ratio)
            cv2.ellipse(frame,(cx_t,cy_t),(22,22),-90,0,360,(30,30,50),2,cv2.LINE_AA)
            cv2.ellipse(frame,(cx_t,cy_t),(22,22),-90,0,ea,t_col,2,cv2.LINE_AA)

        # Combo
        if self._combo>=2:
            labels={2:"DOUBLE!",3:"TRIPLE!",4:"QUAD!!"}
            ct=labels.get(self._combo,f"{self._combo}x COMBO!")
            col=_hsv_bgr((now*90)%360,200,255);sc2=0.72+0.10*abs(math.sin(now*7))
            (tw,_),_=cv2.getTextSize(ct,FONT,sc2,2)
            cv2.putText(frame,ct,((W-tw)//2+2,145),FONT,sc2,(0,0,0),5,cv2.LINE_AA)
            cv2.putText(frame,ct,((W-tw)//2,143),FONT,sc2,col,2,cv2.LINE_AA)

        # Level banner
        if self._lvl_banner and now-self._lvl_banner_t<2.5:
            age=now-self._lvl_banner_t;a=max(0.0,1.0-age/2.5)
            col=_hsv_bgr((age*80)%360,220,255);sc3=1.0+0.15*math.sin(age*8)
            (tw,_),_=cv2.getTextSize(self._lvl_banner,FONT,sc3,3)
            cv2.putText(frame,self._lvl_banner,((W-tw)//2+2,H//3+2),FONT,sc3,(0,0,0),6,cv2.LINE_AA)
            c=tuple(int(cc*a) for cc in col)
            cv2.putText(frame,self._lvl_banner,((W-tw)//2,H//3),FONT,sc3,c,3,cv2.LINE_AA)

        hints="☝Catch  🤌Magnet  ✊Bomb  🖐SlowMo  🤟x2pts"
        (hw,_),_=cv2.getTextSize(hints,FONT_SM,0.32,1)
        cv2.putText(frame,hints,((W-hw)//2,H-14),FONT_SM,0.32,(65,65,65),1,cv2.LINE_AA)

    def _draw_game_over(self,frame,now):
        H,W=frame.shape[:2];age=now-self._game_over_t;alpha=min(1.0,age/0.6)
        ov=np.full((H,W,3),(8,0,18),np.uint8)
        cv2.addWeighted(ov,alpha*0.70,frame,1-alpha*0.70,0,frame)
        bar_h=int(H*0.11)
        cv2.rectangle(frame,(0,0),(W,bar_h),(0,0,0),-1)
        cv2.rectangle(frame,(0,H-bar_h),(W,H),(0,0,0),-1)
        col=_hsv_bgr((now*55)%360,220,255)
        (tw,_),_=cv2.getTextSize("GAME OVER",FONT,1.4,3)
        cv2.putText(frame,"GAME OVER",((W-tw)//2+4,H//2-28),FONT,1.4,(0,0,0),8,cv2.LINE_AA)
        cv2.putText(frame,"GAME OVER",((W-tw)//2,H//2-32),FONT,1.4,col,2,cv2.LINE_AA)
        sc_txt=f"Score: {self.score}     Best: {self.high_score}"
        (sw,_),_=cv2.getTextSize(sc_txt,FONT_SM,0.70,2)
        cv2.putText(frame,sc_txt,((W-sw)//2,H//2+16),FONT_SM,0.70,(230,230,230),2,cv2.LINE_AA)
        time_left_go=max(0.0,4.0-age)
        hint=f"FIST to restart now  —  auto in {time_left_go:.1f}s"
        (rw,_),_=cv2.getTextSize(hint,FONT_SM,0.44,1)
        cv2.putText(frame,hint,((W-rw)//2,H//2+52),FONT_SM,0.44,(120,180,255),1,cv2.LINE_AA)
        bar_w=int(W*0.42);bar_x=(W-bar_w)//2;bar_y=H//2+68
        filled=int(bar_w*(1.0-time_left_go/4.0))
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_w,bar_y+10),(25,25,45),-1)
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+filled,bar_y+10),col,-1)
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_w,bar_y+10),(60,60,100),1)
