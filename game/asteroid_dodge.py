"""
game/asteroid_dodge.py  [v2 — FPS Optimized]
=============================================
Asteroid Dodge — hand controlled survival game.

FPS optimizations:
  - Single np.zeros batch layer (stars + asteroids + particles)
  - cv2.add instead of addWeighted on main layer
  - Stars pre-drawn to a cached background (refreshed every 8 frames)
  - Asteroid shape_pts stored as np.array (no recompute per frame)
  - Particle cap MAX_PARTICLES=80
  - No per-particle frame.copy()
  - Shield bubble: lightweight circle, no copy()
"""

import math, random, time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2, numpy as np

SHIP_W              = 24
SHIP_H              = 36
ASTEROID_BASE_SPEED = 2.8
SPEED_INCREMENT     = 0.35
WAVE_INTERVAL       = 15.0
SPAWN_INTERVAL      = 0.9
SHIELD_DURATION     = 3.0
SHIELD_COOLDOWN     = 6.0
EMP_COOLDOWN        = 8.0
SLOWMO_DURATION     = 2.0
SLOWMO_COOLDOWN     = 5.0
LIVES_START         = 3
HIT_COOLDOWN        = 1.2
COMBO_TIMEOUT       = 4.0
MAX_PARTICLES       = 80   # ← capped


def _hsv_bgr(h, s=220, v=255):
    return cv2.cvtColor(
        np.uint8([[[int(h/2)%180,s,v]]]),
        cv2.COLOR_HSV2BGR)[0][0].tolist()


class _Star:
    __slots__=('x','y','speed','brightness','size')
    def __init__(self,w,h,layer):
        self.x=random.uniform(0,w);self.y=random.uniform(0,h)
        self.speed=[1.2,2.5,4.0][layer]
        self.brightness=random.randint(60,220);self.size=layer+1


class _Particle:
    __slots__=('x','y','vx','vy','life','color','size')
    def __init__(self,x,y,col,sz=4,speed=7.0):
        a=random.uniform(0,6.28);spd=random.uniform(2,speed)
        self.x=float(x);self.y=float(y)
        self.vx=math.cos(a)*spd;self.vy=math.sin(a)*spd
        self.life=1.0;self.color=col;self.size=sz
    def step(self):
        self.x+=self.vx;self.y+=self.vy
        self.vy+=0.12;self.vx*=0.96;self.life-=0.032
    @property
    def alive(self): return self.life>0.03


@dataclass
class _Asteroid:
    x:float; y:float; vx:float; vy:float
    radius:int; hue:float; angle:float=0.0; spin:float=0.0; hp:int=1
    shape_arr: object = None  # np.ndarray of shape (n,2) float32

    @property
    def alive(self): return self.y<900 and -self.radius<self.x<1400 and self.hp>0

    def step(self,dt,ts=1.0):
        self.x+=self.vx*ts; self.y+=self.vy*ts; self.angle+=self.spin*ts


class AsteroidDodge:
    def __init__(self,W,H):
        self.W=W;self.H=H
        self._ship_x=float(W//2);self._ship_y=float(H-80)
        self._ship_smooth=float(W//2);self._target_x=float(W//2)
        self._score=0;self._lives=LIVES_START;self._wave=1
        self._combo=0;self._multiplier=1
        self._game_over=False;self.high_score=0
        self._start_t=time.time();self._spawn_t=time.time()
        self._hit_t=-999.0;self._last_dodge_t=time.time()
        self._wave_next_t=time.time()+WAVE_INTERVAL
        self._shield_active=False;self._shield_end=0.0;self._shield_cd_end=0.0
        self._emp_cd_end=0.0;self._slowmo_end=0.0;self._slowmo_cd_end=0.0
        self._emp_ring=0.0;self._emp_active=False
        self._asteroids:List[_Asteroid]=[]
        self._particles:List[_Particle]=[]
        # 3 parallax star layers
        self._stars:List[List[_Star]]=[
            [_Star(W,H,l) for _ in range(30+l*20)] for l in range(3)]
        self._phase=0.0;self._shake=0.0;self._frame=0
        # Cached star background
        self._star_bg:Optional[np.ndarray]=None
        self._star_bg_frame=-1

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self,hand_control,dt:float):
        dt=min(dt,0.05)
        self._phase+=dt*3.5;self._frame+=1

        if self._game_over:
            self.high_score=max(self.high_score,self._score)
            self._update_particles(dt);return

        now=time.time()
        ts=0.35 if now<self._slowmo_end else 1.0

        # Ship movement
        if hasattr(hand_control,'active') and hand_control.active:
            px,_=getattr(hand_control,'palm_px',(self.W//2,0))
            self._target_x=float(px)
        else:
            self._target_x+=(self.W//2-self._target_x)*0.03

        self._ship_smooth+=(self._target_x-self._ship_smooth)*0.18
        self._ship_x=float(np.clip(self._ship_smooth,SHIP_W+10,self.W-SHIP_W-10))

        # Gestures
        if hasattr(hand_control,'gesture'):
            g=hand_control.gesture
            if g=="FIST" and now>=self._shield_cd_end:
                self._shield_active=True
                self._shield_end=now+SHIELD_DURATION
                self._shield_cd_end=now+SHIELD_DURATION+SHIELD_COOLDOWN
            if g=="ILY" and now>=self._emp_cd_end:
                self._trigger_emp(now)
            if g=="PINCH" and now>=self._slowmo_cd_end:
                self._slowmo_end=now+SLOWMO_DURATION
                self._slowmo_cd_end=now+SLOWMO_DURATION+SLOWMO_COOLDOWN

        if self._shield_active and now>self._shield_end:
            self._shield_active=False

        if self._emp_active:
            self._emp_ring+=28*dt*60
            if self._emp_ring>max(self.W,self.H):
                self._emp_active=False;self._emp_ring=0.0

        if now>self._wave_next_t:
            self._wave+=1;self._wave_next_t=now+WAVE_INTERVAL

        spawn_int=max(0.3,SPAWN_INTERVAL-self._wave*0.08)
        if now-self._spawn_t>=spawn_int:
            self._spawn_t=now;self._spawn_asteroid()

        for a in self._asteroids: a.step(dt,ts)
        self._update_particles(dt)
        self._update_stars(dt,ts)

        sx,sy=int(self._ship_x),int(self._ship_y)
        for a in self._asteroids:
            ax,ay=int(a.x),int(a.y)
            dist=math.hypot(ax-sx,ay-sy)
            if self._emp_active and dist<self._emp_ring:
                self._explode(ax,ay,a.hue,big=True)
                a.hp=0;self._score+=5*self._wave*self._multiplier;continue
            if dist<a.radius+20:
                if self._shield_active:
                    self._explode(ax,ay,200,big=False);a.hp=0
                    self._score+=2*self._wave
                elif now-self._hit_t>HIT_COOLDOWN:
                    self._lives-=1;self._hit_t=now
                    self._combo=0;self._multiplier=1
                    self._shake=min(20,self._shake+12)
                    self._explode(sx,sy,0,big=False);a.hp=0
                    if self._lives<=0:
                        self._game_over=True;self._explode(sx,sy,30,big=True)

        self._asteroids=[a for a in self._asteroids if a.alive]

        if self._frame%30==0:
            self._score+=self._wave*self._multiplier

        if len(self._asteroids)>0 and now-self._hit_t>COMBO_TIMEOUT:
            self._combo+=1
            if self._combo>=10 and self._multiplier<4:
                self._multiplier=min(4,self._multiplier+1);self._combo=0

        self._shake=max(0.0,self._shake-dt*14)

    def _update_stars(self,dt,ts):
        for layer in self._stars:
            for s in layer:
                s.y+=s.speed*ts
                if s.y>self.H:
                    s.y=0;s.x=random.uniform(0,self.W)

    def _update_particles(self,dt):
        for p in self._particles: p.step()
        self._particles=[p for p in self._particles if p.alive]

    def _spawn_asteroid(self):
        x=random.uniform(30,self.W-30);y=-30
        speed=ASTEROID_BASE_SPEED+self._wave*SPEED_INCREMENT+random.uniform(-0.5,0.5)
        vx=random.uniform(-1.2,1.2);vy=speed
        r=random.randint(14,28+self._wave*2)
        hue=random.uniform(0,360);spin=random.uniform(-0.08,0.08)
        hp=1 if self._wave<4 else random.choice([1,1,2])
        n_pts=random.randint(7,11)
        pts=[]
        for i in range(n_pts):
            ang=2*math.pi*i/n_pts;vary=random.uniform(0.6,1.0)
            pts.append([math.cos(ang)*r*vary, math.sin(ang)*r*vary])
        shape_arr=np.array(pts,np.float32)
        self._asteroids.append(_Asteroid(x=x,y=y,vx=vx,vy=vy,radius=r,
                                          hue=hue,spin=spin,hp=hp,shape_arr=shape_arr))

    def _trigger_emp(self,now):
        self._emp_active=True;self._emp_ring=0.0;self._emp_cd_end=now+EMP_COOLDOWN
        for _ in range(15):
            col=_hsv_bgr(random.uniform(180,220),255,255)
            self._particles.append(_Particle(int(self._ship_x),int(self._ship_y),col,sz=5,speed=10))

    def _explode(self,x,y,hue,big=True):
        n=14 if big else 6     # ← reduced from 18/8
        for _ in range(n):
            h=(hue+random.uniform(-30,30))%360;col=_hsv_bgr(h,220,255)
            sz=random.randint(3,6) if big else random.randint(2,4)
            p=_Particle(x,y,col,sz,speed=7 if big else 4)
            self._particles.append(p)
        # Cap total
        self._particles=self._particles[-MAX_PARTICLES:]

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self,frame:np.ndarray):
        now=time.time()
        ox=int(random.uniform(-self._shake,self._shake)) if self._shake>0.5 else 0
        oy=int(random.uniform(-self._shake,self._shake)) if self._shake>0.5 else 0

        # ── Batch layer (stars + asteroids + particles + ship) ────────────
        layer=np.zeros((self.H,self.W,3),np.uint8)

        # Stars — use cached BG refreshed every 8 frames
        if self._frame%8==0 or self._star_bg is None:
            self._star_bg=np.zeros((self.H,self.W,3),np.uint8)
            colors=[(80,80,80),(140,140,140),(200,200,200)]
            for li,sl in enumerate(self._stars):
                col=colors[li]
                for s in sl:
                    x,y=int(s.x),int(s.y)
                    if 0<=x<self.W and 0<=y<self.H:
                        cv2.circle(self._star_bg,(x,y),s.size,col,-1)

        cv2.add(layer,self._star_bg,layer)

        # Asteroids (rotate shape_arr in numpy — fast)
        for a in self._asteroids:
            if a.shape_arr is None: continue
            cos_a=math.cos(a.angle);sin_a=math.sin(a.angle)
            rot=np.array([[cos_a,-sin_a],[sin_a,cos_a]],np.float32)
            pts_rot=(a.shape_arr@rot.T).astype(np.int32)
            pts_rot[:,0]+=int(a.x);pts_rot[:,1]+=int(a.y)
            col=_hsv_bgr(a.hue,180,200);col_dim=[c//3 for c in col]
            cv2.fillPoly(layer,[pts_rot],col_dim)
            cv2.polylines(layer,[pts_rot],True,col,2,cv2.LINE_AA)
            if a.hp>1:
                x_,y_=int(a.x),int(a.y)
                cv2.rectangle(layer,(x_-a.radius,y_-a.radius-8),(x_+a.radius,y_-a.radius-3),(30,30,30),-1)
                cv2.rectangle(layer,(x_-a.radius,y_-a.radius-8),(x_,y_-a.radius-3),(0,200,80),-1)

        # Particles
        for p in self._particles:
            x,y=int(p.x),int(p.y)
            if 0<=x<self.W and 0<=y<self.H:
                col=[int(c*p.life) for c in p.color]
                cv2.circle(layer,(x,y),max(1,int(p.size*p.life)),col,-1,cv2.LINE_AA)

        # EMP ring
        if self._emp_active:
            sx,sy=int(self._ship_x),int(self._ship_y);r=int(self._emp_ring)
            al=max(0.0,1.0-r/max(self.W,self.H))
            col=(int(50*al),int(200*al),int(255*al))
            cv2.circle(layer,(sx,sy),r,col,3,cv2.LINE_AA)
            cv2.circle(layer,(sx,sy),max(0,r-10),[c//2 for c in col],1,cv2.LINE_AA)

        # Ship
        self._draw_ship(layer,now)

        # Screen shake
        if ox or oy:
            layer=np.roll(layer,(oy,ox),axis=(0,1))

        # Slowmo tint — only one addWeighted
        if now<self._slowmo_end:
            vig=np.full_like(layer,(0,20,30))
            cv2.addWeighted(vig,0.12,layer,0.88,0,layer)

        # Composite: addWeighted layer onto frame (70% layer visible)
        cv2.addWeighted(layer,0.70,frame,1.0,0,frame)

        self._draw_hud(frame,now)
        if self._game_over: self._draw_game_over(frame)

    def _draw_ship(self,layer,now):
        cx=int(self._ship_x);cy=int(self._ship_y);ph=self._phase
        if now-self._hit_t<HIT_COOLDOWN and int((now-self._hit_t)*8)%2==1: return
        # Engine flame
        flame_h=int(18+12*math.sin(ph*5))
        for i in range(flame_h,0,-4):
            a=i/flame_h;col=(int(20*a),int(150*a),int(255*a))
            cv2.ellipse(layer,(cx,cy+18+flame_h-i),(5,3),0,0,360,col,-1)
        pts=np.array([[cx,cy-SHIP_H//2],[cx+SHIP_W//2,cy+SHIP_H//2],
                       [cx,cy+SHIP_H//4],[cx-SHIP_W//2,cy+SHIP_H//2]],np.int32)
        cv2.fillPoly(layer,[pts],(25,45,80))
        cv2.polylines(layer,[pts],True,(80,160,255),2,cv2.LINE_AA)
        cv2.ellipse(layer,(cx,cy-8),(6,9),0,0,360,(10,30,60),-1)
        cv2.ellipse(layer,(cx,cy-8),(6,9),0,0,360,(60,160,255),1,cv2.LINE_AA)
        cv2.ellipse(layer,(cx-2,cy-12),(2,3),0,0,360,(180,220,255),-1)
        blink=(255,60,60) if int(ph*2)%2==0 else (60,0,0)
        cv2.circle(layer,(cx-SHIP_W//2,cy+SHIP_H//2),3,blink,-1)
        cv2.circle(layer,(cx+SHIP_W//2,cy+SHIP_H//2),3,(60,60,255),-1)
        if self._shield_active:
            t_left=self._shield_end-now;sr=int(38+6*math.sin(ph*4))
            a=min(1.0,t_left/SHIELD_DURATION)
            col=(int(50*a),int(200*a),int(255*a))
            cv2.circle(layer,(cx,cy),sr,col,2,cv2.LINE_AA)
            cv2.circle(layer,(cx,cy),sr-6,[c//4 for c in col],-1,cv2.LINE_AA)

    def _draw_hud(self,frame,now):
        W,H=self.W,self.H;ph=self._phase
        font=cv2.FONT_HERSHEY_SIMPLEX;font2=cv2.FONT_HERSHEY_DUPLEX

        sc_txt=f"{self._score:07d}"
        (sw,_),_=cv2.getTextSize(sc_txt,font2,0.88,2);cx2=(W-sw)//2
        pad=14;sc_hue=(ph*20+180)%360;sc_border=_hsv_bgr(sc_hue,200,200)
        roi=frame[58-pad:58+10,cx2-pad:cx2+sw+pad]
        if roi.size>0:
            bg=np.full_like(roi,(6,8,18))
            cv2.addWeighted(bg,0.82,roi,0.18,0,roi)
            frame[58-pad:58+10,cx2-pad:cx2+sw+pad]=roi
        cv2.rectangle(frame,(cx2-pad,58-pad),(cx2+sw+pad,60),sc_border,1)
        cv2.line(frame,(cx2-pad+2,58-pad+1),(cx2+sw+pad-2,58-pad+1),sc_border,2)
        cv2.putText(frame,"ASTEROID DODGE",(cx2,58-pad+10),font,0.28,[c//2 for c in sc_border],1,cv2.LINE_AA)
        cv2.putText(frame,sc_txt,(cx2+2,46),font2,0.88,(0,0,0),5,cv2.LINE_AA)
        cv2.putText(frame,sc_txt,(cx2,44),font2,0.88,(80,240,255),2,cv2.LINE_AA)

        if self._multiplier>1:
            mx_col=_hsv_bgr((ph*40)%360,230,255);mx_txt=f"x{self._multiplier}"
            bx=cx2+sw+12
            cv2.rectangle(frame,(bx-2,28),(bx+42,50),[c//4 for c in mx_col],-1)
            cv2.rectangle(frame,(bx-2,28),(bx+42,50),mx_col,1)
            cv2.putText(frame,mx_txt,(bx+2,44),font2,0.68,mx_col,2,cv2.LINE_AA)

        # Lives
        for i in range(LIVES_START):
            lx=W-32-i*32;filled=(i<self._lives)
            col_l=(80,120,255) if filled else (25,25,45)
            pts=self._heart_pts(lx,30,11)
            if filled:
                glow=self._heart_pts(lx,30,15)
                cv2.fillPoly(frame,[glow],[c//5 for c in col_l])
            cv2.fillPoly(frame,[pts],col_l)
            if filled:
                cv2.polylines(frame,[pts],True,[min(255,c+80) for c in col_l],1,cv2.LINE_AA)

        # Left status panel
        pw,pt_h=200,90;px0,py0=10,58
        roi2=frame[py0:py0+pt_h,px0:px0+pw];dark=np.full_like(roi2,(5,8,16))
        cv2.addWeighted(dark,0.84,roi2,0.16,0,roi2);frame[py0:py0+pt_h,px0:px0+pw]=roi2
        pulse_col=_hsv_bgr((ph*25)%360,200,int(100+60*math.sin(ph*2)))
        cv2.rectangle(frame,(px0,py0),(px0+pw,py0+pt_h),pulse_col,1)
        for iy in range(pt_h):
            hue=(iy/pt_h*120+ph*30)%360;c=_hsv_bgr(hue,180,160)
            frame[py0+iy,px0:px0+3]=c
        cv2.putText(frame,"STATUS",(px0+8,py0+12),font,0.28,(70,90,110),1,cv2.LINE_AA)

        def row(label,val,y,col=(150,255,150)):
            cv2.putText(frame,label,(px0+8,y),font,0.33,(60,75,95),1,cv2.LINE_AA)
            cv2.putText(frame,val,(px0+90,y),font,0.38,col,1,cv2.LINE_AA)

        row("WAVE",str(self._wave),py0+26,_hsv_bgr((ph*20+180)%360,200,255))
        row("BEST",str(self.high_score),py0+44,(200,210,80))
        row("COMBO",f"{self._combo}x",py0+62,_hsv_bgr((ph*30)%360,220,255))
        row("TIME",f"{int(now-self._start_t)}s",py0+80,(140,255,160))

        self._draw_powerup_bar(frame,now,W,H)

    def _draw_powerup_bar(self,frame,now,W,H):
        font=cv2.FONT_HERSHEY_SIMPLEX;ph=self._phase
        items=[
            ("SHIELD","✊",self._shield_cd_end,SHIELD_COOLDOWN,(0,200,255),self._shield_active),
            ("EMP","🤟",self._emp_cd_end,EMP_COOLDOWN,(200,100,255),self._emp_active),
            ("SLOW","🤌",self._slowmo_cd_end,SLOWMO_COOLDOWN,(240,200,40),now<self._slowmo_end),
        ]
        bw,bh_bar=155,16;total_w=bw*3+16*2;x0=W//2-total_w//2;y0=H-50
        roi=frame[y0-22:y0+bh_bar+6,x0-8:x0+total_w+8]
        if roi.size>0:
            dark=np.full_like(roi,(6,8,16))
            cv2.addWeighted(dark,0.80,roi,0.20,0,roi)
            frame[y0-22:y0+bh_bar+6,x0-8:x0+total_w+8]=roi
        cv2.rectangle(frame,(x0-8,y0-22),(x0+total_w+8,y0+bh_bar+6),(35,45,65),1)

        for i,(label,icon,cd_end,cd_total,col,active) in enumerate(items):
            bx=x0+i*(bw+16);ready=now>=cd_end
            fill_frac=1.0 if ready else max(0.0,1.0-(cd_end-now)/cd_total)
            cv2.rectangle(frame,(bx,y0),(bx+bw,y0+bh_bar),(12,16,28),-1)
            fill_col=col if ready else [c//4 for c in col]
            fw=int(bw*fill_frac)
            if fw>0: cv2.rectangle(frame,(bx,y0),(bx+fw,y0+bh_bar),fill_col,-1)
            if ready:
                shimmer=int(200+55*math.sin(ph*6+i))
                sh_col=[min(255,c+shimmer//4) for c in col]
                cv2.rectangle(frame,(bx+fw-3,y0),(bx+fw,y0+bh_bar),sh_col,-1)
            border_col=col if (ready or active) else (35,45,60)
            cv2.rectangle(frame,(bx,y0),(bx+bw,y0+bh_bar),border_col,1)
            if active:
                glow=[min(255,c+100) for c in col]
                cv2.rectangle(frame,(bx-2,y0-2),(bx+bw+2,y0+bh_bar+2),glow,2)
            lbl_col=(230,240,255) if ready else (70,80,100)
            cv2.putText(frame,f"{label}",(bx+4,y0-6),font,0.30,lbl_col,1,cv2.LINE_AA)
            if ready:
                ready_col=[min(255,c+60) for c in col]
                (rw,_),_=cv2.getTextSize("READY",font,0.28,1)
                cv2.putText(frame,"READY",(bx+bw-rw-2,y0-6),font,0.28,ready_col,1,cv2.LINE_AA)

    @staticmethod
    def _heart_pts(cx,cy,size):
        pts=[]
        for i in range(25):
            t=2*math.pi*i/24
            x=16*(math.sin(t)**3)
            y=-(13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t))
            pts.append([int(cx+x*size/16),int(cy+y*size/16)])
        return np.array(pts,np.int32)

    def _draw_game_over(self,frame):
        W,H=self.W,self.H
        ov=np.zeros((H,W,3),np.uint8)
        cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
        font2=cv2.FONT_HERSHEY_DUPLEX;font=cv2.FONT_HERSHEY_SIMPLEX
        col=_hsv_bgr((time.time()*80)%360,200,255)
        (tw,_),_=cv2.getTextSize("GAME OVER",font2,1.4,3)
        cv2.putText(frame,"GAME OVER",((W-tw)//2+2,H//2-30),font2,1.4,(0,0,0),6,cv2.LINE_AA)
        cv2.putText(frame,"GAME OVER",((W-tw)//2,H//2-32),font2,1.4,col,3,cv2.LINE_AA)
        sc=f"Score: {self._score}   Best: {self.high_score}"
        (sw,_),_=cv2.getTextSize(sc,font,0.6,2)
        cv2.putText(frame,sc,((W-sw)//2,H//2+10),font,0.6,(200,200,200),1,cv2.LINE_AA)
        cv2.putText(frame,"Press A to restart",((W-175)//2,H//2+48),font,0.44,(100,100,100),1,cv2.LINE_AA)
