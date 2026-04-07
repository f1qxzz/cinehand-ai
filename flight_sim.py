"""
flight_sim.py — Aerial Shooter Flight Simulator  [v2]
======================================================
Real-time physics + SHOOT EM UP game layer.
Controlled by VirtualYoke (dual hands).

New features vs v1:
  - Detailed layered aircraft sprite: fuselage, swept wings, canards,
    tail fins, cockpit glass, dual engine afterburners, gun muzzle flash
  - Scrolling parallax sky (clouds) + ground grid
  - Enemy fighters that pursue & shoot at you
  - Auto-fire laser cannon (throttle > 5%)
  - Particle explosion system
  - Lives (hearts) HUD
  - Wave escalation
  - Screen-shake on hit
  - Invincibility blink after damage
  - High score tracking
"""

import math, random, time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2, numpy as np


# ── Tuneable constants ────────────────────────────────────────────────────────
GRAVITY      = 0.06   # REDUCED: lighter gravity for easier control
DRAG_K       = 0.012  # REDUCED: less drag
ACCEL        = 14.0   # INCREASED: faster throttle response
MAX_SPEED    = 12.0
MIN_SPEED    = 1.0    # REDUCED: can go slower
MAX_PITCH    = 40.0
MAX_ROLL     = 60.0
PITCH_RATE   = 65.0   # INCREASED: more responsive pitch
ROLL_RATE    = 85.0   # INCREASED: more responsive roll
TURN_FACTOR  = 0.65
LIFT_FACTOR  = 0.60
AUTO_LEVEL_RATE = 14.0  # REDUCED: plane stays where you put it longer
WORLD_W      = 2000.0
WORLD_H      = 2000.0

BULLET_SPEED  = 14.0
FIRE_RATE     = 0.18
ENEMY_SPAWN_T = 3.0
LIVES_START   = 3
HIT_COOLDOWN  = 1.5


def _hsv_bgr(h, s=220, v=255):
    return cv2.cvtColor(np.uint8([[[int(h/2)%180,s,v]]]),
                        cv2.COLOR_HSV2BGR)[0][0].tolist()


class _Particle:
    __slots__ = ('x','y','vx','vy','life','color','size')
    def __init__(self, x, y, col, sz=4, speed=8.0):
        a = random.uniform(0,6.28); spd = random.uniform(2, speed)
        self.x=float(x); self.y=float(y)
        self.vx=math.cos(a)*spd; self.vy=math.sin(a)*spd
        self.life=1.0; self.color=col; self.size=sz
    def step(self):
        self.x+=self.vx; self.y+=self.vy
        self.vy+=0.18; self.vx*=0.95; self.life-=0.03
    @property
    def alive(self): return self.life>0.04


@dataclass
class _Bullet:
    x:float; y:float; vx:float; vy:float
    life:float=1.0; friendly:bool=True
    def step(self): self.x+=self.vx; self.y+=self.vy; self.life-=0.012
    @property
    def alive(self): return self.life>0 and -20<self.x<1300 and -20<self.y<740


@dataclass
class _Enemy:
    x:float; y:float; vx:float; vy:float
    hp:int=2; hue:float=0.0; size:int=18
    phase:float=0.0; shoot_t:float=0.0
    def step(self,dt):
        self.x+=self.vx; self.y+=self.vy+math.sin(self.phase)*0.8
        self.phase+=dt*2.5
    @property
    def alive(self): return self.hp>0 and -80<self.x<1360 and -80<self.y<800


@dataclass
class AircraftState:
    x:float=WORLD_W/2; y:float=WORLD_H/2
    speed:float=3.0; yaw:float=0.0
    pitch:float=0.0; roll:float=0.0
    throttle:float=0.3; altitude:float=500.0


class FlightSimulator:
    def __init__(self, frame_w:int, frame_h:int):
        self.W=frame_w; self.H=frame_h
        self.state=AircraftState()
        self._score=0; self._lives=LIVES_START; self._wave=1
        self._enemies:List[_Enemy]=[]
        self._bullets:List[_Bullet]=[]
        self._particles:List[_Particle]=[]
        self._fire_t=0.0; self._spawn_t=0.0; self._hit_t=0.0
        self._game_over=False; self._start_t=time.time()
        self._phase=0.0
        self._scroll1=0.0; self._scroll2=0.0; self._scroll3=0.0
        self._trail:List[Tuple[int,int]]=[]
        self._max_trail=45
        self._shake=0.0
        self.high_score=0

    # ── Update ───────────────────────────────────────────────────────────────
    def update(self, yoke, dt:float):
        dt=min(dt,0.05)
        self._phase+=dt*3.0
        if self._game_over:
            self.high_score=max(self.high_score,self._score); return
        s=self.state; now=time.time()

        if yoke.active:
            target_roll  =  yoke.roll  * MAX_ROLL
            target_pitch = -yoke.pitch * MAX_PITCH
            s.throttle   = max(0.18, yoke.throttle)  # throttle minimum agar tidak stall
            s.roll  += float(np.clip(target_roll  - s.roll,  -ROLL_RATE  * dt, ROLL_RATE  * dt))
            s.pitch += float(np.clip(target_pitch - s.pitch, -PITCH_RATE * dt, PITCH_RATE * dt))
        else:
            # Auto-level: pesawat perlahan kembali ke posisi netral
            s.roll  *= max(0.0, 1.0 - AUTO_LEVEL_RATE * dt)
            s.pitch *= max(0.0, 1.0 - AUTO_LEVEL_RATE * dt * 0.7)

        s.roll  = float(np.clip(s.roll,  -MAX_ROLL,  MAX_ROLL))
        s.pitch = float(np.clip(s.pitch, -MAX_PITCH, MAX_PITCH))
        thrust  = s.throttle * ACCEL
        s.speed += thrust * dt
        s.speed *= (1 - DRAG_K * dt * 60)
        s.speed  = float(np.clip(s.speed, MIN_SPEED, MAX_SPEED))

        yaw_rate=(s.roll/MAX_ROLL)*s.speed*TURN_FACTOR
        s.yaw=(s.yaw+yaw_rate*dt*30)%360

        yaw_r=math.radians(s.yaw); pitch_r=math.radians(s.pitch)
        fwd=s.speed*math.cos(pitch_r); lift=s.speed*math.sin(pitch_r)*LIFT_FACTOR
        s.x+=math.cos(yaw_r)*fwd*dt*30; s.y+=math.sin(yaw_r)*fwd*dt*30
        s.y-=lift*dt*30; s.y+=GRAVITY*dt
        s.x%=WORLD_W; s.y%=WORLD_H
        s.altitude=float(np.clip(WORLD_H-s.y,0,WORLD_H))

        spd_n=s.speed/MAX_SPEED
        self._scroll1=(self._scroll1+spd_n*5)%self.W
        self._scroll2=(self._scroll2+spd_n*2.5)%self.W
        self._scroll3=(self._scroll3+spd_n*1.2)%self.W
        self._shake=max(0.0,self._shake-dt*12)

        cx,cy=self.W//2,self.H//2

        # Auto-fire
        if now-self._fire_t>=FIRE_RATE and s.throttle>0.05:
            self._fire_t=now
            self._bullets.append(_Bullet(x=float(cx),y=float(cy-6),vx=0,vy=-BULLET_SPEED,friendly=True))

        # Enemy spawn
        spawn_int=max(0.7,ENEMY_SPAWN_T-self._wave*0.25)
        if now-self._spawn_t>=spawn_int:
            self._spawn_t=now; self._spawn_enemy()
            if now-self._start_t>30*self._wave: self._wave+=1

        for b in self._bullets: b.step()
        self._bullets=[b for b in self._bullets if b.alive]

        for e in self._enemies:
            e.step(dt)
            if now-e.shoot_t>max(0.8,2.2-self._wave*0.1):
                e.shoot_t=now
                dx=cx-e.x; dy=cy-e.y; dist=max(1,math.hypot(dx,dy))
                spd=5.0+self._wave*0.3
                self._bullets.append(_Bullet(x=e.x,y=e.y,vx=(dx/dist)*spd,vy=(dy/dist)*spd,friendly=False))
        self._enemies=[e for e in self._enemies if e.alive]

        # Player bullets vs enemies
        dead_b=set()
        for bi,b in enumerate(self._bullets):
            if not b.friendly: continue
            for e in self._enemies:
                if math.hypot(b.x-e.x,b.y-e.y)<e.size+8:
                    e.hp-=1; dead_b.add(bi)
                    if e.hp<=0:
                        self._explode(int(e.x),int(e.y),e.hue,big=True)
                        self._score+=10*self._wave; self._shake=min(8,self._shake+4)
                    else:
                        self._explode(int(e.x),int(e.y),e.hue,big=False)
        self._bullets=[b for i,b in enumerate(self._bullets) if i not in dead_b]

        # Enemy vs player
        if now-self._hit_t>HIT_COOLDOWN:
            for b in self._bullets:
                if b.friendly: continue
                if math.hypot(b.x-cx,b.y-cy)<28:
                    self._lives-=1; self._hit_t=now
                    self._shake=min(16,self._shake+10)
                    self._explode(cx,cy,0,big=False); b.life=0
                    if self._lives<=0: self._game_over=True
                    break
            for e in self._enemies:
                if math.hypot(e.x-cx,e.y-cy)<e.size+22:
                    self._lives-=1; self._hit_t=now; e.hp=0
                    self._shake=min(20,self._shake+14)
                    self._explode(cx,cy,20,big=True)
                    if self._lives<=0: self._game_over=True
                    break

        for p in self._particles: p.step()
        self._particles=[p for p in self._particles if p.alive]
        self._trail.append((cx,cy))
        if len(self._trail)>self._max_trail: self._trail.pop(0)

    def _spawn_enemy(self):
        side=random.choice(['top','top','left','right'])
        if side=='top': x=random.uniform(80,self.W-80); y=random.uniform(-50,-10)
        elif side=='left': x=-40; y=random.uniform(60,self.H-60)
        else: x=self.W+40; y=random.uniform(60,self.H-60)
        cx,cy=self.W//2,self.H//2
        dx=cx-x; dy=cy-y; dist=max(1,math.hypot(dx,dy))
        spd=random.uniform(1.5,2.5+self._wave*0.3)
        hp=1 if self._wave<3 else random.choice([1,1,2])
        self._enemies.append(_Enemy(x=x,y=y,vx=(dx/dist)*spd,vy=(dy/dist)*spd,
                                    hp=hp,hue=random.uniform(0,360),
                                    size=random.randint(14,20),shoot_t=time.time()))

    def _explode(self,x,y,hue,big=True):
        n=22 if big else 10
        for _ in range(n):
            h=(hue+random.uniform(-30,30))%360
            col=_hsv_bgr(h,220,255); sz=random.randint(3,8) if big else random.randint(2,5)
            self._particles.append(_Particle(x,y,col,sz,speed=9 if big else 5))
        if big:
            for _ in range(12):
                col=_hsv_bgr(random.uniform(10,40),255,255)
                self._particles.append(_Particle(x,y,col,random.randint(4,10),speed=12))

    # ── Render ───────────────────────────────────────────────────────────────
    def render(self, frame:np.ndarray):
        s=self.state
        ox=int(random.uniform(-self._shake,self._shake)) if self._shake>0.5 else 0
        oy=int(random.uniform(-self._shake,self._shake)) if self._shake>0.5 else 0
        layer=np.zeros((self.H,self.W,3),np.uint8)
        self._draw_background(layer,s)
        self._draw_trail(layer)
        self._draw_bullets(layer)
        self._draw_enemies(layer)
        self._draw_particles(layer)
        self._draw_aircraft(layer,s)
        if ox or oy: layer=np.roll(layer,(oy,ox),axis=(0,1))
        cv2.addWeighted(layer,0.65,frame,1.0,0,frame)
        self._draw_hud(frame,s)
        if self._game_over: self._draw_game_over(frame)

    def _draw_background(self,layer,s:AircraftState):
        W,H=self.W,self.H
        alt_t=min(1.0,s.altitude/WORLD_H)
        sky_col=(max(0,int(80-60*alt_t)),max(0,int(30-20*alt_t)),8)
        horizon=int(H*0.55+s.pitch*2); horizon=max(H//4,min(3*H//4,horizon))
        cv2.rectangle(layer,(0,0),(W,horizon),sky_col,-1)
        cv2.rectangle(layer,(0,horizon),(W,H),(12,45,10),-1)

        # Horizon glow
        for off,a in [(0,0.6),(4,0.25),(8,0.10)]:
            ov=layer.copy()
            cv2.line(ov,(0,horizon+off),(W,horizon+off),(40,180,80),1+off//3,cv2.LINE_AA)
            cv2.addWeighted(ov,a,layer,1-a,0,layer)

        # Scrolling clouds
        spd_n=s.speed/max(1,MAX_SPEED)
        for i,(base_y,gap,col,scroll) in enumerate([
            (30,120,(200,210,220),self._scroll1),
            (65,200,(180,190,200),self._scroll2),
            (100,300,(160,170,180),self._scroll3),
        ]):
            if base_y>=horizon: continue
            for j in range(0,W+gap,gap):
                cx2=int((j-scroll)%(W+gap)); cy2=base_y+(j%3)*12
                rw=40+(j%5)*12; rh=12+(j%3)*4; a=0.12-i*0.03
                ov2=layer.copy()
                cv2.ellipse(ov2,(cx2,cy2),(rw,rh),0,0,360,col,-1,cv2.LINE_AA)
                cv2.addWeighted(ov2,a,layer,1-a,0,layer)

        # Ground grid
        for k in range(0,W+60,60):
            gx=int((k-self._scroll1*2)%(W+60))
            cv2.line(layer,(gx,horizon),(gx-30,H),(20,60,15),1,cv2.LINE_AA)
        for k in range(0,H-horizon+60,40):
            gy=horizon+k; fade=min(255,int(255*k/max(1,H-horizon)))
            cv2.line(layer,(0,gy),(W,gy),(12+fade//20,40+fade//10,12),1,cv2.LINE_AA)

    def _draw_trail(self,layer):
        n=len(self._trail)
        for i in range(n-1):
            a=i/max(n,1); c=int(120*a)
            cv2.line(layer,self._trail[i],self._trail[i+1],(0,c,c//2),max(1,int(a*3)),cv2.LINE_AA)

    def _draw_bullets(self,layer):
        for b in self._bullets:
            x,y=int(b.x),int(b.y)
            if b.friendly:
                cv2.line(layer,(x,y),(x,y+18),(0,60,80),5,cv2.LINE_AA)
                cv2.line(layer,(x,y),(x,y+18),(100,255,255),2,cv2.LINE_AA)
                cv2.circle(layer,(x,y),4,(200,255,255),-1,cv2.LINE_AA)
            else:
                cv2.circle(layer,(x,y),6,(0,80,160),-1,cv2.LINE_AA)
                cv2.circle(layer,(x,y),4,(60,180,255),-1,cv2.LINE_AA)
                cv2.circle(layer,(x,y),2,(255,255,255),-1,cv2.LINE_AA)

    def _draw_enemies(self,layer):
        ph=self._phase
        for e in self._enemies:
            x,y=int(e.x),int(e.y); sz=e.size
            col=_hsv_bgr(e.hue,220,255); dark=[c//3 for c in col]
            cv2.circle(layer,(x,y),sz+10,[c//6 for c in col],-1,cv2.LINE_AA)
            cv2.circle(layer,(x,y),sz+5,[c//4 for c in col],-1,cv2.LINE_AA)
            ang=math.atan2(e.vy,e.vx)
            def ep(l,oa): a=ang+oa; return (int(x+l*math.cos(a)),int(y+l*math.sin(a)))
            nose=ep(sz,0); wl=ep(int(sz*0.8),math.pi*0.65)
            wr=ep(int(sz*0.8),-math.pi*0.65); tail=ep(sz//2,math.pi)
            pts=np.array([nose,wl,tail,wr],np.int32)
            cv2.fillPoly(layer,[pts],dark)
            cv2.polylines(layer,[pts],True,col,2,cv2.LINE_AA)
            cv2.circle(layer,(x,y),sz//3,col,-1,cv2.LINE_AA)
            cv2.circle(layer,(x,y),sz//5,(255,255,255),-1,cv2.LINE_AA)
            ex_col=_hsv_bgr((e.hue+180)%360,255,255)
            er=int(4+3*math.sin(ph*3+e.phase))
            cv2.circle(layer,tail,er,[c//2 for c in ex_col],-1,cv2.LINE_AA)
            if e.hp>1:
                bx0,by0=x-sz,y-sz-10
                cv2.rectangle(layer,(bx0,by0),(bx0+sz*2,by0+5),(30,30,30),-1)
                cv2.rectangle(layer,(bx0,by0),(bx0+sz*2//e.hp+sz,by0+5),(0,200,80),-1)

    def _draw_particles(self,layer):
        for p in self._particles:
            x,y=int(p.x),int(p.y)
            if 0<=x<self.W and 0<=y<self.H:
                col=[int(c*p.life) for c in p.color]
                cv2.circle(layer,(x,y),max(1,int(p.size*p.life)),col,-1,cv2.LINE_AA)

    # ── Aircraft drawing helpers ──────────────────────────────────────────────
    @staticmethod
    def _rot(pts, cx, cy, cos_r, sin_r):
        """Rotate list of (x,y) points around (cx,cy)."""
        out = []
        for x, y in pts:
            dx, dy = x - cx, y - cy
            out.append((int(cx + dx*cos_r - dy*sin_r),
                        int(cy + dx*sin_r + dy*cos_r)))
        return out

    @staticmethod
    def _rp(pts, cx, cy, cos_r, sin_r):
        """Rotate numpy int32 polygon around (cx,cy)."""
        arr = np.array(pts, np.int32)
        dx = arr[:,0] - cx; dy = arr[:,1] - cy
        arr[:,0] = (cx + dx*cos_r - dy*sin_r).astype(np.int32)
        arr[:,1] = (cy + dx*sin_r + dy*cos_r).astype(np.int32)
        return arr

    def _draw_aircraft(self, layer, s: AircraftState):
        cx, cy = self.W//2, self.H//2
        ph  = self._phase
        now = time.time()

        # Invincibility blink
        if now - self._hit_t < HIT_COOLDOWN and int((now - self._hit_t)*8) % 2 == 1:
            return

        # ── Visual transforms driven by physics state ─────────────────────
        # Roll: rotate whole sprite around centre
        roll_deg  = s.roll                          # degrees, +right bank
        roll_r    = math.radians(-roll_deg)         # negate: screen-Y is down
        cos_r, sin_r = math.cos(roll_r), math.sin(roll_r)

        # Pitch: vertical offset + foreshortening
        # pitch >0 = nose down in physics, so sprite moves down on screen
        pitch_norm = s.pitch / MAX_PITCH            # -1..+1
        pitch_oy   = int(pitch_norm * 18)           # pixel shift down/up
        # Foreshortening: when banking hard, aircraft looks narrower
        roll_scale = max(0.30, 1.0 - abs(s.roll / MAX_ROLL) * 0.65)

        # Helper: rotate a raw-coordinate point (relative to 0,0 origin) then place
        def R(x, y):
            # apply pitch offset to y first, then roll-rotate
            y2 = y + pitch_oy
            nx = cx + int(x * roll_scale * cos_r - y2 * sin_r)
            ny = cy + int(x * roll_scale * sin_r + y2 * cos_r)
            return (nx, ny)

        def Rpoly(pts):
            return np.array([R(x - cx, y - cy) for x, y in pts], np.int32)

        thr = s.throttle

        # ── Shadow (rolls with plane) ─────────────────────────────────────
        shad = R(6, 8)
        cv2.ellipse(layer, shad, (int(30*roll_scale), 12),
                    math.degrees(-roll_r), 0, 360, (0, 25, 0), -1, cv2.LINE_AA)

        # ── Dual engine afterburners ──────────────────────────────────────
        er   = int(8 + 10*thr + 4*math.sin(ph*4)*thr)
        ec2  = (int(10*thr), int(200*thr), int(255*thr))
        ec1  = (int(30*thr), int(120*thr), int(220*thr))
        for ex_off in (-10, 10):
            ep = R(ex_off, 32)
            cv2.circle(layer, ep, er+8, [c//4 for c in ec2], -1, cv2.LINE_AA)
            cv2.circle(layer, ep, er+4, [c//2 for c in ec2], -1, cv2.LINE_AA)
            cv2.circle(layer, ep, er,   ec2,                  -1, cv2.LINE_AA)
            if thr > 0.4:
                streak_l = int(30 + 60*thr + 20*math.sin(ph*5))
                # streak direction = "behind" the plane (opposite to nose dir)
                streak_dx = int(sin_r * 6)
                streak_dy = int(cos_r * 6)
                sx, sy = ep
                for si in range(0, streak_l, 6):
                    a = max(0.0, 1.0 - si / streak_l)
                    sw2 = max(1, int(er * a * 0.7))
                    sp = (sx + streak_dx*si//6, sy + streak_dy*si//6)
                    cv2.circle(layer, sp, sw2, [int(c*a) for c in ec1], -1, cv2.LINE_AA)

        # ── Main fuselage ─────────────────────────────────────────────────
        fus = Rpoly([(cx, cy-34), (cx+8, cy-10), (cx+12, cy+28),
                     (cx, cy+22), (cx-12, cy+28), (cx-8, cy-10)])
        cv2.fillPoly(layer, [fus], (30, 50, 80))
        cv2.polylines(layer, [fus], True, (80, 180, 255), 2, cv2.LINE_AA)
        cv2.line(layer, R(0,-34), R(0,22), (50,100,140), 1, cv2.LINE_AA)

        # ── Swept wings ───────────────────────────────────────────────────
        rw_raw = [(cx+8,cy+0),(cx+58,cy+24),(cx+46,cy+32),(cx+12,cy+26)]
        lw_raw = [(2*cx-x, y) for x,y in rw_raw]
        rw = Rpoly(rw_raw); lw = Rpoly(lw_raw)
        cv2.fillPoly(layer, [rw], (25,45,70)); cv2.fillPoly(layer, [lw], (25,45,70))
        cv2.polylines(layer, [rw], True, (60,150,220), 1, cv2.LINE_AA)
        cv2.polylines(layer, [lw], True, (60,150,220), 1, cv2.LINE_AA)
        # Panel lines on wings
        for wraw in [rw_raw, lw_raw]:
            wr = Rpoly(wraw)
            for frac in [0.4, 0.7]:
                p0,p1,p2,p3 = wr[0],wr[1],wr[3],wr[2]
                mA = (int(p0[0]+(p1[0]-p0[0])*frac), int(p0[1]+(p1[1]-p0[1])*frac))
                mB = (int(p2[0]+(p3[0]-p2[0])*frac), int(p2[1]+(p3[1]-p2[1])*frac))
                cv2.line(layer, mA, mB, (40,90,140), 1, cv2.LINE_AA)

        # Nav lights (roll with plane)
        blink_r = (255,80,80) if int(ph*2)%2==0 else (80,0,0)
        cv2.circle(layer, R(58-cx+cx, 24), 5, blink_r, -1, cv2.LINE_AA)
        cv2.circle(layer, R(-(58-cx+cx), 24), 5, (80,80,255), -1, cv2.LINE_AA)
        # simpler: use Rpoly single points
        cv2.circle(layer, Rpoly([(cx+58, cy+24)])[0], 5, blink_r,    -1, cv2.LINE_AA)
        cv2.circle(layer, Rpoly([(cx-58, cy+24)])[0], 5, (80,80,255),-1, cv2.LINE_AA)

        # ── Canard fins ───────────────────────────────────────────────────
        cr = Rpoly([(cx+6,cy-18),(cx+24,cy-4),(cx+20,cy+2),(cx+8,cy-10)])
        cl = Rpoly([(cx-6,cy-18),(cx-24,cy-4),(cx-20,cy+2),(cx-8,cy-10)])
        cv2.fillPoly(layer, [cr], (20,40,65)); cv2.fillPoly(layer, [cl], (20,40,65))
        cv2.polylines(layer, [cr], True, (50,120,180), 1, cv2.LINE_AA)
        cv2.polylines(layer, [cl], True, (50,120,180), 1, cv2.LINE_AA)

        # ── Tail fins ─────────────────────────────────────────────────────
        tfr = Rpoly([(cx+4,cy+22),(cx+18,cy+10),(cx+20,cy+20),(cx+12,cy+30)])
        tfl = Rpoly([(cx-4,cy+22),(cx-18,cy+10),(cx-20,cy+20),(cx-12,cy+30)])
        cv2.fillPoly(layer, [tfr], (20,40,65)); cv2.fillPoly(layer, [tfl], (20,40,65))
        cv2.polylines(layer, [tfr], True, (50,120,180), 1, cv2.LINE_AA)
        cv2.polylines(layer, [tfl], True, (50,120,180), 1, cv2.LINE_AA)

        # ── Ventral intake scoops ─────────────────────────────────────────
        for ox2, sign in [(-5,1),(5,-1)]:
            iv = Rpoly([(cx+ox2,cy-5),(cx+ox2+sign*10,cy+10),
                        (cx+ox2+sign*6,cy+12),(cx+ox2,cy+0)])
            cv2.fillPoly(layer, [iv], (15,30,50))
            cv2.polylines(layer, [iv], True, (40,90,130), 1, cv2.LINE_AA)

        # ── Cockpit glass ─────────────────────────────────────────────────
        cock = R(0, -18)
        cv2.ellipse(layer, cock, (8,12), math.degrees(-roll_r),
                    0, 360, (10,30,50), -1, cv2.LINE_AA)
        cv2.ellipse(layer, cock, (8,12), math.degrees(-roll_r),
                    0, 360, (80,180,255), 1, cv2.LINE_AA)
        glare = R(-2, -22)
        cv2.ellipse(layer, glare, (3,4), math.degrees(-roll_r),
                    0, 360, (200,230,255), -1, cv2.LINE_AA)

        # ── Nose cone ─────────────────────────────────────────────────────
        nose = R(0, -34)
        cv2.circle(layer, nose, 3, (160,230,255), -1, cv2.LINE_AA)
        cv2.circle(layer, nose, 5, (80,140,200),   1, cv2.LINE_AA)

        # ── Muzzle flash ──────────────────────────────────────────────────
        if now - self._fire_t < 0.08:
            muz = R(0, -38)
            cv2.circle(layer, muz, 8,  (200,255,255), -1, cv2.LINE_AA)
            cv2.circle(layer, muz, 14, (40,100,60),   -1, cv2.LINE_AA)

        # ── HUD crosshair (always upright, shows bank angle visually) ────
        # Crosshair arms rotate with the roll so the pilot feels the bank
        for arm_x, arm_y in [(-28,0),(28,0),(0,-20),(0,20)]:
            tip  = R(arm_x, arm_y + pitch_oy)
            base = (int(cx + arm_x*roll_scale*cos_r*0.4 - (arm_y+pitch_oy)*sin_r*0.4),
                    int(cy + arm_x*roll_scale*sin_r*0.4 + (arm_y+pitch_oy)*cos_r*0.4))
            cv2.line(layer, tip, base, (0,200,100), 2, cv2.LINE_AA)

    def _draw_hud(self,frame:np.ndarray,s:AircraftState):
        W,H=self.W,self.H
        font=cv2.FONT_HERSHEY_SIMPLEX; font2=cv2.FONT_HERSHEY_DUPLEX
        ph=self._phase

        # Top-left flight data panel
        pw,pt_h=200,130; px0,py0=10,68
        bg=frame[py0:py0+pt_h,px0:px0+pw].copy()
        dark=np.full_like(bg,(6,10,16))
        cv2.addWeighted(dark,0.80,bg,0.20,0,bg)
        frame[py0:py0+pt_h,px0:px0+pw]=bg
        pulse=int(100+60*math.sin(ph))
        cv2.rectangle(frame,(px0,py0),(px0+pw,py0+pt_h),(0,pulse//2,pulse),1)

        def row(label,val,y,col=(150,255,150)):
            cv2.putText(frame,label,(px0+8,y),font,0.35,(80,100,120),1,cv2.LINE_AA)
            cv2.putText(frame,val,(px0+90,y),font,0.38,col,1,cv2.LINE_AA)

        row("SPEED",f"{int(s.speed/MAX_SPEED*320)} kts",py0+26)
        row("ALT",  f"{int(s.altitude/WORLD_H*15000)} ft",py0+44)
        row("PITCH",f"{s.pitch:+.1f}°",py0+62)
        row("ROLL", f"{s.roll:+.1f}°",py0+80)

        cv2.putText(frame,"THR",(px0+8,py0+100),font,0.35,(80,100,120),1,cv2.LINE_AA)
        bx0,bx1=px0+46,px0+pw-8; by=py0+96
        cv2.rectangle(frame,(bx0,by-9),(bx1,by),(15,25,35),-1)
        fill=int((bx1-bx0)*s.throttle)
        cv2.rectangle(frame,(bx0,by-9),(bx0+fill,by),(0,220,int(80+175*(1-s.throttle))),-1)
        cv2.rectangle(frame,(bx0,by-9),(bx1,by),(40,55,60),1)
        row("WAVE",f"{self._wave}",py0+118,col=(80,200,255))

        # Score (top-centre)
        sc_txt=f"{self._score:07d}"
        (sw,_),_=cv2.getTextSize(sc_txt,font2,0.90,2)
        cx2=(W-sw)//2
        cv2.putText(frame,sc_txt,(cx2+2,44),font2,0.90,(0,0,0),5,cv2.LINE_AA)
        cv2.putText(frame,sc_txt,(cx2,42),font2,0.90,(80,240,255),2,cv2.LINE_AA)

        # Lives (heart icons, top-right)
        for i in range(LIVES_START):
            lx=W-28-i*28; filled=(i<self._lives)
            col_l=(80,100,255) if filled else (30,30,50)
            pts=self._heart_pts(lx,28,10)
            cv2.fillPoly(frame,[pts],col_l)

        self._draw_compass(frame,s.yaw)

        if not (s.roll or s.pitch) and not s.throttle>0.1:
            cv2.putText(frame,"KIRI=GAS | KANAN=KEMUDI | MIRINGKAN TANGAN=ROLL | NAIK/TURUN=PITCH",
                        (W//2-280,H-16),font,0.36,(60,60,60),1,cv2.LINE_AA)

    @staticmethod
    def _heart_pts(cx,cy,size):
        pts=[]
        for i in range(25):
            t=2*math.pi*i/24
            x=16*(math.sin(t)**3)
            y=-(13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t))
            pts.append([int(cx+x*size/16),int(cy+y*size/16)])
        return np.array(pts,np.int32)

    def _draw_compass(self,frame,yaw_deg):
        cx=self.W//2; cy=50; r=34; font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(frame,(cx,cy),r+2,(10,18,28),-1)
        cv2.circle(frame,(cx,cy),r,(40,80,100),1,cv2.LINE_AA)
        for d,label in [(0,'N'),(90,'E'),(180,'S'),(270,'W')]:
            a=math.radians(d-yaw_deg-90)
            tx=int(cx+(r-10)*math.cos(a)); ty=int(cy+(r-10)*math.sin(a))
            col=(80,80,255) if label=='N' else (100,160,180)
            (tw,th),_=cv2.getTextSize(label,font,0.32,1)
            cv2.putText(frame,label,(tx-tw//2,ty+th//2),font,0.32,col,1,cv2.LINE_AA)
        cv2.line(frame,(cx,cy-3),(cx,cy-r+4),(0,220,255),2,cv2.LINE_AA)
        hdg=f"{int(yaw_deg):03d}°"
        (tw,_),_=cv2.getTextSize(hdg,font,0.38,1)
        cv2.putText(frame,hdg,(cx-tw//2,cy+r+14),font,0.38,(160,200,200),1,cv2.LINE_AA)

    def _draw_game_over(self,frame):
        W,H=self.W,self.H
        ov=np.zeros((H,W,3),np.uint8); cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
        font2=cv2.FONT_HERSHEY_DUPLEX; font=cv2.FONT_HERSHEY_SIMPLEX
        col=_hsv_bgr((time.time()*80)%360,200,255)
        (tw,_),_=cv2.getTextSize("GAME OVER",font2,1.4,3)
        cv2.putText(frame,"GAME OVER",((W-tw)//2+2,H//2-20),font2,1.4,(0,0,0),6,cv2.LINE_AA)
        cv2.putText(frame,"GAME OVER",((W-tw)//2,H//2-22),font2,1.4,col,3,cv2.LINE_AA)
        sc=f"Score: {self._score}   Best: {self.high_score}"
        (sw,_),_=cv2.getTextSize(sc,font,0.6,2)
        cv2.putText(frame,sc,((W-sw)//2,H//2+20),font,0.6,(200,200,200),1,cv2.LINE_AA)
        cv2.putText(frame,"Press F to restart",((W-160)//2,H//2+55),font,0.44,(100,100,100),1,cv2.LINE_AA)