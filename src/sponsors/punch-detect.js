// Guest-side punch detection from MediaPipe hand landmarks, run on the guest's own device.
// Only a ~60-byte punch event crosses the network, never video, so a phone on venue Wi-Fi
// still lands hits with data-channel latency. Speeds are monocular estimates, labelled as such.
import {fistScore} from '../physics.js';

const PALM=[0,5,9,13,17],HAND_LENGTH_M=.09,NOMINAL_DISTANCE_M=.55;

export class PunchDetector{
  constructor({minSpeed=.9,hookSpeed=1.2,cooldownMs=450,aspect=16/9}={}){Object.assign(this,{minSpeed,hookSpeed,cooldownMs,aspect,hands:new Map()});}
  // landmarks: 21 normalised image points. Returns a punch event or null.
  update(key,landmarks,timeMs){
    const cx=PALM.reduce((s,i)=>s+landmarks[i].x,0)/PALM.length,cy=PALM.reduce((s,i)=>s+landmarks[i].y,0)/PALM.length;
    // Wrist to middle knuckle is rigid, so its image length tracks distance to the camera.
    const scale=Math.hypot((landmarks[9].x-landmarks[0].x)*this.aspect,landmarks[9].y-landmarks[0].y);
    const previous=this.hands.get(key);const state={cx,cy,scale,time:timeMs,lastHit:previous?.lastHit??-Infinity,speed:0};this.hands.set(key,state);
    if(!previous||scale<.02)return null;
    const dt=(timeMs-previous.time)/1000;if(dt<=0||dt>.25)return null;
    const metresPerUnit=HAND_LENGTH_M/scale;
    const vx=(cx-previous.cx)*this.aspect*metresPerUnit/dt,vy=(cy-previous.cy)*metresPerUnit/dt;
    // Growing hand = approaching camera. d is proportional to 1/scale, so dz/dt = d * (ds/dt)/s.
    const vz=NOMINAL_DISTANCE_M*((scale-previous.scale)/dt)/scale,lateral=Math.hypot(vx,vy),speed=Math.hypot(lateral,Math.max(0,vz));
    // Smooth one step: a single noisy landmark frame should not throw a punch.
    state.speed=previous.speed?previous.speed*.4+speed*.6:speed;
    if(timeMs-state.lastHit<this.cooldownMs||fistScore(landmarks)<.5)return null;
    const straight=vz>this.minSpeed*.6&&state.speed>this.minSpeed,hook=lateral>this.hookSpeed&&Math.abs(vx)>Math.abs(vy);
    if(!straight&&!hook)return null;
    state.lastHit=timeMs;
    // The camera image is unmirrored: the thrower's right hand sits at image-left, and it lands on the
    // viewer-right side of a head that faces them. A hook lands on the side it came from, travelling across.
    const u=hook&&!straight?Math.sign(vx)*.8:Math.max(-1,Math.min(1,(.5-cx)*2.2)),v=Math.max(-1,Math.min(1,(.5-cy)*1.6));
    return {u,v,lateral:hook?Math.max(-1,Math.min(1,-vx/Math.max(lateral,1e-6))):0,speed:+Math.min(4,state.speed).toFixed(2),side:cx<.5?'right':'left',kind:hook&&!straight?'hook':'straight'};
  }
  forget(key){this.hands.delete(key);}
}
