import {slerp} from './fist-pose.js';

// One estimator where there used to be four.
//
// The old POV path stacked an EMA on the rendered landmarks, a lead/gain extrapolator on top of it,
// and a second EMA on velocity inside StrikeTracker, each added to cover a symptom of the one
// before. Their combined phase response was unknowable, and the extrapolator amplified still-hand
// noise ~2.2x because on a motionless fist the sample delta *is* the noise.
//
// A constant-acceleration Kalman filter replaces all of them. It consumes the per-axis variance the
// pose fit actually measured, so depth (always the loose axis) is trusted less than the lateral
// axes without anyone tuning a constant. Its prediction at render time is the extrapolator, and its
// velocity state is what strike detection reads, so position and velocity can no longer disagree.

const clamp=(value,min,max)=>Math.min(max,Math.max(min,value));

class Axis{
  constructor(jerk){this.q=jerk*jerk;this.reset();}
  reset(position=0,variance=1){
    this.x=[position,0,0];
    this.P=[variance,0,0, 0,25,0, 0,0,2500];   // velocity/acceleration start wide open
  }
  step(dt){
    const [p,v,a]=this.x,P=this.P,h=dt*dt/2;
    this.x=[p+v*dt+a*h,v+a*dt,a];
    // P <- F P F^T, F = [[1,dt,dt^2/2],[0,1,dt],[0,0,1]]
    const f=(r,c)=>P[r*3+c];
    const m=[
      f(0,0)+dt*f(1,0)+h*f(2,0), f(0,1)+dt*f(1,1)+h*f(2,1), f(0,2)+dt*f(1,2)+h*f(2,2),
      f(1,0)+dt*f(2,0),          f(1,1)+dt*f(2,1),          f(1,2)+dt*f(2,2),
      f(2,0),                    f(2,1),                    f(2,2),
    ];
    const n=[
      m[0]+dt*m[1]+h*m[2], m[1]+dt*m[2], m[2],
      m[3]+dt*m[4]+h*m[5], m[4]+dt*m[5], m[5],
      m[6]+dt*m[7]+h*m[8], m[7]+dt*m[8], m[8],
    ];
    // continuous white-jerk process noise
    const q=this.q,d3=dt*dt*dt,d4=d3*dt,d5=d4*dt;
    n[0]+=q*d5/20;n[1]+=q*d4/8;n[2]+=q*d3/6;
    n[3]+=q*d4/8; n[4]+=q*d3/3;n[5]+=q*dt*dt/2;
    n[6]+=q*d3/6; n[7]+=q*dt*dt/2;n[8]+=q*dt;
    this.P=n;
  }
  correct(z,r){
    const P=this.P,s=P[0]+r;if(!(s>0))return;
    const k=[P[0]/s,P[3]/s,P[6]/s],y=z-this.x[0];
    this.x=[this.x[0]+k[0]*y,this.x[1]+k[1]*y,this.x[2]+k[2]*y];
    const row=[P[0],P[1],P[2]];
    this.P=P.map((value,i)=>value-k[Math.floor(i/3)]*row[i%3]);
  }
  at(dt){const [p,v,a]=this.x;return [p+v*dt+a*dt*dt/2,v+a*dt,a];}
}

// The pose fit's own information matrix only knows about image noise. It cannot see error in the
// metric shape it was handed, so it reports roughly 2.5x less spread than the estimate actually
// has. Inflating by a measured constant keeps the *relative* weighting between axes -- which is the
// valuable part, since depth is always the loose one -- while making the absolute scale honest.
// Calibrated against the synthetic noise model; re-measure against a real camera before trusting
// the absolute value.
const SHAPE_INFLATION=6.5;

export class PoseFilter{
  // jerk is the process-noise level in m/s^3, held low so a resting fist stays quiet, then raised
  // automatically while the hand is genuinely manoeuvring. A single fixed value cannot do both: set
  // loose enough to track a 120 ms punch it leaves ~1 m/s of velocity noise on a motionless hand,
  // which is most of the way to the 0.4 m/s strike threshold.
  constructor({jerk=90,maxGapMs=260,orientationTau=.05,agility=10,trigger=9}={}){
    this.jerk=jerk;this.maxGapMs=maxGapMs;this.orientationTau=orientationTau;this.agility=agility;this.trigger=trigger;this.reset();
  }
  reset(){this.axes=null;this.time=null;this.quaternion=null;this.started=false;this.boost=1;}
  get ready(){return this.started&&!!this.axes;}
  update(position,variance,timestamp,quaternion){
    if(!Array.isArray(position)||position.some(value=>!Number.isFinite(value))||!Number.isFinite(timestamp))return false;
    const raw=Array.isArray(variance)&&variance.every(Number.isFinite)?variance:[1e-4,1e-4,1e-3];
    const v=raw.map(value=>value*SHAPE_INFLATION);
    if(!this.axes||this.time===null||timestamp-this.time>this.maxGapMs||timestamp<this.time){
      this.axes=[0,1,2].map(k=>{const axis=new Axis(this.jerk);axis.reset(position[k],Math.max(v[k],1e-8));return axis;});
      this.time=timestamp;this.quaternion=quaternion?[...quaternion]:null;this.started=true;return true;
    }
    const dt=clamp((timestamp-this.time)/1000,1e-4,this.maxGapMs/1000);
    // Adaptive process noise. A measurement that lands far outside the predicted spread means the
    // hand is accelerating in a way the constant-acceleration model did not predict -- a punch
    // starting. Raise the process noise for that step so the filter snaps onto it, then decay back
    // so the resting hand returns to being quiet.
    let nis=0;
    for(let k=0;k<3;k++){
      const axis=this.axes[k],predicted=axis.at(dt)[0],s2=axis.P[0]+Math.max(v[k],1e-8);
      nis+=(position[k]-predicted)**2/Math.max(s2,1e-12)/3;
    }
    // `trigger` is a normalised-innovation threshold: 9 means the measurement has to land ~3
    // sigma outside the prediction before the filter loosens, so ordinary noise never boosts it.
    this.boost=clamp(Math.max(this.boost*.55,Math.sqrt(Math.max(nis-this.trigger,0))),1,this.agility);
    for(let k=0;k<3;k++){
      this.axes[k].q=(this.jerk*this.boost)**2;
      this.axes[k].step(dt);this.axes[k].correct(position[k],Math.max(v[k],1e-8));
    }
    this.time=timestamp;
    if(quaternion)this.quaternion=this.quaternion?slerp(this.quaternion,quaternion,1-Math.exp(-dt/this.orientationTau)):[...quaternion];
    return true;
  }
  // Non-mutating evaluation. Render calls this every animation frame with `now`; strike detection
  // calls it with the capture timestamp so impact speed is read at the contact instant, not lagged.
  at(timestamp){
    if(!this.axes)return null;
    const dt=clamp(((Number.isFinite(timestamp)?timestamp:this.time)-this.time)/1000,-.05,.2);
    const s=this.axes.map(axis=>axis.at(dt));
    return {position:s.map(v=>v[0]),velocity:s.map(v=>v[1]),acceleration:s.map(v=>v[2]),quaternion:this.quaternion};
  }
  get speed(){const s=this.at(this.time);return s?Math.hypot(...s.velocity):0;}
}
