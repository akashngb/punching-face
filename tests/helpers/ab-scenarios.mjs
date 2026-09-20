// Ground-truth motions for the estimator A/B. Each returns the true hand state at a time in
// seconds, so velocity and contact are differentiated from the motion itself rather than inferred
// from either estimator under test.
const smooth=x=>x*x*(3-2*x);
const clamp01=x=>Math.min(1,Math.max(0,x));

export const GUARD_DEPTH=.25;
// The synthetic hand's +y axis runs wrist->knuckles, so at zero rotation the knuckles point up.
// pitch = -90deg turns them down-range, which is the resting orientation for a straight punch.
export const FORWARD=-Math.PI/2;

// Head geometry mirrors the debug scene: LeePerrySmith normalised to 0.28 m tall, scaled 2.47x by
// fitTarget and pushed to 0.65 m. Its front surface therefore sits at 0.42 m of depth.
export const HEAD={distance:.65,scale:2.47,radii:[.150,.140,.0915]};

export const SCENARIOS=[
  {
    name:'guard hold',
    detail:'fist still at guard for 6 s — nothing should ever fire',
    duration:6,expectHit:false,
    at:t=>({position:[.06+.002*Math.sin(t*1.7),-.05+.002*Math.sin(t*1.1),-(GUARD_DEPTH+.002*Math.sin(t*.9))],
            rotation:{pitch:FORWARD+.25,yaw:-.18,roll:.05},closure:1}),
  },
  {
    name:'wrist rotation only',
    detail:'held at guard, wrist pitches through 55deg and back; the hand never translates',
    duration:4,expectHit:false,
    at:t=>({position:[.06,-.05,-GUARD_DEPTH],
            rotation:{pitch:FORWARD+.96*Math.sin(Math.PI*clamp01(t/4)*2)**2,yaw:0,roll:0},closure:1}),
  },
  {
    name:'jab',
    detail:'straight to 0.50 m in 120 ms, wrist turning over 50deg, then retract',
    duration:.62,expectHit:true,expectMode:'jab',
    at:t=>{
      const out=clamp01(t/.12),back=clamp01((t-.24)/.22),reach=smooth(out)-smooth(back)*.92;
      return {position:[.06-.02*reach,-.05+.02*reach,-(GUARD_DEPTH+.25*reach)],
              rotation:{pitch:FORWARD+.30-.30*smooth(out),yaw:-.18+.18*smooth(out),roll:.45*smooth(out)},closure:1};
    },
  },
  {
    name:'hook',
    detail:'quadratic arc around the shoulder into the right cheek, knuckles turning to face across',
    duration:.62,expectHit:true,expectMode:'hook',
    at:t=>{
      const out=clamp01(t/.15),back=clamp01((t-.26)/.24),s=smooth(out)-smooth(back)*.92;
      // Bezier: wide out to the right, then in. The chord between samples cuts well inside this.
      const S=[.34,-.05,-.22],C=[.46,-.03,-.52],E=[.10,-.02,-.60];
      const u=1-s,position=[0,1,2].map(k=>u*u*S[k]+2*s*u*C[k]+s*s*E[k]);
      // Knuckles swing from down-range to facing across the body (-x) as the hook comes round.
      return {position,rotation:{pitch:FORWARD,yaw:1.35*s,roll:.25},closure:1};
    },
  },
  {
    name:'uppercut',
    detail:'rises from below the frame with the knuckles up, 0.32 m of lift',
    duration:.62,expectHit:true,expectMode:'uppercut',
    at:t=>{
      const out=clamp01(t/.14),back=clamp01((t-.25)/.23),s=smooth(out)-smooth(back)*.92;
      return {position:[.05,-.30+.32*s,-(GUARD_DEPTH+.23*s)],
              rotation:{pitch:-.30*s,yaw:0,roll:0},closure:1};
    },
  },
];

export function trueVelocity(scenario,t,h=.002){
  const a=scenario.at(Math.max(0,t-h)).position,b=scenario.at(t+h).position;
  return [0,1,2].map(k=>(b[k]-a[k])/((t+h)-Math.max(0,t-h)));
}
