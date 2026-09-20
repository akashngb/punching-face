// Synthetic first-person hand observations with ground truth.
//
// The truth hand is deliberately NOT the estimator's canonical prior: it is a larger hand (92 mm
// knuckle span vs the 84 mm prior) with different finger curl, so anything that only works because
// the prior happens to be right will show up as error here.

export const SOURCE_W=1280,SOURCE_H=720;
export const NOISE_X=2.5/SOURCE_W,NOISE_Y=2.5/SOURCE_H;   // ~2.5 px landmark jitter
export const REL_Z_NOISE=.030;                            // MediaPipe relative-z jitter
export const WORLD_SHAPE_NOISE=.004, WORLD_ROT_NOISE_DEG=3;

export function makeRandom(seed=12345){
  let s=seed>>>0||1;
  const uniform=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};
  const normal=()=>{let u=0,v=0;while(!u)u=uniform();while(!v)v=uniform();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
  return {uniform,normal};
}

// Truth fist, metric, centroid at the origin. Frame: +x pinky->index, +y wrist->knuckles,
// +z palm->back of hand. 92 mm knuckle span.
export const TRUTH_FIST=(()=>{
  const p=Array.from({length:21},()=>[0,0,0]);
  p[0]=[.002,-.061,-.007];
  const k=[[5,.046,.020,.003],[9,.015,.025,.001],[13,-.015,.022,-.003],[17,-.044,.015,-.006]];
  for(const [i,x,y,z] of k)p[i]=[x,y,z];
  for(const [i,x] of [[5,.045],[9,.015],[13,-.015],[17,-.043]]){
    p[i+1]=[x*1.04,.037,-.006];p[i+2]=[x*1.01,.021,-.027];p[i+3]=[x*.94,.001,-.030];
  }
  p[1]=[.044,-.033,-.011];p[2]=[.055,-.009,-.018];p[3]=[.042,.009,-.025];p[4]=[.013,.014,-.029];
  const c=[0,1,2].map(a=>p.reduce((s,q)=>s+q[a]/21,0));
  return p.map(q=>[q[0]-c[0],q[1]-c[1],q[2]-c[2]]);
})();

export const TRUTH_OPEN=(()=>{
  const p=TRUTH_FIST.map(q=>[...q]);
  for(const [i,x] of [[5,.046],[9,.015],[13,-.015],[17,-.044]]){
    p[i+1]=[x*1.03,.056,.005];p[i+2]=[x*1.05,.082,.007];p[i+3]=[x*1.06,.104,.007];
  }
  return p;
})();

export const blendShape=(closure,fist=TRUTH_FIST,open=TRUTH_OPEN)=>
  fist.map((p,i)=>[0,1,2].map(a=>open[i][a]+(p[a]-open[i][a])*closure));

const rx=(p,t)=>[p[0],p[1]*Math.cos(t)-p[2]*Math.sin(t),p[1]*Math.sin(t)+p[2]*Math.cos(t)];
const ry=(p,t)=>[p[0]*Math.cos(t)+p[2]*Math.sin(t),p[1],-p[0]*Math.sin(t)+p[2]*Math.cos(t)];
const rz=(p,t)=>[p[0]*Math.cos(t)-p[1]*Math.sin(t),p[0]*Math.sin(t)+p[1]*Math.cos(t),p[2]];
export const rotate=(p,{pitch=0,yaw=0,roll=0})=>ry(rx(rz(p,roll),pitch),yaw);

// MediaPipe's world frame is y-down and (in this build) z-away-from-camera; the render frame used
// by the truth poses here is y-up, -z into the scene. `WORLD_SIGNS` converts truth -> MediaPipe,
// which is exactly the mapping AxisResolver has to rediscover.
export const WORLD_SIGNS=[1,-1,-1];

/**
 * Observe the truth hand at a pose.
 * @param pose {position:[x,y,z] render space (z negative = in front), rotation:{pitch,yaw,roll}, closure}
 * @returns {landmarks, worldLandmarks, truth:{points, centre, knuckleNormal, closure}}
 */
export function observe(pose,camera,random,{noise=true}={}){
  const {position,rotation={},closure=1}=pose;
  const shape=blendShape(closure);
  const oriented=shape.map(p=>rotate(p,rotation));
  const depth=-position[2];
  const truthPoints=oriented.map(p=>[p[0]+position[0],p[1]+position[1],p[2]+position[2]]);
  const n=noise?random.normal:()=>0;

  const landmarks=truthPoints.map(p=>{
    const d=-p[2];
    const view={x:.5+p[0]/(2*d*camera.kx),y:.5-p[1]/(2*d*camera.ky)};
    // invert the cover crop so the estimator's own cover() puts it back
    const {sourceAspect:s,viewAspect:v}=camera;
    const src=s>v?{x:.5+(view.x-.5)*v/s,y:view.y}:{x:view.x,y:.5+(view.y-.5)*s/v};
    return {x:src.x+n()*NOISE_X,y:src.y+n()*NOISE_Y,z:(d-depth)/(2*depth*camera.kx)+n()*REL_Z_NOISE};
  });

  const rerr={pitch:n()*WORLD_ROT_NOISE_DEG*Math.PI/180,yaw:n()*WORLD_ROT_NOISE_DEG*Math.PI/180};
  const centre=[0,1,2].map(a=>oriented.reduce((s,q)=>s+q[a]/21,0));
  // Emitted as {x,y,z} objects because that is what MediaPipe actually returns. Emitting arrays
  // here once hid a total failure of the real pipeline behind a fully green test suite.
  const worldLandmarks=oriented.map(p=>{
    const q=rotate([p[0]-centre[0],p[1]-centre[1],p[2]-centre[2]],rerr);
    return {x:(q[0]+n()*WORLD_SHAPE_NOISE)*WORLD_SIGNS[0],
            y:(q[1]+n()*WORLD_SHAPE_NOISE)*WORLD_SIGNS[1],
            z:(q[2]+n()*WORLD_SHAPE_NOISE)*WORLD_SIGNS[2],visibility:1};
  });

  const kn=rotate([0,1,0],rotation);
  // The estimator reports the hand's centroid, so truth must be the centroid too — the blended
  // shape is not centred on `position` for anything but a full fist.
  const truthCentre=[0,1,2].map(a=>truthPoints.reduce((sum,q)=>sum+q[a]/21,0));
  return {landmarks,worldLandmarks,truth:{points:truthPoints,centre:truthCentre,knuckleNormal:kn,closure}};
}
