// A/B: legacy per-landmark unprojection vs rigid 6-DOF fit.
//
// Both systems are driven from the SAME noisy observations, at the same inference cadence and the
// same pipeline latency, into the same StrikeTracker and the same HeadCollider. The only thing that
// differs is how image landmarks become a 3D fist. Ground truth comes from the scenario itself, so
// neither estimator is scored against the other.
//
//   node tests/ab-estimator.mjs [--trials N] [--seed N]

import * as THREE from 'three';
import {PinholeCamera,FistPoseEstimator} from '../src/fist-pose.js';
import {PoseFilter} from '../src/fist-filter.js';
import {projectPovHand,handScale} from '../src/pov-tracking.js';
import {fistScore} from '../src/physics.js';
import {HeadCollider,StrikeTracker} from '../src/strike-system.js';
import {observe,makeRandom,rotate} from './helpers/synthetic-hand.mjs';
import {SCENARIOS,trueVelocity,HEAD} from './helpers/ab-scenarios.mjs';

const argv=process.argv.slice(2);
const arg=(name,fallback)=>{const i=argv.indexOf(name);return i>=0?Number(argv[i+1]):fallback;};
const TRIALS=arg('--trials',24),SEED=arg('--seed',20260919);
const INFERENCE_MS=38, LATENCY_MS=55, RENDER_MS=1000/60, WARMUP_SAMPLES=26;
const LINKS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];

const camera=new PinholeCamera({fovDegrees:60,viewAspect:16/9,sourceAspect:16/9});
const VERTEX_COUNT=new THREE.SphereGeometry(1,96,72).attributes.position.count;

// --- scene: the debug head at its real scene size -------------------------------------------
function buildScene(){
  const geometry=new THREE.SphereGeometry(1,96,72);
  geometry.scale(...HEAD.radii);geometry.computeVertexNormals();
  const mesh=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial());
  const target=new THREE.Group();target.position.z=-HEAD.distance;target.scale.setScalar(HEAD.scale);
  target.add(mesh);target.updateMatrixWorld(true);
  const collider=new HeadCollider([mesh],target,()=>null);
  const sweep=s=>collider.sweep({...s,radius:(s.radius??.042)*HEAD.scale,halfLength:(s.halfLength??0)});
  return {target,collider,sweep,vertices:geometry.attributes.position.count};
}

// --- ground truth contact --------------------------------------------------------------------
const worldRadii=HEAD.radii.map(r=>r*HEAD.scale), headCentre=[0,0,-HEAD.distance];
const ellipsoidDepth=p=>Math.sqrt([0,1,2].reduce((s,k)=>s+((p[k]-headCentre[k])/worldRadii[k])**2,0));
function trueContact(scenario){
  let best=null;
  for(let t=0;t<=scenario.duration;t+=.001){
    const state=scenario.at(t);
    // the real striking surface: the knuckle bar, with a fist's own flesh radius
    const knuckles=[[5,9,13,17].map(i=>i)].flat().map(()=>null);void knuckles;
    const d=ellipsoidDepth(state.position);
    if(!best||d<best.d)best={d,t,position:state.position};
    if(d<=1)break;
  }
  if(!best||best.d>1.25)return null;
  // project the fist centre onto the head surface -> where the blow actually lands
  const k=1/best.d;
  const point=[0,1,2].map(a=>headCentre[a]+(best.position[a]-headCentre[a])*k);
  return {time:best.t,point,speed:Math.hypot(...trueVelocity(scenario,best.t)),
          knuckleNormal:rotate([0,1,0],scenario.at(best.t).rotation)};
}

const rigiditySd=points=>{
  const rel=LINKS.map(([a,b])=>{
    const lengths=points.map(frame=>Math.hypot(frame[a][0]-frame[b][0],frame[a][1]-frame[b][1],frame[a][2]-frame[b][2]));
    const mean=lengths.reduce((s,v)=>s+v,0)/lengths.length;
    return mean>1e-6?Math.sqrt(lengths.reduce((s,v)=>s+(v-mean)**2,0)/lengths.length)/mean*100:0;
  });
  return {median:rel.slice().sort((a,b)=>a-b)[rel.length>>1],worst:Math.max(...rel)};
};

// --- one run of one system over one scenario --------------------------------------------------
function run(system,scenario,seed){
  const random=makeRandom(seed),{target,collider,sweep}=buildScene();
  const strikes=new StrikeTracker({minFist:.15,startSpeed:.4,maxGapMs:240});
  const targetWorld=[0,0,-HEAD.distance];
  const state=system==='rigid'
    ?{pose:new FistPoseEstimator({camera:camera}),filter:new PoseFilter()}
    :{guard:{samples:[],scale:0,lastCenter:null,lastScale:0},prev:null,cur:null,rendered:null};

  const impacts=[],positionErrors=[],renderedFrames=[];
  let peakSpeed=0,lastSample=null,captured=-1e9;

  // Warm-up at the guard pose, uncounted. The legacy path needs it to finish its "hold fists still"
  // calibration before the punch, and the rigid path uses it to converge its shape learner and axis
  // resolver -- the same few seconds of showing your fists that both require in the real UI.
  for(let i=0;i<WARMUP_SAMPLES;i++)applySample(system,state,observe(scenario.at(0),camera,random),-(WARMUP_SAMPLES-i)*INFERENCE_MS,scenario,0);
  if(system!=='rigid'){state.prev=null;state.cur=null;state.rendered=null;}

  for(let now=0;now<=scenario.duration*1000+LATENCY_MS;now+=RENDER_MS){
    // a new inference result becomes available LATENCY_MS after its capture instant
    if(now-LATENCY_MS>=captured+INFERENCE_MS){
      captured=now-LATENCY_MS;
      const t=captured/1000;
      if(t>=0&&t<=scenario.duration){
        const truth=scenario.at(t),o=observe(truth,camera,random);
        lastSample=applySample(system,state,o,captured,scenario,t);
        if(lastSample){
          positionErrors.push(Math.hypot(...lastSample.position.map((v,i)=>v-o.truth.centre[i])));
          const event=strikes.update({hand:-1,position:lastSample.position,target:targetWorld,timestamp:captured,
            closed:lastSample.closed,confidence:1,velocity:lastSample.velocity,axis:lastSample.axis,
            halfLength:lastSample.halfLength,knuckleNormal:lastSample.knuckleNormal},sweep);
          if(event)impacts.push(event);
          // Whatever velocity that system actually feeds impact intensity -- the filter's state for
          // the rigid path, the differenced-and-smoothed estimate for legacy.
          const tracked=strikes.hands.get(-1);
          if(tracked)peakSpeed=Math.max(peakSpeed,Math.hypot(...tracked.velocity));
        }
      }
    }
    const frame=render(system,state,now);
    if(frame)renderedFrames.push(frame);
  }
  return {impacts,positionErrors,renderedFrames,peakSpeed,vertexTests:collider.stats.vertexTests,sweeps:collider.stats.sweeps,vertices:VERTEX_COUNT,target};
}

function applySample(system,state,o,timestamp,scenario,t){
  if(system==='rigid'){
    const solved=state.pose.estimate(o.landmarks,o.worldLandmarks);
    if(!solved)return null;
    state.filter.update(solved.centre,solved.variance,timestamp,solved.quaternion);
    const filtered=state.filter.at(timestamp);
    state.offsets=solved.points.map(p=>p.map((v,k)=>v-solved.centre[k]));
    state.points=solved.points;
    const bar=[0,1,2].map(k=>solved.points[5][k]-solved.points[17][k]),barLength=Math.hypot(...bar)||1e-6;
    return {position:filtered?.position??solved.centre,velocity:filtered?.velocity??null,
      closed:solved.closure,axis:bar.map(v=>v/barLength),halfLength:barLength/2*HEAD.scale,
      knuckleNormal:solved.knuckleNormal,points:solved.points};
  }
  // legacy: guard calibration exactly as PovTracking performs it, then per-landmark unprojection
  const scale=handScale(o.landmarks),closed=fistScore(o.landmarks),guard=state.guard;
  const centre={x:(o.landmarks[5].x+o.landmarks[17].x)/2,y:(o.landmarks[5].y+o.landmarks[17].y)/2};
  const stable=guard.lastCenter&&Math.hypot(centre.x-guard.lastCenter.x,centre.y-guard.lastCenter.y)<.03&&Math.abs(scale-guard.lastScale)/scale<.12;
  if(!guard.scale&&closed>.2){
    guard.samples=stable?[...guard.samples.slice(-11),scale]:[scale];
    if(guard.samples.length>=6){const sorted=[...guard.samples].sort((a,b)=>a-b);guard.scale=sorted[sorted.length>>1];}
  }
  guard.lastCenter=centre;guard.lastScale=scale;
  const projected=projectPovHand(o.landmarks,{guardScale:guard.scale||scale,guardDepth:.25,targetDistance:.65,
    fovDegrees:60,viewAspect:16/9,sourceAspect:16/9});
  if(!projected)return null;
  state.prev=state.cur;state.cur={points:projected.points,time:timestamp};
  const c=projected.points[5].clone().add(projected.points[17]).multiplyScalar(.5).lerp(projected.points[9],.2);
  return {position:[c.x,c.y,c.z],velocity:null,closed,axis:null,halfLength:0,knuckleNormal:null,
    points:projected.points.map(p=>[p.x,p.y,p.z])};
}

// Render-rate skeleton, reproducing each system's own display path.
function render(system,state,now){
  if(system==='rigid'){
    const predicted=state.filter?.at(now);
    if(!predicted||!state.offsets)return null;
    const [px,py,pz]=predicted.position;
    return state.offsets.map(p=>[p[0]+px,p[1]+py,p[2]+pz]);
  }
  const {prev,cur}=state;if(!cur)return null;
  const lead=Math.min(Math.max((now-cur.time)/1000,0),.08),sampleDt=prev?Math.max((cur.time-prev.time)/1000,1/120):0;
  if(!state.rendered)state.rendered=cur.points.map(p=>p.clone());
  const alpha=1-Math.exp(-(RENDER_MS/1000)/.045),gain=sampleDt?lead/sampleDt:0;
  for(let i=0;i<cur.points.length;i++){
    const point=cur.points[i],r=state.rendered[i];
    let tx=point.x,ty=point.y,tz=point.z;
    if(gain){const q=prev.points[i];tx+=(point.x-q.x)*gain;ty+=(point.y-q.y)*gain;tz+=(point.z-q.z)*gain;}
    r.x+=(tx-r.x)*alpha;r.y+=(ty-r.y)*alpha;r.z+=(tz-r.z)*alpha;
  }
  return state.rendered.map(p=>[p.x,p.y,p.z]);
}

// --- report -----------------------------------------------------------------------------------
const pct=(a,q)=>{const s=a.slice().sort((x,y)=>x-y);return s.length?s[Math.min(s.length-1,Math.floor(s.length*q))]:NaN;};
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:NaN;

console.log(`\nA/B  legacy per-landmark unprojection  vs  rigid 6-DOF fit`);
console.log(`${TRIALS} trials/scenario · inference ${(1000/INFERENCE_MS).toFixed(0)} fps · pipeline latency ${LATENCY_MS} ms · render 60 fps`);
console.log(`landmark noise 2.5 px @720p · MediaPipe relative-z jitter 0.030 · worldLandmark shape jitter 4 mm + 3° rotation\n`);

const summary=[];
for(const scenario of SCENARIOS){
  const contact=trueContact(scenario);
  let truePeak=0;for(let t=0;t<=scenario.duration;t+=.002)truePeak=Math.max(truePeak,Math.hypot(...trueVelocity(scenario,t)));
  console.log(`── ${scenario.name} ──  ${scenario.detail}`);
  if(scenario.expectHit&&contact)console.log(`   truth: contact at ${(contact.time*1000).toFixed(0)} ms, ${contact.speed.toFixed(2)} m/s, peak ${truePeak.toFixed(2)} m/s`);
  else console.log(`   truth: no contact, peak ${truePeak.toFixed(2)} m/s`);
  const row={scenario:scenario.name};
  for(const system of ['legacy','rigid']){
    const posErr=[],peaks=[],hits=[],modes=[],pointErr=[],speedErr=[],rigid=[];let fired=0,vertexTests=0,sweeps=0;
    for(let trial=0;trial<TRIALS;trial++){
      const r=run(system,scenario,SEED+trial*7919);
      posErr.push(...r.positionErrors);peaks.push(r.peakSpeed);vertexTests+=r.vertexTests;sweeps+=r.sweeps;
      if(r.renderedFrames.length>8)rigid.push(rigiditySd(r.renderedFrames.slice(4)));
      if(r.impacts.length){
        fired++;const first=r.impacts[0];
        modes.push(first.mode);
        if(contact){
          const world=new THREE.Vector3(...first.point);r.target.localToWorld(world);
          pointErr.push(Math.hypot(world.x-contact.point[0],world.y-contact.point[1],world.z-contact.point[2]));
          speedErr.push(first.speed-contact.speed);
        }
      }
      hits.push(r.impacts.length);
    }
    const label=system==='legacy'?'legacy':'rigid ';
    const bone=rigid.length?{median:mean(rigid.map(r=>r.median)),worst:mean(rigid.map(r=>r.worst))}:null;
    let line=`   ${label}  pos ${(mean(posErr)*1000).toFixed(1)}mm (p95 ${(pct(posErr,.95)*1000).toFixed(0)})`;
    if(bone)line+=`  bone-sd ${bone.median.toFixed(1)}%/${bone.worst.toFixed(1)}%`;
    line+=`  peak ${mean(peaks).toFixed(2)} m/s (${((mean(peaks)/truePeak-1)*100>0?'+':'')}${((mean(peaks)/truePeak-1)*100).toFixed(0)}%)`;
    if(scenario.expectHit){
      const top=modes.sort((a,b)=>modes.filter(m=>m===b).length-modes.filter(m=>m===a).length)[0]??'—';
      const right=modes.filter(m=>m===scenario.expectMode).length;
      line+=`  hit ${fired}/${TRIALS}  mode ${top} ${right}/${TRIALS}`;
      if(pointErr.length)line+=`  point ${(mean(pointErr)*1000).toFixed(0)}mm  speed ${mean(speedErr)>0?'+':''}${mean(speedErr).toFixed(2)} m/s`;
    }else{
      line+=`  FALSE STRIKES ${hits.reduce((s,v)=>s+v,0)} over ${TRIALS} runs`;
    }
    // What the pre-broad-phase collider cost: every sweep scanned every vertex.
    const oldCost=sweeps*VERTEX_COUNT;
    line+=`  · collider ${(vertexTests/1000).toFixed(0)}k vs ${(oldCost/1000).toFixed(0)}k tests`+(oldCost?` (${(100-100*vertexTests/oldCost).toFixed(0)}% saved)`:'');
    console.log(line);
    row[system]={posErr:mean(posErr),peaks:mean(peaks),fired,vertexTests:vertexTests/TRIALS,bone};
  }
  console.log('');
  summary.push(row);
}
