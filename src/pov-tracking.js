import * as THREE from 'three';
import {fistScore} from './physics.js';
import {FistPoseEstimator,PinholeCamera} from './fist-pose.js';
import {PoseFilter} from './fist-filter.js';

const clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);

export function handScale(landmarks){
  if(landmarks?.length!==21)return 0;
  return Math.max(distance(landmarks[5],landmarks[17]),distance(landmarks[0],landmarks[9])*1.25);
}

export function coverPoint(point,sourceAspect,viewAspect){
  if(!Number.isFinite(sourceAspect)||!Number.isFinite(viewAspect)||sourceAspect<=0||viewAspect<=0)return {x:point.x,y:point.y,z:point.z};
  return sourceAspect>viewAspect
    ?{x:.5+(point.x-.5)*sourceAspect/viewAspect,y:point.y,z:point.z}
    :{x:point.x,y:.5+(point.y-.5)*viewAspect/sourceAspect,z:point.z};
}

export function projectPovHand(landmarks,{guardScale,guardDepth=.25,targetDistance=.65,fovDegrees=60,viewAspect=16/9,sourceAspect=16/9}={}){
  const scale=handScale(landmarks);if(!scale)return null;
  const depth=clamp(guardDepth*(guardScale||scale)/scale,guardDepth*.45,targetDistance+.35),tan=Math.tan(fovDegrees*Math.PI/360),palmZ=[5,9,13,17].reduce((sum,index)=>sum+(landmarks[index].z||0)/4,0);
  const points=landmarks.map(point=>{
    const screen=coverPoint(point,sourceAspect,viewAspect),zScale=2*depth*tan*viewAspect;
    const pointDepth=clamp(depth+((point.z||0)-palmZ)*zScale,guardDepth*.25,targetDistance+.45);
    return new THREE.Vector3((screen.x*2-1)*pointDepth*tan*viewAspect,(1-screen.y*2)*pointDepth*tan,-pointDepth);
  });
  return {points,depth,scale};
}

export function matchPovHands(detections,hands){
  const result=new Map();if(!detections.length)return result;
  const cost=(detection,hand)=>{
    const anchor=hand.tracked&&hand.imageCenter?hand.imageCenter:hand.guardCenter;
    return anchor?distance(detection.center,anchor):Math.abs(detection.center.x-(hand.side<0?.3:.7));
  };
  if(detections.length===1){const hand=hands.reduce((best,next)=>cost(detections[0],next)<cost(detections[0],best)?next:best,hands[0]);result.set(0,hand);return result;}
  const direct=cost(detections[0],hands[0])+cost(detections[1],hands[1]),crossed=cost(detections[0],hands[1])+cost(detections[1],hands[0]);
  result.set(0,direct<=crossed?hands[0]:hands[1]);result.set(1,direct<=crossed?hands[1]:hands[0]);return result;
}

export class PovTracking{
  // estimator:'rigid' fits a rigid 6-DOF fist to all 21 landmarks and filters it; 'legacy' is the
  // original per-landmark unprojection with depth from apparent hand size. Both are kept so the
  // two can be compared on the same camera feed.
  constructor(video,onStatus,{maskURL,onMask,guardDepth=.25,targetDistance=.65,fovDegrees=60,graceMs=170,estimator='rigid',guardTargets,targetRadius=.32,coverageMin=.45,holdMs=900}={}){
    this.estimator=estimator;this.camera=new PinholeCamera({fovDegrees,viewAspect:16/9,sourceAspect:16/9});
    this.rigid=new Map();
    // The phone lies on the user's stomach tilted toward the ceiling, so at guard the fists sit
    // half-below the bottom edge of the frame at its corners. The guard marks are therefore two
    // partial circles clipped by the bottom corners: the visible top of each fist fills its arc,
    // and nothing downstream (strikes, guard scale) engages until that ritual completes.
    // Measured off the real stomach-cam view: at guard each fist fills roughly a third of the
    // frame, clipped by its side edge as much as the bottom. The marks are correspondingly large
    // and sit into the corners, so each renders as the corner arc the fist actually occupies.
    this.guardTargets=guardTargets??{'-1':{x:.14,y:.78},'1':{x:.86,y:.78}};
    this.targetRadius=targetRadius;this.coverageMin=coverageMin;
    this.cal={active:true,progress:0,holdMs,last:0,samples:new Map(),hands:new Map()};
    this.video=video;this.onStatus=onStatus;this.maskURL=maskURL;this.onMask=onMask;this.guardDepth=guardDepth;this.targetDistance=targetDistance;this.fovDegrees=fovDegrees;this.graceMs=graceMs;
    this.guard=new Map();this.active=false;this.inFlight=0;this.frameCallback=null;this.lastTick=0;
  }
  setGuardDepth(value){if(Number.isFinite(value)&&value>0){this.guardDepth=value;this.recalibrate();}}
  setTargetDistance(value){if(Number.isFinite(value)&&value>this.guardDepth)this.targetDistance=value;}
  recalibrate(){
    this.guard.clear();
    this.cal.active=true;this.cal.progress=0;this.cal.last=0;this.cal.samples.clear();this.cal.hands.clear();
    for(const state of this.rigid.values()){state.pose.reset();state.filter.reset();}
    this.onStatus('Fill the corner arcs with your fists to set your guard.');
  }
  // Debug shortcut: fabricate a completed ritual. Each hand's guard scale comes from its live
  // sample when one exists (so the legacy depth mapping stays roughly honest) and falls back to a
  // typical at-guard apparent scale otherwise; the arc centres become the reacquisition anchors
  // exactly as a real completion would set them.
  skipCalibration(){
    const cal=this.cal;
    for(const side of Object.keys(this.guardTargets)){
      const hand=(this.handsRef??[]).find(candidate=>String(candidate.side)===side);
      const live=hand?.imageCur?handScale(hand.imageCur.points):0;
      const center={...this.guardTargets[side]};
      this.guard.set(Number(side),{scale:live||.42,center});
      if(hand){hand.calibrated=true;hand.guardCenter=center;}
    }
    cal.active=false;cal.progress=0;cal.samples.clear();cal.hands.clear();
    this.onStatus('GUARD SKIPPED — calibration bypassed for debugging.');
  }
  setEstimator(mode){
    if(mode!==this.estimator){this.estimator=mode;this.rigid.clear();for(const hand of this.handsRef??[])hand.pov=null;this.recalibrate();}
  }
  rigidState(side){
    let state=this.rigid.get(side);
    if(!state){state={pose:new FistPoseEstimator({camera:this.camera}),filter:new PoseFilter()};this.rigid.set(side,state);}
    return state;
  }
  async cameras(){return (await navigator.mediaDevices.enumerateDevices()).filter(device=>device.kind==='videoinput');}
  async start(deviceId=''){
    this.onStatus('Starting rear phone camera…');
    try{
      const video={width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:60}};
      if(deviceId)video.deviceId={exact:deviceId};else video.facingMode={ideal:'environment'};
      this.stream=await navigator.mediaDevices.getUserMedia({video,audio:false});
      this.video.srcObject=this.stream;await this.video.play();
      this.worker=new Worker('/pov-worker.js');
      await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(new Error('Phone CV models timed out while loading.')),25000);
        this.worker.onmessage=({data})=>{if(data.type==='ready'){clearTimeout(timeout);resolve();}else if(data.type==='error'){clearTimeout(timeout);reject(new Error(data.message));}};
        this.worker.onerror=event=>{clearTimeout(timeout);reject(new Error(event.message));};
        this.worker.postMessage({type:'init',origin:location.origin,maskURL:this.maskURL,guard:{targets:this.guardTargets,radius:this.targetRadius}});
      });
      this.worker.onmessage=({data})=>{
        this.inFlight=Math.max(0,this.inFlight-1);
        if(data.type==='result'){
          const {mask,...results}=data;
          if(!this.results||results.timestamp>this.results.timestamp)this.results=results;
          this.pipelineLatency=performance.now()-results.timestamp;
          if(mask)this.onMask?.(mask);
          this.scheduleFrame();
        }else if(data.type==='error'){this.onStatus(data.message);this.stop();}
      };
      this.active=true;this.recalibrate();this.scheduleFrame();
    }catch(error){this.stop();throw error;}
  }
  stop(){
    this.active=false;if(this.frameCallback!==null&&this.video.cancelVideoFrameCallback)this.video.cancelVideoFrameCallback(this.frameCallback);
    this.worker?.terminate();this.worker=null;this.inFlight=0;this.results=null;this.appliedTimestamp=0;this.frameCallback=null;this.lastTick=0;this.guard.clear();this.stream?.getTracks().forEach(track=>track.stop());this.stream=null;this.video.srcObject=null;
  }
  scheduleFrame(){
    // Two frames in flight: while the worker chews on frame N, frame N+1 is already captured and
    // queued, so inference throughput is not throttled to one full round trip per result.
    if(!this.active||!this.worker||this.inFlight>=2||this.frameCallback!==null)return;
    const post=async()=>{
      this.frameCallback=null;if(!this.active||!this.worker||this.inFlight>=2)return;if(this.video.readyState<2){this.scheduleFrame();return;}
      this.inFlight++;
      try{const timestamp=performance.now(),bitmap=await createImageBitmap(this.video);this.worker.postMessage({type:'frame',bitmap,timestamp},[bitmap]);}
      catch{this.inFlight--;this.scheduleFrame();}
    };
    this.frameCallback=this.video.requestVideoFrameCallback?this.video.requestVideoFrameCallback(post):setTimeout(post,4);
  }
  applyResults(results,hands,viewAspect,now){
    const wasTracked=new Map(hands.map(hand=>[hand,hand.tracked]));
    this.handsRef=hands;
    const sourceAspect=this.video.videoWidth&&this.video.videoHeight?this.video.videoWidth/this.video.videoHeight:16/9;
    this.camera.set({fovDegrees:this.fovDegrees,viewAspect,sourceAspect});
    // The worker's anchor analysis marks hands whose arms reach the corner regions; a hand it
    // could not anchor is the laptop screen's copy of an arm and must never drive tracking.
    const anchored=results.anchored;
    const detections=results.landmarks
      .map((landmarks,index)=>({landmarks,index,scale:handScale(landmarks),center:{x:(landmarks[5].x+landmarks[17].x)/2,y:(landmarks[5].y+landmarks[17].y)/2}}))
      .filter(detection=>detection.scale&&(anchored?anchored[detection.index]!==false:true))
      .sort((a,b)=>b.scale-a.scale)
      .slice(0,hands.length);
    const assignments=matchPovHands(detections,hands);
    if(this.cal.active)this.cal.hands.clear();
    assignments.forEach((hand,index)=>{
      const detection=detections[index],scale=detection.scale;
      const closed=fistScore(detection.landmarks);
      if(this.cal.active){
        const ring=this.guardTargets[hand.side];
        if(ring){
          // Readiness is judged on what the user can see: how much of the arc the rendered
          // cutout actually fills (measured by the worker on the mask itself), plus the knuckle
          // row having cleared into frame and the hand reading closed. Landmark-extent size
          // checks are gone — estimated off-frame joints made honest guard fists look oversized.
          const knuckles=[5,9,13,17].map(i=>detection.landmarks[i]);
          this.cal.hands.set(hand.side,{
            visible:knuckles.filter(point=>point.x>=.01&&point.x<=.99&&point.y>=.02&&point.y<=.985).length>=2,
            covered:(results.coverage?.[hand.side]??0)>=this.coverageMin,
            coverage:results.coverage?.[hand.side]??0,
            closed,scale,
          });
        }
      }
      // Raw image-space samples power the page's render-rate mask prediction: with these plus
      // velocity, the cutout can be drawn where the fist IS, not where inference last saw it.
      hand.imagePrev=wasTracked.get(hand)&&hand.imageCur?hand.imageCur:null;
      hand.imageCur={points:detection.landmarks,time:results.timestamp};
      if(this.estimator==='rigid'){
        if(this.applyRigid(hand,detection,results,now,wasTracked))return;
        // No worldLandmarks (older worker) or an unsolvable frame: fall through to the legacy path
        // rather than dropping the hand entirely.
      }
      const guardScale=this.guard.get(hand.side)?.scale||0;
      const projected=projectPovHand(detection.landmarks,{guardScale:guardScale||scale,guardDepth:this.guardDepth,targetDistance:this.targetDistance,fovDegrees:this.fovDegrees,viewAspect,sourceAspect});if(!projected)return;
      let pose=hand.pov;
      // A hand that was on the rigid path and lost its worldLandmarks must not keep the rigid
      // render state, or tick() would go on drawing stale offsets.
      if(!pose||pose.mode==='rigid')pose=hand.pov={prev:null,cur:null,rendered:null};
      pose.mode='legacy';
      pose.prev=pose.cur;pose.cur={points:projected.points,time:results.timestamp};
      hand.updated=wasTracked.get(hand)===true;
      hand.tracked=true;hand.visible=true;hand.calibrated=!!guardScale;hand.lastSeen=now;
      hand.confidence=results.handedness?.[detection.index]?.[0]?.score??1;hand.imageCenter=detection.center;
      // Strike physics reads the raw sample (position + capture timestamp), never the
      // prediction-smoothed render pose, so velocity estimates stay honest.
      hand.sampleClosed=closed;hand.sampleTimestamp=results.timestamp;
      const center=projected.points[5].clone().add(projected.points[17]).multiplyScalar(.5).lerp(projected.points[9],.2);
      hand.samplePosition=[center.x,center.y,center.z];
    });
    if(this.cal.active)this.advanceCalibration(hands,results.timestamp);
  }
  // Both fists must sit fully visible and closed inside their rings while the meter fills; any
  // break drains it faster than it fills, so a drive-by pose cannot calibrate. The captured median
  // scale is trustworthy by construction — every sample came from a whole, closed, in-ring fist —
  // and the ring position becomes the hand's reacquisition anchor.
  advanceCalibration(hands,timestamp){
    const cal=this.cal;
    const ready=hands.every(hand=>{const state=cal.hands.get(hand.side);return state?.visible&&state.covered&&state.closed>=.15;});
    const dt=cal.last?clamp(timestamp-cal.last,0,200):0;cal.last=timestamp;
    cal.progress=clamp(cal.progress+(ready?dt:-dt*1.6),0,cal.holdMs);
    if(ready)for(const hand of hands){
      let samples=cal.samples.get(hand.side);if(!samples){samples=[];cal.samples.set(hand.side,samples);}
      samples.push(cal.hands.get(hand.side).scale);if(samples.length>16)samples.shift();
    }
    if(cal.progress<cal.holdMs||hands.some(hand=>(cal.samples.get(hand.side)??[]).length<3))return;
    for(const hand of hands){
      const sorted=[...cal.samples.get(hand.side)].sort((a,b)=>a-b);
      const center={...this.guardTargets[hand.side]};
      this.guard.set(hand.side,{scale:sorted[Math.floor(sorted.length/2)],center});
      hand.calibrated=true;hand.guardCenter=center;
    }
    cal.active=false;cal.progress=0;cal.samples.clear();
    this.onStatus('GUARD SET — this is your guard. Return fists to the corner arcs between punches.');
  }
  // Rigid path: a 6-DOF fist fitted to all 21 landmarks, then a constant-acceleration filter that
  // supplies both the render-rate position and the velocity strike detection reads. The landmark
  // offsets are held fixed relative to the fitted centre, so bone lengths cannot change between
  // frames -- which is what the old per-landmark unprojection could not guarantee.
  applyRigid(hand,detection,results,now,wasTracked){
    const world=results.worldLandmarks?.[detection.index];
    if(!world||world.length!==21)return false;
    const state=this.rigidState(hand.side);
    const solved=state.pose.estimate(detection.landmarks,world);
    if(!solved)return false;
    state.filter.update(solved.centre,solved.variance,results.timestamp,solved.quaternion);
    const filtered=state.filter.at(results.timestamp);
    const centre=filtered?.position??solved.centre;
    let pose=hand.pov;
    if(!pose||pose.mode!=='rigid')pose=hand.pov={mode:'rigid',rendered:null};
    pose.offsets=solved.points.map(p=>new THREE.Vector3(p[0]-solved.centre[0],p[1]-solved.centre[1],p[2]-solved.centre[2]));
    pose.filter=state.filter;pose.time=results.timestamp;
    pose.rendered=pose.rendered??pose.offsets.map(p=>p.clone());
    hand.updated=wasTracked.get(hand)===true;
    hand.tracked=true;hand.visible=true;hand.lastSeen=now;
    // Hand size arrives with the measurement, so there is no guard ritual to wait for; the only
    // warm-up is the few frames the shape learner and axis resolver need.
    hand.calibrated=solved.converged;
    hand.confidence=results.handedness?.[detection.index]?.[0]?.score??1;
    hand.imageCenter=detection.center;
    hand.sampleClosed=solved.closure;hand.sampleTimestamp=results.timestamp;
    hand.samplePosition=[...centre];
    hand.sampleVelocity=filtered?[...filtered.velocity]:null;
    hand.knuckleNormal=solved.knuckleNormal;
    // Capsule along the knuckle bar: the striking surface, not a ball centred on the hand.
    const bar=solved.points[5].map((value,k)=>value-solved.points[17][k]);
    const barLength=Math.hypot(...bar)||1e-6;
    hand.capsuleAxis=bar.map(value=>value/barLength);
    hand.capsuleHalf=barLength/2;
    hand.fistSpan=solved.span;hand.poseResidual=solved.residual;
    return true;
  }
  tick(now,hands,viewAspect){
    if(!this.active)return;
    const frameDt=this.lastTick?clamp((now-this.lastTick)/1000,1/240,.1):1/60;this.lastTick=now;
    for(const hand of hands)hand.updated=false;
    this.scheduleFrame();
    if(this.results&&this.results.timestamp!==this.appliedTimestamp){
      this.appliedTimestamp=this.results.timestamp;
      this.applyResults(this.results,hands,viewAspect,now);
    }
    for(const hand of hands){
      // Brief dropouts (motion blur, self-occlusion) coast on the last velocity instead of
      // hiding the hand; a hand quiet past the grace window is genuinely lost.
      if(hand.tracked&&now-(hand.lastSeen??-Infinity)>this.graceMs){hand.tracked=false;hand.visible=false;hand.pov=null;}
      const pose=hand.pov;
      if(!hand.tracked||!pose)continue;
      if(pose.mode==='rigid'){
        // Position is evaluated from the filter at display time -- that prediction *is* the
        // latency compensation, so there is no separate extrapolator to amplify still-hand noise.
        const predicted=pose.filter?.at(now);
        if(!predicted||!pose.offsets)continue;
        const [px,py,pz]=predicted.position,alpha=1-Math.exp(-frameDt/.035);
        for(let i=0;i<pose.offsets.length;i++){
          const offset=pose.offsets[i],rendered=pose.rendered[i];
          rendered.x+=(offset.x+px-rendered.x)*alpha;
          rendered.y+=(offset.y+py-rendered.y)*alpha;
          rendered.z+=(offset.z+pz-rendered.z)*alpha;
        }
        hand.apply(pose.rendered,hand.sampleClosed??1);
        continue;
      }
      if(!pose.cur)continue;
      // Render-rate motion: extrapolate each landmark along its sample velocity to cover pipeline
      // latency (capped so noise cannot fling the hand), then ease the drawn pose toward that
      // target every animation frame. The skeleton moves at display rate, not inference rate.
      const {prev,cur}=pose,lead=clamp((now-cur.time)/1000,0,.08),sampleDt=prev?Math.max((cur.time-prev.time)/1000,1/120):0;
      if(!pose.rendered)pose.rendered=cur.points.map(point=>point.clone());
      const alpha=1-Math.exp(-frameDt/.045),gain=sampleDt?lead/sampleDt:0;
      for(let i=0;i<cur.points.length;i++){
        const point=cur.points[i],rendered=pose.rendered[i];
        let tx=point.x,ty=point.y,tz=point.z;
        if(gain){const q=prev.points[i];tx+=(point.x-q.x)*gain;ty+=(point.y-q.y)*gain;tz+=(point.z-q.z)*gain;}
        rendered.x+=(tx-rendered.x)*alpha;rendered.y+=(ty-rendered.y)*alpha;rendered.z+=(tz-rendered.z)*alpha;
      }
      hand.apply(pose.rendered,hand.sampleClosed??1);
    }
  }
}
