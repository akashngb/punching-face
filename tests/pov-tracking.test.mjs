import test from 'node:test';
import assert from 'node:assert/strict';
import {arcCoverage,ArmAnchor,armCutout,MaskSmoother} from '../src/arm-mask.js';
import {coverPoint,handScale,matchPovHands,projectPovHand,PovTracking} from '../src/pov-tracking.js';

function hand(cx=.5,palmY=.32,size=.12){
  const points=Array.from({length:21},()=>({x:cx,y:palmY,z:0}));
  points[0]={x:cx,y:palmY+size*1.3,z:0};
  for(let finger=0;finger<5;finger++)for(let joint=0;joint<4;joint++){
    const index=1+finger*4+joint;points[index]={x:cx+(finger-2)*size*.22,y:palmY-joint*size*.22,z:-joint*.005};
  }
  return points;
}

test('first-person projection moves a shrinking fist away from the body camera without mirroring it',()=>{
  const landmarks=hand(.72,.32,.08),guard=hand(.72,.32,.16),projected=projectPovHand(landmarks,{guardScale:handScale(guard),guardDepth:.25,targetDistance:.65,fovDegrees:60,viewAspect:16/9,sourceAspect:16/9});
  assert.ok(Math.abs(projected.depth-.5)<1e-9);
  assert.ok(projected.points[9].x>0,'right side of the phone image stays right in the scene');
  assert.ok(projected.points[9].z<-.45,'smaller apparent hand is farther from the body camera');
});

test('cover projection matches the full-frame cutout crop',()=>{
  assert.deepEqual(coverPoint({x:.25,y:.4,z:0},2,1),{x:0,y:.4,z:0});
  assert.deepEqual(coverPoint({x:.4,y:.25,z:0},1,2),{x:.4,y:0,z:0});
});

test('two hand tracks keep identity when their image positions cross',()=>{
  const left={side:-1,tracked:true,imageCenter:{x:.55,y:.4}},right={side:1,tracked:true,imageCenter:{x:.45,y:.4}},detections=[{center:{x:.56,y:.4}},{center:{x:.44,y:.4}}],match=matchPovHands(detections,[left,right]);
  assert.equal(match.get(0),left);assert.equal(match.get(1),right);
});

test('arm mask keeps skin and sleeve on the landmark-seeded arm but rejects another person blob',()=>{
  const width=32,height=32,labels=new Uint8Array(width*height),landmarks=hand(.5,.28,.12);
  for(let y=5;y<height;y++)for(let x=13;x<=18;x++)labels[y*width+x]=y<17?2:4;
  for(let y=20;y<30;y++)for(let x=2;x<6;x++)labels[y*width+x]=2;
  const {alpha}=armCutout(labels,width,height,[landmarks]);
  assert.equal(alpha[9*width+16],255,'hand skin retained');
  assert.equal(alpha[27*width+16],255,'sleeve corridor retained to frame edge');
  assert.equal(alpha[25*width+3],0,'unseeded person blob rejected');
});

test('a fist the segmenter misses entirely keeps a dimmer landmark-backed core',()=>{
  const width=32,height=32,landmarks=hand(.5,.28,.12);
  const {alpha}=armCutout(new Uint8Array(width*height),width,height,[landmarks]);
  assert.ok(alpha[9*width+16]>0&&alpha[9*width+16]<255,'fist core survives at partial alpha');
  assert.equal(alpha[27*width+16],0,'corridor stays segmentation-gated');
});

test('mask smoothing attacks fast and decays slowly so flicker dims instead of deleting',()=>{
  const smoother=new MaskSmoother();
  const on=new Uint8ClampedArray([255]),off=new Uint8ClampedArray([0]);
  assert.equal(smoother.apply(on)[0],255,'first frame seeds directly');
  const faded=smoother.apply(off)[0];
  assert.ok(faded>100,'one absent frame only dims: '+faded);
  assert.ok(smoother.apply(on)[0]>200,'reappearing pixel recovers fast');
  for(let i=0;i<20;i++)smoother.apply(off);
  assert.equal(smoother.state[0],0,'sustained absence fades to zero');
});

// Closed fist: fingertips tucked back toward the wrist so fistScore reads 1.
function fist(cx=.5,cy=.6,size=.12){
  const points=Array.from({length:21},()=>({x:cx,y:cy,z:0}));
  points[0]={x:cx,y:cy+size*1.3,z:0};
  const curl=[0,-.18,-.1,.05];
  for(let finger=0;finger<5;finger++)for(let joint=0;joint<4;joint++){
    points[1+finger*4+joint]={x:cx+(finger-2)*size*.22,y:cy+curl[joint]*size,z:joint>=2?-.02:0};
  }
  return points;
}

test('corner arcs calibrate when the cutout fills them, gaps allowed',()=>{
  const tracking=new PovTracking({videoWidth:1280,videoHeight:720},()=>{},{estimator:'legacy',holdMs:300});
  const hands=[{side:-1},{side:1}],targets=tracking.guardTargets;
  const guard=side=>fist(targets[side].x,targets[side].y-.07,.35);
  let t=1000;
  // Fists posed correctly but the mask barely reaches the arcs: no progress.
  for(let i=0;i<10;i++)tracking.applyResults({landmarks:[guard('-1'),guard('1')],coverage:{'-1':.2,'1':.2},handedness:[],timestamp:t+=50},hands,16/9,t);
  assert.equal(tracking.cal.active,true,'sparse coverage never calibrates');
  assert.equal(tracking.cal.progress,0);
  // Same pose with the arcs mostly filled — 60% is plenty, gaps are fine.
  assert.ok(guard('-1')[0].y>1,'the wrist really is off-frame');
  for(let i=0;i<12;i++)tracking.applyResults({landmarks:[guard('-1'),guard('1')],coverage:{'-1':.6,'1':.6},handedness:[],timestamp:t+=50},hands,16/9,t);
  assert.equal(tracking.cal.active,false,'filling both arcs completes the ritual');
  assert.ok(tracking.guard.get(-1).scale>0&&tracking.guard.get(1).scale>0,'guard scale captured per hand');
  assert.ok(hands[0].calibrated&&hands[1].calibrated);
  assert.deepEqual(hands[1].guardCenter,targets['1'],'the arc becomes the reacquisition anchor');
});

test('a fist still below the frame edge cannot set the guard even with arc coverage',()=>{
  const tracking=new PovTracking({videoWidth:1280,videoHeight:720},()=>{},{estimator:'legacy',holdMs:300});
  const hands=[{side:-1},{side:1}],targets=tracking.guardTargets;
  let t=1000;
  // Left knuckles have not cleared the bottom edge; the mask lighting the arc is not enough.
  for(let i=0;i<15;i++)tracking.applyResults({landmarks:[fist(targets['-1'].x,1.02,.35),fist(targets['1'].x,targets['1'].y-.07,.35)],coverage:{'-1':.7,'1':.7},handedness:[],timestamp:t+=50},hands,16/9,t);
  assert.equal(tracking.cal.hands.get(-1)?.visible,false,'knuckle row off screen');
  assert.equal(tracking.cal.progress,0,'the meter never fills');
  assert.equal(tracking.cal.active,true);
});

test('arc coverage counts only the on-frame slice of a corner arc',()=>{
  const width=32,height=32,alpha=new Uint8ClampedArray(width*height);
  const target={x:.14,y:.78},radius=.32;
  assert.equal(arcCoverage(alpha,width,height,target,radius),0,'empty mask covers nothing');
  // Light the whole bottom-left corner: full coverage despite most of the circle being off-frame.
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(y>=height*.4&&x<=width*.5)alpha[y*width+x]=255;
  assert.equal(arcCoverage(alpha,width,height,target,radius),1,'fully lit arc reads 100%');
  // Half-dim it: coverage tracks the lit fraction.
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)if((x+y)%2)alpha[y*width+x]=0;
  const half=arcCoverage(alpha,width,height,target,radius);
  assert.ok(half>.3&&half<.7,'checkerboard reads near half: '+half);
});

test('an established arm survives on segmentation alone when hand detection drops',()=>{
  const width=32,height=32,labels=new Uint8Array(width*height),landmarks=hand(.5,.28,.12);
  for(let y=5;y<height;y++)for(let x=13;x<=18;x++)labels[y*width+x]=y<17?2:4;
  const seeded=armCutout(labels,width,height,[landmarks]).alpha;
  assert.equal(seeded[9*width+16],255,'landmarks bootstrap the arm');
  // Detection drops for a frame: no hands at all, but the segmenter still sees the arm.
  const sustained=armCutout(labels,width,height,[],seeded).alpha;
  assert.equal(sustained[9*width+16],255,'hand region persists');
  assert.equal(sustained[27*width+16],255,'sleeve corridor persists');
  // No support and no hands: nothing appears from segmentation alone.
  const cold=armCutout(labels,width,height,[]).alpha;
  assert.ok(cold.every(value=>value===0),'person pixels never bootstrap themselves');
  // Support without person pixels: the arm left the frame, the mask follows it out.
  const gone=armCutout(new Uint8Array(width*height),width,height,[],seeded).alpha;
  assert.ok(gone.every(value=>value===0),'sustain requires live segmentation');
});

// Corner-anchored anti-loop: the below-chin camera sees the laptop screen, so the app's own
// rendered arms reappear mid-frame. Real arms are corner-connected; ghosts are floating islands.
test('corner anchoring keeps the real arm and rejects the on-screen ghost',()=>{
  const width=32,height=32,labels=new Uint8Array(width*height);
  // Real arm: skin column rising from the bottom-left corner strip.
  for(let y=8;y<height;y++)for(let x=2;x<=7;x++)labels[y*width+x]=2;
  // Ghost: a skin patch floating mid-frame — the laptop screen's copy of a fist.
  for(let y=10;y<16;y++)for(let x=17;x<=22;x++)labels[y*width+x]=2;
  const realHand=hand(.15,.35,.12),ghostHand=hand(.62,.38,.1);
  const anchor=new ArmAnchor();
  const {alpha,anchored}=armCutout(labels,width,height,[realHand,ghostHand],null,anchor);
  assert.deepEqual(anchored,[true,false],'only the corner-connected hand is real');
  assert.equal(alpha[12*width+4],255,'real arm rendered');
  assert.ok(alpha.slice(12*width+17,12*width+23).every(value=>value===0),'ghost patch contributes nothing');
});

test('temporal credit carries a corner-connected arm through a segmentation gap, then expires',()=>{
  const width=32,height=32,anchor=new ArmAnchor({maxCredit:3});
  const full=new Uint8Array(width*height),cut=new Uint8Array(width*height);
  for(let y=8;y<height;y++)for(let x=2;x<=7;x++)full[y*width+x]=2;
  // Same arm with its neck severed: rows 24-27 lost by the segmenter.
  for(let y=8;y<height;y++)for(let x=2;x<=7;x++)if(y<24||y>27)cut[y*width+x]=2;
  const arm=[hand(.15,.35,.12)];
  assert.equal(armCutout(full,width,height,arm,null,anchor).anchored[0],true,'established while whole');
  for(let i=0;i<3;i++)assert.equal(armCutout(cut,width,height,arm,null,anchor).anchored[0],true,'credit bridges the gap: frame '+i);
  assert.equal(armCutout(cut,width,height,arm,null,anchor).anchored[0],false,'credit exhausted, upper blob no longer trusted');
});

test('image samples for render-rate prediction carry velocity and reset on reacquisition',()=>{
  const tracking=new PovTracking({videoWidth:1280,videoHeight:720},()=>{},{estimator:'legacy',holdMs:300});
  const hands=[{side:-1},{side:1}],targets=tracking.guardTargets;
  let t=1000;
  const pose=side=>fist(targets[side].x,targets[side].y-.07,.35);
  tracking.applyResults({landmarks:[pose('-1'),pose('1')],handedness:[],timestamp:t},hands,16/9,t);
  assert.equal(hands[0].imagePrev,null,'first sample has no velocity partner');
  assert.ok(hands[0].imageCur.points.length===21);
  tracking.applyResults({landmarks:[pose('-1'),pose('1')],handedness:[],timestamp:t+=50},hands,16/9,t);
  assert.ok(hands[0].imagePrev&&hands[0].imageCur.time>hands[0].imagePrev.time,'consecutive samples pair up');
  // Tracking loss: after the grace window the pair must not bridge the gap with a stale velocity.
  tracking.active=true;
  tracking.tick(t+=1000,hands,16/9);
  assert.equal(hands[0].tracked,false);
  tracking.applyResults({landmarks:[pose('-1'),pose('1')],handedness:[],timestamp:t+=50},hands,16/9,t);
  assert.equal(hands[0].imagePrev,null,'reacquisition starts a fresh pair');
});

test('skipCalibration fabricates a completed ritual from live hands',()=>{
  const tracking=new PovTracking({videoWidth:1280,videoHeight:720},()=>{},{estimator:'legacy',holdMs:300});
  const hands=[{side:-1},{side:1}],targets=tracking.guardTargets;
  let t=1000;
  tracking.applyResults({landmarks:[fist(targets['-1'].x,targets['-1'].y-.07,.35),fist(targets['1'].x,targets['1'].y-.07,.35)],handedness:[],timestamp:t},hands,16/9,t);
  assert.equal(tracking.cal.active,true);
  tracking.skipCalibration();
  assert.equal(tracking.cal.active,false,'ritual bypassed');
  assert.ok(tracking.guard.get(-1).scale>0&&tracking.guard.get(1).scale>0,'guard scales synthesized');
  assert.ok(hands[0].calibrated&&hands[1].calibrated);
  assert.deepEqual(hands[1].guardCenter,targets['1'],'anchors set as a real completion would');
  // Works with no hands ever seen, too: falls back to a default scale.
  const cold=new PovTracking({videoWidth:1280,videoHeight:720},()=>{},{estimator:'legacy'});
  cold.skipCalibration();
  assert.equal(cold.cal.active,false);
  assert.ok(cold.guard.get(-1).scale>0);
});
