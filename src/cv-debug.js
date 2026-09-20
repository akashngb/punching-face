import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VirtualHand} from './hands.js';
import {LINKS} from './arm-mask.js';
import maskURL from './arm-mask.js?url';
import {PovTracking} from './pov-tracking.js';
import {TargetTracking} from './target-camera.js';
import {HeadCollider,StrikeTracker,impactIntensity} from './strike-system.js';
import {normalizeHead} from './surface.js';
import './cv-debug.css';

const root=document.getElementById('cv-debug');
root.innerHTML=`
  <section id="stage" class="stage">
    <div id="mesh"></div><canvas id="arms" aria-hidden="true"></canvas>
    <section class="event-log" aria-label="Impact event log" aria-live="polite">
      <h1>BODY-CAMERA IMPACT LOG</h1>
      <ol id="events"><li>Rest the phone on your stomach facing up at the screen, then raise both fists into the glowing arcs at the bottom corners until the meter fills. That pose is your guard.</li></ol>
    </section>
    <section class="settings" aria-label="Calibration">
      <label>MIN PUNCH SPEED <output id="speed-value">1.2 m/s</output><input id="minspeed" type="range" min="2" max="40" value="12"></label>
      <label>MIN PUNCH REACH <output id="reach-value">8 cm</output><input id="reach" type="range" min="2" max="30" value="8"></label>
      <label>INSTANT-FIRE DEPTH <output id="contact-value">12 cm</output><input id="contact" type="range" min="5" max="45" value="12"></label>
      <label>PUNCH AREA WIDTH <output id="area-value">22 cm</output><input id="area" type="range" min="10" max="60" value="22"></label>
      <label>TARGET CAM FOV <output id="fov-value">60°</output><input id="fov" type="range" min="40" max="100" value="60"></label>
      <details>
        <summary>First-person camera</summary>
        <label>PHONE → GUARD <output id="guard-value">25 cm</output><input id="guard" type="range" min="15" max="35" value="25"></label>
        <label>PHONE → TARGET <output id="target-value">65 cm</output><input id="target" type="range" min="40" max="120" value="65"></label>
        <button id="calibrate">Recalibrate guard</button>
        <button id="skip-calibration">Skip calibration</button>
        <button id="estimator" aria-pressed="true">Estimator: RIGID</button>
      </details>
      <pre id="readout" aria-live="off"></pre>
    </section>
    <section class="cameras" aria-label="Cameras">
      <article class="webcam" aria-label="Target camera preview">
        <video id="target-video" playsinline muted></video>
        <header>TARGET · collision authority</header>
        <footer><span id="target-status" role="status">OFF</span>
          <select id="target-source" aria-label="Target camera"><option value="">Choose camera…</option></select>
          <button id="target-mirror" aria-pressed="false" title="Set to MIRRORED if this camera delivers selfie-style flipped frames — the telltale is hooks reading as the wrong hand and landing on the wrong cheek.">Feed: RAW</button>
          <button id="target-camera">Start</button></footer>
      </article>
      <article class="webcam" aria-label="First-person camera preview">
        <video id="video" playsinline muted></video>
        <header>FIRST-PERSON · rendering</header>
        <footer><span id="status" role="status">OFF</span>
          <select id="camera-source" aria-label="First-person camera"><option value="">Choose camera…</option></select>
          <button id="hand-display" aria-pressed="false">Wireframe</button>
          <button id="camera">Start</button></footer>
      </article>
    </section>
  </section>`;

const $=id=>document.getElementById(id),stage=$('stage'),video=$('video'),targetVideo=$('target-video'),armCanvas=$('arms'),armContext=armCanvas.getContext('2d');
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;$('mesh').append(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#07100d');
const camera=new THREE.PerspectiveCamera(60,1,.01,10);camera.position.set(0,0,0);camera.lookAt(0,0,-.65);
const target=new THREE.Group();target.position.z=-.65;scene.add(target);const targetMeshes=[],targetSize=new THREE.Vector3();
const hands=[new VirtualHand(-1),new VirtualHand(1)];for(const hand of hands){hand.visible=false;scene.add(hand);}
const tracking=new PovTracking(video,setStatus,{maskURL,guardDepth:.25,targetDistance:.65,fovDegrees:camera.fov,onMask});
// A fist seen from behind hides its fingertips, so fistScore reads low on the very hand that is
// punching; the state machine gets a lenient fist gate and a gap tolerance matched to CV cadence.
const strikes=new StrikeTracker({minFist:.15,startSpeed:.4,maxGapMs:240});
const collider=new HeadCollider(targetMeshes,target,()=>null),strikeTracked=new Map();
let regionTarget=null;
// The target camera sits where the head is and looks back at the puncher, so it measures impact
// position and timing in its image plane instead of along its view axis. When it is running it owns
// collision outright and the first-person camera drops to rendering only -- the two are never fused,
// which is what avoids needing an extrinsic calibration between a fixed laptop and a phone strapped
// to a moving person.
const targetTracking=new TargetTracking(targetVideo,setTargetStatus,{contactDepth:.12,targetWidth:.22,fovDegrees:60});
const raycaster=new THREE.Raycaster();
let impactVector=null;
let running=false,targetRunning=false,wireframeHands=false,restarting=false,targetRestarting=false,maskBitmap=null,wasCalibrating=false;

function setStatus(message){$('status').textContent=message;}
function setTargetStatus(message){$('target-status').textContent=message;}
// The head is inflated to fill the frame while fists stay real-sized; sweeping with the fist
// radius scaled up alongside the head keeps hit generosity proportional to what the player sees.
const sweepFist=sweep=>collider.sweep({...sweep,radius:(sweep.radius??.042)*target.scale.x});
function fitTarget(){
  if(!targetSize.y)return;
  const distance=-target.position.z,viewHeight=2*distance*Math.tan(THREE.MathUtils.degToRad(camera.fov)*.5),viewWidth=viewHeight*camera.aspect;
  target.scale.setScalar(Math.max(1,Math.min(viewHeight*.92/targetSize.y,viewWidth*.8/targetSize.x)));
  target.updateMatrixWorld(true);
}
function resize(){const width=stage.clientWidth,height=stage.clientHeight,pixelRatio=Math.min(devicePixelRatio,2);renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();armCanvas.width=Math.round(width*pixelRatio);armCanvas.height=Math.round(height*pixelRatio);fitTarget();}
new ResizeObserver(resize).observe(stage);resize();

function onMask(bitmap){maskBitmap?.close();maskBitmap=bitmap;}
// Composite the *live* video through the latest low-res mask every animation frame. The visible
// arm pixels move at camera rate; only the mask boundary trails inference, and the blur feather
// plus the worker's temporal smoothing keep that trailing edge unobtrusive.
let unionCanvas=null,unionContext=null;
// Composite the *live* video through the mask every animation frame — but the worker mask alone
// trails inference and goes blind exactly during fast punches (motion blur kills detection). So
// the mask drawn each frame is the union of the worker's segmentation mask and hand shapes
// rasterised at the fist's *predicted* current position, extrapolated from the last landmark
// sample and its velocity. The renderer therefore shows the punch at display rate, where the
// fist is now; the segmentation-refined edges catch up a few frames later. Only tracked (and
// therefore anchor-verified) hands draw shapes, so the prediction cannot resurrect a screen ghost.
function drawArms(now){
  const width=armCanvas.width,height=armCanvas.height;
  armContext.clearRect(0,0,width,height);
  if(!tracking.active||video.readyState<2||!video.videoWidth)return;
  const anyHand=hands.some(hand=>hand.tracked&&hand.imageCur);
  if(!maskBitmap&&!anyHand)return;
  const cover=Math.max(width/video.videoWidth,height/video.videoHeight),coverWidth=video.videoWidth*cover,coverHeight=video.videoHeight*cover;
  const offsetX=(width-coverWidth)/2,offsetY=(height-coverHeight)/2;
  if(!unionCanvas||unionCanvas.width!==width||unionCanvas.height!==height){unionCanvas=document.createElement('canvas');unionCanvas.width=width;unionCanvas.height=height;unionContext=unionCanvas.getContext('2d');}
  unionContext.clearRect(0,0,width,height);
  if(maskBitmap){
    const maskCover=Math.max(width/maskBitmap.width,height/maskBitmap.height),maskWidth=maskBitmap.width*maskCover,maskHeight=maskBitmap.height*maskCover;
    unionContext.drawImage(maskBitmap,(width-maskWidth)/2,(height-maskHeight)/2,maskWidth,maskHeight);
  }
  unionContext.fillStyle='#fff';unionContext.strokeStyle='#fff';unionContext.lineCap='round';unionContext.lineJoin='round';
  for(const hand of hands){
    if(!hand.tracked||!hand.imageCur)continue;
    const cur=hand.imageCur,prev=hand.imagePrev;
    const lead=Math.min(Math.max((now-cur.time)/1000,0),.18);
    const sampleDt=prev?Math.max((cur.time-prev.time)/1000,1/120):0;
    const gain=sampleDt?lead/sampleDt:0;
    const points=cur.points.map((point,index)=>{
      let x=point.x,y=point.y;
      if(gain&&prev){const q=prev.points[index];x+=(point.x-q.x)*gain;y+=(point.y-q.y)*gain;}
      return {x:offsetX+x*coverWidth,y:offsetY+y*coverHeight};
    });
    const span=(a,b)=>Math.hypot(points[a].x-points[b].x,points[a].y-points[b].y);
    const scale=Math.max(span(5,17),span(0,9)*1.25);
    if(!scale)continue;
    const palmX=(points[5].x+points[9].x+points[13].x+points[17].x)/4,palmY=(points[5].y+points[9].y+points[13].y+points[17].y)/4;
    unionContext.beginPath();unionContext.arc(palmX,palmY,scale*.7,0,Math.PI*2);unionContext.fill();
    unionContext.lineWidth=scale*.3;
    unionContext.beginPath();
    for(const [a,b] of LINKS){unionContext.moveTo(points[a].x,points[a].y);unionContext.lineTo(points[b].x,points[b].y);}
    unionContext.stroke();
    // wrist corridor toward the arm's frame exit, so the forearm rides along with the prediction
    const wrist=points[0];let dirX=wrist.x-palmX,dirY=wrist.y-palmY;
    const reach=Math.hypot(dirX,dirY)||1;dirX/=reach;dirY/=reach;
    unionContext.lineWidth=scale*.95;
    unionContext.beginPath();unionContext.moveTo(wrist.x,wrist.y);
    unionContext.lineTo(wrist.x+dirX*(width+height),wrist.y+dirY*(width+height));
    unionContext.stroke();
  }
  armContext.drawImage(video,offsetX,offsetY,coverWidth,coverHeight);
  armContext.globalCompositeOperation='destination-in';
  armContext.filter=`blur(${Math.max(2,Math.round(width*.008))}px)`;
  armContext.drawImage(unionCanvas,0,0);
  armContext.filter='none';armContext.globalCompositeOperation='source-over';
}
function clearArms(){maskBitmap?.close();maskBitmap=null;armContext.clearRect(0,0,armCanvas.width,armCanvas.height);}
// Guard marks, drawn with the same cover mapping as the cutout so a fist rendered inside one
// really is at that spot in the puncher camera's frame. The mark centres sit on the bottom frame
// edge, so the canvas clips each circle to the partial arc the tilted stomach camera actually
// affords: the visible top of the fist fills the arc. Outline colour tracks that hand's
// readiness; the fill brightens and the arc-meter closes as the hold runs.
function drawGuardMarks(){
  const cal=tracking.cal;
  if(!cal?.active||video.readyState<2||!video.videoWidth)return;
  const width=armCanvas.width,height=armCanvas.height;
  const cover=Math.max(width/video.videoWidth,height/video.videoHeight),coverWidth=video.videoWidth*cover,coverHeight=video.videoHeight*cover;
  const offsetX=(width-coverWidth)/2,offsetY=(height-coverHeight)/2;
  const progress=cal.progress/cal.holdMs;
  for(const hand of hands){
    const mark=tracking.guardTargets[hand.side];if(!mark)continue;
    const state=cal.hands.get(hand.side);
    const ok=state?.visible&&state.covered&&state.closed>=.15;
    const x=offsetX+mark.x*coverWidth,y=offsetY+mark.y*coverHeight,radius=tracking.targetRadius*coverHeight;
    armContext.save();
    armContext.globalAlpha=.13+.3*progress;
    armContext.fillStyle='#b9ff9e';
    armContext.beginPath();armContext.arc(x,y,radius,0,Math.PI*2);armContext.fill();
    armContext.globalAlpha=1;
    armContext.lineWidth=Math.max(3,width*.004);
    armContext.strokeStyle=ok?'#b9ff9e':state?'#ffef65':'#ffffff88';
    armContext.setLineDash(ok?[]:[12,9]);
    armContext.beginPath();armContext.arc(x,y,radius,0,Math.PI*2);armContext.stroke();
    if(cal.progress>0){
      armContext.setLineDash([]);armContext.strokeStyle='#b9ff9e';armContext.lineWidth*=1.9;
      armContext.beginPath();armContext.arc(x,y,radius,-Math.PI/2,-Math.PI/2+Math.PI*2*progress);armContext.stroke();
    }
    armContext.restore();
  }
}
function logEvent(text,kind='hit'){
  const item=document.createElement('li');item.className=kind;item.textContent=text;$('events').prepend(item);
  while($('events').children.length>8)$('events').lastElementChild.remove();
}
async function loadTarget(){
  const gltf=await new GLTFLoader().loadAsync('/reference/LeePerrySmith.glb');
  gltf.scene.traverse(object=>{if(!object.isMesh)return;normalizeHead(object.geometry);object.material=new THREE.MeshBasicMaterial({color:0x80e66f,transparent:true,opacity:.28,depthWrite:false,side:THREE.DoubleSide});targetMeshes.push(object);const wire=new THREE.LineSegments(new THREE.WireframeGeometry(object.geometry),new THREE.LineBasicMaterial({color:0xc3ffb8,transparent:true,opacity:.2,depthWrite:false}));object.add(wire);});
  target.add(gltf.scene);target.updateMatrixWorld(true);
  new THREE.Box3().setFromObject(target).getSize(targetSize);fitTarget();
  // Real punch offsets are mapped onto the mesh through its actual width, so the "punch area"
  // slider means what it says whatever head is loaded. targetSize is the mesh's LOCAL size —
  // measured before fitTarget scaled the group — and contact points are head-local, so the local
  // width is the right divisor. The old /scale here compressed every marker toward the nose by
  // the display-fit factor (~2.5x), which is much of why hit locations read as unreliable.
  targetTracking.setHeadWidth(targetSize.x);
  // Entry is solved against an ellipsoid standing in for the head, sized from the mesh itself.
  targetTracking.setHeadRadii([targetSize.x/2,targetSize.y/2,targetSize.z/2]);
  // Region labels sized from the mesh too. regionAt's fallback face-target describes a
  // real-scale reconstructed head (a ~6 cm face half-width, shoulders below the chin line);
  // against this differently-normalised bust it relabelled honest low face hits as 'shoulder' —
  // on a mesh that has no shoulders at all. Everything on this mesh is face or head.
  regionTarget={
    center:[0,0,targetSize.z*.15],
    radii:[targetSize.x*.44,targetSize.y*.48,Math.max(.045,targetSize.z*.30)],
    bottom:-targetSize.y,
    front:targetSize.z*.30,
  };
  collider.set(targetMeshes,target,()=>null);
}
loadTarget().catch(error=>logEvent('mesh error: '+error.message,'miss'));

function leaveImpactVector(event){
  const point=target.localToWorld(new THREE.Vector3(...event.point)),direction=new THREE.Vector3(...event.direction).transformDirection(target.matrixWorld).normalize(),length=.055+Math.min(4,event.speed)*.022;
  const group=new THREE.Group(),arrow=new THREE.ArrowHelper(direction,point.clone().addScaledVector(direction,-length),length,0xffef65,.018,.009);
  arrow.line.material.depthTest=false;arrow.cone.material.depthTest=false;arrow.line.renderOrder=20;arrow.cone.renderOrder=20;group.add(arrow);
  const marker=new THREE.Mesh(new THREE.SphereGeometry(.006,14,10),new THREE.MeshBasicMaterial({color:0xffef65,depthTest:false}));marker.position.copy(point);marker.renderOrder=21;group.add(marker);
  // Only the latest hit is shown. A trail of retained arrows made it impossible to tell which
  // marker belonged to the punch you just threw, which is the whole question while debugging.
  clearImpactVector();
  impactVector=group;scene.add(group);
}
function clearImpactVector(){
  if(!impactVector)return;
  impactVector.removeFromParent();
  impactVector.traverse(object=>{object.geometry?.dispose();object.material?.dispose();});
  impactVector=null;
}
// Nearest mesh point to a head-local position — the fallback when the entry ray grazes past the
// actual geometry. A linear scan over ~20k vertices runs once per rare fallback, not per frame.
function snapToMesh(entry){
  let best=null,bestDistance=Infinity;
  const inverse=new THREE.Matrix4().copy(target.matrixWorld).invert(),vertex=new THREE.Vector3();
  for(const mesh of targetMeshes){
    const positions=mesh.geometry.attributes.position;
    if(!positions)continue;
    mesh.updateWorldMatrix(true,false);
    const relative=new THREE.Matrix4().multiplyMatrices(inverse,mesh.matrixWorld);
    for(let i=0;i<positions.count;i++){
      vertex.fromBufferAttribute(positions,i).applyMatrix4(relative);
      const distance=vertex.distanceToSquared(entry);
      if(distance<bestDistance){bestDistance=distance;best=vertex.clone();}
    }
  }
  return best;
}
// A target-camera event arrives fully decided: the pipeline extracted one event per physical
// punch, solved its trajectory, and classified its type and hand from the whole window (see
// docs/target-cv-pipeline.md). Everything is already in the head's own frame. This function only
// drops the decided entry point onto the real mesh and reads off contact quality against the
// actual surface normal.
function registerTargetImpact(contact){
  if(!targetMeshes.length)return;
  target.updateWorldMatrix(true,true);
  if(contact.missed){
    logEvent(`miss: ${contact.hand??''} ${contact.mode??'punch'} passed ${(Math.hypot(contact.aim[0],contact.aim[1])*100).toFixed(0)} cm wide of the head`,'miss');
    clearImpactVector();
    return;
  }
  // Snap the pipeline's ellipsoid entry to real geometry by casting along the punch's own travel
  // direction through that point: a hook comes across and takes the cheek, an uppercut comes up
  // and takes the underside of the chin, a jab comes straight in and takes the front. Casting
  // forward for all three put them all on the nose.
  const travel=new THREE.Vector3(...contact.direction);
  if(travel.lengthSq()<1e-8)travel.set(0,0,-1);
  travel.normalize();
  const entry=new THREE.Vector3(...contact.point);
  const origin=target.localToWorld(entry.clone().addScaledVector(travel,-1.2));
  const direction=travel.clone().transformDirection(target.matrixWorld).normalize();
  raycaster.set(origin,direction);
  const hit=raycaster.intersectObjects(targetMeshes,false)[0];
  // The pipeline already decided this punch crossed the head. A raycast miss here is only the
  // ellipsoid and the mesh disagreeing about the silhouette (the ellipsoid bulges where the chin
  // narrows), so snap to the nearest real surface point instead of contradicting a decided hit
  // with a phantom miss — and never render a marker floating beside the mesh.
  const local=hit?target.worldToLocal(hit.point.clone()):(snapToMesh(entry)??entry.clone());
  // Raycaster hands back a world-space normal; the contact velocity is in head-local space. They
  // coincide only while the head is unrotated, so convert rather than rely on that.
  const inverse=new THREE.Matrix4().copy(target.matrixWorld).invert();
  const normal=hit
    ?(hit.normal?hit.normal.clone():new THREE.Vector3(0,0,1)).transformDirection(inverse).normalize()
    :travel.clone().negate();
  const velocity=new THREE.Vector3(...contact.velocity);
  const normalSpeed=Math.abs(velocity.dot(normal));
  const tangentSpeed=velocity.clone().addScaledVector(normal,-velocity.dot(normal)).length();
  // knuckleNormal is already head-frame — the pipeline owns every frame conversion.
  const knuckleNormal=contact.knuckleNormal;
  const speed=contact.speed;
  registerImpact({
    type:'impact',hand:contact.hand,point:local.toArray(),direction:velocity.clone().normalize().toArray(),
    speed,intensity:impactIntensity(speed),confidence:contact.confidence,
    region:collider.regionAt(local,regionTarget??undefined),mode:contact.mode,
    timestamp:contact.timestamp,normalSpeed,tangentSpeed,
    obliquity:knuckleNormal?Math.acos(Math.min(1,Math.max(-1,new THREE.Vector3(...knuckleNormal).dot(normal.clone().negate()))))*180/Math.PI:undefined,
    source:'target',blurAssisted:contact.stale,bridged:contact.bridged,
  });
}
function registerImpact(event){
  leaveImpactVector(event);const p=event.point,d=event.direction;
  const named=typeof event.hand==='string'?`${event.hand} ${event.mode}`:event.mode;
  const quality=Number.isFinite(event.normalSpeed)
    ?` · ${named} · ${event.normalSpeed.toFixed(1)} m/s in, ${event.tangentSpeed.toFixed(1)} m/s rake${Number.isFinite(event.obliquity)?`, ${event.obliquity.toFixed(0)}° off-square`:''}`
    :` · ${named}`;
  const tag=event.source==='target'?(event.blurAssisted?'TGT*':event.bridged?'TGT+':'TGT '):'POV ';
  logEvent(`${tag}${event.region} hit${event.blind?' (blind)':''}: [${p.map(value=>value.toFixed(3)).join(', ')}] point, [${d.map(value=>value.toFixed(2)).join(', ')}] vector, ${event.intensity.toFixed(0)}/100${quality}`);
}
function readout(){
  const lines=[];
  if(targetTracking.active){
    const d=targetTracking.debug,p=targetTracking.probe,stats=targetTracking.stats;
    // Every stage reported separately. "no hand" and "hand seen but unusable" and "hand fitted but
    // the punch was rejected" are three different faults that used to read identically.
    // Effective inference rate is the single most load-bearing number here: hooks and uppercuts are
    // only visible for the tail of their flight, so how many samples land inside that sliver decides
    // whether they register at all.
    lines.push(`TGT ${p.results} results · ${targetTracking.rate.toFixed(0)} fps · ${Math.round(targetTracking.latency||0)}ms · motion ${d.motion.toFixed(3)}`);
    lines.push(`  1 mediapipe   ${p.rawHands} hand${p.rawHands===1?'':'s'}${p.detectError?`  ERR ${p.detectError.slice(0,40)}`:''}`);
    lines.push(`  2 world lm    ${p.withWorld} usable`+(p.rawHands&&!p.withWorld?'   << worldLandmarks missing':''));
    lines.push(`  3 pose fit    ${p.fitOk} ok / ${p.fitFail} failed`+(p.lastFit?`\n      ${p.lastFit}`:''));
    lines.push(p.crudeDepth!==null
      ?`  raw depth   ${(p.crudeDepth*100).toFixed(1)}cm  closing ${p.crudeClosing.toFixed(2)}m/s   (apparent size, no fit)`
      :'  raw depth   —');
    lines.push(d.depth!==null
      ?`  fitted      depth ${(d.depth*100).toFixed(1)}cm  range ${((d.range??0)*100).toFixed(1)}cm  closing ${d.closing.toFixed(2)}m/s  ${d.key}`
      :'  fitted      — (detection runs on this line; if it is blank nothing can fire)');
    lines.push(`  punch       ${d.phase}  peak ${d.peak.toFixed(2)}m/s  closed ${(d.travel*100).toFixed(0)}cm of range`
      +(d.minDepth!==null?`  apex ${(d.minDepth*100).toFixed(0)}cm`:''));
    lines.push(`  gates       start ${targetTracking.startClosing.toFixed(1)} · peak ${targetTracking.minPeak.toFixed(1)} · reach ${(targetTracking.minTravel*100).toFixed(0)}cm`);
    lines.push(`  hits ${stats.impacts}${stats.stale?` (${stats.stale} censored)`:''}`
      +`  slots ${targetTracking.slots.size}`
      +(stats.merged?`  merged ${stats.merged}`:'')+(stats.refractory?`  suppressed ${stats.refractory}`:'')
      +(stats.rejected?`\n    reject: ${stats.rejected}`:''));
  }else lines.push('TGT  camera off');
  if(tracking.active)for(const hand of hands){
    if(!hand.tracked){lines.push(`POV ${hand.side<0?'L':'R'}  --`);continue;}
    const p=hand.samplePosition??[0,0,0],v=hand.sampleVelocity;
    lines.push(`POV ${hand.side<0?'L':'R'} ${(-p[2]*100).toFixed(1)}cm  ${(v?Math.hypot(...v):0).toFixed(2)}m/s  closed ${(hand.sampleClosed??0).toFixed(2)}`);
  }
  return lines.join('\n');
}
function frame(time){
  const now=time??performance.now();
  // Collision authority. While the target camera is live it owns impacts outright; the
  // first-person camera keeps rendering the view, gloves and arms but stops generating strikes.
  // Two cameras voting on the same event would need them fused, and fusing them would need the
  // phone->laptop transform, which changes every frame because the phone is on a moving person.
  if(targetTracking.active){
    const contact=targetTracking.tick(now);
    if(contact)registerTargetImpact(contact);
    const d=targetTracking.debug,p=targetTracking.probe;
    setTargetStatus(!p.results?'NO FRAMES':!p.rawHands?'NO HAND'
      :d.depth===null?`${p.rawHands} HAND · FIT FAILING`
      :`${(d.depth*100).toFixed(0)}cm ${d.closing>0?'▼':' '}${Math.abs(d.closing).toFixed(1)}m/s · ${targetTracking.stats.impacts} HITS`);
  }
  if(tracking.active){
    tracking.tick(now,hands,stage.clientWidth/stage.clientHeight);
    const calibrating=tracking.cal.active;
    if(wasCalibrating&&!calibrating)logEvent('guard set — return fists to the corner arcs between punches.');
    wasCalibrating=calibrating;
    for(const hand of hands){
      const was=strikeTracked.get(hand)===true;
      if(!calibrating&&!targetTracking.active&&hand.tracked&&hand.updated&&hand.calibrated){
        const event=strikes.update({hand:hand.side,position:hand.samplePosition,target:collider.targetWorld().toArray(),
          timestamp:hand.sampleTimestamp,closed:hand.sampleClosed,confidence:hand.confidence,
          // Rigid path only: filtered velocity, the knuckle-bar capsule, and the direction the
          // knuckles face. The legacy path leaves these null and StrikeTracker falls back.
          velocity:hand.sampleVelocity,axis:hand.capsuleAxis,halfLength:(hand.capsuleHalf??0)*target.scale.x,
          knuckleNormal:hand.knuckleNormal},sweepFist);
        if(event)registerImpact(event);
      }else if(!targetTracking.active&&!hand.tracked&&was){
        // Tracking loss mid-punch is the self-occluded fist arriving; sweep the blind window.
        // Only needed when the first-person camera is on its own -- the target camera sees the
        // fist best at exactly the moment this hack exists to paper over.
        const event=strikes.release(hand.side,now,sweepFist);
        if(event)registerImpact(event);
      }
      strikeTracked.set(hand,hand.tracked);
    }
    if(calibrating){
      const states=hands.map(hand=>tracking.cal.hands.get(hand.side));
      const message=!states.some(Boolean)?'FILL THE CORNER ARCS WITH YOUR FISTS'
        :states.some(state=>!state)?'SHOW YOUR OTHER FIST'
        :states.some(state=>!state.visible)?'RAISE FISTS INTO THE ARCS'
        :states.some(state=>!state.covered)?`FILL THE ARCS — ${states.map(state=>Math.round((state?.coverage??0)*100)+'%').join(' / ')}`
        :states.some(state=>state.closed<.15)?'CLOSE YOUR FISTS'
        :`HOLD… ${Math.round(tracking.cal.progress/tracking.cal.holdMs*100)}%`;
      setStatus(message);
    }else{
      const tracked=hands.filter(hand=>hand.tracked),ready=tracked.filter(hand=>hand.calibrated);
      setStatus(tracked.length?(ready.length===tracked.length?`${tracked.length} ARM${tracked.length===1?'':'S'} READY · ${Math.round(tracking.pipelineLatency||0)} ms`:'HOLD GUARD STILL'):'SHOW FISTS IN GUARD');
    }
  }
  $('readout').textContent=readout();
  const showRings=tracking.active&&tracking.cal.active;
  armCanvas.style.display=(wireframeHands&&!showRings)?'none':'block';for(const hand of hands)hand.line.visible=wireframeHands&&hand.tracked;
  if(!wireframeHands)drawArms(now);else if(showRings)armContext.clearRect(0,0,armCanvas.width,armCanvas.height);
  if(showRings)drawGuardMarks();
  renderer.render(scene,camera);requestAnimationFrame(frame);
}
frame();

// Both cameras are ordinary getUserMedia devices on this machine -- Continuity Camera makes the
// iPhone just another entry in the list -- so each picker gets the full list and you choose which
// is which. Device labels stay blank until some camera permission has been granted once, which is
// why the first Start may show "Camera 1/2/3".
async function populateCameras(){
  const devices=await tracking.cameras();
  for(const [id,placeholder,live] of [['camera-source','Choose camera…',tracking],['target-source','Choose camera…',targetTracking]]){
    const select=$(id),wanted=select.value||live.stream?.getVideoTracks()[0]?.getSettings().deviceId||'';
    select.replaceChildren(new Option(placeholder,''),...devices.map((device,index)=>new Option(device.label||`Camera ${index+1}`,device.deviceId)));
    select.value=devices.some(device=>device.deviceId===wanted)?wanted:'';
  }
}
async function startCamera(){
  $('camera').disabled=true;
  try{await tracking.start($('camera-source').value);running=true;$('camera').textContent='Stop';await populateCameras();}
  catch(error){setStatus('CAMERA ERROR');logEvent('first-person camera: '+error.message,'miss');}
  finally{$('camera').disabled=false;}
}
function stopCamera(){tracking.stop();strikes.reset();strikeTracked.clear();clearImpactVector();for(const hand of hands){hand.visible=false;hand.tracked=false;hand.pov=null;}clearArms();running=false;$('camera').textContent='Start';setStatus('OFF');}
async function startTargetCamera(){
  $('target-camera').disabled=true;
  try{
    await targetTracking.start($('target-source').value);
    targetRunning=true;$('target-camera').textContent='Stop';
    logEvent('target camera live — it now owns collision; the phone renders only.');
    await populateCameras();
  }catch(error){setTargetStatus('CAMERA ERROR');logEvent('target camera: '+error.message,'miss');}
  finally{$('target-camera').disabled=false;}
}
function stopTargetCamera(){
  targetTracking.stop();targetRunning=false;clearImpactVector();
  $('target-camera').textContent='Start';setTargetStatus('OFF');
  logEvent('target camera stopped — collision falls back to the first-person camera.');
}

$('hand-display').onclick=()=>{wireframeHands=!wireframeHands;$('hand-display').textContent=wireframeHands?'Cutout':'Wireframe';$('hand-display').setAttribute('aria-pressed',String(wireframeHands));};
$('camera').onclick=()=>running?stopCamera():startCamera();
$('camera-source').onchange=async()=>{if(!running||restarting)return;restarting=true;stopCamera();await startCamera();restarting=false;};
$('target-camera').onclick=()=>targetRunning?stopTargetCamera():startTargetCamera();
// A mirrored feed cannot be auto-detected — every geometric measurement, MediaPipe's label
// included, flips coherently — so the interpretation is a remembered user choice. Wrong-handed
// hooks landing on the wrong cheek are the telltale.
function applyMirror(mirrored,announce){
  targetTracking.setMirrored(mirrored);
  $('target-mirror').textContent=mirrored?'Feed: MIRRORED':'Feed: RAW';
  $('target-mirror').setAttribute('aria-pressed',String(mirrored));
  try{localStorage.setItem('cv-debug.target-mirrored',mirrored?'1':'0');}catch{}
  if(announce)logEvent(`target feed interpreted as ${mirrored?'MIRRORED (selfie-style)':'RAW (unmirrored)'} — hooks should now name the hand that threw them.`);
}
$('target-mirror').onclick=()=>applyMirror(!targetTracking.mirrored,true);
try{if(localStorage.getItem('cv-debug.target-mirrored')==='1')applyMirror(true,false);}catch{}
$('target-source').onchange=async()=>{if(!targetRunning||targetRestarting)return;targetRestarting=true;stopTargetCamera();await startTargetCamera();targetRestarting=false;};
$('contact').oninput=()=>{const value=Number($('contact').value);$('contact-value').textContent=value+' cm';targetTracking.setContactDepth(value/100);};
$('minspeed').oninput=()=>{const value=Number($('minspeed').value)/10;$('speed-value').textContent=value.toFixed(1)+' m/s';targetTracking.minPeak=value;targetTracking.startClosing=Math.min(value*.7,value);};
$('reach').oninput=()=>{const value=Number($('reach').value);$('reach-value').textContent=value+' cm';targetTracking.minTravel=value/100;};
$('area').oninput=()=>{const value=Number($('area').value);$('area-value').textContent=value+' cm';targetTracking.setTargetWidth(value/100);};
$('fov').oninput=()=>{const value=Number($('fov').value);$('fov-value').textContent=value+'°';targetTracking.setFov(value);};
$('calibrate').onclick=()=>{tracking.recalibrate();strikes.reset();strikeTracked.clear();clearImpactVector();};
$('skip-calibration').onclick=()=>{if(!tracking.cal.active)return;tracking.skipCalibration();logEvent('calibration skipped — guard synthesized from live hands (debug).');};
$('estimator').onclick=()=>{
  const rigid=tracking.estimator!=='rigid';
  tracking.setEstimator(rigid?'rigid':'legacy');
  $('estimator').textContent=`Estimator: ${rigid?'RIGID':'LEGACY'}`;
  $('estimator').setAttribute('aria-pressed',String(rigid));
  strikes.reset();strikeTracked.clear();clearImpactVector();
  for(const hand of hands){hand.pov=null;hand.sampleVelocity=null;hand.capsuleAxis=null;hand.knuckleNormal=null;hand.fistSpan=0;}
  logEvent(`estimator → ${rigid?'rigid 6-DOF fit':'legacy per-landmark unprojection'}`,'hit');
};
$('guard').oninput=()=>{const value=Number($('guard').value);$('guard-value').textContent=value+' cm';tracking.setGuardDepth(value/100);strikes.reset();};
$('target').oninput=()=>{const value=Number($('target').value);$('target-value').textContent=value+' cm';target.position.z=-value/100;tracking.setTargetDistance(value/100);fitTarget();strikes.reset();};
populateCameras().catch(()=>{});
navigator.mediaDevices?.addEventListener?.('devicechange',()=>populateCameras().catch(()=>{}));
window.addEventListener('pagehide',()=>{tracking.stop();targetTracking.stop();});
if(import.meta.hot)import.meta.hot.dispose(()=>{tracking.stop();targetTracking.stop();});
