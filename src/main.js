import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { NewtonFaceDynamics } from './newton-dynamics.js';
import { FaceDynamics, clamp } from './physics.js';
import { ImpactReferences } from './impact-references.js';
import { ArmCapture } from './arm-capture.js';
import { FaceCapture } from './face-capture.js';
import { ScannedArm } from './scanned-arm.js';
import { HeadGlasses } from './head-accessories.js';
import { HeadHair, remapHairRoots } from './head-hair.js';
import { SurfaceAppearance, weldTexturedSurface } from './surface-appearance.js';
import { normalizeHead, refineSurface } from './surface.js';
import { VirtualHand, Tracking, makePhotoFace, cropFacePortrait } from './hands.js';
import { HeadCollider, StrikeTracker } from './strike-system.js';
import './style.css';

const $=id=>document.getElementById(id);
document.querySelector('#app').innerHTML=`
<header><div class="brand"><svg viewBox="0 0 32 32" fill="none"><path d="M26 7 16 2 5 8v16l11 6 10-6V14l-10-5-5 3v9l5 3 5-3v-4l-5-3" stroke="#46663a" stroke-width="2.5" stroke-linejoin="round"/></svg><span class="wordmark">PUNCHING FACE</span></div><div class="header-right"><div id="meshy-panel"><label class="meshy-check"><input id="meshy-toggle" type="checkbox"><span>Photo → 3D self</span></label><label class="meshy-check"><input id="compress-toggle" type="checkbox"><span>Compress GLB</span></label><button id="beat-yourself" class="small" disabled>Beat yourself</button></div><button id="fullscreen" class="small icon-btn" aria-label="Toggle fullscreen" title="Fullscreen">⤢</button><button id="capture-open" class="small">Capture guide ↗</button></div></header>
<main><aside class="panel left"><section class="panel-section"><h2>Face</h2><div class="face-info"><strong id="model-name">Reference head</strong><small id="model-kind">Public mesh reference</small></div><button id="scan-face" class="full primary" style="margin-bottom:7px">Record / upload head video</button><button id="import-face" class="full small" style="margin-bottom:7px">↑ Upload GLB head</button><button id="reference" class="full small">↺ Reset to reference</button><input id="face-file" type="file" accept=".glb,.json"><p id="photo-count" class="muted">Saved multiview photos</p><p id="mesh-count" class="muted">Loading reconstruction…</p></section>
<section class="panel-section"><h2>Camera</h2><button id="camera" class="full primary">Connect laptop webcam</button><video id="webcam" class="camera-preview" playsinline muted></video><p id="tracking-status" class="muted">Connect your camera, then hold your hands still in guard while they calibrate.</p><button id="calibrate" class="full small" disabled>Recalibrate guard</button><label class="controls-label" for="hand-display">Hand display</label><select id="hand-display"><option value="wireframe">Tracking wireframe</option></select><button id="scan-arms" class="full small" style="margin-top:7px">Scan my arms</button><p id="arm-appearance" class="muted">Personal arm meshes: awaiting capture.</p></section>
<section class="panel-section"><h2>Room</h2><button id="import-room" class="full small">↑ Add room panorama</button><input id="room-file" type="file" accept=".jpg,.jpeg,.png"><p id="room-label" class="muted">Studio environment · placeholder</p><div id="room-fields" class="room-fields"><label class="controls-label">Heading <output id="room-yaw-value">0°</output></label><input id="room-yaw" aria-label="Room heading" type="range" min="-180" max="180" value="0"><label class="controls-label">Scale <output id="room-scale-value">1×</output></label><input id="room-scale" aria-label="Room scale" type="range" min=".2" max="3" step=".01" value="1"><label class="controls-label">Height <output id="room-height-value">0 m</output></label><input id="room-height" aria-label="Room height" type="range" min="-3" max="3" step=".01" value="0"><button id="reset-room" class="small full">Restore studio</button></div></section></aside>
<section class="stage-shell"><div id="stage" class="stage"></div><div class="stage-top"><div class="scene-name" id="scene-name">Reference head <span>CPU Poisson surface</span></div><div id="stage-status" class="stage-status">● DEMO INPUT</div></div><div class="view-switch"><button id="view-mesh" class="active">Surface</button><button id="view-clay">Geometry</button><button id="view-wire">Wireframe</button></div><div class="reticle"></div><div id="impact-label" class="impact-label">CONTACT REGISTERED</div>
<div class="stage-bottom"><div class="hint"><span><kbd>Q</kbd>Left hook</span><span><kbd>E</kbd>Right hook</span><span><kbd>Space</kbd>Uppercut</span><span><kbd>Drag</kbd>Inspect</span></div><div class="bottom-line"><span id="view-label">FIRST-PERSON · VIRTUAL HANDS</span><span>Calibrate in guard · close fist · strike the mesh</span></div></div><div id="toast" class="toast" role="status"></div><div id="busy" class="busy-overlay"><div><div class="spinner"></div><div id="busy-text">Preparing surface…</div><p class="muted">Processed on this computer.</p></div></div></section>
<aside class="panel right"><section class="panel-section"><h2>Contact response</h2><div class="metric-grid"><div class="metric"><strong id="impacts">00</strong><small>Contacts</small></div><div class="metric"><strong><span id="speed">0.0</span><em>m/s*</em></strong><small>Last speed</small></div></div><canvas id="signal" class="signal" width="400" height="100"></canvas><div class="controls-label"><span>Peak deformation</span><output id="compression">0.0 mm</output></div><label class="controls-label">Softness <output id="softness-value">60%</output></label><input id="softness" aria-label="Softness" type="range" min="0" max="1" value=".6" step=".01"><label class="controls-label">Distance <output id="distance-value">30 cm</output></label><input id="distance" aria-label="Target distance" type="range" min=".24" max=".8" value=".30" step=".01"><div class="row"><button id="left-hook" class="small">↗ Left hook</button><button id="right-hook" class="small">Right hook ↖</button></div>
<label class="check">Hold peak deformation <input id="hold-peak" type="checkbox"></label><button id="resume-impact" class="small full">Release deformation</button><label class="check">Head recoil <input id="head-recoil" type="checkbox" checked></label><label class="check">Slow motion <input id="slow-motion" type="checkbox" checked></label><p id="region-readout" class="muted">Facial impact rig · ready</p><p id="physics-engine" class="muted">Select a reconstructed photo model.</p></section>
<section class="panel-section"><h2>Surface & rig</h2><label class="check">Wireframe <input id="wire" type="checkbox"></label><label class="check">Rig markers <input id="rig" type="checkbox"></label><label class="check">Sculpt <input id="sculpt" type="checkbox"></label><div class="row"><button id="undo" class="small">↶ Undo</button><button id="redo" class="small">Redo ↷</button></div><label class="controls-label">Jaw <output id="jaw-value">0%</output></label><input id="jaw" aria-label="Jaw opening" type="range" min="0" max="1" step=".01" value="0"><label class="controls-label">Smile <output id="smile-value">0%</output></label><input id="smile" aria-label="Lip corner pull" type="range" min="0" max="1" step=".01" value="0"><details><summary>Accessories & alignment</summary><label class="check">3D glasses <input id="glasses" type="checkbox" disabled></label><p id="eye-detail-status" class="muted" role="status"></p><label class="check">Strand hair <input id="hair-visible" type="checkbox" disabled></label><details id="hair-controls"><summary>Hair style</summary><p id="hair-description" class="muted">Rebuild a head scan with AI hair analysis to add editable strands.</p><label class="controls-label" for="hair-type">Type</label><select id="hair-type" disabled><option value="straight">Straight</option><option value="wavy">Wavy</option><option value="curly">Curly</option><option value="coily">Coily</option><option value="braided">Braided</option><option value="locs">Locs</option></select><label class="controls-label">Top length, mm</label><input id="hair-length" aria-label="Hair top length" type="range" min="1" max="450" step="1" disabled><label class="controls-label">Curl</label><input id="hair-curl" aria-label="Hair curl tightness" type="range" min="0" max="1" step=".01" disabled><label class="controls-label">Root lift, mm</label><input id="hair-lift" aria-label="Hair root lift" type="range" min="0" max="12" step=".1" disabled><label class="controls-label">Frizz</label><input id="hair-frizz" aria-label="Hair frizz" type="range" min="0" max="1" step=".01" disabled></details><label class="controls-label">Brow raise</label><input id="brow" aria-label="Brow raise" type="range" min="0" max="1" step=".01" value="0"><label class="controls-label">Lid compression</label><input id="squint" aria-label="Lid compression" type="range" min="0" max="1" step=".01" value="0"><label class="controls-label">Face heading</label><input id="face-yaw" aria-label="Face heading" type="range" min="-180" max="180" value="0"><label class="controls-label">Face tilt</label><input id="face-pitch" aria-label="Face tilt" type="range" min="-180" max="180" value="0"><button id="first-person" class="small full">Reset view</button><button id="impact-references" class="small full" style="margin-top:7px">Impact reference library</button></details></section>
<section class="panel-section"><div class="row"><button id="reset" class="small">Reset face</button><button id="export" class="small primary">Export GLB ↗</button></div><button id="save" class="small full" style="margin-top:7px">Save editable session</button><details><summary>Reconstruction evidence</summary><div id="stats-detail"></div></details></section></aside></main>
<dialog id="capture-dialog"><button class="close" id="capture-close" aria-label="Close capture guide">×</button><span class="eyebrow">Bring yourself into the scene</span><h1>Capture a face. Then a room.</h1><div class="capture-grid"><div class="capture-card"><h2>Try your face now</h2><p class="muted">A frontal image becomes a textured landmark mesh. Fast likeness preview; estimated depth and no back of head. Use Record for multiview geometry.</p><button id="snapshot" class="full primary">Use webcam portrait</button><button id="photo-import" class="full small" style="margin-top:9px">↑ Choose a face photo</button><input id="photo-file" type="file" accept="image/*"></div><div class="capture-card"><h2>Scan for fidelity</h2><p class="muted">Keep a neutral expression and even light. Have a helper move a phone around your still head, including both profiles, ears, chin, and crown. Keep overlapping views.</p><button id="record" class="full">Record a face scan</button><p class="muted">Saves face photographs, recovers camera angles, builds a connected mesh and bakes the captured texture. Astra supplies modeling advice; Newton simulates tissue contact.</p></div><div class="capture-card"><h2>Bring your own head</h2><p class="muted">Skip capture. Upload a GLB head or bust from Sketchfab, Meshy, Ready Player Me, Blender, etc. The mesh is auto-scaled and wired into the impact rig.</p><button id="glb-import" class="full primary">↑ Upload a GLB head</button><p class="muted">Multi-part meshes (hair, eyes, teeth as separate objects) are supported: the largest surface becomes the interactive face. Y-up, facing +Z works best.</p></div></div><ol><li>Capture 60–150 overlapping sharp face images with fixed exposure. A multiview capture is needed to preserve unseen features.</li><li>Press Create 3D face. Inspect Geometry and Wireframe to check depth. Gray areas mark estimated, unseen head surfaces.</li><li>Import a 360° room panorama for the surrounding view. A panorama supplies rotation only; it does not measure room depth.</li></ol><p class="note">Your laptop camera drives virtual hand articulation. A room scan supplies novel background views; it cannot reveal your hands’ hidden surfaces. Capture each arm separately to reconstruct its appearance. Monocular depth and automatic rigging remain estimates.</p><div id="capture-result" class="muted" role="status"></div></dialog>`;

const scene=new THREE.Scene();scene.background=new THREE.Color('#f6f7f2');scene.fog=new THREE.Fog('#f6f7f2',3,9);
const camera=new THREE.PerspectiveCamera(50,1,.01,30);camera.position.set(0,0,0);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance',preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
$('stage').appendChild(renderer.domElement);

const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,-.30);controls.enablePan=false;controls.enableDamping=true;controls.minDistance=.22;controls.maxDistance=1.6;controls.maxPolarAngle=Math.PI*.9;
const ambient=new THREE.HemisphereLight(0xffffff,0xdcdfd6,2.7);scene.add(ambient);
const key=new THREE.DirectionalLight(0xfff2e0,2.4);key.position.set(-1,1.6,1);scene.add(key);
const rim=new THREE.DirectionalLight(0xe4e9ff,1.3);rim.position.set(.8,.4,-1.3);scene.add(rim);
const fill=new THREE.DirectionalLight(0xffffff,.9);fill.position.set(.7,0,1);scene.add(fill);
const studio=new THREE.Group();scene.add(studio);
// Lightweight white room: four flush surfaces + subtle gridlines etched into
// the floor and each wall. Clear 3D volume, near-zero scene weight.
function panel(x,y,z,sx,sy,sz){const o=new THREE.Mesh(new THREE.BoxGeometry(sx,sy,sz),new THREE.MeshStandardMaterial({color:0xfafbf6,roughness:1}));o.position.set(x,y,z);studio.add(o);return o;}
panel(0,-1.28,-1,8,.02,8);
panel(0,.6,-3.3,8,4,.02);
panel(-2.3,.6,-.6,.02,4,6);
panel(2.3,.6,-.6,.02,4,6);
function wallGrid(size,divisions,rotX,rotZ,px,py,pz){const g=new THREE.GridHelper(size,divisions,0x4a5548,0x8b948a);g.material.transparent=true;g.material.opacity=.9;g.material.depthWrite=false;g.rotation.set(rotX,0,rotZ);g.position.set(px,py,pz);studio.add(g);return g;}
wallGrid(8,32,0,0,0,-1.268,-1);
wallGrid(8,32,Math.PI/2,0,0,.6,-3.28);
wallGrid(6,24,0,Math.PI/2,-2.28,.6,-.6);
wallGrid(6,24,0,Math.PI/2,2.28,.6,-.6);
const target=new THREE.Group();target.position.set(0,.015,-.30);scene.add(target);
const headPivot=new THREE.Group();target.add(headPivot);
const handGroup=new THREE.Group();scene.add(handGroup);const hands=[new VirtualHand(-1),new VirtualHand(1)];hands.forEach(h=>handGroup.add(h));
const rigMarkers=new THREE.Group();headPivot.add(rigMarkers);rigMarkers.visible=false;
let mesh,wireMesh,clayMesh,dynamics,sourceBytes,sourceName='Reference head',sourceTransform,stats={},photoData=null,representation='mesh',roomSplat=null,roomBaseScale=1,roomTexture=null;
let impacts=0,lastSpeed=0,normalClock=0,peak=0,mode='demo',demo=null,toastTimer,revision=0;
let surfaceAppearance=null,headGlasses=null,headHair=null;
let meshMatchesSource=true,impactHeld=false,watchPeak=false,previousDisplacement=0;
const raycaster=new THREE.Raycaster();const screen=new THREE.Vector2();
const tracking=new Tracking($('webcam'),message=>{$('tracking-status').textContent=message;},{autoCalibrate:true,targetDistance:.30});
const strikeTracker=new StrikeTracker(),headCollider=new HeadCollider([],headPivot,()=>dynamics);
const referenceLibrary=new ImpactReferences();$('impact-references').onclick=()=>referenceLibrary.open();
const scannedArms=new Map();
const armCapture=new ArmCapture(tracking,async bundle=>{
  const arm=new ScannedArm(bundle);scannedArms.get(bundle.side)?.dispose();scannedArms.set(bundle.side,arm);scene.add(arm);tracking.setArmProfile(arm.profile);
  $('arm-appearance').textContent=`Captured meshes: ${[...scannedArms.keys()].join(' + ')}. Auto rig; inspect joints in motion.`;
  toast(`${bundle.side} arm loaded from reconstructed capture.`);
});
$('scan-arms').onclick=()=>armCapture.open();
const trace=new Array(150).fill(0);
const busy=(active,message)=>{$('busy').classList.toggle('active',active);if(message)$('busy-text').textContent=message;};
function toast(message){$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),5000);}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);}
function disposeMesh(m){if(!m)return;m.removeFromParent();m.geometry.dispose();if(m.material.map)m.material.map.dispose();m.material.dispose();}
function meshControls(available){for(const id of ['left-hook','right-hook','wire','rig','sculpt','undo','redo','jaw','smile','brow','squint','reset','export','save','hold-peak','resume-impact'])$(id).disabled=!available;}
function geometryFromData(data){
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));g.setIndex(data.indices);
  if(data.colors){const cs=new Float32Array(data.colors);const c=new THREE.Color();for(let i=0;i<cs.length;i+=3){c.setRGB(cs[i],cs[i+1],cs[i+2],THREE.SRGBColorSpace);cs.set([c.r,c.g,c.b],i);}g.setAttribute('color',new THREE.BufferAttribute(cs,3));}
  g.computeVertexNormals();return g;
}
function installGlasses(spec){
  headGlasses?.dispose();headGlasses=spec?new HeadGlasses(spec):null;
  if(headGlasses)headPivot.add(headGlasses);
  $('glasses').disabled=!headGlasses;$('glasses').checked=!!headGlasses;
}
$('glasses').onchange=()=>{if(headGlasses)headGlasses.visible=$('glasses').checked;};
const hairFields={'hair-length':'lengthMm','hair-curl':'curlTightness','hair-lift':'rootLiftMm','hair-frizz':'frizz'};
function installHair(spec){
  headHair?.dispose();headHair=spec?new HeadHair(mesh.geometry,spec):null;
  if(headHair)headPivot.add(headHair);
  for(const id of ['hair-visible','hair-type',...Object.keys(hairFields)])$(id).disabled=!headHair;
  $('hair-visible').checked=!!headHair;
  $('hair-description').textContent=spec?`Detected: ${spec.parameters.style||spec.parameters.type}. ${headHair.strandCount.toLocaleString()} ${spec.mode==='photo-strands'?'photo-guided fibers following captured locks':spec.mode==='photo-detail'?'photo-guided detail strands':'estimated strands'}.`:'Rebuild a head scan with AI hair analysis to add editable strands.';
  if(spec?.mode==='photo-strands')$('hair-description').textContent+=' Individual fibers and unseen crown detail remain estimated.';
  if(spec){$('hair-type').value=spec.parameters.type;for(const [id,key] of Object.entries(hairFields))$(id).value=spec.parameters[key]??0;}
}
function restoreAccessoryVisibility(accessories){
  if(headGlasses){headGlasses.visible=accessories?.visibility?.glasses??true;$('glasses').checked=headGlasses.visible;}
  if(headHair){headHair.visible=accessories?.visibility?.hair??true;$('hair-visible').checked=headHair.visible;}
}
$('hair-visible').onchange=()=>{if(headHair)headHair.visible=$('hair-visible').checked;};
for(const [id,key] of Object.entries(hairFields))$(id).onchange=()=>{if(headHair){headHair.rebuild(mesh.geometry,{[key]:Number($(id).value)});revision++;}};
$('hair-type').onchange=()=>{if(headHair){const type=$('hair-type').value,curl={straight:0,wavy:.35,curly:.65,coily:.9,braided:.6,locs:.6}[type];headHair.rebuild(mesh.geometry,{type,curlTightness:curl});$('hair-curl').value=curl;revision++;}};

function installMesh(g,material,meta={}){
  installGlasses(null);installHair(null);
  dynamics?.dispose?.();meshControls(true);capturedFaceId=null;
  if(g.attributes.position.count<4000){const refined=refineSurface(g,2);g.dispose();g=refined;}
  surfaceAppearance?.dispose();surfaceAppearance=null;
  impactHeld=false;watchPeak=false;
  meshMatchesSource=true;$('view-mesh').disabled=false;$('view-wire').disabled=false;$('view-clay').disabled=false;
  if(wireMesh){wireMesh.removeFromParent();wireMesh.material.dispose();wireMesh=null;}if(clayMesh){clayMesh.removeFromParent();clayMesh.material.dispose();clayMesh=null;}
  disposeMesh(mesh);mesh=new THREE.Mesh(g,material??new THREE.MeshStandardMaterial({vertexColors:true,roughness:.86,metalness:0,side:THREE.DoubleSide}));mesh.name='Editable face';headPivot.add(mesh);clayMesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0xbfc2bd,roughness:.8,side:THREE.DoubleSide}));clayMesh.visible=false;headPivot.add(clayMesh);
  $('physics-engine').textContent='Preview springs + facial impact rig';dynamics=new FaceDynamics(g);dynamics.softness=Number($('softness').value);
  headCollider.set([mesh],headPivot,()=>dynamics);strikeTracker.reset();
  wireMesh=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:0xb9eb9a,wireframe:true,transparent:true,opacity:.28,depthWrite:false,polygonOffset:true,polygonOffsetFactor:-1}));headPivot.add(wireMesh);
  wireMesh.visible=$('wire').checked;
  while(rigMarkers.children.length){const m=rigMarkers.children.pop();m.geometry.dispose();m.material.dispose();}
  for(const [name,x,y,z] of [['brow',-.034,.07,.062],['brow',.034,.07,.062],['squint',-.038,.035,.070],['squint',.038,.035,.070],['smile',-.026,-.025,.067],['smile',.026,-.025,.067],['jaw',0,-.077,.062]]){
    const ball=new THREE.Mesh(new THREE.SphereGeometry(.0037,12,8),new THREE.MeshBasicMaterial({color:0xcaf4a4,depthTest:false}));ball.position.set(x,y,z);ball.userData.control=name;ball.renderOrder=9;rigMarkers.add(ball);
  }
  stats=meta.stats??meta;sourceTransform=meta.transform??sourceTransform;
  $('eye-detail-status').textContent=stats.eyeDetail?.summary??'';
  $('mesh-count').textContent=`${(g.index.count/3).toLocaleString()} triangles · editable`;
  $('stats-detail').textContent=JSON.stringify(meta.stats??meta,null,2);
  $('model-name').textContent=sourceName;
  $('photo-count').textContent=Number.isFinite(stats.registeredViews)?`${stats.registeredViews} recovered photo views`:photoData?'Single photo preview':sourceName==='Reference head'?'Public reference mesh':'Imported mesh';
  document.querySelector('.footer-credit')?.classList.toggle('hidden',sourceName!=='Reference head');
  $('scene-name').replaceChildren(document.createTextNode(sourceName),Object.assign(document.createElement('span'),{textContent:stats.source==='Single image landmark proxy'?'Landmark depth estimate':'Editable mesh'}));
  for(const name of ['jaw','smile','brow','squint']){$(name).value=0;if($(name+'-value'))$(name+'-value').textContent='0%';}
  setView('mesh');revision++;peak=0;window.__labReady=true;
}
function setView(view){
  representation='mesh';
  if(mesh)mesh.visible=view!=='clay'&&!surfaceAppearance;
  if(surfaceAppearance)surfaceAppearance.visible=view!=='clay';
  if(clayMesh)clayMesh.visible=view==='clay';
  if(wireMesh)wireMesh.visible=$('wire').checked||view==='wire';
  for(const name of ['mesh','clay','wire'])$('view-'+name).classList.toggle('active',view===name);
}
async function reference(){
  busy(true,'Loading reference mesh…');
  try{const data=await fetch('/reference/face-poisson.json').then(r=>r.json());sourceName='Reference head';sourceBytes=null;photoData=null;installMesh(geometryFromData(data),null,data);$('model-kind').textContent='Public mesh · legacy spring preview';$('physics-engine').textContent='Reference springs + facial impact rig';}
  catch(e){toast(e.message);}finally{busy(false);}
}

function firstPerson(){camera.position.set(0,0,0);controls.target.set(0,.015,-Number($('distance').value));camera.lookAt(controls.target);controls.update();}
function contact(point,direction,speed,source,mode='hook',region='face',intensity=clamp(speed*32,0,100)){
  if(!dynamics)return false;
  impactHeld=false;watchPeak=true;previousDisplacement=0;const affected=dynamics.impulse(point,direction,speed,mode);if(!affected)return false;
  impacts++;lastSpeed=speed;peak=0;$('impacts').textContent=String(impacts).padStart(2,'0');$('speed').textContent=speed.toFixed(1);
  $('impact-label').textContent=`${region.toUpperCase()} CONTACT`;$('impact-label').classList.remove('flash');void $('impact-label').offsetWidth;$('impact-label').classList.add('flash');
  window.__lastContact={point:point.toArray(),direction:direction.toArray(),speed,intensity,region,affected,source,time:performance.now()};return true;
}
function trackStrike(hand,timestamp,source){
  if(!mesh||!hand.visible||(source==='webcam'&&(!hand.tracked||!hand.calibrated)))return;
  const event=strikeTracker.update({hand:hand.side,position:hand.center.toArray(),target:headCollider.targetWorld().toArray(),timestamp,closed:hand.closed,confidence:hand.confidence||1},sweep=>headCollider.sweep(sweep));
  if(!event)return;
  const body=event.region==='shoulder'||event.region==='chest';
  contact(new THREE.Vector3(...event.point),new THREE.Vector3(...event.direction),clamp(event.speed,0,4),source,body?'body':event.mode,event.region,event.intensity);
}
// Prefer the Newton cage vertex if we have one, then the adaptive rig anchor from detectAnchors
// (so Meshy/upload heads land on the real cheek/chin), and only as a last resort the reference-frame default.
function rigGoal(index,fallback){const cage=dynamics.cage?.positions;if(cage)return new THREE.Vector3(cage[index*3],cage[index*3+1],cage[index*3+2]);const a=dynamics.impactRig?.anchors?.[index];if(a)return new THREE.Vector3(a[0],a[1],a[2]);return fallback.clone();}
function hook(side){if(!mesh||!meshMatchesSource)return;if(tracking.active){toast('Disconnect webcam to run a demo hook.');return;}firstPerson();strikeTracker.reset(side);headPivot.updateWorldMatrix(true,false);const goal=rigGoal(side<0?50:280,new THREE.Vector3(side*.045,-.005,.055));headPivot.localToWorld(goal);demo={side,type:'hook',start:performance.now()/1000,previousT:0,goal};}
function uppercut(){if(!mesh||!meshMatchesSource)return;if(tracking.active){toast('Disconnect webcam to run a demo uppercut.');return;}firstPerson();strikeTracker.reset(1);headPivot.updateWorldMatrix(true,false);const goal=rigGoal(152,new THREE.Vector3(0,-.062,.05));headPivot.localToWorld(goal);demo={side:1,type:'uppercut',start:performance.now()/1000,previousT:0,goal};}
function scriptedImpact(event){
  if(!mesh||!dynamics)return false;
  let localPoint,localDir;
  if(event.type==='uppercut'){
    localPoint=rigGoal(152,new THREE.Vector3(0,-.062,.048));
    localDir=new THREE.Vector3(0,.85,-.5).normalize();
  }else if(event.type==='jab'){
    // Nose tip (rig index 1 is Newton-cage-only); adaptive rig doesn't ship an anchor for it,
    // so fall through to the default which is close to the reference nose tip.
    localPoint=rigGoal(1,new THREE.Vector3(0,.005,.062));
    localDir=new THREE.Vector3(0,0,-1);
  }else{
    const s=event.side==='left'?-1:1;
    localPoint=rigGoal(s<0?50:280,new THREE.Vector3(s*.045,-.005,.055));
    localDir=new THREE.Vector3(-s*.75,-.05,-.6).normalize();
  }
  // Raycast from the incoming direction so the impulse lands on the mesh surface
  // rather than a raw cage anchor which may sit fractionally inside the skin.
  headPivot.updateWorldMatrix(true,false);
  const worldPoint=headPivot.localToWorld(localPoint.clone());
  const worldDir=localDir.clone().transformDirection(headPivot.matrixWorld);
  raycaster.set(worldPoint.clone().addScaledVector(worldDir,-.12),worldDir);raycaster.far=.30;
  const hit=raycaster.intersectObject(mesh,false)[0];
  const finalLocal=hit?headPivot.worldToLocal(hit.point.clone()):localPoint;
  raycaster.far=Infinity;
  const speed=clamp(event.speed??1.8,.9,3.4);
  const mode=event.type==='uppercut'?'uppercut':event.type==='jab'?'jab':'hook';
  return contact(finalLocal,localDir,speed,'demo',mode,'face');
}

function drawTrace(){const canvas=$('signal'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,400,100);ctx.strokeStyle='#d8dfd0';ctx.lineWidth=1;for(let y=25;y<100;y+=25){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(400,y);ctx.stroke();}ctx.beginPath();trace.forEach((v,i)=>{const x=i/(trace.length-1)*400,y=92-Math.min(v/.018,1)*80;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.lineWidth=2;ctx.strokeStyle='#749b54';ctx.stroke();}
let lastTime=performance.now();
function frame(time){
  const dt=Math.min((time-lastTime)/1000,1/30);lastTime=time;const now=time/1000;
  if(tracking.active){
    tracking.tick(time,hands);mode='webcam';
    const tracked=hands.filter(hand=>hand.tracked),ready=tracked.filter(hand=>hand.calibrated);
    $('tracking-status').textContent=tracked.length?(ready.length===tracked.length?`${ready.length} hand${ready.length===1?'':'s'} ready · close your fist and punch`:'Hold still in guard · calibrating…'):'Searching for your hands…';
  }
  else{
    mode='demo';const restZ=-Math.max(.12,Number($('distance').value)-.20);
    for(const hand of hands){hand.visible=true;const side=hand.side;
      if(demo&&demo.side===side){const t=(now-demo.start)/.72;
        if(t>1){demo=null;hand.demoPose(new THREE.Vector3(side*.11,-.10,restZ),0);}
        else{
          const a=clamp(t/.52,0,1),b=clamp((t-.52)/.48,0,1);const strike=Math.sin(a*Math.PI/2)*(1-b);
          const goal=demo.goal;
          // With the webcam disabled the CV pipeline can't supply a fist score. Ramp the mesh
          // into a closed fist as the strike winds up (a→1) and hold it through the impact,
          // opening back up as the follow-through decays (b→1) so the recovery reads as relaxed.
          const closedAmount=clamp(a-b*0.6,0,1);
          if(demo.type==='uppercut'){
            const x=side*.05*(1-strike)+goal.x*strike,y=-.32*(1-strike)+(goal.y-.005)*strike,z=restZ*(1-strike)+(goal.z-.02)*strike;
            hand.demoPose(new THREE.Vector3(x,y,z),closedAmount);
          }else{
            const x=side*.24*(1-strike)+(goal.x-side*.020)*strike,y=-.13*(1-strike)+(goal.y+.012)*strike,z=restZ*(1-strike)+(goal.z-.023)*strike;
            hand.demoPose(new THREE.Vector3(x,y,z),closedAmount);
          }
        }
      }else{hand.demoPose(new THREE.Vector3(side*.11,-.10+Math.sin(now*1.4+side)*.002,restZ),0);}
      trackStrike(hand,time,'demo');
    }
  }
  if(tracking.active)for(const h of hands){if(h.tracked&&h.updated)trackStrike(h,time,'webcam');else if(!h.tracked)strikeTracker.reset(h.side);}
  if(dynamics){
    if(!impactHeld){dynamics.step(dt*($('slow-motion').checked?.38:1));
      if(watchPeak&&$('hold-peak').checked&&previousDisplacement>.0005&&dynamics.impactRig.hasPeaked&&dynamics.maxDisplacement<previousDisplacement){impactHeld=true;watchPeak=false;}
      previousDisplacement=dynamics.maxDisplacement;
    }
    const recoil=$('head-recoil').checked?.55:0;headPivot.rotation.set(dynamics.recoil.x*recoil+Number($('face-pitch').value)*Math.PI/180,dynamics.recoil.y*recoil+Number($('face-yaw').value)*Math.PI/180,dynamics.recoil.z*recoil);
    normalClock++;if(normalClock%3===0){mesh.geometry.computeVertexNormals();mesh.geometry.computeBoundingSphere();trace.push(dynamics.maxDisplacement);trace.shift();drawTrace();}
    peak=Math.max(peak,dynamics.maxDisplacement);$('compression').textContent=(peak*1000).toFixed(1)+' mm';if(normalClock%6===0)$('region-readout').textContent=Object.entries(dynamics.regionPeaks).map(([name,value])=>name+': '+(value*1000).toFixed(1)+' mm').join(' · ');
  }
  if(surfaceAppearance)surfaceAppearance.updateSurface(mesh.geometry.attributes.position.array,mesh.geometry.attributes.normal.array);
  if(headHair)headHair.updateSurface(mesh.geometry);
  for(const h of hands){
    const arm=scannedArms.get(h.side<0?'left':'right');
    if(arm)arm.updateFromHand(h);
    h.line.visible=!arm;
  }
  if(!scannedArms.size)$('view-label').textContent='FIRST-PERSON · TRACKING WIREFRAMES';
  if(scannedArms.size&&normalClock%15===0){
    const visible=[...scannedArms].filter(([,arm])=>arm.visible).map(([side])=>side);
    $('arm-appearance').textContent=!tracking.active?'Captured arms loaded. Connect the webcam to drive them.':!tracking.bodyFrame?'Captured arms loaded. Calibrate with your face, shoulders, elbows and wrists visible.':visible.length?`Following your ${visible.join(' and ')} arm. Capture proportions retained; pose and skin weights are estimates.`:'Personal arms hidden: show the matching shoulder, elbow, wrist and hand clearly.';
    $('view-label').textContent='FIRST-PERSON · CAPTURED ARM MESHES';
  }
  controls.update();renderer.render(scene,camera);
}
renderer.setAnimationLoop(frame);
new ResizeObserver(()=>{const {width,height}=$('stage').getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();}).observe($('stage'));

$('view-mesh').onclick=()=>setView('mesh');$('view-clay').onclick=()=>setView('clay');$('view-wire').onclick=()=>setView('wire');
$('wire').onchange=()=>{if(wireMesh)wireMesh.visible=$('wire').checked&&representation==='mesh';};$('rig').onchange=()=>rigMarkers.visible=$('rig').checked;
$('sculpt').onchange=()=>{controls.enabled=!$('sculpt').checked;toast($('sculpt').checked?'Drag on the face: up pulls, down pushes.':'Inspect mode enabled.');};
$('undo').onclick=()=>{if(dynamics?.undo())revision++;};$('redo').onclick=()=>{if(dynamics?.redo())revision++;};
for(const key of ['jaw','smile','brow','squint'])$(key).oninput=()=>{if(dynamics)dynamics.rig[key]=Number($(key).value);if($(key+'-value'))$(key+'-value').textContent=Math.round(Number($(key).value)*100)+'%';};
$('resume-impact').onclick=()=>{impactHeld=false;watchPeak=false;};
$('softness').oninput=()=>{if(dynamics)dynamics.softness=Number($('softness').value);$('softness-value').textContent=Math.round(Number($('softness').value)*100)+'%';};
$('distance').oninput=()=>{const distance=Number($('distance').value);target.position.z=-distance;tracking.setTargetDistance(distance);strikeTracker.reset();$('distance-value').textContent=Math.round(distance*100)+' cm';};
$('left-hook').onclick=()=>hook(-1);$('right-hook').onclick=()=>hook(1);$('reset').onclick=()=>{impactHeld=false;watchPeak=false;dynamics?.reset();for(const id of ['jaw','smile','brow','squint']){$(id).value=0;if($(id+'-value'))$(id+'-value').textContent='0%';}peak=0;revision++;};$('first-person').onclick=firstPerson;
$('reference').onclick=reference;$('import-face').onclick=()=>$('face-file').click();
$('glb-import').onclick=()=>{$('capture-dialog').close();$('face-file').click();};
window.addEventListener('keydown',e=>{if(e.repeat||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||document.querySelector('dialog[open]'))return;if(e.code==='KeyQ')hook(-1);if(e.code==='KeyE')hook(1);if(e.code==='Space'){e.preventDefault();uppercut();}if(e.code==='KeyR')$('reset').click();if(e.code==='KeyW'){$('wire').checked=!$('wire').checked;$('wire').onchange();}});

let drag=null;
function pick(e){const rect=renderer.domElement.getBoundingClientRect();screen.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(screen,camera);return raycaster.intersectObject(mesh,false)[0];}
renderer.domElement.addEventListener('pointerdown',e=>{
  if(!mesh||representation!=='mesh')return;
  if(rigMarkers.visible){const rect=renderer.domElement.getBoundingClientRect();screen.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(screen,camera);const marker=raycaster.intersectObjects(rigMarkers.children)[0];if(marker){const name=marker.object.userData.control;$(name).focus();toast(`${name} control selected — adjust its slider.`);return;}}
  if(!$('sculpt').checked)return;const hit=pick(e);if(!hit)return;dynamics.remember();drag={y:e.clientY,point:headPivot.worldToLocal(hit.point.clone())};renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener('pointermove',e=>{if(!drag)return;const delta=clamp((drag.y-e.clientY)*.00012,-.001,.001);dynamics.sculpt(drag.point,delta);drag.y=e.clientY;revision++;});
renderer.domElement.addEventListener('pointerup',()=>{drag=null;});renderer.domElement.addEventListener('pointercancel',()=>{drag=null;});

const CAMERA_WANTED_KEY='contact-camera-wanted';
async function cameraToggle(){
  if(tracking.active){tracking.stop();strikeTracker.reset();sessionStorage.removeItem(CAMERA_WANTED_KEY);$('camera').textContent='Connect laptop webcam';$('webcam').classList.remove('active');$('calibrate').disabled=true;$('tracking-status').textContent='Camera off. Virtual hands are in demo mode.';$('stage-status').textContent='● DEMO INPUT';return;}
  $('camera').disabled=true;
  try{
    await tracking.start();demo=null;firstPerson();$('camera').textContent='Disconnect webcam';$('webcam').classList.add('active');$('calibrate').disabled=false;$('stage-status').textContent='● WEBCAM · VIRTUAL POV';
    // Persist intent so a Vite HMR reload (this project reloads on every save) can bring
    // the camera back automatically instead of silently dropping the stream on the user.
    sessionStorage.setItem(CAMERA_WANTED_KEY,'1');
  }
  catch(e){toast(`Camera unavailable: ${e.message}`);$('tracking-status').textContent='Camera could not start. Allow camera access in the browser, then reconnect.';sessionStorage.removeItem(CAMERA_WANTED_KEY);}
  finally{$('camera').disabled=false;}
}
$('camera').onclick=cameraToggle;$('calibrate').onclick=()=>{try{tracking.calibrate();strikeTracker.reset();}catch(e){toast(e.message);}};
// Auto-reconnect after HMR reloads. getUserMedia proceeds without a fresh user gesture once
// this document has already been granted camera permission, so the reconnect is silent.
if(sessionStorage.getItem(CAMERA_WANTED_KEY)==='1')queueMicrotask(()=>cameraToggle().catch(()=>{}));
window.addEventListener('pagehide',()=>{tracking.stop();dynamics?.dispose?.();});
if(import.meta.hot){
  // The old module's MediaStream must be released before the new one grabs the device
  // again, or the browser hands out overlapping tracks and one ends up orphaned.
  import.meta.hot.dispose(()=>{tracking.stop();dynamics?.dispose?.();});
}

$('face-file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;e.target.value='';busy(true,'Importing face asset…');
  try{
    if(file.size>180_000_000)throw new Error('Use a GLB or editable session smaller than 180 MB.');
    const ext=file.name.split('.').pop().toLowerCase();
    if(ext==='json'){await restoreSession(JSON.parse(await file.text()));return;}
    if(ext==='glb'){
      const gltf=await new GLTFLoader().parseAsync(await file.arrayBuffer(),'');const meshes=[];gltf.scene.updateMatrixWorld(true);gltf.scene.traverse(o=>{if(o.isMesh)meshes.push(o);});
      const faces=meshes.filter(m=>!['eyeglasses','hair'].includes(m.userData.accessory));
      if(!faces.length)throw new Error('This GLB contains no meshes to import.');
      // A third-party head/bust often ships as multiple sub-meshes (hair, eyes, teeth).
      // Pick the largest by triangle count as the interactive face surface; the app's
      // physics rig works on a single geometry, so we take the dominant part.
      const m=faces.length===1?faces[0]:faces.reduce((a,b)=>((b.geometry.index?.count??b.geometry.attributes.position.count)>(a.geometry.index?.count??a.geometry.attributes.position.count)?b:a));
      if(faces.length>1)toast(`GLB has ${faces.length} meshes; using the largest (${m.name||'unnamed'}) as the face surface.`);
      const g=m.geometry.clone();g.applyMatrix4(m.matrixWorld);
      if(m.userData.coordinateSystem!=='punching-face-head-metres-v1'){
        normalizeHead(g);
      }
      sourceName=file.name;photoData=null;sourceBytes=null;
      const uploadedAnchors=m.userData.coordinateSystem==='punching-face-head-metres-v1'?null:detectAnchors(g);
      if(m.material.map&&m.userData.appearance){
        const recovered=weldTexturedSurface(g);recovered.atlas.stats=m.userData.appearanceStats;installMesh(recovered.geometry,null,m.userData.reconstruction??{source:'Imported textured mesh'});
        surfaceAppearance=new SurfaceAppearance(mesh.geometry,recovered.atlas,m.material.map.clone(),m.material.roughnessMap?.clone());headPivot.add(surfaceAppearance);g.dispose();setView('mesh');
      }else installMesh(g,m.material.clone(),m.userData.reconstruction??{source:'Imported mesh'});
      if(uploadedAnchors)dynamics.impactRig.setAnchors(uploadedAnchors);
      installGlasses(m.userData.accessories?.glasses);
      installHair(remapHairRoots(m.userData.accessories?.hair,g.attributes.position.array,mesh.geometry.attributes.position.array));restoreAccessoryVisibility(m.userData.accessories);
      for(const name of ['jaw','smile','brow','squint']){
        const index=m.morphTargetDictionary?.[name];if(index===undefined)continue;
        dynamics.rig[name]=clamp(m.morphTargetInfluences[index]||0,0,1);$(name).value=dynamics.rig[name];if($(name+'-value'))$(name+'-value').textContent=Math.round(dynamics.rig[name]*100)+'%';
      }
      $('model-kind').textContent=surfaceAppearance?'Imported textured mesh · preview physics':'Imported triangle mesh';$('physics-engine').textContent='Imported preview · no Newton cage attached';return;
    }
    throw new Error('Choose a GLB mesh or saved session JSON.');
  }catch(err){toast(err.message);console.error(err);}finally{busy(false);}
};

async function exportGLB(){
  if(!mesh)return;busy(true,'Exporting mesh and expression controls…');
  try{
    let g;if(surfaceAppearance)g=surfaceAppearance.exportGeometry(dynamics.rest,dynamics.exportMorphs());else{g=mesh.geometry.clone();g.setAttribute('position',new THREE.BufferAttribute(dynamics.rest.slice(),3));g.morphAttributes.position=dynamics.exportMorphs();g.morphTargetsRelative=true;g.computeVertexNormals();}
    const out=new THREE.Mesh(g,(surfaceAppearance?.material??mesh.material).clone());out.name='Punching Face editable face';out.updateMorphTargets();out.morphTargetInfluences=Object.values(dynamics.rig);out.userData={coordinateSystem:'punching-face-head-metres-v1',source:sourceName,appearance:surfaceAppearance?'Captured photographic texture, lighting retained':'Vertex colors or portrait texture',rig:'Heuristic jaw, smile, brow, squint fields. Not anatomically fitted.',reconstruction:stats.stats??stats,accessories:{glasses:headGlasses?.spec??null,hair:headHair?.spec??null,visibility:{glasses:headGlasses?.visible??false,hair:headHair?.visible??false}}};
    out.userData.appearanceStats=surfaceAppearance?.atlas.stats;
    out.userData.accessories.hair=remapHairRoots(headHair?.spec,dynamics.rest,g.attributes.position.array);
    const hairSource=mesh.geometry.clone();hairSource.setAttribute('position',new THREE.BufferAttribute(dynamics.rest.slice(),3));hairSource.computeVertexNormals();
    const exportedHair=headHair?new HeadHair(hairSource,headHair.spec):null;
    const bundle=new THREE.Group();bundle.name='Editable head and accessories';bundle.add(out);if(exportedHair){exportedHair.visible=headHair.visible;bundle.add(exportedHair);}const exportedGlasses=headGlasses?new HeadGlasses(headGlasses.spec):null;if(exportedGlasses){exportedGlasses.visible=headGlasses.visible;bundle.add(exportedGlasses);}
    const result=await new GLTFExporter().parseAsync(bundle,{binary:true,onlyVisible:false});exportedGlasses?.dispose();exportedHair?.dispose();hairSource.dispose();
    const saved=await fetch('/api/save?type=glb',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:result});
    if(!saved.ok)throw new Error('Local export could not be saved. Check the reconstruction server.');
    download(new Blob([result],{type:'model/gltf-binary'}),'punching-face.glb');g.dispose();out.material.dispose();toast('Saved .local/exports/punching-face.glb with geometry, appearance, and four morph targets.');
  }catch(e){toast(e.message);console.error(e);}finally{busy(false);}
}
$('export').onclick=exportGLB;
function sessionData(){
  const g=mesh.geometry.clone();g.setAttribute('position',new THREE.BufferAttribute(dynamics.rest.slice(),3));
  // Known demo assets are restored from local provenance. Imported baked meshes
  // must carry their own texture and UV mapping to survive a saved-session reload.
  const portableAppearance=!!surfaceAppearance;
  let storedPhoto=photoData;const material=portableAppearance?surfaceAppearance.material:mesh.material;
  if(!storedPhoto&&material.map?.image){const image=material.map.image,c=document.createElement('canvas');c.width=image.width;c.height=image.height;c.getContext('2d').drawImage(image,0,0);storedPhoto=c.toDataURL('image/png');}
  let roughnessPhoto=null;if(material.roughnessMap?.image){const image=material.roughnessMap.image,c=document.createElement('canvas');c.width=image.width;c.height=image.height;c.getContext('2d').drawImage(image,0,0);roughnessPhoto=c.toDataURL('image/png');}
  const data={roughnessPhoto,roughnessFlipY:material.roughnessMap?.flipY,format:'punching-face-session',version:1,physics:dynamics instanceof NewtonFaceDynamics?{binding:dynamics.binding,cage:dynamics.cage,id:capturedFaceId}:null,sourceTransform,name:sourceName,geometry:g.toJSON(),original:Array.from(dynamics.original),rig:{...dynamics.rig},softness:dynamics.softness,photo:storedPhoto,appearanceAtlas:portableAppearance?surfaceAppearance.atlas:null,textureFlipY:material.map?.flipY,unlit:material.isMeshBasicMaterial,accessories:{glasses:headGlasses?.spec??null,hair:headHair?.spec??null,visibility:{glasses:headGlasses?.visible??false,hair:headHair?.visible??false}},stats,distance:Number($('distance').value),yaw:Number($('face-yaw').value),pitch:Number($('face-pitch').value),material:material.toJSON()};g.dispose();return data;
}
async function restoreSession(data){
  if(data.format!=='punching-face-session'||data.version!==1||!data.geometry?.data?.attributes?.position)throw new Error('This is not a supported Punching Face session.');
  const arr=data.geometry.data.attributes.position.array;if(!Array.isArray(arr)||arr.length>900000||arr.length%3||!arr.every(Number.isFinite))throw new Error('Invalid session geometry.');
  const g=new THREE.BufferGeometryLoader().parse(data.geometry);
  let mat=data.unlit?new THREE.MeshBasicMaterial({vertexColors:!!g.attributes.color,side:THREE.DoubleSide,toneMapped:false}):new THREE.MeshStandardMaterial({vertexColors:!!g.attributes.color,roughness:.86,side:THREE.DoubleSide});
  if(data.photo){const texture=await new THREE.TextureLoader().loadAsync(data.photo);texture.colorSpace=THREE.SRGBColorSpace;texture.flipY=data.textureFlipY??true;mat.map=texture;}
  const roughnessTexture=data.roughnessPhoto?await new THREE.TextureLoader().loadAsync(data.roughnessPhoto):null;if(roughnessTexture)roughnessTexture.flipY=data.roughnessFlipY??true;
  sourceName=data.name||'Saved face';photoData=data.photo??null;sourceBytes=null;
  installMesh(g,mat,data.stats);if(data.original?.length===dynamics.rest.length&&data.original.every(Number.isFinite))dynamics.original=new Float32Array(data.original);
  for(const key of ['jaw','smile','brow','squint']){dynamics.rig[key]=clamp(Number(data.rig?.[key])||0,0,1);$(key).value=dynamics.rig[key];if($(key+'-value'))$(key+'-value').textContent=Math.round(dynamics.rig[key]*100)+'%';}
  $('distance').value=clamp(Number(data.distance)||.30,.24,.8);$('distance').oninput();$('face-yaw').value=Number(data.yaw)||0;$('face-pitch').value=Number(data.pitch)||0;$('softness').value=clamp(Number(data.softness)||.6,0,1);$('softness').oninput();
  $('model-kind').textContent='Restored editable session';
  if(data.appearanceAtlas&&mat.map){surfaceAppearance=new SurfaceAppearance(mesh.geometry,data.appearanceAtlas,mat.map.clone(),roughnessTexture);headPivot.add(surfaceAppearance);setView('mesh');$('model-kind').textContent='Restored textured mesh · welded physics';}
  if(data.physics?.id){const old=dynamics;dynamics=new NewtonFaceDynamics(mesh.geometry,data.physics.binding,data.physics.cage,physicsStatus);dynamics.rest=old.rest;dynamics.original=old.original;dynamics.rig=old.rig;capturedFaceId=data.physics.id;await dynamics.connect(capturedFaceId);}
  installGlasses(data.accessories?.glasses);installHair(data.accessories?.hair);restoreAccessoryVisibility(data.accessories);
  firstPerson();toast('Editable session restored. Re-import a room asset separately.');
}
$('save').onclick=async()=>{if(!mesh)return;try{
  const json=JSON.stringify(sessionData());const r=await fetch('/api/save?type=json',{method:'POST',headers:{'Content-Type':'application/json'},body:json});if(!r.ok)throw new Error('Local save failed.');
  try{sessionStorage.setItem('punching-face-dev-recovery',json);}catch{sessionStorage.setItem('punching-face-dev-recovery',JSON.stringify({format:'punching-face-session-pointer'}));}
  download(new Blob([json],{type:'application/json'}),'punching-face-session.json');toast('Saved editable session and Photographs link locally.');
}catch(e){toast(e.message);}};

$('capture-open').onclick=()=>$('capture-dialog').showModal();$('capture-close').onclick=()=>$('capture-dialog').close();
async function photoFace(image){busy(true,'Estimating a portrait mesh…');try{const result=await makePhotoFace(image);photoData=result.photo;sourceName='Your portrait preview';sourceBytes=null;installMesh(result.geometry,result.material,{source:'Single image landmark proxy',limitation:'Estimated depth. Single-view estimate. Unseen surfaces unavailable.'});$('model-kind').textContent='Photo proxy · estimated depth';$('capture-dialog').close();toast('Your portrait is ready. This is a frontal mesh preview, not a complete scan.');}catch(e){$('capture-result').textContent=e.message;toast(e.message);}finally{busy(false);}}
$('snapshot').onclick=async()=>{if(!tracking.active){await cameraToggle();}if(tracking.active)await photoFace($('webcam'));};
$('photo-import').onclick=()=>$('photo-file').click();$('photo-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;const url=URL.createObjectURL(file);try{const image=new Image();image.src=url;await image.decode();await photoFace(image);}finally{URL.revokeObjectURL(url);e.target.value='';}};
let capturedFaceId=null;
function physicsStatus(message){$('physics-engine').textContent=message+(dynamics?.ready?' + facial impact rig':'');const ready=!!dynamics?.ready;$('left-hook').disabled=!ready;$('right-hook').disabled=!ready;}
async function loadPhotoFace(id){
  busy(true,'Loading your photo model and initializing Newton…');
  try{
    // Re-baking repacks UV islands. A decoded image cached under the old URL
    // must never be paired with a newly fetched atlas for the same capture.
    const loadVersion=Date.now();
    const asset=name=>`/api/face-asset?id=${id}&asset=${name}&v=${loadVersion}`;
    let data,atlas,binding,cage,textureBytes;
    const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
    for(let attempt=0;attempt<4;attempt++){
      [data,atlas,binding,cage]=await Promise.all(['mesh.json','texture-atlas.json','physics-binding.json','physics-cage.json'].map(name=>fetch(asset(name),{cache:'no-store'}).then(async r=>{if(!r.ok)throw new Error('Build the photo model to create '+name);return r.json();})));
      const response=await fetch(asset('appearance.png'),{cache:'no-store'});if(!response.ok)throw new Error('The captured texture could not be loaded.');textureBytes=await response.arrayBuffer();
      const matches=(!atlas.textureSha256||await digest(textureBytes)===atlas.textureSha256)&&(!atlas.positionsSha256||await digest(new Float32Array(data.positions).buffer)===atlas.positionsSha256);
      if(matches)break;
      if(attempt===3)throw new Error('The capture is still publishing its rebuilt texture. The existing model has been retained.');
      await new Promise(resolve=>setTimeout(resolve,500));
    }
    const textureURL=URL.createObjectURL(new Blob([textureBytes],{type:'image/png'}));let texture;
    try{texture=await new THREE.TextureLoader().loadAsync(textureURL);}finally{URL.revokeObjectURL(textureURL);}
    texture.colorSpace=THREE.SRGBColorSpace;
    sourceName='Your photo model';sourceBytes=null;photoData=null;sourceTransform=data.transform;
    installMesh(geometryFromData(data),null,data);
    const roughnessTexture=atlas.roughnessTexture?await new THREE.TextureLoader().loadAsync(asset('appearance-roughness.png')):null;
    surfaceAppearance=new SurfaceAppearance(mesh.geometry,atlas,texture,roughnessTexture);headPivot.add(surfaceAppearance);capturedFaceId=id;sessionStorage.setItem('punching-face-active-capture',id);installGlasses(data.accessories?.glasses);installHair(data.accessories?.hair);
    dynamics=new NewtonFaceDynamics(mesh.geometry,binding,cage,physicsStatus);dynamics.softness=Number($('softness').value);
    for(let i=0;i<rigMarkers.children.length;i++){const index=[70,300,159,386,61,291,152][i];rigMarkers.children[i].position.fromArray(cage.rigAnchors[index]);}
    $('model-kind').textContent=data.stats.templateFit?(data.stats.orbitCoverage?.registeredRearViews>=3?'Fitted full head · captured rear views':'Fitted full head · rear shape estimated'):(data.stats.includesHairCapture?'Captured face + hair · predicted back':'Photo face · estimated rear shape');$('photo-count').textContent=`${data.stats.registeredViews} recovered photo views${data.stats.astra?' · Astra reviewed':' · local reconstruction'}`;
    $('face-yaw').value=0;$('face-pitch').value=0;firstPerson();setView('mesh');physicsStatus('Starting Newton CPU solver…');await dynamics.connect(id);
    toast('Photo model and Newton ready. Inspect Geometry or Wireframe, then try a hook.');
  }catch(e){toast(e.message);throw e;}finally{busy(false);}
}
const faceCapture=new FaceCapture(loadPhotoFace,async id=>{if(capturedFaceId===id){capturedFaceId=null;sessionStorage.removeItem('punching-face-dev-recovery');await reference();}});

function openFaceScan(){if(tracking.active)cameraToggle();$('capture-dialog').close();faceCapture.open().catch(e=>toast(e.message));}
$('scan-face').onclick=openFaceScan;$('record').onclick=openFaceScan;

function resetRoom(){if(roomSplat){roomSplat.removeFromParent();roomSplat.dispose();roomSplat=null;}roomTexture?.dispose();roomTexture=null;scene.background=new THREE.Color('#f6f7f2');studio.visible=true;scene.fog=new THREE.Fog('#f6f7f2',3,9);$('room-label').textContent='Studio environment · placeholder';$('room-fields').classList.remove('active');}
$('reset-room').onclick=resetRoom;$('import-room').onclick=()=>$('room-file').click();
$('room-file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;e.target.value='';busy(true,'Loading room context…');
  try{
    if(file.size>250_000_000)throw new Error('Use a panorama smaller than 250 MB for this prototype.');
    if(/\.(jpg|jpeg|png)$/i.test(file.name)){
      const url=URL.createObjectURL(file);let tex;try{tex=await new THREE.TextureLoader().loadAsync(url);}finally{URL.revokeObjectURL(url);}
      resetRoom();tex.mapping=THREE.EquirectangularReflectionMapping;tex.colorSpace=THREE.SRGBColorSpace;roomTexture=tex;scene.background=tex;studio.visible=false;scene.fog=null;
      $('room-label').textContent=`${file.name} · panorama, rotation only`;toast('Panorama loaded. Use a 2:1 equirectangular image for a correct 360° view.');
    }else throw new Error('Choose a JPG or PNG panorama.');
    $('room-fields').classList.add('active');for(const id of ['room-yaw','room-height'])$(id).value=0;$('room-scale').value=1;
  }catch(err){toast(err.message);console.error(err);}finally{busy(false);}
};
$('room-yaw').oninput=()=>{const rad=Number($('room-yaw').value)*Math.PI/180;if(roomSplat)roomSplat.rotation.y=rad;scene.backgroundRotation.y=rad;$('room-yaw-value').textContent=$('room-yaw').value+'°';};
$('room-scale').oninput=()=>{if(roomSplat){roomSplat.scale.setScalar(roomBaseScale*Number($('room-scale').value));roomSplat.position.copy(roomSplat.userData.basePosition).multiplyScalar(Number($('room-scale').value));roomSplat.position.y+=Number($('room-height').value);}$('room-scale-value').textContent=$('room-scale').value+'×';};
$('room-height').oninput=()=>{if(roomSplat)roomSplat.position.y=roomSplat.userData.basePosition.y*Number($('room-scale').value)+Number($('room-height').value);$('room-height-value').textContent=$('room-height').value+' m';};

// Fullscreen (immersive) toggle: hides side panels for a clean scene view. The
// header — and the wordmark — stays visible so the brand doesn't disappear.
$('fullscreen').onclick=()=>{document.body.classList.toggle('immersive');const on=document.body.classList.contains('immersive');$('fullscreen').setAttribute('aria-pressed',on?'true':'false');$('fullscreen').title=on?'Exit fullscreen':'Fullscreen';};

// Read-only diagnostics and deterministic fixture interactions for browser QA.
window.__punchingFace={get state(){return {ready:!!mesh,representation,impacts,lastSpeed,revision,vertices:mesh?.geometry.attributes.position.count,triangles:mesh?.geometry.index.count/3,maxDisplacement:dynamics?.maxDisplacement,peak,rig:dynamics?{...dynamics.rig}:{},hair:headHair?{visible:headHair.visible,strands:headHair.strandCount,type:headHair.spec.parameters.type,parameters:headHair.spec.parameters,vertices:headHair.geometry.attributes.position.count}:null,glasses:headGlasses?{visible:headGlasses.visible,meshes:headGlasses.children.length,source:headGlasses.spec.source}:null,cameraActive:tracking.active,calibrated:!!tracking.calibration,trackedHands:hands.filter(h=>h.tracked).length,lastTrackingTimestamp:tracking.appliedTimestamp,bodyTracking:{ready:!!tracking.poseReady,poseTimestamp:tracking.results?.pose?.timestamp,bodyCalibrated:!!tracking.bodyFrame,orientation:tracking.bodyFrame?.orientationSource,visiblePersonalArms:[...scannedArms].filter(([,arm])=>arm.visible).map(([side])=>side)},stats,physics:dynamics?.physicsInfo,physicsMetrics:dynamics?.lastMetrics,physicsError:dynamics?.error,impactHeld,regions:dynamics?.regionPeaks,scannedArms:[...scannedArms.keys()],mode,room:!!roomSplat||!!roomTexture};},get appearance(){return surfaceAppearance?{vertices:surfaceAppearance.mapping.length,textureSize:surfaceAppearance.material.map.image.width}:null;},get positions(){return mesh?.geometry.attributes.position.array.slice();},get rest(){return dynamics?.rest.slice();},sessionData,restoreSession,
get strike(){return hands.map(hand=>({hand:hand.side,phase:strikeTracker.state(hand.side),tracked:hand.tracked,calibrated:hand.calibrated}));}};
firstPerson();
const recovery=sessionStorage.getItem('punching-face-dev-recovery');
async function recover(){
  try{const data=await fetch('/api/face-captures').then(r=>r.json());const active=sessionStorage.getItem('punching-face-active-capture');
    if(active&&data.captures.some(v=>v.id===active)){
      const stable=await fetch(`/api/face-asset?id=${active}&asset=mesh.json`);
      if(stable.ok){await loadPhotoFace(active);return;}
    }
    const latest=data.captures.find(v=>v.photoModel&&v.status==='complete'&&!v.testFixture);if(latest){await loadPhotoFace(latest.id);return;}
    if(recovery){let saved=JSON.parse(recovery);if(saved.format==='punching-face-session-pointer'){const r=await fetch('/api/saved-session');if(!r.ok)throw new Error('Saved session unavailable.');saved=await r.json();}await restoreSession(saved);return;}
    await reference();
  }catch(e){toast(e.message);console.error(e);if(!mesh)await reference();}
}
recover();

const meshyState={busy:false,lastGLB:null,taskId:null};
async function meshyCapturePhoto(){
  if(!tracking.active){await cameraToggle();}
  if(!tracking.active)throw new Error('Camera required for photo capture.');
  const video=$('webcam');
  if(!video.videoWidth)throw new Error('Camera stream is not ready yet.');
  const full=document.createElement('canvas');full.width=video.videoWidth;full.height=video.videoHeight;
  full.getContext('2d').drawImage(video,0,0,full.width,full.height);
  const crop=await cropFacePortrait(full);
  return await new Promise(resolve=>crop.toBlob(resolve,'image/jpeg',0.92));
}
// Meshy can still hallucinate a bust from a masked input. If we see a real waist (narrow horizontal
// cross-section between wider shoulders and wider head), drop every triangle whose vertices are all
// below it. One linear pass over vertices and one over triangles — sub-millisecond for 30K tris.
function sliceBust(g){
  g.computeBoundingBox();
  const box=g.boundingBox,h=box.max.y-box.min.y,w=box.max.x-box.min.x;
  const BINS=48,minX=new Float32Array(BINS).fill(Infinity),maxX=new Float32Array(BINS).fill(-Infinity);
  const pos=g.attributes.position;
  for(let i=0;i<pos.count;i++){
    const y=pos.getY(i),x=pos.getX(i);
    const b=Math.min(BINS-1,Math.max(0,Math.floor((y-box.min.y)/h*BINS)));
    if(x<minX[b])minX[b]=x;if(x>maxX[b])maxX[b]=x;
  }
  const widthAt=b=>maxX[b]-minX[b];
  let bestB=-1,bestW=Infinity;
  for(let b=2;b<Math.floor(BINS*0.65);b++){const wb=widthAt(b);if(isFinite(wb)&&wb<bestW){bestW=wb;bestB=b;}}
  if(bestB<0)return;
  // A real waist has a wider region on BOTH sides (head above, shoulders below). A plain chin on a head-only
  // model has nothing wider below it — that's how we tell them apart.
  let widestAbove=0,widestBelow=0;
  for(let b=bestB+1;b<BINS;b++){const wb=widthAt(b);if(isFinite(wb)&&wb>widestAbove)widestAbove=wb;}
  for(let b=0;b<bestB;b++){const wb=widthAt(b);if(isFinite(wb)&&wb>widestBelow)widestBelow=wb;}
  if(widestBelow<bestW*1.3||widestAbove<bestW*1.3)return;
  const yCut=box.min.y+(bestB+0.5)/BINS*h;
  const idx=g.index;
  if(idx){
    const src=idx.array,keep=new (src.constructor)(src.length);let n=0;
    for(let i=0;i<src.length;i+=3){
      const a=src[i],b=src[i+1],c=src[i+2];
      if(pos.getY(a)>=yCut||pos.getY(b)>=yCut||pos.getY(c)>=yCut){keep[n++]=a;keep[n++]=b;keep[n++]=c;}
    }
    g.setIndex(new THREE.BufferAttribute(keep.slice(0,n),1));
  }else{
    const attrs=g.attributes,names=Object.keys(attrs),out={};
    for(const k of names)out[k]=[];
    for(let i=0;i<pos.count;i+=3){
      if(pos.getY(i)>=yCut||pos.getY(i+1)>=yCut||pos.getY(i+2)>=yCut){
        for(let k=0;k<3;k++)for(const nm of names){const a=attrs[nm],sz=a.itemSize;for(let s=0;s<sz;s++)out[nm].push(a.array[(i+k)*sz+s]);}
      }
    }
    for(const nm of names)g.setAttribute(nm,new THREE.Float32BufferAttribute(out[nm],attrs[nm].itemSize));
  }
  g.computeBoundingBox();
}
// Don't reshape the mesh. Bust or no bust, keep the whole thing — sliceBust tore across the chin
// and made the head look chewed off. Instead: detect where the actual features (nose, chin, eyes,
// cheeks, mouth) live on this specific mesh and hand those positions to the impact rig. Then the
// rig anchors follow the mesh instead of the mesh being forced to fit the anchors.
const REFERENCE_HEAD_HEIGHT=0.28;
// Walk the mesh geometry to find facial landmarks. Assumes normalizeHead has been run so the mesh
// is bbox-centered at origin, Y-up, face at +Z. Returns anchors keyed the same way as the rig's
// hardcoded defaults so FaceImpactRig.setAnchors just accepts them.
function detectAnchors(g){
  const pos=g.attributes.position,N=pos.count;
  g.computeBoundingBox();
  const b=g.boundingBox;
  // Ignore any retained torso below the normalized head window.
  const headBottom=Math.max(b.min.y,b.max.y-REFERENCE_HEAD_HEIGHT*1.08),headHeight=b.max.y-headBottom;
  // Nose tip = front-most vertex, ignoring hair fringe (only look above chin height, below brow).
  let noseIdx=-1,noseZ=-Infinity;
  for(let i=0;i<N;i++){
    const y=pos.getY(i);
    if(y<headBottom+headHeight*.25||y>b.max.y-headHeight*.18)continue;
    const z=pos.getZ(i);
    if(z>noseZ){noseZ=z;noseIdx=i;}
  }
  if(noseIdx<0){noseIdx=0;for(let i=0;i<N;i++)if(pos.getZ(i)>pos.getZ(noseIdx))noseIdx=i;noseZ=pos.getZ(noseIdx);}
  const noseX=pos.getX(noseIdx),noseY=pos.getY(noseIdx);
  // Chin = lowest Y among front-facing vertices near the midline (X close to nose X).
  const frontThreshold=b.min.z+(noseZ-b.min.z)*.55;
  let chinY=Infinity,chinZ=frontThreshold;
  for(let i=0;i<N;i++){
    const z=pos.getZ(i); if(z<frontThreshold)continue;
    if(Math.abs(pos.getX(i)-noseX)>0.05)continue;
    const y=pos.getY(i);
    if(y<headBottom)continue;
    if(y<chinY){chinY=y;chinZ=z;}
  }
  if(!isFinite(chinY))chinY=headBottom;
  // Face height = chin-to-brow. Nose sits roughly halfway between chin and brow, so brow ≈ 2*(nose - chin) + chin.
  const faceH=Math.max(0.05, 2*(noseY-chinY));
  const eyeY=chinY+faceH*0.62;   // eyeline
  const mouthY=chinY+faceH*0.22; // mouth centerline
  const cheekY=chinY+faceH*0.42; // cheekbone
  // Cheeks = widest front-facing X at cheek Y level.
  let cheekL=0,cheekR=0;
  const cheekTol=faceH*0.14;
  const cheekFront=b.min.z+(noseZ-b.min.z)*.45;
  for(let i=0;i<N;i++){
    if(pos.getZ(i)<cheekFront)continue;
    if(Math.abs(pos.getY(i)-cheekY)>cheekTol)continue;
    const x=pos.getX(i);
    if(x<cheekL)cheekL=x;
    if(x>cheekR)cheekR=x;
  }
  if(cheekL===0&&cheekR===0){cheekL=-faceH*0.4;cheekR=faceH*0.4;}
  // Return anchors in the frame the impact rig expects (see /src/impact-rig.js:setAnchors).
  const faceZ=noseZ*0.75;
  const mouthZ=noseZ*0.85;
  return {
    1:  [noseX,          noseY,   noseZ],       // nose tip — used by jab / Newton cage index 1
    152:[0,              chinY,   chinZ],
    13: [0,              mouthY+0.005, mouthZ],
    14: [0,              mouthY-0.005, mouthZ],
    50: [cheekL*0.90,    cheekY,  faceZ],
    280:[cheekR*0.90,    cheekY,  faceZ],
    61: [cheekL*0.45,    mouthY,  mouthZ],
    291:[cheekR*0.45,    mouthY,  mouthZ],
    159:[cheekL*0.55,    eyeY,    faceZ],
    386:[cheekR*0.55,    eyeY,    faceZ],
  };
}
async function loadMeshyGLB(bytes){
  const gltf=await new GLTFLoader().parseAsync(bytes.slice(0),'');
  const meshes=[];gltf.scene.updateMatrixWorld(true);gltf.scene.traverse(o=>{if(o.isMesh)meshes.push(o);});
  if(!meshes.length)throw new Error('Meshy returned an empty scene.');
  const m=meshes.length===1?meshes[0]:meshes.reduce((a,b)=>((b.geometry.index?.count??b.geometry.attributes.position.count)>(a.geometry.index?.count??a.geometry.attributes.position.count)?b:a));
  const g=m.geometry.clone();g.applyMatrix4(m.matrixWorld);
  normalizeHead(g);
  sourceName='Your Meshy head';photoData=null;sourceBytes=null;
  installMesh(g,m.material.clone(),{source:'Meshy image-to-3d',limitation:'Textured full-head estimate from a single frontal photograph.'});
  dynamics.impactRig.setAnchors(detectAnchors(g));
  $('model-kind').textContent='Meshy AI head · textured';
  $('physics-engine').textContent='Meshy import · preview springs + facial impact rig';
  firstPerson();
}
async function maybeCompressGLB(bytes){
  if(!$('compress-toggle').checked)return bytes;
  const before=bytes.byteLength;
  busy(true,'Compressing GLB with gltf-transform…');
  const [{WebIO},{dedup,prune,weld,quantize},{ALL_EXTENSIONS}]=await Promise.all([import('@gltf-transform/core'),import('@gltf-transform/functions'),import('@gltf-transform/extensions')]);
  const io=new WebIO().registerExtensions(ALL_EXTENSIONS);
  const doc=await io.readBinary(new Uint8Array(bytes));
  await doc.transform(dedup(),prune(),weld(),quantize());
  const out=await io.writeBinary(doc);
  const after=out.byteLength;
  toast(`Compressed: ${(before/1024).toFixed(0)} KB → ${(after/1024).toFixed(0)} KB (${(100-after/before*100).toFixed(0)}% smaller)`);
  return out.buffer.slice(out.byteOffset,out.byteOffset+out.byteLength);
}
async function meshyBuildFromPhoto(){
  if(meshyState.busy)return;
  meshyState.busy=true;busy(true,'Capturing your photo…');
  try{
    const blob=await meshyCapturePhoto();
    busy(true,'Meshy is texturing your head (up to a few minutes)…');
    const response=await fetch('/api/meshy-headshot',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob});
    if(!response.ok){let message='Meshy request failed.';try{message=(await response.json()).error||message;}catch{}throw new Error(message);}
    meshyState.taskId=response.headers.get('X-Meshy-Task-Id')||null;
    const rawBytes=await response.arrayBuffer();meshyState.lastGLB=rawBytes;
    const bytes=await maybeCompressGLB(rawBytes);
    await loadMeshyGLB(bytes);
    $('beat-yourself').disabled=false;
    toast('Meshy head loaded. Try a hook, or press "Beat yourself".');
    return true;
  }catch(e){toast('Meshy: '+e.message);console.error(e);return false;}
  finally{meshyState.busy=false;busy(false);}
}
$('meshy-toggle').onchange=async()=>{
  if(!$('meshy-toggle').checked)return;
  const ok=await meshyBuildFromPhoto();
  if(!ok)$('meshy-toggle').checked=false;
};
$('beat-yourself').onclick=async()=>{
  if(meshyState.busy)return;
  if(!meshyState.lastGLB){await meshyBuildFromPhoto();if(!meshyState.lastGLB)return;}
  else{busy(true,'Loading your Meshy head…');try{const bytes=await maybeCompressGLB(meshyState.lastGLB);await loadMeshyGLB(bytes);}catch(e){toast(e.message);return;}finally{busy(false);}}
  firstPerson();
  setTimeout(()=>scriptedImpact({type:'uppercut',side:'left',speed:2.4}),300);
  setTimeout(()=>scriptedImpact({type:'hook',side:'left',speed:2.1}),1100);
  setTimeout(()=>scriptedImpact({type:'hook',side:'right',speed:2.1}),1800);
};

// LiveKit arena hook (src/sponsors/arena-host.js; guarded by tests/sponsors-hook.test.mjs). A remote
// participant lands a punch through the same contact() path as a tracked fist. u,v are -1..1 across the
// visible face; the surface point comes from raycasting the real mesh, never from a guessed position.
window.__punchingFace.remotePunch=(punch={})=>{
  if(!mesh||!dynamics)return false;
  // Never trust the caller: a NaN speed serialises to null and takes the physics session down.
  const number=(value,fallback,low,high)=>Number.isFinite(+value)?clamp(+value,low,high):fallback;
  const u=number(punch.u,0,-1,1),v=number(punch.v,0,-1,1),lateral=number(punch.lateral,0,-1,1),speed=number(punch.speed,1.2,0,4);
  headPivot.updateWorldMatrix(true,false);mesh.geometry.computeBoundingBox();const box=mesh.geometry.boundingBox,centre=box.getCenter(new THREE.Vector3()),half=box.getSize(new THREE.Vector3()).multiplyScalar(.5);
  // Aim within the FACE, not the head: with a cranium and hair the bounding box centre sits near the brow.
  // The physics cage carries the measured landmarks (10 forehead, 152 chin, 234/454 face sides).
  const cage=dynamics.cage?.positions;let cx=centre.x,cy=centre.y,hx=half.x*.75,hy=half.y*.75;
  if(cage?.length>=1365){cx=(cage[234*3]+cage[454*3])/2;cy=(cage[10*3+1]+cage[152*3+1])/2;hx=Math.abs(cage[454*3]-cage[234*3])/2*.85;hy=Math.abs(cage[10*3+1]-cage[152*3+1])/2*.85;}
  raycaster.set(headPivot.localToWorld(new THREE.Vector3(cx+u*hx,cy+v*hy,box.max.z+.25)),new THREE.Vector3(0,0,-1).transformDirection(headPivot.matrixWorld));raycaster.far=Infinity;
  const hit=raycaster.intersectObject(mesh,false)[0];if(!hit)return false;
  const mode=({uppercut:'uppercut',jab:'jab',hook:'hook'})[String(punch.mode||'').toLowerCase()]||'hook';
  const landed=contact(headPivot.worldToLocal(hit.point.clone()),new THREE.Vector3(lateral*.6,0,-1).normalize(),speed,'remote',mode);
  if(landed)$('impact-label').textContent=String(punch.label||'REMOTE CONTACT').slice(0,40);
  return landed;
};
