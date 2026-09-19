import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { NewtonFaceDynamics } from './newton-dynamics.js';
import { FaceDynamics, clamp, sweptEllipsoid } from './physics.js';
import { ImpactReferences } from './impact-references.js';
import { ArmCapture } from './arm-capture.js';
import { FaceCapture } from './face-capture.js';
import { ScannedArm } from './scanned-arm.js';
import { HeadGlasses } from './head-accessories.js';
import { HeadHair, remapHairRoots } from './head-hair.js';
import { SurfaceAppearance, weldTexturedSurface } from './surface-appearance.js';
import { refineSurface } from './surface.js';
import { VirtualHand, Tracking, makePhotoFace } from './hands.js';
import './style.css';

const $=id=>document.getElementById(id);
document.querySelector('#app').innerHTML=`
<header><div class="brand"><svg viewBox="0 0 32 32" fill="none"><path d="M26 7 16 2 5 8v16l11 6 10-6V14l-10-5-5 3v9l5 3 5-3v-4l-5-3" stroke="#46663a" stroke-width="2.5" stroke-linejoin="round"/></svg><span class="wordmark">CONTACT</span><span class="divider"></span><span class="eyebrow header-sub">Spatial interaction lab</span></div><div class="header-right"><span class="version">RESEARCH BUILD / 0.1</span><span class="tiny"><i class="live-dot"></i>Local session</span><button id="capture-open" class="small">Capture guide ↗</button></div></header>
<main><aside class="panel left"><span class="eyebrow">Experiment 001</span><h1>From capture<br>to contact.</h1><p class="intro">A face you can reconstruct,<br>reshape, and reach.</p>
<div class="section-header"><h2>Face asset</h2><span class="number">01</span></div><div class="asset"><div class="asset-icon">◉</div><div><strong id="model-name">Reference head</strong><small id="model-kind">Public mesh reference</small></div></div>
<button id="scan-face" class="full primary" style="margin-bottom:9px">Record / upload head video</button><div class="row"><button id="import-face" class="small">↑ Import mesh</button><button id="reference" class="small">↺ Reference</button></div><input id="face-file" type="file" accept=".glb,.json"><div class="step-list"><div class="step done"><b>1</b><span><strong>Photographs</strong><span id="photo-count">Saved multiview photos</span></span></div><div class="step done" id="mesh-step"><b>2</b><span><strong>Editable surface</strong><span id="mesh-count">Loading reconstruction…</span></span></div><div class="step done"><b>3</b><span><strong>Interaction rig</strong>Tissue contact + facial impact rig</span></div></div>
<section class="section"><div class="section-header"><h2>Your perspective</h2><span class="number">02</span></div><button id="camera" class="full primary">Connect laptop webcam</button><video id="webcam" class="camera-preview" playsinline muted></video><p id="tracking-status" class="tracker-note">Connect your camera for tracking. Scan each arm to add your own geometry.</p><button id="calibrate" class="full small" disabled>Calibrate guard position</button><button id="scan-arms" class="full small" style="margin-top:9px">Scan my arms</button><p id="arm-appearance" class="muted">Personal arm meshes: awaiting capture.</p><label class="check">Show demo hand shapes <input id="demo-hands" type="checkbox"></label></section>
<section class="section"><div class="section-header"><h2>Room context</h2><span class="number">03</span></div><button id="import-room" class="full small">↑ Add room panorama</button><input id="room-file" type="file" accept=".jpg,.jpeg,.png"><p id="room-label" class="muted">Studio environment · placeholder</p><div id="room-fields" class="room-fields"><label class="controls-label">Room heading <output id="room-yaw-value">0°</output></label><input id="room-yaw" aria-label="Room heading" type="range" min="-180" max="180" value="0"><label class="controls-label">Room scale <output id="room-scale-value">1×</output></label><input id="room-scale" aria-label="Room scale" type="range" min=".2" max="3" step=".01" value="1"><label class="controls-label">Room height <output id="room-height-value">0 m</output></label><input id="room-height" aria-label="Room height" type="range" min="-3" max="3" step=".01" value="0"><button id="reset-room" class="small full">Restore studio</button></div></section>
<details><summary>One-word commands</summary><form id="command" class="command"><input id="command-input" aria-label="Command" placeholder="wireframe" autocomplete="off"><button>↵</button></form><p class="muted">mesh · geometry · wireframe · rig · left · right · reset · export<br>Runs locally. No LLM calls.</p></details><div class="footer-credit">Reference: <a href="/reference/LICENSE.txt" target="_blank">Lee Perry-Smith / CC BY 3.0</a>.<br>Public reference mesh, separate from your photographs.</div></aside>
<section class="stage-shell"><div id="stage" class="stage"></div><div class="stage-top"><div><div class="scene-caption">INTERACTION VOLUME</div><div class="scene-name" id="scene-name">Reference head <span>CPU Poisson surface</span></div></div><div id="stage-status" class="stage-status">● DEMO INPUT</div></div><div class="view-switch"><button id="view-mesh" class="active">Surface</button><button id="view-clay">Geometry</button><button id="view-wire">Wireframe</button></div><div class="reticle"></div><div class="target-label">TARGET / 01<br><span id="distance-label">0.55 m from view</span><br>EDITABLE SURFACE</div><div id="impact-label" class="impact-label">CONTACT REGISTERED</div><div class="stage-bottom"><div class="hint"><span><kbd>Q</kbd>Left hook</span><span><kbd>E</kbd>Right hook</span><span><kbd>Drag</kbd>Inspect</span></div><div class="bottom-line"><span id="view-label">FIRST-PERSON · VIRTUAL HANDS</span><span><span id="fps">—</span> FPS · WEBGL 2</span></div></div><div id="toast" class="toast" role="status"></div><div id="busy" class="busy-overlay"><div><div class="spinner"></div><div id="busy-text">Preparing surface…</div><p class="muted">Processed on this computer.</p></div></div></section>
<aside class="panel right"><div class="section-header"><h2>Contact response</h2><span class="tag">LIVE</span></div><p class="muted">Cheek compression, jaw shift, and facial recovery.<br><span id="physics-engine">Select a reconstructed photo model.</span></p><div class="metric-grid"><div class="metric"><strong id="impacts">00</strong><small>Contacts</small></div><div class="metric"><strong><span id="speed">0.0</span><em>m/s*</em></strong><small>Last input speed</small></div></div><canvas id="signal" class="signal" width="400" height="100"></canvas><div class="legend"><span>Surface displacement</span></div><div class="controls-label"><span>Peak deformation</span><output id="compression">0.0 mm</output></div><label class="controls-label">Softness <output id="softness-value">60%</output></label><input id="softness" aria-label="Softness" type="range" min="0" max="1" value=".6" step=".01"><label class="controls-label">Target distance <output id="distance-value">55 cm</output></label><input id="distance" aria-label="Target distance" type="range" min=".35" max=".8" value=".55" step=".01"><div class="row"><button id="left-hook" class="small">↗ Left hook</button><button id="right-hook" class="small">Right hook ↖</button></div>
<label class="check">Hold peak deformation <input id="hold-peak" type="checkbox"></label><button id="resume-impact" class="small full">Release deformation</button><button id="impact-references" class="small full" style="margin-top:8px">Impact reference library</button><label class="check">Head recoil <input id="head-recoil" type="checkbox"></label><label class="check">Slow motion <input id="slow-motion" type="checkbox" checked></label><div id="region-readout" class="note">Facial impact rig · ready</div><section class="section"><div class="section-header"><h2>Surface & rig</h2><span class="number">04</span></div><label class="check">3D glasses <input id="glasses" type="checkbox" disabled></label><p id="eye-detail-status" class="muted" role="status"></p><label class="check">Strand hair <input id="hair-visible" type="checkbox" disabled></label><details id="hair-controls"><summary>Hair style & detail</summary><p id="hair-description" class="muted">Rebuild a head scan with AI hair analysis to add editable strands.</p><label class="controls-label" for="hair-type">Hair type</label><select id="hair-type" disabled><option value="straight">Straight</option><option value="wavy">Wavy</option><option value="curly">Curly</option><option value="coily">Coily</option><option value="braided">Braided</option><option value="locs">Locs</option></select><label class="controls-label">Top length, mm</label><input id="hair-length" aria-label="Hair top length" type="range" min="1" max="450" step="1" disabled><label class="controls-label">Curl tightness</label><input id="hair-curl" aria-label="Hair curl tightness" type="range" min="0" max="1" step=".01" disabled><label class="controls-label">Root lift, mm</label><input id="hair-lift" aria-label="Hair root lift" type="range" min="0" max="12" step=".1" disabled><label class="controls-label">Frizz</label><input id="hair-frizz" aria-label="Hair frizz" type="range" min="0" max="1" step=".01" disabled><p class="muted">Photo detail follows your captured hair. Selecting a different type previews an estimated hairstyle.</p></details><label class="check">Wireframe overlay <input id="wire" type="checkbox"></label><label class="check">Rig control markers <input id="rig" type="checkbox"></label><label class="check">Sculpt surface <input id="sculpt" type="checkbox"></label><div class="row"><button id="undo" class="small">↶ Undo</button><button id="redo" class="small">Redo ↷</button></div><p class="muted" id="sculpt-hint">Sculpt: drag up to pull, down to push.</p><label class="controls-label">Jaw opening <output id="jaw-value">0%</output></label><input id="jaw" aria-label="Jaw opening" type="range" min="0" max="1" step=".01" value="0"><label class="controls-label">Lip corner pull <output id="smile-value">0%</output></label><input id="smile" aria-label="Lip corner pull" type="range" min="0" max="1" step=".01" value="0"><details><summary>More rig & alignment</summary><label class="controls-label">Brow raise</label><input id="brow" aria-label="Brow raise" type="range" min="0" max="1" step=".01" value="0"><label class="controls-label">Lid compression</label><input id="squint" aria-label="Lid compression" type="range" min="0" max="1" step=".01" value="0"><label class="controls-label">Face heading</label><input id="face-yaw" aria-label="Face heading" type="range" min="-180" max="180" value="0"><label class="controls-label">Face tilt</label><input id="face-pitch" aria-label="Face tilt" type="range" min="-180" max="180" value="0"><button id="first-person" class="small full">Reset first-person view</button></details><p class="note">Expression controls follow the measured face landmarks. Tissue layers and material properties are estimates; this is not an anatomical muscle simulation.</p></section>
<section class="section"><div class="row"><button id="reset" class="small">Reset face</button><button id="export" class="small primary">Export GLB ↗</button></div><button id="save" class="small full" style="margin-top:7px">Save editable session</button><details><summary>Reconstruction evidence</summary><div id="stats-detail"></div></details><p class="muted">* Webcam speed and reach are estimates, not measured physical quantities.</p></section></aside></main>
<dialog id="capture-dialog"><button class="close" id="capture-close" aria-label="Close capture guide">×</button><span class="eyebrow">Bring yourself into the scene</span><h1>Capture a face. Then a room.</h1><div class="capture-grid"><div class="capture-card"><h2>Try your face now</h2><p class="muted">A frontal image becomes a textured landmark mesh. Fast likeness preview; estimated depth and no back of head. Use Record for multiview geometry.</p><button id="snapshot" class="full primary">Use webcam portrait</button><button id="photo-import" class="full small" style="margin-top:9px">↑ Choose a face photo</button><input id="photo-file" type="file" accept="image/*"></div><div class="capture-card"><h2>Scan for fidelity</h2><p class="muted">Keep a neutral expression and even light. Have a helper move a phone around your still head, including both profiles, ears, chin, and crown. Keep overlapping views.</p><button id="record" class="full">Record a face scan</button><p class="muted">Saves face photographs, recovers camera angles, builds a connected mesh and bakes the captured texture. Astra supplies modeling advice; Newton simulates tissue contact.</p></div></div><ol><li>Capture 60–150 overlapping sharp face images with fixed exposure. A multiview capture is needed to preserve unseen features.</li><li>Press Create 3D face. Inspect Geometry and Wireframe to check depth. Gray areas mark estimated, unseen head surfaces.</li><li>Import a 360° room panorama for the surrounding view. A panorama supplies rotation only; it does not measure room depth.</li></ol><p class="note">Your laptop camera drives virtual hand articulation. A room scan supplies novel background views; it cannot reveal your hands’ hidden surfaces. Capture each arm separately to reconstruct its appearance. Monocular depth and automatic rigging remain estimates.</p><div id="capture-result" class="muted" role="status"></div></dialog>`;

const scene=new THREE.Scene();scene.background=new THREE.Color('#172120');scene.fog=new THREE.Fog('#172120',2.5,8);
const camera=new THREE.PerspectiveCamera(50,1,.01,30);camera.position.set(0,0,0);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance',preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
$('stage').appendChild(renderer.domElement);

const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,-.55);controls.enablePan=false;controls.enableDamping=true;controls.minDistance=.22;controls.maxDistance=1.6;controls.maxPolarAngle=Math.PI*.9;
const ambient=new THREE.HemisphereLight(0xd6e5d9,0x544b3b,2.3);scene.add(ambient);
const key=new THREE.DirectionalLight(0xffdcc2,3.4);key.position.set(-1,1.6,1);scene.add(key);
const rim=new THREE.DirectionalLight(0xadcab0,2.2);rim.position.set(.8,.4,-1.3);scene.add(rim);
const fill=new THREE.DirectionalLight(0xe2ecff,.75);fill.position.set(.7,0,1);scene.add(fill);
const studio=new THREE.Group();scene.add(studio);
function box(x,y,z,sx,sy,sz,color){const o=new THREE.Mesh(new THREE.BoxGeometry(sx,sy,sz),new THREE.MeshStandardMaterial({color,roughness:1}));o.position.set(x,y,z);studio.add(o);return o;}
box(0,-1.28,-1,8,.06,8,'#3b4138');box(0,.6,-3.3,8,4,.08,'#283a34');box(-2.3,.6,-.6,.08,4,6,'#34453c');box(2.3,.6,-.6,.08,4,6,'#26352f');
const grid=new THREE.GridHelper(8,32,0x65765b,0x4a5849);grid.position.y=-1.243;grid.position.z=-1;studio.add(grid);grid.material.transparent=true;grid.material.opacity=.4;
// A quiet studio placeholder, clearly separate from an imported room scan.
for(let i=0;i<6;i++)box(-1.85+i*.74,.5,-3.24,.009,2.9,.015,'#4a5e4c');
box(-1.1,.5,-3.19,1.1,1.1,.04,'#748274');box(-1.1,.5,-3.15,1.02,1.02,.02,'#b2b8a0');box(-1.1,.5,-3.12,.015,1.05,.03,'#4d6352');box(-1.1,.5,-3.12,1.05,.015,.03,'#4d6352');
box(1.14,-.60,-2.9,1.1,.06,.48,'#677255');for(const x of [.66,1.62])box(x,-.9,-2.9,.035,.6,.38,'#485c47');
for(let i=0;i<5;i++)box(.83+i*.08,-.43,-2.93,.05,.30+(i%2)*.07,.18,['#879071','#647561','#777b63'][i%3]);
const target=new THREE.Group();target.position.set(0,.015,-.55);scene.add(target);
const headPivot=new THREE.Group();target.add(headPivot);
const handGroup=new THREE.Group();scene.add(handGroup);const hands=[new VirtualHand(-1),new VirtualHand(1)];hands.forEach(h=>handGroup.add(h));
const rigMarkers=new THREE.Group();headPivot.add(rigMarkers);rigMarkers.visible=false;
let mesh,wireMesh,clayMesh,dynamics,sourceBytes,sourceName='Reference head',sourceTransform,stats={},photoData=null,representation='mesh',roomSplat=null,roomBaseScale=1,roomTexture=null;
let impacts=0,lastSpeed=0,normalClock=0,peak=0,mode='demo',demo=null,toastTimer,revision=0;
let surfaceAppearance=null,headGlasses=null,headHair=null;
let meshMatchesSource=true,impactHeld=false,watchPeak=false,previousDisplacement=0;
const raycaster=new THREE.Raycaster();const screen=new THREE.Vector2();
const tracking=new Tracking($('webcam'),message=>{$('tracking-status').textContent=message;});
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
  document.querySelector('.footer-credit').classList.toggle('hidden',sourceName!=='Reference head');
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
function contact(point,direction,speed,source){
  if(!dynamics)return false;
  impactHeld=false;watchPeak=true;previousDisplacement=0;const affected=dynamics.impulse(point,direction,speed);if(!affected)return false;
  impacts++;lastSpeed=speed;peak=0;$('impacts').textContent=String(impacts).padStart(2,'0');$('speed').textContent=speed.toFixed(1);
  $('impact-label').textContent=source==='webcam'?'TRACKED CONTACT':'DEMO CONTACT';$('impact-label').classList.remove('flash');void $('impact-label').offsetWidth;$('impact-label').classList.add('flash');
  window.__lastContact={point:point.toArray(),direction:direction.toArray(),speed,affected,source,time:performance.now()};return true;
}
function checkContact(hand,dt,now,source){
  if(!mesh||!hand.visible||hand.closed<.5||now-hand.lastHit<.45)return;
  const velocity=hand.center.clone().sub(hand.previous).divideScalar(Math.max(dt,1/120));const speed=velocity.length();if(speed<.35)return;
  headPivot.updateWorldMatrix(true,false);
  const from=headPivot.worldToLocal(hand.previous.clone()),to=headPivot.worldToLocal(hand.center.clone());
  mesh.geometry.computeBoundingBox();const bounds=mesh.geometry.boundingBox;
  const center=bounds.getCenter(new THREE.Vector3()),radii=bounds.getSize(new THREE.Vector3()).multiplyScalar(.5);
  if(sweptEllipsoid(from.toArray(),to.toArray(),center.toArray(),radii.toArray(),.037)===null)return;
  // Narrow phase samples the real mesh along the motion and radial probe rays.
  // This is a small swept-sphere approximation, not a medical contact solver.
  const travel=to.clone().sub(from),length=travel.length(),dir=travel.clone().normalize();
  let hit=null;
  for(const offset of [[0,0,0],[.026,0,0],[-.026,0,0],[0,.026,0],[0,-.026,0]]){
    const start=from.clone().add(new THREE.Vector3(...offset));
    const worldStart=headPivot.localToWorld(start);const worldDir=dir.clone().transformDirection(headPivot.matrixWorld);
    raycaster.set(worldStart,worldDir);raycaster.far=length+.045;
    const h=raycaster.intersectObject(mesh,false)[0];if(h){hit=h;break;}
  }
  raycaster.far=Infinity;
  if(!hit){
    // A hook can graze the cheek without the fist centre ray intersecting it.
    // Test the swept fist radius against the actual densely sampled surface.
    const p=mesh.geometry.attributes.position.array,n=mesh.geometry.attributes.normal.array,denom=Math.max(travel.lengthSq(),1e-12);let best=.037*.037,closest=-1;
    for(let i=0;i<p.length;i+=3){
      if(dynamics.binding&&!dynamics.binding.active[i/3])continue;
      const px=p[i]-from.x,py=p[i+1]-from.y,pz=p[i+2]-from.z,t=clamp((px*travel.x+py*travel.y+pz*travel.z)/denom,0,1);
      const d=(px-travel.x*t)**2+(py-travel.y*t)**2+(pz-travel.z*t)**2;if(d<best){best=d;closest=i;}
    }
    if(closest>=0)hit={point:headPivot.localToWorld(new THREE.Vector3(...p.slice(closest,closest+3))),face:{normal:new THREE.Vector3(...n.slice(closest,closest+3))}};
  }
  if(!hit)return;
  const point=headPivot.worldToLocal(hit.point.clone());
  const localDir=velocity.clone().normalize().transformDirection(new THREE.Matrix4().copy(headPivot.matrixWorld).invert());
  const inward=hit.face.normal.clone().normalize();if(inward.dot(localDir)<0)inward.negate();localDir.multiplyScalar(.35).addScaledVector(inward,.65).normalize();
  if(contact(point,localDir,clamp(speed,0,4),source))hand.lastHit=now;
}
function hook(side){if(!mesh||!meshMatchesSource)return;if(tracking.active){toast('Disconnect webcam to run a demo hook.');return;}firstPerson();headPivot.updateWorldMatrix(true,false);const anchor=dynamics.cage?.positions;const index=side<0?50:280;const goal=anchor?new THREE.Vector3(...anchor.slice(index*3,index*3+3)):new THREE.Vector3(side*.045,-.005,.055);headPivot.localToWorld(goal);demo={side,start:performance.now()/1000,previousT:0,goal};}

function drawTrace(){const canvas=$('signal'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,400,100);ctx.strokeStyle='#d8dfd0';ctx.lineWidth=1;for(let y=25;y<100;y+=25){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(400,y);ctx.stroke();}ctx.beginPath();trace.forEach((v,i)=>{const x=i/(trace.length-1)*400,y=92-Math.min(v/.018,1)*80;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.lineWidth=2;ctx.strokeStyle='#749b54';ctx.stroke();}
let lastTime=performance.now(),frames=0,fpsTime=lastTime;
function frame(time){
  const dt=Math.min((time-lastTime)/1000,1/30);lastTime=time;const now=time/1000;
  if(tracking.active){tracking.tick(time,hands);mode='webcam';}
  else{
    mode='demo';
    for(const hand of hands){hand.visible=true;const side=hand.side;
      if(demo&&demo.side===side){const t=(now-demo.start)/.72;
        if(t>1){demo=null;hand.demoPose(new THREE.Vector3(side*.11,-.10,-.32));}
        else{
          const a=clamp(t/.52,0,1),b=clamp((t-.52)/.48,0,1);const strike=Math.sin(a*Math.PI/2)*(1-b);
          const goal=demo.goal;const x=side*.24*(1-strike)+(goal.x-side*.020)*strike,y=-.13*(1-strike)+(goal.y+.012)*strike,z=-.24*(1-strike)+(goal.z-.023)*strike;
          hand.demoPose(new THREE.Vector3(x,y,z));checkContact(hand,dt,now,'demo');
        }
      }else{hand.demoPose(new THREE.Vector3(side*.11,-.10+Math.sin(now*1.4+side)*.002,-.32));}
    }
  }
  if(tracking.active&&tracking.calibration)for(const h of hands)if(h.tracked&&h.updated)checkContact(h,h.sampleDt,now,'webcam');
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
    for(const child of h.children)child.visible=!arm&&$('demo-hands').checked;
  }
  if(scannedArms.size&&normalClock%15===0){
    const visible=[...scannedArms].filter(([,arm])=>arm.visible).map(([side])=>side);
    $('arm-appearance').textContent=!tracking.active?'Captured arms loaded. Connect the webcam to drive them.':!tracking.bodyFrame?'Captured arms loaded. Calibrate with your face, shoulders, elbows and wrists visible.':visible.length?`Following your ${visible.join(' and ')} arm. Capture proportions retained; pose and skin weights are estimates.`:'Personal arms hidden: show the matching shoulder, elbow, wrist and hand clearly.';
    $('view-label').textContent='FIRST-PERSON · CAPTURED ARM MESHES';
  }
  controls.update();renderer.render(scene,camera);frames++;if(time-fpsTime>750){$('fps').textContent=Math.round(frames*1000/(time-fpsTime));frames=0;fpsTime=time;}
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
$('distance').oninput=()=>{target.position.z=-Number($('distance').value);$('distance-value').textContent=Math.round(Number($('distance').value)*100)+' cm';$('distance-label').textContent=Number($('distance').value).toFixed(2)+' m from view';};
$('left-hook').onclick=()=>hook(-1);$('right-hook').onclick=()=>hook(1);$('reset').onclick=()=>{impactHeld=false;watchPeak=false;dynamics?.reset();for(const id of ['jaw','smile','brow','squint']){$(id).value=0;if($(id+'-value'))$(id+'-value').textContent='0%';}peak=0;revision++;};$('first-person').onclick=firstPerson;
$('reference').onclick=reference;$('import-face').onclick=()=>$('face-file').click();
window.addEventListener('keydown',e=>{if(e.repeat||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||document.querySelector('dialog[open]'))return;if(e.code==='KeyQ')hook(-1);if(e.code==='KeyE')hook(1);if(e.code==='KeyR')$('reset').click();if(e.code==='KeyW'){$('wire').checked=!$('wire').checked;$('wire').onchange();}});

let drag=null;
function pick(e){const rect=renderer.domElement.getBoundingClientRect();screen.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(screen,camera);return raycaster.intersectObject(mesh,false)[0];}
renderer.domElement.addEventListener('pointerdown',e=>{
  if(!mesh||representation!=='mesh')return;
  if(rigMarkers.visible){const rect=renderer.domElement.getBoundingClientRect();screen.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(screen,camera);const marker=raycaster.intersectObjects(rigMarkers.children)[0];if(marker){const name=marker.object.userData.control;$(name).focus();toast(`${name} control selected — adjust its slider.`);return;}}
  if(!$('sculpt').checked)return;const hit=pick(e);if(!hit)return;dynamics.remember();drag={y:e.clientY,point:headPivot.worldToLocal(hit.point.clone())};renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener('pointermove',e=>{if(!drag)return;const delta=clamp((drag.y-e.clientY)*.00012,-.001,.001);dynamics.sculpt(drag.point,delta);drag.y=e.clientY;revision++;});
renderer.domElement.addEventListener('pointerup',()=>{drag=null;});renderer.domElement.addEventListener('pointercancel',()=>{drag=null;});

async function cameraToggle(){
  if(tracking.active){tracking.stop();$('camera').textContent='Connect laptop webcam';$('webcam').classList.remove('active');$('calibrate').disabled=true;$('tracking-status').textContent='Camera off. Virtual hands are in demo mode.';$('stage-status').textContent='● DEMO INPUT';return;}
  $('camera').disabled=true;
  try{await tracking.start();demo=null;firstPerson();$('camera').textContent='Disconnect webcam';$('webcam').classList.add('active');$('calibrate').disabled=false;$('stage-status').textContent='● WEBCAM · VIRTUAL POV';}
  catch(e){toast(`Camera unavailable: ${e.message}`);$('tracking-status').textContent='Camera could not start. Allow camera access in the browser, then reconnect.';}
  finally{$('camera').disabled=false;}
}
$('camera').onclick=cameraToggle;$('calibrate').onclick=()=>{try{tracking.calibrate();}catch(e){toast(e.message);}};
window.addEventListener('pagehide',()=>{tracking.stop();dynamics?.dispose?.();});
if(import.meta.hot)import.meta.hot.dispose(()=>dynamics?.dispose?.());

$('face-file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;e.target.value='';busy(true,'Importing face asset…');
  try{
    if(file.size>180_000_000)throw new Error('Use a GLB or editable session smaller than 180 MB.');
    const ext=file.name.split('.').pop().toLowerCase();
    if(ext==='json'){await restoreSession(JSON.parse(await file.text()));return;}
    if(ext==='glb'){
      const gltf=await new GLTFLoader().parseAsync(await file.arrayBuffer(),'');const meshes=[];gltf.scene.updateMatrixWorld(true);gltf.scene.traverse(o=>{if(o.isMesh)meshes.push(o);});
      const faces=meshes.filter(m=>!['eyeglasses','hair'].includes(m.userData.accessory));
      if(faces.length!==1)throw new Error('Import a GLB containing one face mesh and optional hair/glasses.');
      const m=faces[0],g=m.geometry.clone();g.applyMatrix4(m.matrixWorld);
      if(m.userData.coordinateSystem!=='contact-head-metres-v1'){
        g.computeBoundingBox();const box=g.boundingBox,center=box.getCenter(new THREE.Vector3()),scale=.28/box.getSize(new THREE.Vector3()).y;g.translate(-center.x,-center.y,-center.z);g.scale(scale,scale,scale);
      }
      sourceName=file.name;photoData=null;sourceBytes=null;
      if(m.material.map&&m.userData.appearance){
        const recovered=weldTexturedSurface(g);recovered.atlas.stats=m.userData.appearanceStats;installMesh(recovered.geometry,null,m.userData.reconstruction??{source:'Imported textured mesh'});
        surfaceAppearance=new SurfaceAppearance(mesh.geometry,recovered.atlas,m.material.map.clone(),m.material.roughnessMap?.clone());headPivot.add(surfaceAppearance);g.dispose();setView('mesh');
      }else installMesh(g,m.material.clone(),m.userData.reconstruction??{source:'Imported mesh'});
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
    const out=new THREE.Mesh(g,(surfaceAppearance?.material??mesh.material).clone());out.name='Contact editable face';out.updateMorphTargets();out.morphTargetInfluences=Object.values(dynamics.rig);out.userData={coordinateSystem:'contact-head-metres-v1',source:sourceName,appearance:surfaceAppearance?'Captured photographic texture, lighting retained':'Vertex colors or portrait texture',rig:'Heuristic jaw, smile, brow, squint fields. Not anatomically fitted.',reconstruction:stats.stats??stats,accessories:{glasses:headGlasses?.spec??null,hair:headHair?.spec??null,visibility:{glasses:headGlasses?.visible??false,hair:headHair?.visible??false}}};
    out.userData.appearanceStats=surfaceAppearance?.atlas.stats;
    out.userData.accessories.hair=remapHairRoots(headHair?.spec,dynamics.rest,g.attributes.position.array);
    const hairSource=mesh.geometry.clone();hairSource.setAttribute('position',new THREE.BufferAttribute(dynamics.rest.slice(),3));hairSource.computeVertexNormals();
    const exportedHair=headHair?new HeadHair(hairSource,headHair.spec):null;
    const bundle=new THREE.Group();bundle.name='Editable head and accessories';bundle.add(out);if(exportedHair){exportedHair.visible=headHair.visible;bundle.add(exportedHair);}const exportedGlasses=headGlasses?new HeadGlasses(headGlasses.spec):null;if(exportedGlasses){exportedGlasses.visible=headGlasses.visible;bundle.add(exportedGlasses);}
    const result=await new GLTFExporter().parseAsync(bundle,{binary:true,onlyVisible:false});exportedGlasses?.dispose();exportedHair?.dispose();hairSource.dispose();
    const saved=await fetch('/api/save?type=glb',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:result});
    if(!saved.ok)throw new Error('Local export could not be saved. Check the reconstruction server.');
    download(new Blob([result],{type:'model/gltf-binary'}),'contact-face.glb');g.dispose();out.material.dispose();toast('Saved .local/exports/contact-face.glb with geometry, appearance, and four morph targets.');
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
  const data={roughnessPhoto,roughnessFlipY:material.roughnessMap?.flipY,format:'contact-session',version:1,physics:dynamics instanceof NewtonFaceDynamics?{binding:dynamics.binding,cage:dynamics.cage,id:capturedFaceId}:null,sourceTransform,name:sourceName,geometry:g.toJSON(),original:Array.from(dynamics.original),rig:{...dynamics.rig},softness:dynamics.softness,photo:storedPhoto,appearanceAtlas:portableAppearance?surfaceAppearance.atlas:null,textureFlipY:material.map?.flipY,unlit:material.isMeshBasicMaterial,accessories:{glasses:headGlasses?.spec??null,hair:headHair?.spec??null,visibility:{glasses:headGlasses?.visible??false,hair:headHair?.visible??false}},stats,distance:Number($('distance').value),yaw:Number($('face-yaw').value),pitch:Number($('face-pitch').value),material:material.toJSON()};g.dispose();return data;
}
async function restoreSession(data){
  if(data.format!=='contact-session'||data.version!==1||!data.geometry?.data?.attributes?.position)throw new Error('This is not a supported Contact session.');
  const arr=data.geometry.data.attributes.position.array;if(!Array.isArray(arr)||arr.length>900000||arr.length%3||!arr.every(Number.isFinite))throw new Error('Invalid session geometry.');
  const g=new THREE.BufferGeometryLoader().parse(data.geometry);
  let mat=data.unlit?new THREE.MeshBasicMaterial({vertexColors:!!g.attributes.color,side:THREE.DoubleSide,toneMapped:false}):new THREE.MeshStandardMaterial({vertexColors:!!g.attributes.color,roughness:.86,side:THREE.DoubleSide});
  if(data.photo){const texture=await new THREE.TextureLoader().loadAsync(data.photo);texture.colorSpace=THREE.SRGBColorSpace;texture.flipY=data.textureFlipY??true;mat.map=texture;}
  const roughnessTexture=data.roughnessPhoto?await new THREE.TextureLoader().loadAsync(data.roughnessPhoto):null;if(roughnessTexture)roughnessTexture.flipY=data.roughnessFlipY??true;
  sourceName=data.name||'Saved face';photoData=data.photo??null;sourceBytes=null;
  installMesh(g,mat,data.stats);if(data.original?.length===dynamics.rest.length&&data.original.every(Number.isFinite))dynamics.original=new Float32Array(data.original);
  for(const key of ['jaw','smile','brow','squint']){dynamics.rig[key]=clamp(Number(data.rig?.[key])||0,0,1);$(key).value=dynamics.rig[key];if($(key+'-value'))$(key+'-value').textContent=Math.round(dynamics.rig[key]*100)+'%';}
  $('distance').value=clamp(Number(data.distance)||.55,.35,.8);$('distance').oninput();$('face-yaw').value=Number(data.yaw)||0;$('face-pitch').value=Number(data.pitch)||0;$('softness').value=clamp(Number(data.softness)||.6,0,1);$('softness').oninput();
  $('model-kind').textContent='Restored editable session';
  if(data.appearanceAtlas&&mat.map){surfaceAppearance=new SurfaceAppearance(mesh.geometry,data.appearanceAtlas,mat.map.clone(),roughnessTexture);headPivot.add(surfaceAppearance);setView('mesh');$('model-kind').textContent='Restored textured mesh · welded physics';}
  if(data.physics?.id){const old=dynamics;dynamics=new NewtonFaceDynamics(mesh.geometry,data.physics.binding,data.physics.cage,physicsStatus);dynamics.rest=old.rest;dynamics.original=old.original;dynamics.rig=old.rig;capturedFaceId=data.physics.id;await dynamics.connect(capturedFaceId);}
  installGlasses(data.accessories?.glasses);installHair(data.accessories?.hair);restoreAccessoryVisibility(data.accessories);
  firstPerson();toast('Editable session restored. Re-import a room asset separately.');
}
$('save').onclick=async()=>{if(!mesh)return;try{
  const json=JSON.stringify(sessionData());const r=await fetch('/api/save?type=json',{method:'POST',headers:{'Content-Type':'application/json'},body:json});if(!r.ok)throw new Error('Local save failed.');
  try{sessionStorage.setItem('contact-dev-recovery',json);}catch{sessionStorage.setItem('contact-dev-recovery',JSON.stringify({format:'contact-session-pointer'}));}
  download(new Blob([json],{type:'application/json'}),'contact-session.json');toast('Saved editable session and Photographs link locally.');
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
    surfaceAppearance=new SurfaceAppearance(mesh.geometry,atlas,texture,roughnessTexture);headPivot.add(surfaceAppearance);capturedFaceId=id;sessionStorage.setItem('contact-active-capture',id);installGlasses(data.accessories?.glasses);installHair(data.accessories?.hair);
    dynamics=new NewtonFaceDynamics(mesh.geometry,binding,cage,physicsStatus);dynamics.softness=Number($('softness').value);
    for(let i=0;i<rigMarkers.children.length;i++){const index=[70,300,159,386,61,291,152][i];rigMarkers.children[i].position.fromArray(cage.rigAnchors[index]);}
    $('model-kind').textContent=data.stats.templateFit?(data.stats.orbitCoverage?.registeredRearViews>=3?'Fitted full head · captured rear views':'Fitted full head · rear shape estimated'):(data.stats.includesHairCapture?'Captured face + hair · predicted back':'Photo face · estimated rear shape');$('photo-count').textContent=`${data.stats.registeredViews} recovered photo views${data.stats.astra?' · Astra reviewed':' · local reconstruction'}`;
    $('face-yaw').value=0;$('face-pitch').value=0;firstPerson();setView('mesh');physicsStatus('Starting Newton CPU solver…');await dynamics.connect(id);
    toast('Photo model and Newton ready. Inspect Geometry or Wireframe, then try a hook.');
  }catch(e){toast(e.message);throw e;}finally{busy(false);}
}
const faceCapture=new FaceCapture(loadPhotoFace,async id=>{if(capturedFaceId===id){capturedFaceId=null;sessionStorage.removeItem('contact-dev-recovery');await reference();}});

function openFaceScan(){if(tracking.active)cameraToggle();$('capture-dialog').close();faceCapture.open().catch(e=>toast(e.message));}
$('scan-face').onclick=openFaceScan;$('record').onclick=openFaceScan;

function resetRoom(){if(roomSplat){roomSplat.removeFromParent();roomSplat.dispose();roomSplat=null;}roomTexture?.dispose();roomTexture=null;scene.background=new THREE.Color('#172120');studio.visible=true;scene.fog=new THREE.Fog('#172120',2.5,8);$('room-label').textContent='Studio environment · placeholder';$('room-fields').classList.remove('active');}
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

$('command').onsubmit=e=>{e.preventDefault();const command=$('command-input').value.trim().toLowerCase();const commands={mesh:()=>setView('mesh'),geometry:()=>setView('clay'),wireframe:()=>setView('wire'),rig:()=>{$('rig').checked=!$('rig').checked;$('rig').onchange();},left:()=>hook(-1),right:()=>hook(1),reset:()=>$('reset').click(),export:exportGLB};if(commands[command])commands[command]();else toast('Available: mesh, geometry, wireframe, rig, left, right, reset, export.');$('command-input').value='';};

// Read-only diagnostics and deterministic fixture interactions for browser QA.
window.__contactLab={get state(){return {ready:!!mesh,representation,impacts,lastSpeed,revision,vertices:mesh?.geometry.attributes.position.count,triangles:mesh?.geometry.index.count/3,maxDisplacement:dynamics?.maxDisplacement,peak,rig:dynamics?{...dynamics.rig}:{},hair:headHair?{visible:headHair.visible,strands:headHair.strandCount,type:headHair.spec.parameters.type,parameters:headHair.spec.parameters,vertices:headHair.geometry.attributes.position.count}:null,glasses:headGlasses?{visible:headGlasses.visible,meshes:headGlasses.children.length,source:headGlasses.spec.source}:null,cameraActive:tracking.active,calibrated:!!tracking.calibration,trackedHands:hands.filter(h=>h.tracked).length,lastTrackingTimestamp:tracking.appliedTimestamp,bodyTracking:{ready:!!tracking.poseReady,poseTimestamp:tracking.results?.pose?.timestamp,bodyCalibrated:!!tracking.bodyFrame,orientation:tracking.bodyFrame?.orientationSource,visiblePersonalArms:[...scannedArms].filter(([,arm])=>arm.visible).map(([side])=>side)},stats,physics:dynamics?.physicsInfo,physicsMetrics:dynamics?.lastMetrics,physicsError:dynamics?.error,impactHeld,regions:dynamics?.regionPeaks,scannedArms:[...scannedArms.keys()],mode,room:!!roomSplat||!!roomTexture};},get appearance(){return surfaceAppearance?{vertices:surfaceAppearance.mapping.length,textureSize:surfaceAppearance.material.map.image.width}:null;},get positions(){return mesh?.geometry.attributes.position.array.slice();},get rest(){return dynamics?.rest.slice();},sessionData,restoreSession};
firstPerson();
const recovery=sessionStorage.getItem('contact-dev-recovery');
async function recover(){
  try{const data=await fetch('/api/face-captures').then(r=>r.json());const active=sessionStorage.getItem('contact-active-capture');
    if(active&&data.captures.some(v=>v.id===active)){
      const stable=await fetch(`/api/face-asset?id=${active}&asset=mesh.json`);
      if(stable.ok){await loadPhotoFace(active);return;}
    }
    const latest=data.captures.find(v=>v.photoModel&&v.status==='complete'&&!v.testFixture);if(latest){await loadPhotoFace(latest.id);return;}
    if(recovery){let saved=JSON.parse(recovery);if(saved.format==='contact-session-pointer'){const r=await fetch('/api/saved-session');if(!r.ok)throw new Error('Saved session unavailable.');saved=await r.json();}await restoreSession(saved);return;}
    await reference();
  }catch(e){toast(e.message);console.error(e);if(!mesh)await reference();}
}
recover();

// LiveKit arena hook (src/sponsors/arena-host.js; guarded by tests/sponsors-hook.test.mjs). A remote
// participant lands a punch through the same contact() path as a tracked fist. u,v are -1..1 across the
// visible face; the surface point comes from raycasting the real mesh, never from a guessed position.
window.__contactLab.remotePunch=(punch={})=>{
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
  const landed=contact(headPivot.worldToLocal(hit.point.clone()),new THREE.Vector3(lateral*.6,0,-1).normalize(),speed,'remote');
  if(landed)$('impact-label').textContent=String(punch.label||'REMOTE CONTACT').slice(0,40);
  return landed;
};
