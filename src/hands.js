import * as THREE from 'three';
import { FilesetResolver, HandLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';
import { clamp, fistScore } from './physics.js';
import {calibrateBodyFrame,matchHandsToBody,retargetCapturedArm} from './arm-pose.js';

const LINKS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const up=new THREE.Vector3(0,1,0);
export function segment(mesh,a,b,radius){
  mesh.position.copy(a).add(b).multiplyScalar(.5);
  const d=new THREE.Vector3().subVectors(b,a);
  mesh.quaternion.setFromUnitVectors(up,d.clone().normalize());
  mesh.scale.set(radius,d.length(),radius);
}

export class VirtualHand extends THREE.Group {
  constructor(side){
    super();this.side=side;this.center=new THREE.Vector3();this.previous=new THREE.Vector3();this.lastHit=-10;this.closed=1;this.tracked=false;
    const skin=new THREE.MeshStandardMaterial({color:0xb98d72,roughness:.76});
    const wrist=new THREE.MeshStandardMaterial({color:0x3c5250,roughness:.9});
    this.joints=Array.from({length:21},()=>{const m=new THREE.Mesh(new THREE.SphereGeometry(1,10,8),skin);m.scale.setScalar(.009);this.add(m);return m;});
    this.links=LINKS.map(()=>{const m=new THREE.Mesh(new THREE.CylinderGeometry(1,1,1,10),skin);this.add(m);return m;});
    this.forearm=new THREE.Mesh(new THREE.CylinderGeometry(.75,1,1,18),skin);this.add(this.forearm);
    this.sleeve=new THREE.Mesh(new THREE.CylinderGeometry(.8,1,1,18),wrist);this.add(this.sleeve);
    this.palm=new THREE.Mesh(new THREE.SphereGeometry(1,20,16),skin);this.add(this.palm);
    this.demoPose(new THREE.Vector3(side*.19,-.17,-.25));
  }
  apply(points,closed,armPose=null){
    this.armPose=armPose;
    this.previous.copy(this.center);this.closed=closed;
    const center=new THREE.Vector3().addVectors(points[5],points[17]).multiplyScalar(.5).lerp(points[9],.2);
    this.center.copy(center);
    points.forEach((p,i)=>{this.joints[i].position.copy(p);this.joints[i].scale.setScalar(i===0?.022:(i%4===0?.008:.01));});
    LINKS.forEach(([a,b],i)=>segment(this.links[i],points[a],points[b],a===0?.015:.009));
    this.palm.position.copy(points[0]).lerp(points[9],.53);
    this.palm.scale.set(.035,.047,.019);
    this.palm.quaternion.setFromUnitVectors(up,new THREE.Vector3().subVectors(points[9],points[0]).normalize());
    const elbow=armPose?.joints[1]??new THREE.Vector3(this.side*.17,-.25,-.04);
    const cuff=points[0].clone().lerp(elbow,.74);
    segment(this.forearm,points[0],cuff,.031);segment(this.sleeve,cuff,elbow,.039);
  }
  demoPose(center){
    const points=[new THREE.Vector3(center.x,center.y-.07,center.z+.025)];
    for(let f=0;f<5;f++)for(let j=0;j<4;j++){
      const x=center.x+(f-2)*.015*this.side;
      points.push(new THREE.Vector3(x,center.y+[-.012,.022,.016,-.004][j]-(f===0?.024:0),center.z+[.015,.004,-.022,-.026][j]));
    }
    this.apply(points,1);
  }
}

export class Tracking {
  constructor(video,onStatus){this.video=video;this.onStatus=onStatus;this.active=false;this.calibration=null;this.bodyFrame=null;this.armProfiles=new Map();this.guardWidths=[];this.lastFrame=-1;this.lastTick=0;this.results=null;this.stream=null;this.worker=null;this.busy=false;}
  setArmProfile(profile){this.armProfiles.set(profile.side,profile);this.calibration=null;this.bodyFrame=null;this.onStatus('Personal arm loaded. Show your face, shoulders, elbows and wrists, then calibrate guard.');}
  async start(){
    this.onStatus('Starting local hand tracking…');
    try{
      this.stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}},audio:false});
      this.video.srcObject=this.stream;await this.video.play();
      // Inference lives in a worker so synchronous MediaPipe calls cannot block
      // rendering and impact integration on the main thread.
      this.worker=new Worker('/tracking-worker.js');
      await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(new Error('Hand model initialization timed out.')),25000);
        this.worker.onmessage=({data})=>{
          if(data.type==='ready'){clearTimeout(timeout);resolve();}
          if(data.type==='error'){clearTimeout(timeout);reject(new Error(data.message));}
        };
        this.worker.onerror=e=>{clearTimeout(timeout);reject(new Error(e.message));};
        this.worker.postMessage({type:'init',origin:location.origin});
      });
      this.worker.onmessage=({data})=>{if(data.type==='poseReady'){this.poseReady=true;this.onPoseReady?.();return;}this.busy=false;if(data.type==='result'){this.results=data;this.receivedAt=performance.now();if(data.capture)this.onCapture?.(data);}else if(data.type==='error'){this.onStatus(data.message);this.stop();}};
      this.active=true;this.calibration=null;await this.enableBody();this.onStatus('Webcam connected · show your face, shoulders, elbows and hands, then calibrate guard');
    }catch(e){this.stop();throw e;}
  }
  stop(){this.active=false;this.worker?.terminate();this.worker=null;this.busy=false;this.results=null;this.poseReady=false;this.appliedTimestamp=0;this.calibration=null;this.bodyFrame=null;this.captureRequest=null;this.lastFrame=-1;this.poseInitReject?.(new Error('Camera disconnected while loading body tracking.'));this.stream?.getTracks().forEach(t=>t.stop());this.video.srcObject=null;}
  calibrate(){
    const hands=this.results?.landmarks;if(!hands?.length)throw new Error('Show an open hand to the camera before calibrating.');
    const pose=this.results.pose;this.bodyFrame=calibrateBodyFrame(pose?.worldLandmarks?.[0],pose?.landmarks?.[0],[...this.armProfiles.values()]);
    if(this.armProfiles.size&&(!this.bodyFrame||!Number.isFinite(pose.timestamp)||this.results.timestamp-pose.timestamp>180))throw new Error('Personal arms need a clear view of your face, shoulders, elbows and wrists to calibrate.');
    this.calibration=hands.map(lm=>Math.hypot(lm[5].x-lm[17].x,lm[5].y-lm[17].y)).reduce((a,b)=>a+b,0)/hands.length;
    this.onStatus(this.armProfiles.size?'Personal proportions calibrated · body and palm tracking drive the captured meshes':'Guard calibrated · close your fist and move across the target');
  }
  async enableBody(){if(this.poseReady)return;if(!this.active)throw new Error('Connect your webcam first.');if(this.poseInitPromise)return this.poseInitPromise;
    this.poseInitPromise=new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Body model initialization timed out.')),25000);this.poseInitReject=e=>{clearTimeout(timer);reject(e);};this.onPoseReady=()=>{clearTimeout(timer);resolve();};this.worker.postMessage({type:'enablePose'});}).finally(()=>{this.poseInitPromise=null;this.poseInitReject=null;this.onPoseReady=null;});return this.poseInitPromise;
  }
  async tick(now,hands){
    if(!this.active)return;
    for(const h of hands)h.updated=false;
    if(!this.busy&&now-this.lastTick>45&&this.video.readyState>=2&&this.video.currentTime!==this.lastFrame){
      this.busy=true;this.lastTick=now;this.lastFrame=this.video.currentTime;
      try{const bitmap=await createImageBitmap(this.video);if(this.worker){this.worker.postMessage({type:'frame',bitmap,timestamp:now,capture:this.captureRequest,trackBody:this.armProfiles.size>0},[bitmap]);this.captureRequest=null;}else bitmap.close();}catch{this.busy=false;}
    }
    if(!this.results||now-this.results.timestamp>350){for(const h of hands){h.tracked=false;h.visible=false;h.armPose=null;}return;}
    if(this.appliedTimestamp===this.results.timestamp)return;
    const sampleDt=this.appliedTimestamp?clamp((this.results.timestamp-this.appliedTimestamp)/1000,1/60,.15):1/30;
    this.appliedTimestamp=this.results.timestamp;
    const previouslyTracked=new Map(hands.map(h=>[h.side,h.tracked]));
    for(const h of hands){h.tracked=false;h.visible=false;h.armPose=null;}
    const body=this.results.pose,assignments=matchHandsToBody(this.results.landmarks,body?.landmarks?.[0],this.video.videoWidth/this.video.videoHeight||1);
    this.results.landmarks.forEach((lm,i)=>{
      // Mirror horizontal motion into the user's body frame. This creates a
      // virtual view, not recovered RGB of unseen hand surfaces.
      const centerX=(lm[5].x+lm[17].x)/2;
      const matched=assignments.get(i),side=matched?(matched==='left'?-1:1):(this.results.handedness[i]?.[0]?.categoryName==='Left'?1:-1);
      const h=hands.find(h=>h.side===side);if(!h)return;
      const profile=this.armProfiles.get(side<0?'left':'right');
      if(this.armProfiles.size&&!profile)return;
      if(profile){
        if(!matched||!body||!Number.isFinite(body.timestamp)||this.results.timestamp-body.timestamp>180)return;
        const pose=retargetCapturedArm(profile,this.bodyFrame,body.worldLandmarks?.[0],body.landmarks?.[0],this.results.worldLandmarks?.[i]);if(!pose)return;
        h.visible=true;h.tracked=true;h.apply(pose.joints.slice(3),fistScore(lm),pose);h.sampleDt=sampleDt;h.updated=previouslyTracked.get(h.side)===true;if(!h.updated)h.previous.copy(h.center);return;
      }
      const width=Math.hypot(lm[5].x-lm[17].x,lm[5].y-lm[17].y);
      const ratio=this.calibration?width/this.calibration:1;
      const depth=clamp(.30+(ratio-1)*.40,.15,.80);
      const base=new THREE.Vector3((.5-centerX)*.85,clamp((.55-lm[9].y)*.65,-.30,.16),-depth);
      const unit=.075/Math.max(width,.025);
      const points=lm.map(p=>new THREE.Vector3(base.x-(p.x-centerX)*unit,base.y-(p.y-lm[9].y)*unit,base.z+(p.z-lm[9].z)*unit));
      h.visible=true;h.tracked=true;h.apply(points,fistScore(lm));h.sampleDt=sampleDt;
      h.updated=previouslyTracked.get(h.side)===true;
      if(!h.updated)h.previous.copy(h.center);
    });
  }
}

let faceDetector;
export async function makePhotoFace(image){
  if(!faceDetector){const files=await FilesetResolver.forVisionTasks('/wasm');faceDetector=await FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'/models/face_landmarker.task',delegate:'CPU'},runningMode:'IMAGE',numFaces:1});}
  const found=faceDetector.detect(image);const lm=found.faceLandmarks?.[0];
  if(!lm)throw new Error('No face detected. Use a well-lit, frontal neutral portrait.');
  const canonical=await fetch('/models/canonical_face_model.obj').then(r=>r.text());
  const indices=canonical.split('\n').filter(l=>l.startsWith('f ')).flatMap(l=>l.trim().split(/\s+/).slice(1).map(v=>Number(v.split('/')[0])-1));
  const minY=Math.min(...lm.slice(0,468).map(p=>p.y)),maxY=Math.max(...lm.slice(0,468).map(p=>p.y));
  const scale=.24/(maxY-minY),cx=(lm[234].x+lm[454].x)/2,cy=(minY+maxY)/2;
  const aspect=image.videoWidth?image.videoWidth/image.videoHeight:image.width/image.height;
  const pos=[],uv=[];
  lm.slice(0,468).forEach(p=>{pos.push((p.x-cx)*scale*aspect,-(p.y-cy)*scale,-p.z*scale*aspect+.025);uv.push(p.x,1-p.y);});
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(indices);geometry.computeVertexNormals();
  const canvas=document.createElement('canvas');canvas.width=image.videoWidth||image.width;canvas.height=image.videoHeight||image.height;canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  return {geometry,material:new THREE.MeshStandardMaterial({map:texture,roughness:.87,side:THREE.DoubleSide}),photo:canvas.toDataURL('image/jpeg',.92)};
}
