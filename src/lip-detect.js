// Find a head's real features by looking at it.
//
// Anchors are the weak point everywhere else: the reference head runs on a
// hardcoded table, `detectAnchors` guesses the mouth from nose-and-chin
// proportions, and both put the lip line 14–19 mm out. Searching the mesh for a
// crease is worse — the deepest recess between nose and chin is the fold UNDER
// the lower lip, not the lip seam.
//
// So stop inferring. Render the head and run the same MediaPipe FaceLandmarker
// the capture path already uses on that render, then raycast each landmark back
// onto the surface. Whatever the head is — photo proxy, fitted scan, Sketchfab
// bust, Meshy generation — if it reads as a face, its mouth is found where the
// mouth actually is.
//
// NOTHING IS MOVED OR REPARENTED. An earlier version lifted `headPivot` into a
// scratch scene to render it, which silently dropped its parent's offset
// (`target` sits at z=-0.55 and slides with the distance slider): the camera was
// framed in one space and the head rendered in another, so the detector saw a
// badly-framed head and every projected coordinate inherited the error. Here the
// scene's other objects are hidden in place instead, which keeps one coordinate
// frame across the render, the raycast and the projection — the property the
// whole method depends on.

import * as THREE from 'three';
import {ensureFaceDetector} from './hands.js';

const ANCHOR_KEYS=[1,13,14,50,61,152,159,280,291,386];
// The mouth aperture contour, in ring order.
export const INNER_LIP=[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];
const RENDER_SIZE=512;
// How much of the frame the head fills. A face that is too tight or too loose in
// frame is a common reason for the landmarker to decline, so try a few.
const FRAMINGS=[1.45,1.9,1.15,2.5];
// Outer face landmarks: temples, jaw corners, forehead and chin. Used to frame
// the second pass on the face alone.
const FACE_EXTENT_LANDMARKS=[10,152,234,454,162,389];
// A little margin so the crop is not clipped at the hairline or the chin.
const FACE_FRAMING=1.5;

// The last render handed to the detector, kept for diagnosis when a head is
// rejected. Only populated when a caller asks for debug output.
let lastRender=null;
export const lastDetectorRender=()=>lastRender;

export async function detectFaceOnMesh({renderer,scene,mesh,headPivot,size=RENDER_SIZE,debug=false}){
  if(!renderer||!scene||!mesh||!headPivot)return null;
  let detector;
  try{detector=await ensureFaceDetector();}catch{return null;}

  mesh.updateMatrixWorld(true);
  mesh.geometry.computeBoundingBox();
  const local=mesh.geometry.boundingBox;
  if(!local)return null;
  const localSize=local.getSize(new THREE.Vector3());
  const centre=local.getCenter(new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
  const worldScale=new THREE.Vector3();
  mesh.matrixWorld.decompose(new THREE.Vector3(),new THREE.Quaternion(),worldScale);
  const extent=Math.max(localSize.x*worldScale.x,localSize.y*worldScale.y);
  if(!(extent>0))return null;

  // Look down the head's OWN forward axis, so a head turned by the yaw/pitch
  // sliders or mid-recoil is still seen square-on without moving it.
  const forward=new THREE.Vector3(0,0,1).transformDirection(mesh.matrixWorld).normalize();
  const up=new THREE.Vector3(0,1,0).transformDirection(mesh.matrixWorld).normalize();

  const hidden=[],headParts=new Set();
  for(let o=headPivot;o;o=o.parent)headParts.add(o);
  headPivot.traverse(o=>headParts.add(o));
  scene.traverse(o=>{
    if(o===scene||o.isLight||headParts.has(o))return;
    if(o.visible){o.visible=false;hidden.push(o);}
  });
  const background=scene.background,fog=scene.fog;
  scene.background=new THREE.Color(0x8d9299);scene.fog=null;
  const previousTarget=renderer.getRenderTarget();

  const camera=new THREE.PerspectiveCamera(35,1,extent/50,extent*50);
  const target=new THREE.WebGLRenderTarget(size,size);
  const pixels=new Uint8Array(size*size*4);
  const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
  const context=canvas.getContext('2d',{willReadFrequently:true});
  const image=context.createImageData(size,size);

  // Render from a given camera and ask the landmarker what it sees.
  const attempt=(camera)=>{
    camera.updateMatrixWorld(true);
    renderer.setRenderTarget(target);
    renderer.render(scene,camera);
    renderer.setRenderTarget(previousTarget??null);
    renderer.readRenderTargetPixels(target,0,0,size,size,pixels);
    // readRenderTargetPixels hands back rows bottom-up; canvases are top-down.
    for(let row=0;row<size;row++){
      const from=(size-1-row)*size*4;
      image.data.set(pixels.subarray(from,from+size*4),row*size*4);
    }
    context.putImageData(image,0,0);
    let found=null;
    try{found=detector.detect(canvas)?.faceLandmarks?.[0];}catch{found=null;}
    if(debug)lastRender={image:canvas.toDataURL('image/jpeg',.85),detected:!!found};
    return found&&found.length>=468?found:null;
  };
  const aim=(at,fill)=>{
    const distance=fill/(2*Math.tan(camera.fov*Math.PI/360));
    camera.up.copy(up);
    camera.position.copy(at).addScaledVector(forward,distance);
    camera.lookAt(at);
    return camera;
  };

  let landmarks=null,used=null;
  try{
    // Pass one: find the face anywhere on the mesh. The bounding box includes
    // neck and shoulders, so the face itself is a small part of this frame.
    for(const framing of FRAMINGS){
      landmarks=attempt(aim(centre,extent*framing));
      if(landmarks){used=framing;break;}
    }
    // Pass two: reframe on the FACE and look again. This is what makes the lip
    // landmarks precise — MediaPipe crops the face out of the frame before
    // fitting the mesh, so a face filling 25% of a 512 px render is fitted from
    // a fraction of the pixels a tightly framed one gets.
    if(landmarks){
      const raycaster=new THREE.Raycaster();
      const span=new THREE.Box3();
      let hits=0;
      for(const key of FACE_EXTENT_LANDMARKS){
        const l=landmarks[key];
        raycaster.setFromCamera(new THREE.Vector2(l.x*2-1,-(l.y*2-1)),camera);
        const found=raycaster.intersectObject(mesh,false);
        if(found.length){span.expandByPoint(found[0].point);hits++;}
      }
      if(hits>=4&&!span.isEmpty()){
        const faceCentre=span.getCenter(new THREE.Vector3());
        const faceSize=span.getSize(new THREE.Vector3());
        const faceExtent=Math.max(faceSize.x,faceSize.y);
        if(faceExtent>0){
          const closer=attempt(aim(faceCentre,faceExtent*FACE_FRAMING));
          if(closer){landmarks=closer;used='face';}
          else aim(centre,extent*used);   // keep camera and landmarks in step
        }
      }
    }
  }finally{
    renderer.setRenderTarget(previousTarget??null);
    target.dispose();
    scene.background=background;scene.fog=fog;
    for(const o of hidden)o.visible=true;
  }
  if(!landmarks)return null;

  // One frame for everything: the camera above, the mesh exactly where it has
  // been all along.
  const raycaster=new THREE.Raycaster();
  const inverse=new THREE.Matrix4().copy(mesh.matrixWorld).invert();
  const toNdc=l=>new THREE.Vector2(l.x*2-1,-(l.y*2-1));
  const hit=l=>{
    raycaster.setFromCamera(toNdc(l),camera);
    const found=raycaster.intersectObject(mesh,false);
    return found.length?found[0].point.clone().applyMatrix4(inverse):null;
  };

  const anchors={};
  for(const key of ANCHOR_KEYS){
    const point=hit(landmarks[key]);
    if(point)anchors[key]=[point.x,point.y,point.z];
  }
  if(!anchors[13]||!anchors[14]||!anchors[152])return null;

  const lipPolygon=INNER_LIP.map(i=>{const n=toNdc(landmarks[i]);return {x:n.x,y:n.y};});
  const matrixWorld=mesh.matrixWorld.clone();
  const project=(localPoint)=>{
    const v=localPoint.clone().applyMatrix4(matrixWorld).project(camera);
    return {x:v.x,y:v.y};
  };

  return {
    anchors,project,lipPolygon,framing:used,
    // Sanity figures a caller can assert on, and an image to look at when a head
    // comes out wrong. No landmark data is drawn unless it is asked for.
    mouthWidthNdc:Math.max(...lipPolygon.map(p=>p.x))-Math.min(...lipPolygon.map(p=>p.x)),
    debugImage:debug?annotate(canvas,landmarks,size):null,
  };
}

/** Draw the detected contour over the render, for eyeballing a bad head. */
function annotate(canvas,landmarks,size){
  const context=canvas.getContext('2d');
  context.strokeStyle='#19ff9b';context.lineWidth=2;context.beginPath();
  INNER_LIP.forEach((i,n)=>{
    const x=landmarks[i].x*size,y=landmarks[i].y*size;
    n?context.lineTo(x,y):context.moveTo(x,y);
  });
  context.closePath();context.stroke();
  context.fillStyle='#ff3b6b';
  for(const i of [1,13,14,61,291,152]){
    context.beginPath();context.arc(landmarks[i].x*size,landmarks[i].y*size,3,0,Math.PI*2);context.fill();
  }
  return canvas.toDataURL('image/jpeg',.85);
}

/** Even-odd point-in-polygon, in the detector's normalised device space. */
export function insidePolygon(polygon,x,y){
  let inside=false;
  for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
    const a=polygon[i],b=polygon[j];
    if((a.y>y)!==(b.y>y)&&x<(b.x-a.x)*(y-a.y)/(b.y-a.y)+a.x)inside=!inside;
  }
  return inside;
}
