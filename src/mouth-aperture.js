// Open the mouth by removing the triangles that seal it.
//
// Every head here is watertight — `build_photo_face.py` rejects anything else —
// so the lips are a continuous surface and the jaw rig can only stretch skin
// across them. To part the lips the sealing band has to go.
//
// CRITICAL: this removes FACES ONLY and never touches the vertex buffer. The
// vertex count is what `NewtonFaceDynamics` validates its binding against
// (src/newton-dynamics.js), what the physics cage indexes into, and what the
// UVs, the impact rig and the speech rig are all addressed by. Rewriting the
// index buffer leaves every one of those intact.
//
// Two strategies, chosen by what the mesh actually is:
//
//   exact      The MediaPipe canonical face seals the aperture with 18 known
//              triangles spanning the inner-lip ring, and a landmark proxy mesh
//              carries that topology verbatim. When those faces are present we
//              remove precisely them — no guessing at all.
//   geometric  A fitted scan, whose anchors are real MediaPipe landmarks
//              triangulated in 3D by build_photo_face.py. Take the triangles
//              inside the lip aperture that straddle the lip line.
//
// The geometric pass runs ONLY when the caller vouches for the anchors
// (`trustAnchors`). Measured on the reference head, guessed anchors put the cut
// 6 mm low and slit the chin: the default anchor table says the mouth is at
// y=-0.041 and `detectAnchors` proportions say -0.051, while the lips are near
// -0.035. Searching for the crease does not rescue it either — the deepest
// recess between nose and chin is the labiomental fold UNDER the lower lip, not
// the lip seam, so a z-minimum search lands lower still. A head whose mouth
// cannot be located keeps its lips sealed and reads as stretching, which is
// wrong-looking in no way at all.

import * as THREE from 'three';
import {insidePolygon} from './lip-detect.js';

// MediaPipe inner-lip contour. The two rings share their corner vertices.
const UPPER=[78,191,80,81,82,13,312,311,310,415,308];
const LOWER=[78,95,88,178,87,14,317,402,318,324,308];
const INNER=new Set([...UPPER,...LOWER]);
const UPPER_ONLY=new Set(UPPER.filter(v=>!LOWER.includes(v)));
const LOWER_ONLY=new Set(LOWER.filter(v=>!UPPER.includes(v)));
const EXPECTED_EXACT=18;

const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));

/**
 * @returns {{strategy:string,removed:number,ring:number[],centre:number[],width:number,height:number}|null}
 *          null when no aperture could be identified, in which case the
 *          geometry is left exactly as it was.
 */
export function openMouthAperture(geometry,anchors,{trustAnchors=false,detection=null}={}){
  if(geometry.userData?.mouthAperture)return geometry.userData.mouthAperture;
  const index=geometry.getIndex();
  if(!index)return null;
  const faces=index.array,count=faces.length/3;
  const position=geometry.attributes.position;
  // Detection first: it is measured on this exact mesh, so it beats both the
  // canonical-topology shortcut and anything derived from anchors.
  const doomed=projectedFaces(faces,count,position,detection)
    ??exactFaces(faces,count,position)
    ??(trustAnchors?geometricFaces(faces,count,position,anchors):null);
  if(!doomed||!doomed.size)return null;

  const kept=new Uint32Array((count-doomed.size)*3);
  let write=0;
  for(let f=0;f<count;f++){
    if(doomed.has(f))continue;
    kept[write++]=faces[f*3];kept[write++]=faces[f*3+1];kept[write++]=faces[f*3+2];
  }
  geometry.setIndex(new THREE.BufferAttribute(kept,1));
  geometry.computeVertexNormals();

  // The hole's rim, for sizing whatever gets put behind it.
  const rim=new Set();
  for(const f of doomed)for(let k=0;k<3;k++)rim.add(faces[f*3+k]);
  const ring=[...rim];
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity,sx=0,sy=0,sz=0;
  for(const v of ring){
    const x=position.getX(v),y=position.getY(v),z=position.getZ(v);
    minX=Math.min(minX,x);maxX=Math.max(maxX,x);
    minY=Math.min(minY,y);maxY=Math.max(maxY,y);
    sx+=x;sy+=y;sz+=z;
  }
  const aperture={
    strategy:doomed.strategy,
    removed:doomed.size,
    ring,
    centre:[sx/ring.length,sy/ring.length,sz/ring.length],
    width:maxX-minX,
    height:maxY-minY,
  };
  geometry.userData=geometry.userData||{};
  geometry.userData.mouthAperture=aperture;
  return aperture;
}

/**
 * Triangles that fall inside the detected inner-lip contour.
 *
 * This is the general case and the only one that works on an arbitrary head.
 * The contour comes from running the face landmarker on a render of this mesh
 * (src/lip-detect.js), so it is measured rather than inferred, and the test is
 * a polygon rather than a box — a wide mouth, a small mouth or a crooked one
 * all cut correctly.
 *
 * Only forward-facing triangles are eligible: the contour is a 2D silhouette,
 * so without that the back of the skull behind the mouth is cut away too.
 */
function projectedFaces(faces,count,position,detection){
  if(!detection?.lipPolygon?.length||!detection.project)return null;
  const {lipPolygon,project}=detection;
  const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3();
  const ab=new THREE.Vector3(),ac=new THREE.Vector3(),normal=new THREE.Vector3();
  const centroid=new THREE.Vector3();
  const found=new Set();
  for(let f=0;f<count;f++){
    const i0=faces[f*3],i1=faces[f*3+1],i2=faces[f*3+2];
    a.fromBufferAttribute(position,i0);
    b.fromBufferAttribute(position,i1);
    c.fromBufferAttribute(position,i2);
    // Face the camera, or this cuts a matching hole out of the back of the head.
    ab.subVectors(b,a);ac.subVectors(c,a);normal.crossVectors(ab,ac);
    if(normal.z<=0)continue;
    centroid.copy(a).add(b).add(c).multiplyScalar(1/3);
    const p=project(centroid);
    if(insidePolygon(lipPolygon,p.x,p.y))found.add(f);
  }
  if(!found.size)return null;
  found.strategy='detected';
  return found;
}

/** The 18 canonical aperture triangles, if this mesh still carries them. */
function exactFaces(faces,count,position){
  if(position.count<468)return null;
  const found=new Set();
  for(let f=0;f<count;f++){
    const a=faces[f*3],b=faces[f*3+1],c=faces[f*3+2];
    if(!INNER.has(a)||!INNER.has(b)||!INNER.has(c))continue;
    const hasUpper=UPPER_ONLY.has(a)||UPPER_ONLY.has(b)||UPPER_ONLY.has(c);
    const hasLower=LOWER_ONLY.has(a)||LOWER_ONLY.has(b)||LOWER_ONLY.has(c);
    if(hasUpper&&hasLower)found.add(f);
  }
  // Require the whole band. A partial match means this is not that topology
  // (or it has already been subdivided), and a half-open mouth is worse than
  // falling through to the geometric pass.
  if(found.size!==EXPECTED_EXACT)return null;
  found.strategy='exact';
  return found;
}

/** Triangles inside the lip aperture that straddle the lip line. */
function geometricFaces(faces,count,position,anchors){
  const upper=anchors?.[13],lower=anchors?.[14];
  if(!upper||!lower)return null;
  const my=(upper[1]+lower[1])/2,mx=(upper[0]+lower[0])/2,mz=(upper[2]+lower[2])/2;
  const cornerL=anchors[61],cornerR=anchors[291];
  // Half-width of the opening. Pull in from the corners: the very corner of a
  // mouth stays shut, and cutting to it leaves a gash across the cheeks.
  const halfWidth=(cornerL&&cornerR?Math.abs(cornerR[0]-cornerL[0])/2:.030)*.82;
  // Lips only meet over a shallow band, so the vertical catch is deliberately
  // tight — this is the seam, not the whole mouth region.
  const halfHeight=Math.max(.004,Math.abs(upper[1]-lower[1]))*1.6+.0035;
  const found=new Set();
  for(let f=0;f<count;f++){
    let cx=0,cy=0,cz=0,above=0,below=0,inside=true;
    for(let k=0;k<3;k++){
      const v=faces[f*3+k];
      const x=position.getX(v),y=position.getY(v),z=position.getZ(v);
      cx+=x/3;cy+=y/3;cz+=z/3;
      if(Math.abs(x-mx)>halfWidth||Math.abs(y-my)>halfHeight*2.2){inside=false;break;}
      if(y>my)above++;else below++;
    }
    if(!inside)continue;
    // Must bridge the lip line, sit within the aperture, and face forward —
    // never catch the back of the head on a mesh whose midline wraps around.
    if(!above||!below)continue;
    if(Math.abs(cx-mx)>halfWidth||Math.abs(cy-my)>halfHeight)continue;
    if(cz<mz-.045)continue;
    found.add(f);
  }
  if(!found.size)return null;
  found.strategy='geometric';
  return found;
}

/**
 * A dark void behind the lips. Without it the mouth hole looks straight through
 * a double-sided head at the inside of the skull, which reads as a bug rather
 * than a mouth. This does not deform with the face; it only has to stay darker
 * than everything around it and stay occluded when the lips are shut.
 */
export class MouthCavity extends THREE.Mesh{
  constructor({centre,width,height}){
    const depth=clamp(width*.55,.012,.045);
    super(
      new THREE.SphereGeometry(.5,20,14),
      new THREE.MeshBasicMaterial({color:0x140d0e,side:THREE.BackSide,toneMapped:false}),
    );
    this.name='Mouth interior';
    this.userData.accessory='mouth-cavity';
    // Wide and shallow, tucked just behind the lip plane so a closed mouth
    // never shows it and an open one is full of it.
    this.scale.set(Math.max(width*1.15,.02),Math.max(height*3.2,.022),depth*2);
    this.position.set(centre[0],centre[1],centre[2]-depth*.55);
    this.renderOrder=-1;
  }
  dispose(){this.removeFromParent();this.geometry.dispose();this.material.dispose();}
}
