import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {openMouthAperture} from '../src/mouth-aperture.js';

// The real MediaPipe canonical face: this is the topology the webcam and photo
// paths hand to installMesh, verbatim.
function canonicalFace(){
  const text=readFileSync(new URL('../public/models/canonical_face_model.obj',import.meta.url),'utf8');
  const position=[],index=[];
  for(const line of text.split('\n')){
    if(line.startsWith('v '))position.push(...line.trim().split(/\s+/).slice(1,4).map(Number));
    else if(line.startsWith('f '))index.push(...line.trim().split(/\s+/).slice(1).map(v=>Number(v.split('/')[0])-1));
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(position,3));
  geometry.setIndex(index);
  return geometry;
}

const boundaryEdges=geometry=>{
  const index=geometry.getIndex().array,seen=new Map();
  for(let f=0;f<index.length;f+=3){
    const t=[index[f],index[f+1],index[f+2]];
    for(const [a,b] of [[t[0],t[1]],[t[1],t[2]],[t[2],t[0]]]){
      const key=a<b?`${a}:${b}`:`${b}:${a}`;
      seen.set(key,(seen.get(key)||0)+1);
    }
  }
  return [...seen.entries()].filter(([,c])=>c===1).map(([k])=>k.split(':').map(Number));
};
// Boundary edges whose midpoint is near the mouth. The canonical face is an open
// frontal shell, so its silhouette is boundary too — only the mouth matters here.
const boundaryAtMouth=geometry=>{
  const p=geometry.attributes.position;
  const mouth=[(p.getX(13)+p.getX(14))/2,(p.getY(13)+p.getY(14))/2];
  // The raw canonical .obj is in its own units, not metres, so size the window
  // off the model itself. makePhotoFace normalises before any of this is used.
  const box=new THREE.Box3().setFromBufferAttribute(p);
  const span=box.max.y-box.min.y;
  return boundaryEdges(geometry).filter(([a,b])=>{
    const x=(p.getX(a)+p.getX(b))/2,y=(p.getY(a)+p.getY(b))/2;
    return Math.abs(x-mouth[0])<span*.14&&Math.abs(y-mouth[1])<span*.05;
  }).length;
};

test('the canonical face starts with its mouth sealed',()=>{
  const g=canonicalFace();
  assert.equal(g.attributes.position.count,468);
  // It is an open frontal shell, so its outline is boundary — but the mouth is not.
  assert.ok(boundaryEdges(g).length>0,'the face outline is an open boundary');
  assert.equal(boundaryAtMouth(g),0,'the mouth itself starts sealed');
});

test('cutting the aperture opens the mouth and removes exactly the sealing band',()=>{
  const g=canonicalFace();
  const before=g.getIndex().count/3;
  const aperture=openMouthAperture(g,null);
  assert.ok(aperture,'the canonical topology must be recognised');
  assert.equal(aperture.strategy,'exact');
  assert.equal(aperture.removed,18,'the sealing band is 18 triangles');
  assert.equal(g.getIndex().count/3,before-18);
  assert.ok(boundaryAtMouth(g)>0,'there must now be a hole at the mouth');
});

test('the vertex buffer is untouched, which is what keeps the Newton binding valid',()=>{
  const g=canonicalFace();
  const before=g.attributes.position.array.slice();
  const count=g.attributes.position.count;
  openMouthAperture(g,null);
  assert.equal(g.attributes.position.count,count,'vertex count must not change');
  assert.deepEqual(g.attributes.position.array,before,'no vertex may move');
});

test('the aperture sits at the mouth and is wider than it is tall',()=>{
  const g=canonicalFace();
  const {centre,width,height}=openMouthAperture(g,null);
  const box=new THREE.Box3().setFromBufferAttribute(g.attributes.position);
  assert.ok(centre[1]<(box.min.y+box.max.y)/2,'the mouth is in the lower half of the face');
  assert.ok(Math.abs(centre[0])<width,'the mouth straddles the midline');
  assert.ok(width>height,'a closed mouth is a wide slit');
});

test('cutting twice is a no-op',()=>{
  const g=canonicalFace();
  const first=openMouthAperture(g,null);
  const faces=g.getIndex().count;
  const second=openMouthAperture(g,null);
  assert.equal(g.getIndex().count,faces,'a second pass must not remove anything further');
  assert.equal(second,first,'the cached aperture is returned as-is');
});

test('a head with no recognisable mouth is left exactly alone',()=>{
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute([0,0,0, 1,0,0, 0,1,0, 1,1,0],3));
  g.setIndex([0,1,2, 1,3,2]);
  const faces=g.getIndex().count;
  assert.equal(openMouthAperture(g,null),null,'no anchors and no canonical topology means no cut');
  assert.equal(g.getIndex().count,faces,'the geometry must be untouched');
});

test('the geometric pass cuts a slit that straddles the lip line',()=>{
  // A small sealed grid standing in for a lip region: two rows of lip vertices
  // with a band of triangles bridging them, exactly what a sealed mouth is.
  const position=[],index=[];
  const cols=9;
  for(let row=0;row<4;row++)for(let c=0;c<cols;c++){
    position.push(-.04+c*.01, -.055+row*.01, .06);
  }
  for(let row=0;row<3;row++)for(let c=0;c<cols-1;c++){
    const a=row*cols+c,b=a+1,d=a+cols,e=d+1;
    index.push(a,b,d, b,e,d);
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(position,3));
  g.setIndex(index);
  const before=g.getIndex().count/3;
  const anchors={13:[0,-.030,.06],14:[0,-.038,.06],61:[-.035,-.034,.06],291:[.035,-.034,.06]};
  const aperture=openMouthAperture(g,anchors,{trustAnchors:true});
  assert.ok(aperture,'a sealed lip band must be found');
  assert.equal(aperture.strategy,'geometric');
  assert.ok(aperture.removed>0&&aperture.removed<before,'some but not all of the band is cut');
  assert.ok(boundaryEdges(g).length>0,'the slit must actually be open');
  assert.ok(aperture.width>aperture.height,'and it must be a slit, not a gash');
});

test('anchors the caller will not vouch for never cut anything',()=>{
  // Guessed anchors put the cut 6 mm low on the reference head and slit the chin.
  // A head whose mouth is only estimated keeps its lips sealed.
  const position=[],index=[];
  const cols=9;
  for(let row=0;row<4;row++)for(let c=0;c<cols;c++)position.push(-.04+c*.01,-.055+row*.01,.06);
  for(let row=0;row<3;row++)for(let c=0;c<cols-1;c++){
    const a=row*cols+c,b=a+1,d=a+cols,e=d+1;
    index.push(a,b,d, b,e,d);
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(position,3));
  g.setIndex(index);
  const faces=g.getIndex().count;
  const anchors={13:[0,-.030,.06],14:[0,-.038,.06],61:[-.035,-.034,.06],291:[.035,-.034,.06]};
  assert.equal(openMouthAperture(g,anchors),null,'untrusted anchors must not cut');
  assert.equal(g.getIndex().count,faces,'the geometry must be untouched');
});
