import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {HeadHair,remapHairRoots} from '../src/head-hair.js';
function fixture(type='wavy'){
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute([-.02,.13,-.01,.02,.13,-.01,0,.13,-.04],3));g.setIndex([0,1,2]);g.computeVertexNormals();
 const spec={seed:12,hairlineY:.1,rootTriangles:Array.from({length:12},()=>[0,1,2]).flat(),rootWeights:Array.from({length:12},()=>[.3,.3,.4]).flat(),parameters:{type,colorSrgb:[.08,.06,.04],lengthMm:50,sideLengthMm:10,density:1,curlTightness:.7,curlRadiusMm:4,frizz:.3,waveLengthMm:14,rootLiftMm:3}};
 return {g,spec};
}
test('hair types change finite 3D geometry and saved controls reproduce it exactly',()=>{
 const shapes=[];
 for(const type of ['straight','wavy','curly','coily','braided','locs']){
  const {g,spec}=fixture(type),hair=new HeadHair(g,spec),copy=new HeadHair(g,JSON.parse(JSON.stringify(hair.spec)));
  const points=hair.geometry.attributes.position.array;assert.ok(points.every(Number.isFinite));assert.deepEqual(points,copy.geometry.attributes.position.array);shapes.push(points.slice());
  hair.rebuild(g,{lengthMm:95,rootLiftMm:8});assert.notDeepEqual(hair.geometry.attributes.position.array,points);hair.dispose();copy.dispose();g.dispose();
 }
 assert.notDeepEqual(shapes[0],shapes[1]);assert.notDeepEqual(shapes[1],shapes[2]);
});
test('scalp-bound hair translates and rotates with its surface, then returns to rest',()=>{
 const {g,spec}=fixture(),hair=new HeadHair(g,spec),before=hair.geometry.attributes.position.array.slice();
 g.translate(.015,-.004,.002);hair.updateSurface(g);
 const after=hair.geometry.attributes.position.array;for(let i=0;i<after.length;i++)assert.ok(Math.abs(after[i]-before[i]-[.015,-.004,.002][i%3])<1e-7);
 g.rotateZ(.15);g.computeVertexNormals();hair.updateSurface(g);assert.ok(after.every(Number.isFinite));
 hair.dispose();g.dispose();
});
test('GLB vertex reordering rebinds hair without attaching to the wrong face vertices',()=>{
 const {g,spec}=fixture(),old=g.attributes.position.array,next=new Float32Array([...old.slice(6),...old.slice(0,6)]);
 const mapped=remapHairRoots(spec,old,next);assert.deepEqual(mapped.rootTriangles.slice(0,3),[1,2,0]);
 assert.deepEqual(spec.rootTriangles.slice(0,3),[0,1,2]);assert.throws(()=>remapHairRoots(spec,old,new Float32Array(9)),/do not match/);
});
test('malformed imported root weights cannot corrupt the renderer',()=>{
 const {g,spec}=fixture();spec.rootWeights[0]=NaN;assert.throws(()=>new HeadHair(g,spec),/Invalid/);
});
test('binary GLB retains independent strand geometry, material and editable specification',async()=>{
 const {GLTFExporter}=await import('three/addons/exporters/GLTFExporter.js');
 const {GLTFLoader}=await import('three/addons/loaders/GLTFLoader.js');
 const previous=globalThis.FileReader;
 globalThis.FileReader=class {readAsArrayBuffer(blob){blob.arrayBuffer().then(result=>{this.result=result;this.onloadend?.();});}};
 try{
  const {g,spec}=fixture('curly'),hair=new HeadHair(g,spec);hair.userData.groom=hair.spec;
  const binary=await new GLTFExporter().parseAsync(hair,{binary:true});const parsed=await new GLTFLoader().parseAsync(binary,'');
  const restored=parsed.scene.children[0];assert.equal(restored.userData.accessory,'hair');assert.equal(restored.userData.groom.parameters.type,'curly');
  assert.deepEqual(restored.geometry.attributes.position.array,hair.geometry.attributes.position.array);
  assert.equal(restored.material.isMeshPhysicalMaterial,true);hair.dispose();g.dispose();
 }finally{globalThis.FileReader=previous;}
});
test('photo-guided detail stays close to captured locks and skips unseen roots without losing bindings',()=>{
 const {g,spec}=fixture();spec.mode='photo-detail';
 spec.photoGuides={directions:Array.from({length:12},()=>[1,0,0]).flat(),colors:Array.from({length:12},()=>[.12,.08,.05]).flat(),confidence:Array.from({length:12},(_,i)=>i%2?.8:0)};
 const hair=new HeadHair(g,spec);assert.equal(hair.strandCount,6);assert.equal(hair.material.isMeshBasicMaterial,true);
 const root=new THREE.Vector3(-.02*.3+.02*.3,.13,-.01*.6-.04*.4);
 const points=hair.geometry.attributes.position.array;for(let i=0;i<points.length;i+=3)assert.ok(new THREE.Vector3().fromArray(points,i).distanceTo(root)<.004);
 const before=points.slice();g.translate(.01,0,0);hair.updateSurface(g);
 for(let i=0;i<points.length;i++)assert.ok(Math.abs(points[i]-before[i]-(i%3===0?.01:0))<1e-7);
 const saved=new HeadHair(g,JSON.parse(JSON.stringify(hair.spec)));assert.equal(saved.material.isMeshBasicMaterial,true);hair.dispose();saved.dispose();g.dispose();
});
