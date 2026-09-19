import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {capturePoseSignature,compareCapturePose,armViewCoverage} from '../src/capture-quality.js';
const p=(x,y,z)=>({x,y,z});
function fixture(){const body=Array.from({length:33},()=>p(0,0,0));body[12]=p(-.19,.21,.04);body[14]=p(-.20,.40,-.15);body[16]=p(-.14,.24,-.36);const hand=[p(0,0,0)];for(let f=0;f<5;f++)for(let j=0;j<4;j++)hand.push(p((2-f)*.015,-.03-j*.025,-.012));return {body,hand};}
const turn=(points,q)=>points.map(p=>{const v=new THREE.Vector3(p.x,p.y,p.z).applyQuaternion(q);return {x:v.x,y:v.y,z:v.z};});
test('whole-arm rotation adds viewpoint coverage without failing the rigid-pose check',()=>{
  const f=fixture(),start=capturePoseSignature(f.body,f.hand,'right');
  const q=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),Math.PI/3),rotated=capturePoseSignature(turn(f.body,q),turn(f.hand,q),'right');
  assert.equal(compareCapturePose(start,rotated).ok,true);assert.ok(armViewCoverage([start,rotated])>55);
});
test('wrist twisting is rejected and cannot count as new arm viewpoints',()=>{
  const f=fixture(),start=capturePoseSignature(f.body,f.hand,'right'),axis=new THREE.Vector3().copy(f.body[16]).sub(f.body[14]).normalize();
  const q=new THREE.Quaternion().setFromAxisAngle(axis,Math.PI/3),twisted=capturePoseSignature(f.body,turn(f.hand,q),'right');
  assert.equal(compareCapturePose(start,twisted).ok,false);assert.ok(armViewCoverage([start,twisted])<1e-4);
});
test('invalid and changed elbow poses are rejected before accepting another scan frame',()=>{
  const f=fixture(),start=capturePoseSignature(f.body,f.hand,'right');f.body[14].z-=.30;
  assert.equal(compareCapturePose(start,capturePoseSignature(f.body,f.hand,'right')).ok,false);
  assert.equal(compareCapturePose(start,capturePoseSignature([],f.hand,'right')).ok,false);
});
