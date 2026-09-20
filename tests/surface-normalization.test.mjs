import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {normalizeHead} from '../src/surface.js';

test('bust normalization sizes the head and retains the torso below the view',()=>{
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute([
    -1,5,-1,1,5,1,-1,3.5,1,1,3.5,-1, // head
    -3,1,0,3,1,0,-1,0,0,1,0,0,0,0,0, // shoulders and chest
  ],3));
  normalizeHead(geometry);const box=geometry.boundingBox,size=box.getSize(new THREE.Vector3());
  assert.ok(Math.abs(box.max.y-.14)<1e-6,'head top is framed around the target origin');
  assert.ok(size.y>.4,'the torso is retained instead of shrinking the head to fit the whole bust');
});
