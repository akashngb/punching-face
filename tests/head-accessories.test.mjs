import test from 'node:test';
import assert from 'node:assert/strict';
import {HeadGlasses} from '../src/head-accessories.js';
import * as THREE from 'three';
function fixture(){
 const ring=x=>Array.from({length:16},(_,i)=>[x+.024*Math.cos(i*Math.PI/8),.04+.016*Math.sin(i*Math.PI/8),.024]);
 return {rims:[ring(-.037),ring(.037)],bridge:[[-.013,.043,.025],[0,.048,.03],[.013,.043,.025]],temples:[[[.06,.046,.023],[.08,.046,-.05],[.08,.03,-.1]],[[-.06,.046,.023],[-.08,.046,-.05],[-.08,.03,-.1]]],frameColor:[.05,.05,.05],radius:.0015,lensTint:0};
}
test('glasses have independent three-dimensional geometry that round-trips through a saved specification',()=>{
 const spec=fixture(),g=new HeadGlasses(spec);const restored=new HeadGlasses(JSON.parse(JSON.stringify(g.spec)));
 const bounds=new THREE.Box3().setFromObject(g);assert.ok(bounds.max.z-bounds.min.z>.1);
 assert.equal(g.children.length,7);g.children.forEach((m,i)=>{assert.ok(m.geometry.attributes.position.array.every(Number.isFinite));assert.deepEqual(m.geometry.attributes.position.array,restored.children[i].geometry.attributes.position.array);assert.equal(m.userData.accessory,'eyeglasses');assert.equal(m.geometry.morphAttributes.position,undefined);});
 spec.rims[0][0][0]=.8;assert.notEqual(g.spec.rims[0][0][0],.8);g.dispose();restored.dispose();
});
test('invalid eyewear cannot install non-finite vertices into the renderer',()=>{
 const spec=fixture();spec.rims[0][3][2]=Infinity;assert.throws(()=>new HeadGlasses(spec),/Invalid/);
});
