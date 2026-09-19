import test from 'node:test';
import assert from 'node:assert/strict';
import {FaceImpactRig} from '../src/impact-rig.js';

const anchors={13:[0,-.040,.020],14:[0,-.042,.020],152:[0,-.100,0],50:[-.055,-.004,0],280:[.055,-.004,0],61:[-.03,-.041,.015],291:[.03,-.041,.015],159:[-.037,.040,.010],386:[.037,.040,.010]};
const point={x:-.05,y:-.004,z:0},direction={x:.7,y:0,z:-.7};
function rig(points){const rest=new Float32Array(points.flat());return {rest,rig:new FaceImpactRig(rest,anchors)};}
const magnitude=(offset,index)=>Math.hypot(...offset.slice(index*3,index*3+3));

test('a cheek punch moves the lips and jaw, squeezes the eyelids, and leaves the back and scalp attached',()=>{
  const points=[anchors[50],anchors[61],anchors[152],[-.037,.044,.01],[-.037,.030,.01],[0,.14,-.04],[0,0,-.20],[0,-.15,0]];
  const {rest,rig:r}=rig(points);r.trigger(rest,point,direction,1,.6);r.step(.12);
  assert.ok(magnitude(r.offset,0)>.015,'cheek must visibly compress and shear');
  assert.ok(magnitude(r.offset,1)>.015,'lips must follow the cheek');
  assert.ok(magnitude(r.offset,2)>.010,'the jaw silhouette must respond');
  assert.ok(r.offset[0]>0&&r.offset[3]>0&&r.offset[6]>0,'transport follows the hook');
  const eyeGap=(rest[10]+r.offset[10])-(rest[13]+r.offset[13]);
  assert.ok(eyeGap>0&&eyeGap<.014*.7,'eyelids squeeze without crossing');
  for(const i of [5,6,7])assert.equal(magnitude(r.offset,i),0,'unaffected attachments remain fixed');
});

test('left and right hooks mirror their facial deformation',()=>{
  const points=[anchors[50],anchors[61],anchors[152],[-.037,.035,.01],[-.085,-.055,-.03]];
  const left=rig(points),right=rig(points.map(([x,y,z])=>[-x,y,z]));
  left.rig.trigger(left.rest,point,direction,1,.6);
  right.rig.trigger(right.rest,{...point,x:-point.x},{...direction,x:-direction.x},1,.6);
  left.rig.step(.12);right.rig.step(.12);
  left.rig.offset.forEach((value,i)=>assert.ok(Math.abs(value*(i%3===0?-1:1)-right.rig.offset[i])<1e-7));
});

test('repeated and alternating hooks stay bounded and recover without changing the rest pose',()=>{
  const {rest,rig:r}=rig(Object.values(anchors)),original=rest.slice();
  let peak=0;
  for(let frame=0;frame<240;frame++){
    if(frame%6===0){const side=frame%12===0?-1:1;r.trigger(rest,{...point,x:side*.05},direction,4,1);}
    r.step(1/120);
    assert.ok(r.offset.every(Number.isFinite));
    for(let i=0;i<rest.length/3;i++)peak=Math.max(peak,magnitude(r.offset,i));
  }
  assert.ok(peak>.015&&peak<.033);
  for(let i=0;i<150;i++)r.step(1/120);
  assert.ok(r.offset.every(v=>v===0));assert.deepEqual(rest,original);
  r.trigger(rest,point,direction,1,.6);r.step(.12);r.reset();
  assert.equal(r.events.length,0);assert.ok(r.offset.every(v=>v===0));
});

test('speed and softness scale the response and paused time preserves a held pose',()=>{
  const points=Object.values(anchors),peaks=[];
  for(const [speed,softness] of [[.4,0],[.8,0],[.8,1]]){
    const {rest,rig:r}=rig(points);r.trigger(rest,point,direction,speed,softness);r.step(.06);
    assert.equal(r.hasPeaked,false);r.step(.06);assert.equal(r.hasPeaked,true);
    const held=r.offset.slice();r.step(0);assert.deepEqual(r.offset,held);
    peaks.push(Math.max(...points.map((_,i)=>magnitude(r.offset,i))));
  }
  assert.ok(peaks[0]<peaks[1]&&peaks[1]<peaks[2]);
});
