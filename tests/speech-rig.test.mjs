import test from 'node:test';
import assert from 'node:assert/strict';
import {FaceSpeechRig} from '../src/speech-rig.js';
import {createSyntheticSignal} from '../src/omni/mouth-signal.js';

// Same reference-frame anchor table the impact-rig tests use.
const anchors={13:[0,-.040,.020],14:[0,-.042,.020],152:[0,-.100,0],50:[-.055,-.004,0],280:[.055,-.004,0],61:[-.03,-.041,.015],291:[.03,-.041,.015],159:[-.037,.040,.010],386:[.037,.040,.010]};
// chin, lower lip, mouth centre, both corners, then a scalp, an occiput and a neck
// vertex that speech must never touch.
const points=[anchors[152],anchors[14],[0,-.041,.020],anchors[61],anchors[291],[0,.14,-.04],[0,0,-.20],[0,-.15,0]];
const CHIN=0,LIP=1,MOUTH=2,CORNER_L=3,CORNER_R=4,SCALP=5,BACK=6,NECK=7;

const build=(pts=points)=>{const rest=new Float32Array(pts.flat());return {rest,rig:new FaceSpeechRig(rest,anchors)};};
const magnitude=(offset,index)=>Math.hypot(...offset.slice(index*3,index*3+3));
// Drive the rig to a settled pose; the follow filter is ~25 ms.
const settle=(rig,signal,duck=1)=>{rig.set(signal);for(let i=0;i<40;i++)rig.step(1/60,duck);};

test('a silent face is left untouched, exactly',()=>{
  const {rig}=build();
  rig.set({open:0,spread:0,round:0});
  for(let i=0;i<10;i++)rig.step(1/60);
  assert.ok(rig.offset.every(v=>v===0),'silence must not move a single vertex');
});

test('opening the mouth swings the jaw and leaves the skull, occiput and neck fixed',()=>{
  const {rig}=build();
  settle(rig,{open:1});
  assert.ok(magnitude(rig.offset,CHIN)>.004,'the chin must carry the jaw swing');
  assert.ok(magnitude(rig.offset,LIP)>.001,'the lower lip must follow the jaw');
  for(const i of [SCALP,BACK,NECK])assert.equal(magnitude(rig.offset,i),0,`vertex ${i} must stay attached`);
  // A hinge below and behind the mouth sweeps the chin downward.
  assert.ok(rig.offset[CHIN*3+1]<0,'the chin must travel down, not up');
});

test('spread and round move the mouth corners in opposite directions',()=>{
  const wide=build(),tight=build();
  settle(wide.rig,{spread:1});
  settle(tight.rig,{round:1});
  const wideL=wide.rig.offset[CORNER_L*3],wideR=wide.rig.offset[CORNER_R*3];
  const tightL=tight.rig.offset[CORNER_L*3],tightR=tight.rig.offset[CORNER_R*3];
  assert.ok(wideL<0&&wideR>0,'spread pulls both corners outward');
  assert.ok(tightL>0&&tightR<0,'round gathers both corners inward');
});

test('every shape is mirror-symmetric about the midline',()=>{
  const left=build(),right=build(points.map(([x,y,z])=>[-x,y,z]));
  for(const signal of [{open:1},{spread:1},{round:1}]){
    left.rig.reset();right.rig.reset();
    settle(left.rig,signal);settle(right.rig,signal);
    left.rig.offset.forEach((value,i)=>{
      const mirrored=value*(i%3===0?-1:1);
      assert.ok(Math.abs(mirrored-right.rig.offset[i])<1e-7,`asymmetry at ${i} for ${JSON.stringify(signal)}`);
    });
  }
});

test('no shape, or any combination of them, exceeds the travel budget',()=>{
  const {rig}=build();
  settle(rig,{open:1,spread:1,round:1});
  for(let i=0;i<points.length;i++){
    assert.ok(magnitude(rig.offset,i)<.03,'a speaking mouth must stay well inside the impact clamp');
  }
});

test('ducking scales the whole rig toward rest so a punch reads over a sentence',()=>{
  const loud=build(),ducked=build();
  settle(loud.rig,{open:1});
  settle(ducked.rig,{open:1},.15);
  assert.ok(magnitude(ducked.rig.offset,CHIN)<magnitude(loud.rig.offset,CHIN)*.4,'a landed punch must win');
});

test('a head with no mouth line goes inert instead of guessing',()=>{
  const rest=new Float32Array(points.flat());
  const rig=new FaceSpeechRig(rest,{152:[0,-.1,0]});   // chin only, no lips
  settle(rig,{open:1,spread:1,round:1});
  assert.ok(rig.offset.every(v=>v===0),'without a mouth line nothing may move');
});

test('no shape disturbs the eyes, however faintly',()=>{
  // Found on the real 13k-vertex head, not on this fixture: an earlier mask faded
  // out at eye level, so spreading the lips stirred the eyelids a little.
  const eyes=[anchors[159],anchors[386],[0,.055,.02]];
  const {rest,rig}=build([...points,...eyes]);
  for(const signal of [{open:1},{spread:1},{round:1},{open:1,spread:1,round:1}]){
    rig.reset();settle(rig,signal);
    for(let i=points.length;i<points.length+eyes.length;i++){
      assert.equal(magnitude(rig.offset,i),0,`eye vertex ${i} moved for ${JSON.stringify(signal)}`);
    }
  }
  assert.ok(rest.length>0);
});

test('reset returns the rig to silence',()=>{
  const {rig}=build();
  settle(rig,{open:1,spread:1});
  rig.reset();
  assert.ok(rig.offset.every(v=>v===0));
  assert.equal(rig.open,0);
});

test('the mock stand-in envelope opens while speaking and closes after',()=>{
  const signal=createSyntheticSignal();
  signal.speakFor(1);
  let peak=0;
  for(let i=0;i<60;i++)peak=Math.max(peak,signal.read(1/60).open);
  assert.ok(peak>.3,'a speaking stand-in must actually open the mouth');
  signal.stop();
  let value=1;
  for(let i=0;i<60;i++)value=signal.read(1/60).open;
  assert.ok(value<.02,'it must fall shut once the utterance ends');
});
