import test from 'node:test';
import assert from 'node:assert/strict';
import {PunchExtractor,firstContact,classifyMode,resolveHand,toHead} from '../src/punch-events.js';
import {makeRandom} from './helpers/synthetic-hand.mjs';

// Camera-frame observations straight into the core: x right, y up, z = metres in front of the
// lens. 60 deg FOV at 4:3, matching the shell's defaults.
const KX=Math.tan(Math.PI/6)*(4/3),KY=Math.tan(Math.PI/6);
const project=p=>({u:.5+p[0]/(2*p[2]*KX),v:.5-p[1]/(2*p[2]*KY)});

/**
 * Drive an extractor over a camera-frame trajectory. `traj(tSec)` returns [x,y,z] or null while
 * the hand is out of play; `visible` can kill landmark samples (blur, frame exit) while `blob`
 * keeps the motion stream alive.
 */
function run(extractor,traj,{duration=900,step=1000/60,noise=.0015,seed=7,label='Right',score=.9,
  closure=1,kn=null,wristDx=null,chirality=null,visible=null,blob=null,detour=null}={}){
  const random=makeRandom(seed),events=[];
  const at=(option,t)=>typeof option==='function'?option(t):option;
  for(let ms=0;ms<=duration;ms+=step){
    const t=ms/1000;
    let p=traj?traj(t):null;
    if(detour)p=detour(t,p)??p;
    if(p&&(!visible||visible(t,p))){
      const {u,v}=project(p);
      if(u>-.5&&u<1.5&&v>-.5&&v<1.5){
        const jitter=[p[0]+random.normal()*noise,p[1]+random.normal()*noise,p[2]+random.normal()*noise*5];
        const det={u,v,span:.084/Math.max(p[2],.05)};
        const slot=extractor.assign([det],ms).get(det);
        extractor.push(slot,{t:ms,u,v,p:jitter,closure,kn:at(kn,t),label:at(label,t),score,
          wristDx:at(wristDx,t),chirality:at(chirality,t)});
      }
    }
    if(blob){
      const b=blob(t);
      if(b)extractor.pushBlob({t:ms,kx:KX,ky:KY,...b});
    }
    const event=extractor.tick(ms+8,{gain:1,headRadii:[.15,.14,.09]});
    if(event)events.push(event);
  }
  return events;
}

// The standard shadow punch: fast out, brief hold at full reach, retraction — the same shape the
// target-camera tests use.
const sCurve=(t,{out,hold,back})=>t<0?0:t<out?(t/out)**.7:t<out+hold?1:Math.max(0,1-(t-out-hold)/back);
const straight=({x=0,y=0,from=.70,to=.34,out=.13,hold=.08,back=.22,at=0}={})=>t=>{
  const s=sCurve(t-at,{out,hold,back});
  return [x,y,from+(to-from)*s];
};
const lerpPath=(from,to,{out=.16,hold=.07,back=.24,at=0}={})=>t=>{
  const s=sCurve(t-at,{out,hold,back});
  return [0,1,2].map(k=>from[k]+(to[k]-from[k])*s);
};

test('one punch, one event; the retraction that follows never fires a second one',()=>{
  const x=new PunchExtractor();
  const events=run(x,straight(),{duration:1400});
  assert.equal(events.length,1,`expected exactly one event, got ${events.length}`);
  assert.ok(events[0].closing>1.5,`peak closing carried (${events[0].closing?.toFixed(2)} m/s)`);
  assert.equal(events[0].mode,'jab');
});

test('a settle-dip when the fist re-enters guard is not a punch',()=>{
  const x=new PunchExtractor();
  // full punch, then a slow 4 cm overshoot-and-settle around guard depth
  const traj=t=>{
    if(t<.6)return straight()(t);
    const wobble=Math.sin((t-.6)*2*Math.PI)* .02;
    return [0,0,.70-Math.max(0,.04-(t-.6)*.08)+wobble*0];
  };
  const events=run(x,traj,{duration:1600});
  assert.equal(events.length,1,'the guard settle after retraction must not re-fire');
});

test('two punches with a withdrawal between them are two events',()=>{
  const x=new PunchExtractor();
  const first=straight({at:0}),second=straight({at:.55});
  const events=run(x,t=>t<.55?first(t):second(t),{duration:1400});
  assert.equal(events.length,2,`two swings, two impacts (got ${events.length})`);
});

test('a slow reach toward the camera is not a punch',()=>{
  const x=new PunchExtractor();
  const events=run(x,t=>[0,0,Math.max(.25,.70-.45*Math.min(1,t/3))],{duration:2600});
  assert.equal(events.length,0);
});

test('a fast twitch that goes nowhere is not a punch',()=>{
  const x=new PunchExtractor();
  const events=run(x,straight({from:.50,to:.465,out:.05,hold:.04,back:.1}),{duration:700});
  assert.equal(events.length,0,'3.5 cm of travel is under the reach gate');
});

test('a hand sweeping laterally past at constant range is not a punch',()=>{
  const x=new PunchExtractor();
  const events=run(x,t=>{
    const angle=-.9+1.8*Math.min(1,t/.45),radius=.42;
    return [radius*Math.sin(angle),.02,radius*Math.cos(angle)];
  },{duration:900});
  assert.equal(events.length,0,'moving across the target is not moving into it');
});

test('landmarks that die at full extension still land the punch, once',()=>{
  const x=new PunchExtractor();
  const events=run(x,straight({from:.75,to:.30}),{duration:1000,visible:(t,p)=>p[2]>=.40});
  assert.equal(events.length,1,'losing the hand mid-extension is evidence of a punch, not absence of one');
  assert.equal(events[0].stale,true);
});

test('the motion stream carries a landmark-dead punch to a confirmed apex',()=>{
  const x=new PunchExtractor();
  const traj=straight({from:.72,to:.32,out:.14,hold:.06,back:.20});
  let lastZ=null,lastT=null;
  const events=run(x,traj,{
    duration:1100,
    visible:(t,p)=>p[2]>=.52,          // landmarks die early in the approach
    blob:t=>{
      const p=traj(t);if(p[2]>=.55)return null;   // blob picks up as the fist blurs in
      const {u,v}=project(p);
      const expand=lastZ!==null&&t>lastT?Math.log(lastZ/p[2])/(t-lastT):0;
      lastZ=p[2];lastT=t;
      return {u,v,mass:.4,spread:.1,expand};
    },
  });
  assert.equal(events.length,1,`blob bridge must land the punch exactly once (got ${events.length})`);
  assert.ok(events[0].bridged,'and record that the motion stream carried it');
});

test('an instant-fire deep punch does not fire again at its own apex',()=>{
  const x=new PunchExtractor();
  const events=run(x,straight({from:.70,to:.06,out:.16,hold:.10,back:.26}),{duration:1400});
  assert.equal(events.length,1,`deep punch: one event (got ${events.length})`);
  assert.equal(events[0].instant,true,'and it fired on the contact-depth crossing, not the apex');
});

test('after an instant fire the fist must withdraw before the next punch registers',()=>{
  const x=new PunchExtractor();
  const first=straight({from:.70,to:.08,out:.13,hold:.06,back:.20}),second=straight({from:.68,to:.30,at:.65});
  const events=run(x,t=>t<.65?first(t):second(t),{duration:1600});
  assert.equal(events.length,2,`instant fire then a fresh punch after withdrawal (got ${events.length})`);
});

test('a committed jab is not an instant fire — it reports its landing, not its flight',()=>{
  // The live symptom: jabs "landed" mid-flight (logged low and lateral, as shoulder hits) and the
  // actual face contact never registered. A 20 cm instant-fire line crossed ~80 ms before full
  // extension; at the 12 cm default the crossing IS the landing, so a jab stopping at 22 cm goes
  // through the apex path and reports its extension point.
  const x=new PunchExtractor();
  const events=run(x,straight({from:.70,to:.22,out:.14,hold:.07,back:.22}),{duration:1100});
  assert.equal(events.length,1);
  assert.ok(!events[0].instant,'apex path, not instant');
  assert.ok(events[0].depth<.26,`reported at full extension (${(events[0].depth*100).toFixed(0)} cm), not along the flight`);
});

test('a two-frame depth stutter mid-flight does not fire the punch early',()=>{
  // Near the camera the depth estimate gets noisy exactly as the fist blurs in; one or two fitted
  // samples bouncing above the confirm line used to read as "the retreat", firing the event at a
  // point along the punch's travel and consuming the real apex.
  const x=new PunchExtractor();
  const events=run(x,straight({from:.70,to:.34,out:.13,hold:.08,back:.22}),{duration:1100,
    detour:(t,p)=>p&&t>=.09&&t<.14?[p[0],p[1],p[2]+.06]:null});
  assert.equal(events.length,1,`one punch, one event (got ${events.length})`);
  assert.ok(events[0].depth<.40,`fired at the real apex (${(events[0].depth*100).toFixed(0)} cm), not at the stutter`);
});

test('a jab converging from a low guard lands where it stopped, not down its flight path',()=>{
  // A guard-anchored jab sweeps ~20 cm laterally on its way in, but closes 30+ cm of depth doing
  // it: its lateral motion is en route, not through the face. The entry may only walk sideways
  // when the sweep dominates the depth closed — otherwise a centred jab reported at the chin.
  const x=new PunchExtractor();
  const events=run(x,lerpPath([.10,-.22,.62],[-.02,-.05,.30],{out:.15,hold:.08,back:.24}),
    {duration:1100,label:'Right'});
  assert.equal(events.length,1);
  const e=events[0];
  assert.equal(e.missed,false);
  assert.ok(Math.abs(e.point[0]-.02)<.05&&e.point[1]>-.115,
    `lands near the extension point (got [${e.point.map(v=>v.toFixed(3))}])`);
  assert.ok(e.point[2]>.05,'on the front of the face, not walked to a silhouette edge');
});

test('a jab-cross combo from the two sides is two events',()=>{
  const x=new PunchExtractor();
  // right jab from the puncher's right (image left, camera -x), then left cross from the other
  // side. Labels follow the vendored landmarker's real convention on an unmirrored feed (it
  // names the PHYSICAL hand), and each arm trails its wrist toward its own shoulder.
  const jab=straight({x:-.12,at:0}),cross=straight({x:.12,at:.34});
  const events=run(x,t=>t<.34?jab(t):cross(t),{duration:1200,
    label:t=>t<.34?'Right':'Left',wristDx:t=>t<.34?-.04:.04});
  assert.equal(events.length,2,`a 340 ms one-two is two punches (got ${events.length})`);
  assert.equal(events[0].hand,'right');
  assert.equal(events[1].hand,'left');
});

test('one punch fragmented across identities is still one event',()=>{
  for(const teleportAt of [.12,.19,.26]){
    const x=new PunchExtractor();
    const hook=lerpPath([-.42,.02,.38],[-.06,.02,.30],{out:.16,hold:.07,back:.26});
    const events=run(x,hook,{duration:1400,detour:(t,p)=>{
      if(p&&Math.abs(t-teleportAt)<1/120)return [p[0]+.28,p[1]+.18,p[2]];
      return null;
    }});
    assert.ok(events.length<=1,`fragment at ${teleportAt}s produced ${events.length} events`);
    assert.equal(events.length,1,`fragment at ${teleportAt}s must still land the punch`);
  }
});

test('the motion stream carries identity through a blur gap: reacquired landmarks rejoin their slot',()=>{
  // The main live duplicate: landmarks die mid-arc, and a straight-line extrapolation of the
  // pre-blur velocity misses where the hook actually went — reacquisition forked a second
  // identity, which then fired its own view of the punch. The blobs saw the fist the whole way;
  // their measured flow steers the slot's prediction along the arc, so the reacquired landmarks
  // land back inside the association gate of the identity they belong to.
  const x=new PunchExtractor();
  const hook=lerpPath([.34,.02,.36],[.06,.02,.30],{out:.16,hold:.07,back:.24});
  let lastU=null,lastV=null,lastZ=null,lastT=null;
  const events=run(x,hook,{duration:1100,label:'Right',
    visible:t=>t<.05||t>=.28,             // landmarks dead through the whole fast mid-arc
    blob:t=>{
      const p=hook(t);const {u,v}=project(p);
      const b={u,v,mass:.4,spread:.1,
        du:lastU!==null&&t>lastT?(u-lastU)/(t-lastT):0,
        dv:lastV!==null&&t>lastT?(v-lastV)/(t-lastT):0,
        expand:lastZ!==null&&t>lastT?Math.log(lastZ/p[2])/(t-lastT):0};
      lastU=u;lastV=v;lastZ=p[2];lastT=t;
      return t>=.05&&t<.30?b:null;
    }});
  assert.equal(events.length,1,`one hook, one event (got ${events.length})`);
  assert.equal(x.slots.size,1,'the blur gap must not fork identity');
  assert.equal(x.stats.refractory,0,'identity, not the arbitration backstop, absorbed the gap');
  assert.ok([...x.slots.values()][0].firstObsT<100,
    'the surviving identity is the one born BEFORE the gap — reacquisition rejoined it');
  assert.equal(events[0].stale,false,
    'carried to a confirmed apex on live evidence, not censored mid-flight at the blur');
  assert.ok(events[0].bridged,'the motion stream carried the window');
  assert.equal(events[0].mode,'hook');
});

test('a censored fragment and its reacquired continuation are one punch, however far blur moved it',()=>{
  // The escape hatch the image-gated dedup could never close: the first identity dies to blur and
  // fires censored; the fist reappears far away (here: no motion stream at all, and a hook's arc
  // defeats the straight-line prediction), and the fresh identity completes the punch. The two
  // sightings never coexisted on screen — one fist wearing two identities in turn — so the second
  // candidate is suppressed however far apart the image positions are.
  const x=new PunchExtractor();
  const traj=t=>{
    if(t<.15){const s=t/.15;return [.30-.16*s,0,.60-.18*s];}            // approach, then blur
    if(t<.38)return null;                                                // 230 ms of nothing
    if(t<.50){const s=(t-.38)/.12;return [-.305+.185*s,-.178+.108*s,.44-.20*s];}
    if(t<.58)return [-.12,-.07,.24];                                     // landing
    const s=Math.min(1,(t-.58)/.20);return [-.12-.13*s,-.07-.08*s,.24+.16*s];
  };
  const events=run(x,traj,{duration:1200,label:'Right',noise:.001});
  assert.equal(events.length,1,`one physical punch, one event (got ${events.length})`);
  assert.ok(x.stats.refractory>=1,'the duplicate view was formed and suppressed, not never seen');
});

test('a mid-flight punch with no geometry votes takes its hand by elimination',()=>{
  // A blur-truncated punch can arrive with every within-window vote mute: born fast mid-frame
  // (entry side untrusted), travel nearly straight (azimuth silent), labels flickering, no rigid
  // fits (no wrist trail, no chirality). But the OTHER fist is sitting in guard on camera — and
  // the fist visibly at rest is not the one that just landed.
  const x=new PunchExtractor(),random=makeRandom(31),events=[];
  const swing=t=>{
    if(t<.30)return null;                                    // never seen before mid-flight
    const s=sCurve(t-.30,{out:.10,hold:.08,back:.22});
    return [.09-.03*s,.01,.48-.18*s];
  };
  for(let ms=0;ms<=900;ms+=1000/60){
    const t=ms/1000,batch=[];
    for(const [p,label] of [[[-.18,-.02,.55],'Left'],[swing(t),Math.floor(ms/(1000/60))%2?'Left':'Right']]){
      if(!p)continue;
      const {u,v}=project(p);
      const jitter=[p[0]+random.normal()*.0015,p[1]+random.normal()*.0015,p[2]+random.normal()*.008];
      batch.push({u,v,obs:{t:ms,u,v,p:jitter,closure:1,label,score:.9}});
    }
    const assignment=x.assign(batch,ms);
    for(const d of batch)x.push(assignment.get(d),d.obs);
    const event=x.tick(ms+8,{gain:1,headRadii:[.15,.14,.09]});
    if(event)events.push(event);
  }
  assert.equal(events.length,1,`only the punching hand scores (got ${events.length})`);
  assert.equal(events[0].hand,'left',
    `the fist resting at low u is the right hand, so the punch is the left (votes ${JSON.stringify(events[0].handVotes)}, lineage ${JSON.stringify(events[0].lineage)})`);
});

test('shifting the guard around at constant distance never fires — a punch must arrive',()=>{
  // The live symptom: guard up, forearms vertical, shifting the arms around at an unchanged
  // distance from the screen logged hook after hook. Any inward lateral move closes range by the
  // lateral slack r−z (6-11 cm from a normal guard — squarely inside the travel gates), the
  // shift back brackets it, and the edge-born softening lowered the speed bar into fidget
  // territory. What no guard shift can fake is ARRIVAL: its range minimum is the guard distance
  // itself, half a metre from the face.
  const x=new PunchExtractor(),random=makeRandom(9);const events=[];
  const shift=t=>{
    const x0=-.34,x1=-.02,phase=(t%1.0)/1.0;   // brisk ~1.4 m/s lateral, constant .55 m depth
    let lat;
    if(phase<.19)lat=x0+(x1-x0)*(phase/.19);
    else if(phase<.5)lat=x1;
    else if(phase<.69)lat=x1+(x0-x1)*((phase-.5)/.19);
    else lat=x0;
    return [lat,-.05,.55];
  };
  for(let ms=0;ms<=3600;ms+=1000/60){
    const p=shift(ms/1000);
    const {u,v}=project(p);
    const det={u,v};const slot=x.assign([det],ms).get(det);
    x.push(slot,{t:ms,u,v,p:[p[0]+random.normal()*.0015,p[1]+random.normal()*.0015,p[2]+random.normal()*.008],closure:1,label:'Left',score:.9});
    const e=x.tick(ms+8,{gain:1,headRadii:[.15,.14,.09]});
    if(e)events.push(e);
  }
  assert.equal(events.length,0,`guard shifting fired ${events.length} phantom events`);
  assert.match(x.stats.rejected,/from the head/,'and the readout names the arrival gate');
});

test('one jab held out at the lens is ONE event, however long the hold',()=>{
  // The live symptom: a single left jab held at full extension logged seven events over a few
  // seconds — hooks, uppercuts, both hands, one at 15.3 m/s. At 15-20 cm the rigid fit fails
  // chronically, the anchor-less span-depth oscillates as fingers clip the frame, every
  // fabricated dip crossed the instant-fire line and every fabricated rise cleared the re-arm.
  // Fabricated depth can now do neither: qualification and re-arm withdrawal are rigid-anchored.
  const x=new PunchExtractor(),random=makeRandom(5);const events=[];
  const trueP=t=>{
    if(t<.18){const s=sCurve(t,{out:.13,hold:.05,back:1});return [-.02,0,.62-.44*s];}
    return [-.02,0,.18];                        // held still at 18 cm, forever
  };
  for(let ms=0;ms<=6000;ms+=1000/60){
    const t=ms/1000,p=trueP(t),held=t>.35;
    const sq=Math.tanh(3*Math.sin(2*Math.PI*t));  // span-depth square-ish swing, what clipping does
    const z=held?.215+.085*sq+random.normal()*.02:p[2]+random.normal()*.008;
    const {u,v}=project(p);
    const det={u,v};const slot=x.assign([det],ms).get(det);
    x.push(slot,{t:ms,u:u+random.normal()*.01,v:v+random.normal()*.01,
      p:[(u*2-1)*z*KX+random.normal()*.003,(1-v*2)*z*KY+random.normal()*.003,z],
      closure:1,label:Math.floor(ms/300)%2?'Left':'Right',score:.8,quality:held?'loose':'rigid'});
    if(held)x.pushBlob({t:ms,u:u+.03*Math.sin(t*4),v:v+.06,mass:.3,spread:.12,
      expand:1.2*Math.sin(2*Math.PI*.6*t),kx:KX,ky:KY});     // the user's own arm/body swaying
    const e=x.tick(ms+8,{gain:1,headRadii:[.15,.14,.09]});
    if(e)events.push(e);
  }
  assert.equal(events.length,1,
    `one punch, one event (got ${events.length}: ${events.map(e=>e.hand+' '+e.mode).join(', ')})`);
  assert.equal(events[0].mode,'jab');
});

test('a fast reach that stops well short of the head is not a punch, until the knob says so',()=>{
  const far=new PunchExtractor();
  const events=run(far,straight({from:.85,to:.55}),{duration:900});
  assert.equal(events.length,0,'stopped 55 cm out — nothing arrived');
  assert.match(far.stats.rejected,/from the head/);
  const lenient=new PunchExtractor({strikeRange:.70});
  assert.equal(run(lenient,straight({from:.85,to:.55}),{duration:900}).length,1,
    'the strike-range knob widens what counts as arriving');
});

test('a chest-height strike is ignored outright — no hit, no miss, no log line',()=>{
  const x=new PunchExtractor();
  const events=run(x,straight({y:-.30,from:.70,to:.32}),{duration:900});
  assert.equal(events.length,0,'nothing below the head is reported');
  assert.equal(x.stats.ignored,1,'but it is counted for the readout');
});

test('a hook reports its terminal tangent, not the mid-arc chord',()=>{
  const x=new PunchExtractor();
  // Genuine arc: the fist swings on a circle, so its travel direction rotates continuously. A
  // right hook seen from the target: enters wide on the image left, curls in across the face.
  const centre=[-.06,.02,.62],R=.30;
  const arc=t=>{
    const s=sCurve(t,{out:.20,hold:.06,back:.20});
    const theta=(-65+60*s)*Math.PI/180;
    return [centre[0]+R*Math.sin(theta),centre[1],centre[2]-R*Math.cos(theta)];
  };
  const events=run(x,arc,{duration:1100});
  assert.equal(events.length,1);
  const d=events[0].direction;
  // Terminal tangent is almost pure lateral (head -x for a right hook); the chord from entry to
  // apex would carry |dz| ~ 0.5. The solver must read the end of the arc, not its average.
  assert.ok(d[0]<-.85,`drives across the face (dx ${d[0].toFixed(2)})`);
  assert.ok(Math.abs(d[2])<.4,`terminal tangent, not the chord (dz ${d[2].toFixed(2)})`);
  assert.equal(events[0].mode,'hook');
  assert.equal(events[0].hand,'right');
  assert.ok(events[0].curvature>.4,`the arc itself is measured (${events[0].curvature?.toFixed(2)} rad)`);
});

test('an uppercut classifies as an uppercut and lands under the chin',()=>{
  const x=new PunchExtractor();
  const events=run(x,lerpPath([.03,-.36,.42],[.03,-.08,.26],{out:.16,hold:.07,back:.24}),
    {duration:1100,label:'Right',score:.9});
  assert.equal(events.length,1);
  const e=events[0];
  assert.equal(e.mode,'uppercut');
  assert.ok(e.direction[1]>.5,`drives upward (dy ${e.direction[1].toFixed(2)})`);
  assert.ok(e.point[1]<-.02,`lands low on the head (y ${e.point[1].toFixed(3)})`);
  assert.equal(e.hand,'right','the vendored landmarker names the physical hand — no selfie flip');
});

test('a left hook mirrors: enters image-right, drives head +x, classified left',()=>{
  const x=new PunchExtractor();
  const events=run(x,lerpPath([.34,.02,.36],[.06,.02,.30],{out:.16,hold:.07,back:.24}),
    {duration:1100,label:'Left',score:.9});
  assert.equal(events.length,1);
  assert.equal(events[0].mode,'hook');
  assert.equal(events[0].hand,'left');
  assert.ok(events[0].direction[0]>.5,`travels toward head +x (dx ${events[0].direction[0].toFixed(2)})`);
});

test('impact point sits on the surface the punch came in through',()=>{
  const radii=[.15,.14,.09];
  const zone=point=>{
    const n=point.map((v,k)=>v/radii[k]),biggest=n.map(Math.abs).reduce((b,v,i,a)=>a[b]>v?b:i,0);
    return biggest===0?'cheek':biggest===1?(point[1]>0?'forehead':'chin'):'front';
  };
  const jab=run(new PunchExtractor(),straight({from:.70,to:.30}),{duration:900})[0];
  assert.equal(zone(jab.point),'front',`jab on the front (got ${zone(jab.point)})`);
  const hook=run(new PunchExtractor(),lerpPath([-.34,.02,.36],[-.06,.02,.30]),{duration:900})[0];
  assert.equal(zone(hook.point),'cheek',`hook on the cheek (got ${zone(hook.point)})`);
  const upper=run(new PunchExtractor(),lerpPath([.03,-.34,.40],[.03,-.06,.24]),{duration:900})[0];
  assert.equal(zone(upper.point),'chin',`uppercut on the chin (got ${zone(upper.point)})`);
});

test('punch to the puncher\'s right lands at head +x, and high lands high',()=>{
  const right=run(new PunchExtractor(),straight({x:-.09,from:.70,to:.34}),{duration:900})[0];
  assert.ok(right.point[0]>.03,`camera -x is the puncher's right, head +x (got ${right.point[0].toFixed(3)})`);
  const high=run(new PunchExtractor(),straight({y:.09,from:.70,to:.34}),{duration:900})[0];
  assert.ok(high.point[1]>.03,`the vertical axis never flips (got ${high.point[1].toFixed(3)})`);
});

test('a guard hand parked in frame does not fire while the other punches',()=>{
  const x=new PunchExtractor(),random=makeRandom(17),events=[];
  const swing=straight({x:.10,from:.70,to:.30});
  for(let ms=0;ms<=1000;ms+=1000/60){
    const t=ms/1000;
    for(const [p,label] of [[[-.18,-.02,.55],'Left'],[swing(t),'Right']]){
      const {u,v}=project(p);
      const jitter=[p[0]+random.normal()*.0015,p[1]+random.normal()*.0015,p[2]+random.normal()*.008];
      const det={u,v,span:.084/p[2]};
      const slot=x.assign([det],ms).get(det);
      x.push(slot,{t:ms,u,v,p:jitter,closure:1,label,score:.9});
    }
    const event=x.tick(ms+8,{gain:1,headRadii:[.15,.14,.09]});
    if(event)events.push(event);
  }
  assert.equal(events.length,1,`only the punching hand scores (got ${events.length})`);
  assert.ok(x.slots.size>=2,'both hands hold their own slot');
});

test('after an event no nearby slot is left mid-approach to fire its own copy',()=>{
  const x=new PunchExtractor();
  const hook=lerpPath([-.42,.02,.38],[-.06,.02,.30],{out:.16,hold:.07,back:.26});
  run(x,hook,{duration:1400,detour:(t,p)=>{
    if(p&&Math.abs(t-.12)<1/120)return [p[0]+.28,p[1]+.18,p[2]];
    return null;
  }});
  assert.ok(x.stats.impacts<=1);
  for(const slot of x.slots.values())
    assert.ok(slot.rearm!==null||slot.consumedUntil>-1e8||slot.samples.length<3,
      'an unconsumed mid-approach orphan would fire its own version of the punch later');
});

test('an open hand pushed forward is not a punch',()=>{
  const x=new PunchExtractor();
  const events=run(x,straight(),{duration:900,closure:.05});
  assert.equal(events.length,0,'closure evidence is required');
  assert.match(x.stats.rejected,/open hand/);
});

test('a right hook first detected mid-frame still attributes to the right hand',()=>{
  // The failure seen live: hooks are often first detected mid-arc near frame centre, where the
  // entry-side vote is meaningless — right hooks were logged as left. The wrist trail, the fitted
  // hand's chirality and the travel azimuth must carry it instead.
  const x=new PunchExtractor();
  // First sample already at u ~ .35 — inside the centre band where the entry-side vote is mute —
  // and still sweeping across as it closes on the head.
  const events=run(x,lerpPath([-.09,.02,.40],[.11,.02,.28],{out:.15,hold:.07,back:.24}),
    {duration:1100,label:'Left',wristDx:-.05,chirality:.16});
  assert.equal(events.length,1);
  assert.equal(events[0].mode,'hook');
  assert.equal(events[0].hand,'right',
    `wrist trail + chirality + azimuth outvote the useless centre entry (votes ${JSON.stringify(events[0].handVotes)})`);
});

test('a rising hook with upward knuckles stays a hook, not an uppercut',()=>{
  // Hooks often carry lift and a rotated fist; the old first-match uppercut branch took any
  // knuckle normal with n.y > .45 at its word.
  const x=new PunchExtractor();
  const events=run(x,lerpPath([-.34,-.06,.36],[-.06,.06,.30],{out:.16,hold:.07,back:.24}),
    {duration:1100,label:'Left',kn:[-.55,.55,-.63]});
  assert.equal(events.length,1);
  assert.equal(events[0].mode,'hook',`overwhelming lateral travel wins (got ${events[0].mode})`);
});

test('a blur-censored shallow hook still classifies as a hook',()=>{
  // Landmarks die mid-arc, so the terminal tangent degrades toward the chord — which for a
  // shallow hook is forward enough to have read as a jab.
  const x=new PunchExtractor();
  const events=run(x,lerpPath([-.30,.02,.55],[-.04,.02,.30],{out:.16,hold:.07,back:.24}),
    {duration:1100,visible:(t,p)=>p[2]>=.40,label:'Left'});
  assert.equal(events.length,1);
  assert.equal(events[0].stale,true,'the punch was censored');
  assert.equal(events[0].mode,'hook',`the image sweep keeps it a hook (got ${events[0].mode})`);
});

test('a hook pulled just short grazes the cheek; one stopped a fist-width out misses',()=>{
  // First-contact is honest about pulled hooks: a stop within a fist of the silhouette lands on
  // it, a stop clearly wide of the head is a miss, and neither teleports anywhere.
  const near=new PunchExtractor();
  const nearEvents=run(near,lerpPath([-.42,.02,.38],[-.19,.02,.31],{out:.15,hold:.07,back:.24}),
    {duration:1100,label:'Left'});
  assert.equal(nearEvents.length,1);
  assert.equal(nearEvents[0].missed,false,'a stop grazing the silhouette lands');
  assert.ok(nearEvents[0].point[0]>.12,`on the cheek it was driving toward (x ${nearEvents[0].point[0].toFixed(3)})`);
  const wide=new PunchExtractor();
  const wideEvents=run(wide,lerpPath([-.44,.02,.38],[-.27,.02,.31],{out:.15,hold:.07,back:.24}),
    {duration:1100,label:'Left'});
  assert.equal(wideEvents.length,1);
  assert.equal(wideEvents[0].missed,true,'a stop a fist-width wide of the head is a miss');
});

test('a hook with follow-through lands on the cheek it came in through, not past the centreline',()=>{
  // The live symptom: hooks landing on the WRONG side of the face. A hook's fist decelerates at
  // or past the centreline (follow-through), and solving the impact from the stopping point put
  // the marker on the far cheek. First-contact takes the earliest touch of the trajectory.
  const x=new PunchExtractor();
  // right hook sweeping THROUGH centre: enters via head +x, stops at head -x
  const events=run(x,lerpPath([-.30,.02,.40],[.14,.02,.30],{out:.17,hold:.06,back:.24}),
    {duration:1100,label:'Left'});
  assert.equal(events.length,1);
  const e=events[0];
  assert.equal(e.missed,false);
  assert.ok(e.point[0]>.12,`entry-side cheek despite the follow-through (got [${e.point.map(v=>v.toFixed(3))}], apex ${e.apexPoint[0].toFixed(3)})`);
  assert.equal(e.mode,'hook');
});

test('a shallow hook — more forward drive than sweep — still lands on its entry side',()=>{
  // Real hooks at a laptop close 25-40 cm of depth while sweeping 15-25 cm laterally. An earlier
  // gate required the sweep to dominate depth before sideways entry applied, so every real hook
  // fell back to its stopping point — the follow-through side. First-contact has no such branch.
  const x=new PunchExtractor();
  const events=run(x,lerpPath([-.28,.02,.62],[.02,.02,.30],{out:.16,hold:.07,back:.24}),
    {duration:1100,label:'Left'});
  assert.equal(events.length,1);
  const e=events[0];
  assert.equal(e.missed,false);
  assert.ok(e.point[0]>.03,`first touch on the entry-side half (got ${e.point[0].toFixed(3)}, stop was ${e.apexPoint[0].toFixed(3)})`);
});

test('a hook seen only from the centreline onward still lands on the side the face would have stopped it',()=>{
  // The live symptom, stated by the user: "if I swing such that it hits screen-left, my hook
  // ends with my fist on the right of the screen — but in a real scenario it would have been
  // stopped by the face." Blur eats the entry-side flight, so every MEASURED sample sits at or
  // past the centreline; the entry is reconstructed backward along the arrival direction to
  // where the trajectory first crossed the face.
  for(const cut of [.05,.06,.07]){
    const x=new PunchExtractor();
    // left hook with follow-through; landmarks only exist from `cut` seconds in
    const events=run(x,lerpPath([.30,.02,.46],[-.14,.02,.23],{out:.15,hold:.06,back:.24}),
      {duration:1100,label:'Left',seed:11,visible:t=>t>=cut});
    assert.equal(events.length,1,`cut ${cut}: the truncated hook must register (got ${events.length})`);
    const e=events[0];
    assert.equal(e.mode,'hook');
    assert.equal(e.hand,'left');
    assert.ok(e.point[0]<-.12,
      `cut ${cut}: lands screen-left where the face would have stopped it (got [${e.point.map(v=>v.toFixed(3))}])`);
  }
});

test('a hook whose retraction re-crosses the closest range still anchors on the punch, not the pull-back',()=>{
  // The range profile of a follow-through hook dips twice: the forward crossing and the
  // retraction re-cross, which can dip LOWER. Anchoring the apex on the global minimum hung the
  // terminal window and the direction on the pull-back — the reported travel reversed and the
  // entry flipped sides. The contact is the FIRST arrival into the deepest zone.
  const x=new PunchExtractor();
  // right hook, fully tracked, retraction retracing the arc
  const events=run(x,lerpPath([-.30,.02,.40],[.14,.02,.30],{out:.17,hold:.06,back:.24}),
    {duration:1100,label:'Right',seed:7});
  assert.equal(events.length,1);
  const e=events[0];
  assert.ok(e.direction[0]<-.5,`the punch's travel, not the retraction's (dx ${e.direction[0].toFixed(2)})`);
  assert.ok(e.point[0]>.12,`entry-side cheek (got [${e.point.map(v=>v.toFixed(3))}])`);
});

test('an uppercut seen only at its top still classifies and lands under the chin',()=>{
  // The live symptom: an uppercut's rise is low, fast, blurred and partly below the frame, so
  // the landmarker locks on only at the top where the upward velocity is spent. The measured
  // direction reads near-pure -z ("right jab · [-0.20,-0.08,-0.98]"), every travel-gated
  // uppercut branch fails, and the marker lands high on the skull. The fist's ORIENTATION is
  // measured at the apex where tracking is good — knuckles-up carries the classification, and
  // the classification carries the entry down to the chin.
  const traj=t=>{
    if(t<.10){const s=(t/.10)**.7;return [.02,-.30+.25*s,.46-.08*s];}      // the unseen rise
    if(t<.17){const s=(t-.10)/.07;return [.02,-.05+.005*s,.38-.12*s];}     // the seen top: forward drive, no vertical
    if(t<.23)return [.02,-.045,.26];
    const s=Math.min(1,(t-.23)/.24);return [.02,-.045-.20*s,.26+.16*s];
  };
  const x=new PunchExtractor();
  const events=run(x,traj,{duration:1000,visible:t=>t>=.10,label:'Left',kn:[.1,.85,-.5]});
  assert.equal(events.length,1,`the top-only uppercut must register (got ${events.length})`);
  const e=events[0];
  assert.equal(e.mode,'uppercut',`knuckles-up carries it despite direction dz ${e.direction[2].toFixed(2)} (got ${e.mode})`);
  assert.equal(e.missed,false);
  assert.ok(e.point[1]<-.11,`lands under the chin, not where the top was measured (got [${e.point.map(v=>v.toFixed(3))}])`);
});

test('a hook fragment first seen mid-arc, moving fast, still registers',()=>{
  // Tracking picks hooks up mid-flight. Born-fast is judged on full 3D speed — a mid-arc hook
  // moves 3-5 m/s while closing range slowly, and judging it on range rate kept the relaxed
  // reach gate away from exactly the punches that needed it.
  const x=new PunchExtractor();
  const events=run(x,lerpPath([-.12,.02,.34],[.02,.02,.28],{out:.10,hold:.06,back:.20}),
    {duration:900,label:'Left'});
  assert.equal(events.length,1,`the visible tail closes only ~7 cm, but it was born at speed (got ${events.length})`);
});

test('a hook that blurs out in its tangential phase still censors and lands',()=>{
  // The censor gate reads the trailing window's PEAK closing: this fragment closed hard early,
  // then swept tangentially just before vanishing — judged on its last sample it read as a hand
  // drifting to a stop and died silently.
  const x=new PunchExtractor();
  const traj=t=>{
    if(t<.12){const s=(t/.12)**.7;return [.30-.25*s,.02,.55-.22*s];}
    if(t<.18){const s=(t-.12)/.06;return [.05-.12*s,.02,.33];}
    return [-.07,.02,.33];
  };
  const events=run(x,traj,{duration:900,visible:t=>t<.18,label:'Right'});
  assert.equal(events.length,1,`the fragment was a punch losing its landing to blur (got ${events.length})`);
  assert.equal(events[0].stale,true);
});

test('an uppercut offset from centre lands on the chin, never beside the head',()=>{
  // Real uppercuts drift laterally and finish above face height. They were reading as misses or
  // as entry points on the ellipsoid's flank where the real mesh has no surface.
  const x=new PunchExtractor();
  const events=run(x,lerpPath([.09,-.34,.42],[.09,-.10,.26],{out:.16,hold:.07,back:.24}),
    {duration:1100,label:'Right'});
  assert.equal(events.length,1);
  const e=events[0];
  assert.equal(e.missed,false,'a chin-bound trajectory is a hit');
  assert.equal(e.mode,'uppercut');
  assert.ok(e.point[1]<-.09,`lands low (y ${e.point[1].toFixed(3)})`);
  const onSilhouette=(e.point[0]/.15)**2+(e.point[1]/.14)**2<=1.0001;
  assert.ok(onSilhouette,`the entry stays on the head silhouette (point [${e.point.map(v=>v.toFixed(3))}])`);
});

test('the extractor holds up at 20 fps inference, where a punch is five samples long',()=>{
  const x=new PunchExtractor();
  const events=run(x,straight({out:.15,hold:.09,back:.24}),{duration:1200,step:50});
  assert.equal(events.length,1,`one punch, one event at 20 fps (got ${events.length})`);
  assert.equal(events[0].mode,'jab');
  const hook=new PunchExtractor();
  const hookEvents=run(hook,lerpPath([-.34,.02,.36],[-.06,.02,.30],{out:.18,hold:.08,back:.26}),
    {duration:1200,step:1000/24});
  assert.equal(hookEvents.length,1,`and a hook still lands at 24 fps (got ${hookEvents.length})`);
  assert.equal(hookEvents[0].mode,'hook');
});

test('firstContact takes the earliest touch of the path, never the stopping point',()=>{
  const radii=[.15,.14,.09];
  // a hook path sweeping in from wide: enters through the +x cheek where it first crossed
  const swept=firstContact([[.25,.02],[.19,.02],[.13,.02],[.07,.02],[.01,.02]],radii);
  assert.ok(swept&&Math.abs(swept[0]-.15)<.01,`entry at the silhouette crossing, not deeper (got ${swept?.[0].toFixed(3)})`);
  // a follow-through path crossing the whole face: still the FIRST touch, not where it stopped
  const through=firstContact([[.20,.02],[.08,.02],[-.04,.02],[-.12,.02]],radii);
  assert.ok(through&&through[0]>.14,'first contact is on the side it came in through');
  // a path already inside: lands at its earliest point
  const inside=firstContact([[.02,-.01],[.01,-.02],[0,-.03]],radii);
  assert.ok(inside&&Math.abs(inside[0]-.02)<1e-6&&inside[2]>.08,'a jab lands where it arrives, on the front');
  // a path skimming just wide grazes on; one clearly wide misses
  const grazed=firstContact([[.20,.02],[.185,.02]],radii);
  assert.ok(grazed&&grazed[0]>.14,'a skim within a fist-width snaps onto the silhouette');
  assert.equal(firstContact([[.30,.02],[.26,.02]],radii),null,'clearly wide of the head misses');
});

test('the same jab lands in the same place, run after run',()=>{
  // The live symptom: identical jabs marked on entirely opposite parts of the mesh. Root cause
  // was a single near-apex direction probe that could read the retraction and report the punch
  // travelling OUT of the face (~1 run in 8), which sent the app's raycast through the back of
  // the skull. Directions must always drive into the head and points must cluster.
  const points=[],directions=[];
  for(let seed=1;seed<=24;seed++){
    const x=new PunchExtractor();
    const events=run(x,straight({x:-.03,y:-.02,from:.68,to:.33}),
      {duration:900,seed:seed*7919,noise:.003,label:'Left'});
    assert.equal(events.length,1,`seed ${seed}: one jab, one event`);
    points.push(events[0].point);directions.push(events[0].direction);
  }
  for(const d of directions)
    assert.ok(d[2]<-.5,`every direction drives into the face (got dz ${d[2].toFixed(2)})`);
  const mean=[0,1,2].map(k=>points.reduce((s,p)=>s+p[k],0)/points.length);
  for(const p of points){
    const spread=Math.hypot(p[0]-mean[0],p[1]-mean[1]);
    assert.ok(spread<.02,`every marker within 2 cm of the cluster centre (got ${(spread*100).toFixed(1)} cm at [${p.map(v=>v.toFixed(3))}])`);
  }
});

test('classification is image-first: the sweep outranks the depth-fitted direction',()=>{
  // A depth-starved terminal (rigid fit dead, depth inertial) collapses velocity-z and turns a
  // straight jab's fitted direction into lateral/vertical noise — which read jabs as hooks and
  // uppercuts live. The image plane is the trustworthy witness: a jab barely moves across it, a
  // hook crosses it, an uppercut climbs it.
  assert.equal(classifyMode({direction:[.9,.1,-.3],sweep:{du:.03,dv:.01}}),'jab',
    'a lateral 3D direction with no image sweep behind it is a jab, not a hook');
  assert.equal(classifyMode({direction:[-.7,0,-.6],sweep:{du:-.35,dv:.02}}),'hook',
    'a real hook shows both');
  assert.equal(classifyMode({direction:[.2,.3,-.9],knuckleNormal:[0,.6,-.6],sweep:{du:.02,dv:-.02}}),'jab',
    'ordinary jab form angles the fist up ~45° — that must not read as an uppercut');
  assert.equal(classifyMode({direction:[0,.05,-1],knuckleNormal:[.1,.85,-.5],sweep:{du:.02,dv:-.03}}),'uppercut',
    'an emphatic vertical fist still carries a top-only uppercut');
  assert.equal(classifyMode({direction:[.6,.3,-.5],sweep:{du:.05,dv:.30}}),'uppercut',
    'upward image motion outranks a noisy lateral direction');
  assert.equal(classifyMode({direction:[.1,.3,-.9],knuckleNormal:[0,.05,-.99],sweep:{du:.02,dv:.20}}),'jab',
    'knuckles square at the camera are never an uppercut, whatever the sweep says');
  // The measured image axis decides the vertical question outright, whichever way the
  // depth-derived evidence leans.
  assert.equal(classifyMode({direction:[.1,.4,-.9],knuckleNormal:[0,.7,-.7],sweep:{du:.02,dv:.20},
    handAxis:{du:-.02,dv:.16}}),'jab','a foreshortened fist axis means the knuckles face the camera');
  // Orientation may not overrule clean travel: near the lens the wrist projects far below the
  // knuckles under perspective, so a dead-straight jab measures an upward axis (this logged as
  // LEFT UPPERCUT live, at 10 cm range, with the knuckles plainly facing the camera). An upward
  // axis needs corroboration — entry from below, or an upward image sweep — when travel says
  // the punch drove straight in. A truncated uppercut always supplies one.
  assert.equal(classifyMode({direction:[0,.02,-1],knuckleNormal:[0,.1,-.99],sweep:{du:.01,dv:-.02},
    handAxis:{du:.03,dv:.80},range:.14}),'jab',
    'at close range an upward axis is perspective, not an uppercut — it cannot overrule dead-straight travel');
  assert.equal(classifyMode({direction:[0,.02,-1],knuckleNormal:[0,.1,-.99],sweep:{du:.01,dv:-.02},
    handAxis:{du:.03,dv:.80},entry:[0,-.30,.4]}),'uppercut','entry from below corroborates it');
  assert.equal(classifyMode({direction:[0,.02,-1],knuckleNormal:[0,.1,-.99],sweep:{du:.01,dv:.20},
    handAxis:{du:.03,dv:.80}}),'uppercut','so does an upward image sweep');
  assert.equal(classifyMode({direction:[-.9,.1,-.3],sweep:{du:-.35,dv:.20},handAxis:{du:-.1,dv:.62}}),'hook',
    'overwhelming lateral travel still beats an upward-tilted fist');
  // Which FACE of the hand shows separates guard from uppercut: both are axis-up, but guard is
  // edge-on (facing ~.1) where an uppercut shows the back of the fist (facing ~1.0).
  assert.equal(classifyMode({direction:[0,.5,-.85],knuckleNormal:[0,.9,-.3],sweep:{du:.02,dv:.18},
    handAxis:{du:0,dv:1,facing:.12}}),'jab','an edge-on vertical fist is a guard, not an uppercut');
  assert.equal(classifyMode({direction:[-.5,.05,-.85],sweep:{du:-.05,dv:.01},
    handAxis:{du:-.85,dv:0,facing:1}}),'hook','a back-on fist with knuckles to the side is a hook shape');
});

test('fallback samples never steer the reported direction',()=>{
  // The mechanism behind "jabs logged as uppercuts": a fallback sample's depth is frozen, so a
  // velocity fitted through one reads dz ~ 0 — and for a straight punch, whose entire velocity
  // IS dz, the normalised direction then collapses onto leftover lateral/vertical noise, which
  // reads as a rise. Travel is solved on rigid samples only; fallbacks keep the trajectory
  // continuous but never speak about where the punch was going.
  const x=new PunchExtractor(),random=makeRandom(7);const events=[];
  const jab=straight({y:.12,from:.66,to:.24,out:.12,hold:.08,back:.22});
  let frozen=null;
  for(let ms=0;ms<=900;ms+=1000/60){
    const t=ms/1000,p=jab(t);const {u,v}=project(p);
    const det={u,v};const slot=x.assign([det],ms).get(det);
    // the rigid fit dies through the terminal phase, as it does on a fast jab blurring in
    if(p[2]>.38){
      frozen=p[2];
      x.push(slot,{t:ms,u,v,p:[p[0]+random.normal()*.002,p[1]+random.normal()*.002,p[2]+random.normal()*.01],
        closure:1,label:'Left',score:.9,kn:[0,.6,-.8],quality:'rigid'});
    }else{
      x.push(slot,{t:ms,u:u+random.normal()*.012,v:v+random.normal()*.012,
        p:[(u*2-1)*frozen*KX,(1-v*2)*frozen*KY,frozen],
        closure:1,label:'Left',score:.7,kn:null,quality:'span'});
    }
    const e=x.tick(ms+8,{gain:1,headRadii:[.15,.14,.09]});
    if(e)events.push(e);
  }
  assert.equal(events.length,1);
  const e=events[0];
  assert.equal(e.mode,'jab',`a straight punch stays a jab (got ${e.mode} via ${e.why})`);
  assert.ok(e.direction[2]<-.85,`and still drives into the face (dz ${e.direction[2].toFixed(2)})`);
});

test('a soft jab at 30 fps with a failing close-range fit still registers, at an honest speed',()=>{
  // The live symptom: a real jab rejected with "peak 1.13 < 1.20". At 30 fps with the rigid fit
  // failing intermittently near the fist ("hand too close"), speed read off the blended range
  // series was garbage in both directions: frozen-depth samples flattened it below the gate, and
  // a frozen plateau ending in one rigid catch-up read as a staircase that inflated a 1.5 m/s
  // jab to 3.4. Peak closing is now a windowed LINEAR slope over rigid samples only — sparse
  // noisy points cannot mint phantom slope through an exactly-determined quadratic — and the
  // speed gates are calibrated to that honest, slightly conservative measure.
  for(const seed of [9,17,25,33]){
    const x=new PunchExtractor(),random=makeRandom(seed);const events=[];
    const jab=t=>{const s=sCurve(t,{out:.24,hold:.10,back:.28});return [0,.05,.52-.36*s];};
    let anchor=null,frame=0;
    for(let ms=0;ms<=1200;ms+=1000/30){
      frame++;
      const t=ms/1000,p=jab(t);const {u,v}=project(p);
      const det={u,v};const slot=x.assign([det],ms).get(det);
      if(p[2]>=.42||frame%3===0){anchor=p[2];
        x.push(slot,{t:ms,u,v,p:[p[0]+random.normal()*.002,p[1]+random.normal()*.002,p[2]+random.normal()*.012],
          closure:1,label:'Left',score:.9,quality:'rigid'});
      }else{
        x.push(slot,{t:ms,u:u+random.normal()*.008,v:v+random.normal()*.008,
          p:[(u*2-1)*anchor*KX,(1-v*2)*anchor*KY,anchor],closure:1,label:'Left',score:.7,quality:'span'});
      }
      const e=x.tick(ms+8,{gain:1,headRadii:[.15,.14,.09]});
      if(e)events.push(e);
    }
    assert.equal(events.length,1,`seed ${seed}: the soft jab must register (reject: ${x.stats.rejected})`);
    assert.equal(events[0].mode,'jab');
    assert.ok(events[0].closing>.9&&events[0].closing<2.8,
      `seed ${seed}: reported speed stays honest for a ~1.5 m/s punch (got ${events[0].closing.toFixed(2)})`);
  }
});

test('a straight jab above the low-slung camera is a jab, not an uppercut',()=>{
  // The live symptom: knuckles visibly pointing at the camera, event logged "uppercut". A laptop
  // camera sits below the face, so fists ride ABOVE the image centre — and an approaching
  // point's image position diverges radially away from the centre, so every high straight punch
  // "rises" in the image (dv ~ +.23 for a dead-straight jab here) with zero vertical motion.
  // The classifier removes the outward-radial component of the sweep before reading it.
  const x=new PunchExtractor();
  const events=run(x,straight({y:.14,from:.68,to:.30}),{duration:900,label:'Left'});
  assert.equal(events.length,1);
  assert.equal(events[0].mode,'jab',`perspective divergence is not a rise (got ${events[0].mode})`);
});

test('classification helpers behave at the boundaries',()=>{
  assert.equal(classifyMode({direction:[0,.8,-.6]}),'uppercut');
  assert.equal(classifyMode({direction:[-.9,0,-.3]}),'hook');
  assert.equal(classifyMode({direction:[0,0,-1]}),'jab');
  assert.equal(classifyMode({direction:[-.6,.1,-.6],knuckleNormal:[-.8,0,-.4]}),'hook');
  assert.equal(resolveHand({entryU:.2,labels:[]}).hand,'right');
  assert.equal(resolveHand({entryU:.8,labels:[]}).hand,'left');
  assert.equal(resolveHand({labels:[{label:'Right',score:.9}]}).hand,'right','the vendored landmarker names the physical hand');
  assert.equal(resolveHand({wristDx:-.05,labels:[]}).hand,'right','wrist trails toward the right shoulder at camera -x');
  assert.equal(resolveHand({chirality:.15,labels:[]}).hand,'right','a right hand measures positive in the fit frame');
  assert.equal(resolveHand({chirality:-.15,labels:[]}).hand,'left');
  assert.equal(resolveHand({restU:.25,labels:[]}).hand,'right','a fist that rested at low u is the right hand');
  assert.equal(resolveHand({otherRestU:.25,labels:[]}).hand,'left','the fist resting elsewhere is not the one that landed');
  assert.equal(resolveHand({restU:.75,labels:[{label:'Right',score:.9}]}).hand,'left','rest lineage outvotes the label');
  assert.deepEqual(toHead([1,2,3]),[-1,2,3]);
});
