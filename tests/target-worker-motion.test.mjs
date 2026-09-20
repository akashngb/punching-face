import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

// The target worker is a classic worker (MediaPipe's loader needs importScripts), so its motion
// functions cannot be imported as a module. Evaluate the source with stubbed worker globals and
// pull the pure functions out — the canvas-dependent capture path stays untested here, but the
// blob analysis and the similarity-flow fit are exactly the maths that must not silently rot.
const source=readFileSync(fileURLToPath(new URL('../public/target-worker.js',import.meta.url)),'utf8')
  .replace("importScripts('/vendor/vision_bundle.cjs');","");
const factory=new Function('self','importScripts','OffscreenCanvas',
  source+'\nreturn {componentsOf,blobFlow,matchPoint,MOTION_W};');
const {componentsOf,blobFlow}=factory({exports:{},postMessage(){}},()=>{},class{});

const W=96,H=72;
// A textured square whose pattern is anchored to its own frame, so scaling the square really
// rescales the texture the block matcher sees — a flat square would have nothing to match on.
function frame({cx,cy,half}){
  const grey=new Uint8Array(W*H).fill(20);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    if(Math.abs(x-cx)>half||Math.abs(y-cy)>half)continue;
    const u=(x-cx)/half,v=(y-cy)/half;
    grey[y*W+x]=Math.round(140+70*Math.sin(u*4.7)*Math.sin(v*4.7));
  }
  return grey;
}
function diffOf(a,b){
  const diff=new Uint8Array(a.length);
  for(let i=0;i<a.length;i++){const d=Math.abs(a[i]-b[i]);if(d>=18)diff[i]=d;}
  return diff;
}

test('frame differencing finds the moving mass as one blob near its true position',()=>{
  const a=frame({cx:30,cy:36,half:10}),b=frame({cx:34,cy:36,half:10});
  const {blobs}=componentsOf(diffOf(a,b),W,H);
  assert.ok(blobs.length>=1,'the moving square is a component');
  const blob=blobs[0],cx=blob.sx/blob.mass;
  assert.ok(Math.abs(cx-32)<6,`centroid between the two positions (got x=${cx.toFixed(1)})`);
});

test('similarity flow reads pure translation as translation, not expansion',()=>{
  const a=frame({cx:30,cy:36,half:10}),b=frame({cx:34,cy:36,half:10});
  const {labels,blobs}=componentsOf(diffOf(a,b),W,H);
  const flow=blobFlow(a,b,W,H,labels,blobs[0]);
  assert.ok(flow,'enough texture to match');
  assert.ok(flow.du>2&&flow.du<6,`rightward ~4 px (got ${flow.du.toFixed(1)})`);
  assert.ok(Math.abs(flow.scalePerFrame)<.12,`no phantom looming (got ${flow.scalePerFrame.toFixed(3)})`);
});

test('similarity flow reads expansion as looming with the right sign',()=>{
  const a=frame({cx:48,cy:36,half:10}),b=frame({cx:48,cy:36,half:13});
  const {labels,blobs}=componentsOf(diffOf(a,b),W,H);
  const flow=blobFlow(a,b,W,H,labels,blobs[0]);
  assert.ok(flow,'enough texture to match');
  assert.ok(flow.scalePerFrame>.12,`a growing square is approaching (got ${flow.scalePerFrame.toFixed(3)})`);
  assert.ok(Math.abs(flow.du)<2.5&&Math.abs(flow.dv)<2.5,'and it is not misread as travel');
  const shrink=blobFlow(b,a,W,H,componentsOf(diffOf(b,a),W,H).labels,componentsOf(diffOf(b,a),W,H).blobs[0]);
  assert.ok(shrink&&shrink.scalePerFrame<-.12,`a shrinking one is retreating (got ${shrink?.scalePerFrame.toFixed(3)})`);
});
