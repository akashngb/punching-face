// Local body-part segmentation of saved captures. Original photographs and the
// original reconstruction attempt are retained. Only public weights are fetched.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import * as tf from '@tensorflow/tfjs';
import {createSegmenter,SupportedModels} from '@tensorflow-models/body-segmentation';
import {PNG} from 'pngjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const folder=path.resolve(process.argv[2]||'');
if(!process.argv[2]||!fs.existsSync(path.join(folder,'capture.json')))throw new Error('Usage: node scripts/segment_arm.mjs CAPTURE_FOLDER [--pilot]');
const capture=JSON.parse(fs.readFileSync(path.join(folder,'capture.json'))),pilot=process.argv.includes('--pilot');
const out=path.join(folder,'segmented');for(const name of ['images','masks','originals','parts'])fs.mkdirSync(path.join(out,name),{recursive:true});
globalThis.ImageData??=class ImageData{constructor(data,width,height){this.data=data;this.width=width;this.height=height;}};
// The wrapper checks this browser type before its Tensor3D branch in Node.
globalThis.ImageBitmap??=class ImageBitmap{};
await tf.setBackend('cpu');await tf.ready();
const modelBase='https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/100/';
const cache=path.join(root,'.local/models/bodypix-mobilenet100');fs.mkdirSync(cache,{recursive:true});
async function cached(name){
  if(path.basename(name)!==name)throw new Error('Unexpected model shard path.');
  const target=path.join(cache,name);
  if(!fs.existsSync(target)){const response=await fetch(modelBase+name);if(!response.ok)throw new Error(`Model download failed: ${response.status}`);const data=Buffer.from(await response.arrayBuffer());fs.writeFileSync(target+'.tmp',data);fs.renameSync(target+'.tmp',target);}
  return fs.readFileSync(target);
}
const model=JSON.parse(await cached('model-stride16.json')),weightSpecs=[],chunks=[],hashes={};
for(const group of model.weightsManifest){weightSpecs.push(...group.weights);for(const name of group.paths){const bytes=await cached(name);chunks.push(bytes);hashes[name]=crypto.createHash('sha256').update(bytes).digest('hex');}}
const weights=Buffer.concat(chunks),weightData=weights.buffer.slice(weights.byteOffset,weights.byteOffset+weights.byteLength);
tf.io.registerLoadRouter(url=>url==='punching-face-bodypix-local'?tf.io.fromMemory({modelTopology:model.modelTopology,weightSpecs,weightData}):null);
const segmenter=await createSegmenter(SupportedModels.BodyPix,{architecture:'MobileNetV1',outputStride:16,multiplier:1,quantBytes:4,modelUrl:'punching-face-bodypix-local'});
const indices=pilot?[0,Math.floor(capture.frames.length/2),capture.frames.length-1]:capture.frames.map((_,i)=>i),metrics=[];
for(const index of indices){
  const frame=capture.frames[index],name=frame.filename;if(path.basename(name)!==name)throw new Error('Invalid frame filename.');
  const started=performance.now(),original=path.join(folder,'originals',name),png=PNG.sync.read(fs.readFileSync(original));
  const previous=PNG.sync.read(fs.readFileSync(path.join(folder,'masks',name+'.png'))),rgb=new Uint8Array(png.width*png.height*3);
  for(let i=0;i<png.width*png.height;i++)rgb.set(png.data.subarray(i*4,i*4+3),i*3);
  const input=tf.tensor3d(rgb,[png.height,png.width,3],'int32');
  const people=await segmenter.segmentPeople(input,{multiSegmentation:false,segmentBodyParts:true,flipHorizontal:false,internalResolution:.5,segmentationThreshold:.65});input.dispose();
  const labels=await people[0].mask.toImageData(),mask=new PNG({width:png.width,height:png.height}),parts=new PNG({width:png.width,height:png.height});let kept=0,originalCount=0,faceRemoved=0,torsoRemoved=0;
  for(let i=0;i<png.width*png.height;i++){
    const j=i*4,label=labels.data[j],inPerson=labels.data[j+3]>128,inCrop=previous.data[j]>128;
    // Use semantic arm labels within the selected arm's existing pose crop.
    // This avoids relying on a second model's left/right classification.
    const keep=inCrop&&inPerson&&label>=2&&label<=11;
    if(inCrop){originalCount++;if(inPerson&&label<2)faceRemoved++;if(inPerson&&(label===12||label===13))torsoRemoved++;}
    if(keep)kept++;else png.data[j]=png.data[j+1]=png.data[j+2]=0;png.data[j+3]=keep?255:0;mask.data[j]=mask.data[j+1]=mask.data[j+2]=keep?255:0;mask.data[j+3]=255;
    parts.data[j]=label;parts.data[j+1]=parts.data[j+2]=0;parts.data[j+3]=inPerson?255:0;
  }
  fs.writeFileSync(path.join(out,'images',name),PNG.sync.write(png));fs.writeFileSync(path.join(out,'masks',name+'.png'),PNG.sync.write(mask));fs.writeFileSync(path.join(out,'parts',name),PNG.sync.write(parts));
  const linked=path.join(out,'originals',name);if(!fs.existsSync(linked))fs.linkSync(original,linked);
  const metric={frame:name,keptPixels:kept,retainedFraction:kept/Math.max(originalCount,1),facePixelsRemoved:faceRemoved,torsoPixelsRemoved:torsoRemoved,seconds:Math.round((performance.now()-started)/10)/100};metrics.push(metric);console.log(JSON.stringify(metric));
}
segmenter.dispose();
const evidence={method:'BodyPix MobileNetV1 1.0, stride 16, float weights, CPU; semantic arm labels intersected with the selected pose crop',sourceFolder:folder,sourceModel:modelBase+'model-stride16.json',weightHashes:hashes,pilot,frames:metrics,limitation:'Predicted segmentation. Occlusion, articulation and insufficient viewpoint coverage can still prevent reconstruction.'};
fs.writeFileSync(path.join(out,pilot?'pilot-segmentation.json':'segmentation.json'),JSON.stringify(evidence,null,2));
if(!pilot)fs.writeFileSync(path.join(out,'capture.json'),JSON.stringify({...capture,segmentation:evidence}));
console.log(`Saved ${metrics.length} segmented views to ${out}`);
