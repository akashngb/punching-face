/* MediaPipe's WASM loader uses importScripts; keep this a classic worker. As a module worker it
   hits importScripts' TypeError and falls back to self.import, which does not exist. */
self.exports={};
importScripts('/vendor/vision_bundle.cjs');
const {FilesetResolver,HandLandmarker,ImageSegmenter}=self.exports;

// Landmarks come off the full-resolution frame (the landmarker crops hand ROIs from it), but the
// segmenter only feeds a 256-wide model, so it reads a downscaled copy. That also keeps the mask —
// and every JS pass over it — at ~37k pixels instead of the camera's 920k.
const SEG_WIDTH=256;
let armCutout,arcCoverage,smoother,anchor,guard,hands,segmenter,segCanvas,segContext,maskCanvas,maskContext;

async function create(task,files,options){
  try{return await task.createFromOptions(files,{...options,baseOptions:{...options.baseOptions,delegate:'GPU'}});}
  catch{return task.createFromOptions(files,{...options,baseOptions:{...options.baseOptions,delegate:'CPU'}});}
}

self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      const module=await import(new URL(data.maskURL,data.origin).href);
      armCutout=module.armCutout;arcCoverage=module.arcCoverage;smoother=new module.MaskSmoother();anchor=new module.ArmAnchor();guard=data.guard??null;
      const files=await FilesetResolver.forVisionTasks(`${data.origin}/wasm`);
      // Confidence sits low so a motion-blurred fist mid-punch keeps its track instead of dropping out.
      // Presence/tracking thresholds sit very low so a guard fist half-clipped by the frame edge
      // keeps its track; fresh detections stay at .45 so noise cannot seed phantom hands.
      hands=await create(HandLandmarker,files,{baseOptions:{modelAssetPath:`${data.origin}/models/hand_landmarker.task`},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.45,minHandPresenceConfidence:.3,minTrackingConfidence:.3});
      segmenter=await create(ImageSegmenter,files,{baseOptions:{modelAssetPath:`${data.origin}/models/selfie_multiclass_256x256.tflite`},runningMode:'VIDEO',outputCategoryMask:true,outputConfidenceMasks:false});
      self.postMessage({type:'ready'});return;
    }
    if(data.type!=='frame')return;
    try{
      const result=hands.detectForVideo(data.bitmap,data.timestamp);
      const segHeight=Math.max(64,Math.round(SEG_WIDTH*data.bitmap.height/data.bitmap.width));
      if(!segCanvas||segCanvas.height!==segHeight){segCanvas=new OffscreenCanvas(SEG_WIDTH,segHeight);segContext=segCanvas.getContext('2d');}
      segContext.drawImage(data.bitmap,0,0,SEG_WIDTH,segHeight);
      const segmented=segmenter.segmentForVideo(segCanvas,data.timestamp),category=segmented.categoryMask;
      try{
        const width=category.width,height=category.height;
        // smoother.state (last frame's smoothed mask) doubles as the sustain support: regions the
        // landmarks established stay alive on segmentation alone through detection dropouts.
        const cut=armCutout(category.getAsUint8Array(),width,height,result.landmarks,smoother.state,anchor);
        const alpha=smoother.apply(cut.alpha);
        if(!maskCanvas||maskCanvas.width!==width||maskCanvas.height!==height){maskCanvas=new OffscreenCanvas(width,height);maskContext=maskCanvas.getContext('2d');}
        const pixels=maskContext.createImageData(width,height);
        for(let i=0;i<alpha.length;i++){const offset=i*4;pixels.data[offset]=pixels.data[offset+1]=pixels.data[offset+2]=255;pixels.data[offset+3]=alpha[i];}
        maskContext.putImageData(pixels,0,0);
        // Only the low-res mask travels back. The page composites it over the *live* video element
        // every animation frame, so the visible arm pixels never wait on inference.
        // Calibration judges arc fill on this exact mask, so what the user sees green is what counts.
        let coverage=null;
        if(guard?.targets){coverage={};for(const side of Object.keys(guard.targets))coverage[side]=arcCoverage(alpha,width,height,guard.targets[side],guard.radius);}
        const mask=maskCanvas.transferToImageBitmap();
        // worldLandmarks are MediaPipe's metric, hand-centred 3D estimate. The rigid pose estimator
        // needs them for shape and orientation; without them the page must re-derive depth from
        // apparent hand size, which cannot separate distance from rotation.
        self.postMessage({type:'result',landmarks:result.landmarks,worldLandmarks:result.worldLandmarks,handedness:result.handedness,anchored:cut.anchored,coverage,timestamp:data.timestamp,mask},[mask]);
      }finally{segmented.close();}
    }finally{data.bitmap.close();}
  }catch(error){self.postMessage({type:'error',message:error.message});}
};
