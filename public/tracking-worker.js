/* MediaPipe's WASM loader uses importScripts; keep this a classic worker. */
self.exports={};
importScripts('/vendor/vision_bundle.cjs');
const {FilesetResolver,HandLandmarker,PoseLandmarker}=self.exports;
let detector,poseDetector,origin,frameCount=0,lastPose;
self.onmessage=async({data})=>{
  try{
    if(data.type==='init'){
      origin=data.origin;
      const files=await FilesetResolver.forVisionTasks(`${data.origin}/wasm`);
      detector=await HandLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:`${data.origin}/models/hand_landmarker.task`,delegate:'CPU'},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.55,minTrackingConfidence:.5});
      self.postMessage({type:'ready'});
    }else if(data.type==='enablePose'){
      const files=await FilesetResolver.forVisionTasks(`${origin}/wasm`);
      poseDetector=await PoseLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:`${origin}/models/pose_landmarker.task`,delegate:'CPU'},runningMode:'VIDEO',numPoses:1,outputSegmentationMasks:true});
      self.postMessage({type:'poseReady'});
    }else if(data.type==='frame'){
      try{
        const result=detector.detectForVideo(data.bitmap,data.timestamp);let capture;
        if(poseDetector&&(data.capture||data.trackBody||frameCount++%4===0)){
          poseDetector.detectForVideo(data.bitmap,data.timestamp,pose=>{
            lastPose={landmarks:pose.landmarks,worldLandmarks:pose.worldLandmarks,timestamp:data.timestamp};
            if(data.capture&&pose.segmentationMasks?.[0]){
              const mask=pose.segmentationMasks[0];capture={mask:new Float32Array(mask.getAsFloat32Array()),maskWidth:mask.width,maskHeight:mask.height,pose:lastPose,side:data.capture};
            }
          });
        }
        if(capture){const canvas=new OffscreenCanvas(data.bitmap.width,data.bitmap.height);canvas.getContext('2d').drawImage(data.bitmap,0,0);capture.image=await canvas.convertToBlob({type:'image/png'});}
        self.postMessage({type:'result',landmarks:result.landmarks,worldLandmarks:result.worldLandmarks,handedness:result.handedness,pose:lastPose,capture,timestamp:data.timestamp});
      }finally{data.bitmap.close();}
    }
  }catch(e){self.postMessage({type:'error',message:e.message});}
};
