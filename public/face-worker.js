self.exports={};
importScripts('/vendor/vision_bundle.cjs');
const {FilesetResolver,FaceLandmarker,ImageSegmenter}=self.exports;
let detector,segmenter,faceQuality,headBox,previousThumbnail;
self.onmessage=async({data})=>{
  try{
    if(data.type==='reset'){headBox=null;previousThumbnail=null;return;}
    if(data.type==='init'){
      ({faceQuality}=await import(new URL(data.qualityURL,data.origin).href));
      const files=await FilesetResolver.forVisionTasks('/wasm');
      detector=await FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'/models/face_landmarker.task',delegate:'CPU'},runningMode:'IMAGE',numFaces:2,outputFacialTransformationMatrixes:true,minFaceDetectionConfidence:.6});
      segmenter=await ImageSegmenter.createFromOptions(files,{baseOptions:{modelAssetPath:'/models/selfie_multiclass_256x256.tflite',delegate:'CPU'},runningMode:'IMAGE',outputCategoryMask:true,outputConfidenceMasks:false});
      self.postMessage({type:'ready'});return;
    }
    if(data.type==='frame'){
      try{
        const {bitmap}=data,w=bitmap.width,h=bitmap.height,result=detector.detect(bitmap);
        if(result.faceLandmarks?.length>1){self.postMessage({type:'frame',ok:false,message:'Keep only one person in the frame.'});return;}
        // Landmarks constrain facial fitting only. They must not be a condition
        // for retaining the profile or back of the head during an orbit.
        const quality=faceQuality(result,w,h,null);
        if(quality.ok){
          const xs=quality.oval.map(p=>p.x*w),ys=quality.oval.map(p=>p.y*h);
          const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
          if(Math.abs(quality.yaw)<35||!headBox)headBox={cx:(x0+x1)/2,top:y0-(y1-y0)*.85,bottom:y1+(y1-y0)*.12,width:Math.max((x1-x0)*1.65,(y1-y0)*1.15),height:y1-y0};
        }
        if(!headBox){self.postMessage({type:'frame',ok:false,message:'Start facing the camera, with the entire head visible, to locate the head.'});return;}
        const canvas=new OffscreenCanvas(w,h),ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(bitmap,0,0);
        const pixels=ctx.getImageData(0,0,w,h),segmented=segmenter.segment(bitmap),category=segmented.categoryMask;
        let count=0,sharpness=0,brightness=0,bounds=[w,h,0,0];
        try{
          const labels=category.getAsUint8Array(),mw=category.width,mh=category.height;
          // Follow the hair component within the calibrated head area. Keeping
          // its vertical limits avoids including exposed shoulders or arms.
          let sumX=0,hairCount=0;
          for(let y=0;y<mh;y++)for(let x=0;x<mw;x++)if(labels[y*mw+x]===1&&y*h/mh>headBox.top&&y*h/mh<headBox.bottom&&Math.abs(x*w/mw-headBox.cx)<headBox.width*.75){sumX+=x*w/mw;hairCount++;}
          const cx=hairCount>20?headBox.cx*.4+(sumX/hairCount)*.6:headBox.cx;
          const polygon=new OffscreenCanvas(w,h),pc=polygon.getContext('2d',{willReadFrequently:true});
          if(quality.ok){pc.fillStyle='white';pc.beginPath();quality.oval.forEach((p,i)=>i?pc.lineTo(p.x*w,p.y*h):pc.moveTo(p.x*w,p.y*h));pc.closePath();pc.fill();}
          const faceAlpha=pc.getImageData(0,0,w,h).data;
          for(let y=0;y<h;y++)for(let x=0;x<w;x++){
            const i=(y*w+x)*4,label=labels[Math.min(mh-1,Math.floor(y*mh/h))*mw+Math.min(mw-1,Math.floor(x*mw/w))];
            const inHead=Math.abs(x-cx)<headBox.width*.5&&y>headBox.top&&y<headBox.bottom;
            const keep=faceAlpha[i+3]>200||(inHead&&(label===1||label===3||(label===2&&y<headBox.bottom-headBox.height*.04)));
            if(keep){pixels.data[i+3]=255;count++;brightness+=pixels.data[i];if(x>0&&x<w-1)sharpness+=Math.abs(pixels.data[i-4]-2*pixels.data[i]+pixels.data[i+4]);bounds=[Math.min(bounds[0],x),Math.min(bounds[1],y),Math.max(bounds[2],x),Math.max(bounds[3],y)];}
            else pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=pixels.data[i+3]=0;
          }
        }finally{segmented.close();}
        if(count<12000){self.postMessage({type:'frame',ok:false,message:'Keep the whole head visible and move closer for more detail.'});return;}
        if(bounds[0]<8||bounds[1]<8||bounds[2]>w-9||bounds[3]>h-9){self.postMessage({type:'frame',ok:false,message:'The head is cropped. Move back so hair, ears, and chin fit inside the frame.'});return;}
        if(sharpness/count<1.6){self.postMessage({type:'frame',ok:false,message:'Motion blur: turn more slowly and pause at each angle.'});return;}
        if(brightness/count<22||brightness/count>235){self.postMessage({type:'frame',ok:false,message:'Use even light so the head is neither dark nor washed out.'});return;}
        ctx.putImageData(pixels,0,0);
        const small=new OffscreenCanvas(64,48),sc=small.getContext('2d',{willReadFrequently:true});sc.drawImage(canvas,0,0,64,48);const thumbnail=sc.getImageData(0,0,64,48).data;
        let difference=0;if(previousThumbnail)for(let i=0;i<thumbnail.length;i+=4)difference+=Math.abs(thumbnail[i]-previousThumbnail[i]);
        if(previousThumbnail&&difference/(64*48)<1.4){self.postMessage({type:'frame',ok:false,message:'View already saved. Continue your slow rotation.'});return;}
        previousThumbnail=thumbnail;
        const blob=await canvas.convertToBlob({type:'image/png'});
        self.postMessage({type:'frame',ok:true,yaw:quality.ok?quality.yaw:null,pitch:quality.ok?quality.pitch:null,landmarks:quality.ok?quality.landmarks:null,irisLandmarks:quality.ok?quality.irisLandmarks:null,viewKind:quality.ok?'face':'head-only',blob});
      }finally{data.bitmap.close();}
    }
  }catch(e){self.postMessage({type:'error',message:e.message});}
};
