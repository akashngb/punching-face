import { fistScore } from './physics.js';
import {capturePoseSignature,compareCapturePose,armViewCoverage} from './capture-quality.js';
const $=id=>document.getElementById(id);
const poseIds={left:[11,13,15],right:[12,14,16]};
const blobURL=blob=>new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(blob);});

export class ArmCapture {
  constructor(tracking,onArmReady){
    this.tracking=tracking;this.onArmReady=onArmReady;this.frames=[];this.running=false;this.side='left';this.sectors=new Set();this.saved=new Map();this.savedCount=0;this.token=0;this.signatures=[];
    document.body.insertAdjacentHTML('beforeend',`<dialog id="arm-dialog"><button class="close" id="arm-close" aria-label="Close arm scan">×</button><span class="eyebrow">Personal geometry · multiview capture</span><h1>Scan your arm, one side at a time.</h1><p class="muted">Keep your shoulder, elbow, wrist, and closed fist visible. Hold that entire arm rigid and slowly turn your torso to expose every side. Use even lighting and keep your other arm out of the way.</p><div class="row"><button id="arm-left" class="active">Left arm</button><button id="arm-right">Right arm</button></div><label class="controls-label" style="margin-top:10px">Elbow to wrist, cm (blank = estimated 27 cm)</label><input id="arm-length" aria-label="Elbow to wrist length in centimetres" type="number" min="12" max="50" step=".1" placeholder="27 (estimate)"><p class="muted">Live framing — keep the entire selected arm inside this view.</p><video id="arm-live" aria-label="Live arm framing" autoplay playsinline muted style="width:100%;aspect-ratio:16/9;object-fit:contain;background:#192c24;border-radius:8px;transform:scaleX(-1)"></video><p class="muted">Last accepted segmentation crop</p><canvas id="arm-preview" width="640" height="360" style="width:100%;max-height:160px;object-fit:contain;background:#192c24;border-radius:8px"></canvas><div id="arm-state" class="note">Connect the webcam, then prepare your arm. Captures are saved only on this computer.</div><div class="row" style="margin-top:15px"><button id="arm-start" class="primary">Start capture</button><button id="arm-stop" disabled>Finish capture</button></div><p class="muted">Aim for 60–100 sharp views. The preview outlines an approximate arm crop using body segmentation and joint positions; inspect it before training. The reconstruction will be rejected if camera poses or geometry cannot be recovered.</p><button id="arm-train" class="full" disabled>Reconstruct this arm</button><button id="arm-import" class="full small" style="margin-top:8px">Import a reconstructed arm bundle</button><input id="arm-bundle" type="file" accept=".json"><p id="arm-result" class="muted" role="status"></p></dialog>`);
    $('arm-close').onclick=()=>{this.stop().catch(e=>$('arm-state').textContent=e.message);$('arm-dialog').close();};
    for(const side of ['left','right'])$('arm-'+side).onclick=async()=>{if(this.running||this.saving)return;try{await this.persist();this.side=side;this.frames=[];this.savedCount=0;this.signatures=[];this.sectors.clear();$('arm-left').classList.toggle('active',side==='left');$('arm-right').classList.toggle('active',side==='right');this.showSaved();}catch(e){$('arm-state').textContent=e.message;}};
    $('arm-start').onclick=()=>this.start().catch(e=>$('arm-state').textContent=e.message);
    $('arm-stop').onclick=()=>this.stop().catch(e=>$('arm-state').textContent=e.message);$('arm-train').onclick=()=>this.train();
    $('arm-import').onclick=()=>$('arm-bundle').click();$('arm-bundle').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const data=JSON.parse(await file.text());await this.onArmReady(data);$('arm-dialog').close();}catch(e){$('arm-result').textContent=e.message;}};
    tracking.onCapture=data=>this.accept(data).catch(e=>$('arm-state').textContent=e.message);
    this.loadSaved().catch(e=>$('arm-result').textContent=`Saved-capture list unavailable: ${e.message}`);
  }
  open(){$('arm-dialog').showModal();this.showLive();}
  showLive(){const video=$('arm-live');video.srcObject=this.tracking.stream;if(video.srcObject)video.play().catch(()=>{});}
  async start(){
    await this.persist();await this.tracking.enableBody();this.frames=[];this.savedCount=0;this.signatures=[];this.sectors.clear();this.running=true;this.token++;this.started=performance.now();$('arm-start').disabled=true;$('arm-stop').disabled=false;$('arm-train').disabled=true;
    this.showLive();$('arm-live').scrollIntoView({block:'center'});
    $('arm-state').textContent='Capturing. Keep the entire arm and fist rigid while turning slowly.';
    this.timer=setInterval(()=>{if(this.frames.length>=100||performance.now()-this.started>90000){this.stop().catch(e=>$('arm-state').textContent=e.message);return;}this.tracking.captureRequest=this.side;},650);
  }
  async stop(){clearInterval(this.timer);this.running=false;this.token++;this.tracking.captureRequest=null;$('arm-start').disabled=true;$('arm-stop').disabled=true;
    try{await this.persist();this.showSaved();}finally{$('arm-start').disabled=false;}
  }
  showSaved(){const draft=this.saved.get(this.side);$('arm-train').disabled=!draft||draft.frames<12||draft.status!=='captured';$('arm-state').textContent=draft?`${draft.frames} ${this.side}-arm views saved locally. ${draft.status==='captured'?'Ready for camera-pose verification.':draft.message}`:'No saved capture for this arm yet.';}
  async loadSaved(){const response=await fetch('/api/arm-drafts');if(!response.ok)throw new Error('Check the reconstruction server.');const data=await response.json();for(const draft of data.drafts)if(!this.saved.has(draft.side))this.saved.set(draft.side,draft);if(!this.running)this.showSaved();for(const draft of this.saved.values())if(draft.status==='running')this.watch(draft.id);}
  async persist(){
    if(this.saving)return this.saving;if(!this.frames.length||this.savedCount===this.frames.length)return;
    const side=this.side,frames=this.frames.slice(),forearmCm=Number($('arm-length').value)||null;
    $('arm-state').textContent=`Saving ${frames.length} ${side}-arm images on this computer…`;$('arm-train').disabled=true;
    this.saving=(async()=>{const response=await fetch('/api/arm-draft',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({side,forearmCm,frames})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Capture save failed. Keep this page open; the images remain in memory.');this.saved.set(side,{id:result.id,side,frames:frames.length,status:'captured',message:'Images saved locally.'});this.savedCount=frames.length;})().finally(()=>this.saving=null);return this.saving;
  }
  async accept(data){
    if(!this.running||data.capture.side!==this.side)return;
    const token=this.token;
    const cap=data.capture,pose=cap.pose.landmarks?.[0];if(!pose)return;
    const ids=poseIds[this.side],joints=ids.map(i=>pose[i]);
    const distances=data.landmarks.map(h=>Math.hypot(h[0].x-joints[2].x,h[0].y-joints[2].y));
    const index=distances.indexOf(Math.min(...distances));
    const hand=data.landmarks[index];
    const missing=joints.flatMap((p,i)=>(p.visibility<.65||p.x<.03||p.x>.97||p.y<.03||p.y>.97)?[['shoulder','elbow','wrist'][i]]:[]);
    if(!hand||distances[index]>.15)missing.push('hand');
    if(missing.length){$('arm-state').textContent=`Your ${this.side} ${missing.join(', ')} ${missing.length>1?'are':'is'} not clearly visible. Step back and show the whole selected arm.`;return;}
    if(fistScore(hand)<.75){$('arm-state').textContent='Close the selected hand into a fist and keep that pose throughout this scan.';return;}
    const signature=capturePoseSignature(cap.pose.worldLandmarks?.[0],data.worldLandmarks?.[index],this.side),compatible=compareCapturePose(this.signatures[0],signature);
    if(!compatible.ok){$('arm-state').textContent=compatible.message;return;}
    const eye=pose[0],palmCenter={x:(hand[5].x+hand[17].x)/2,y:(hand[5].y+hand[17].y)/2};
    if(eye?.visibility>.6&&Math.hypot(palmCenter.x-eye.x,palmCenter.y-eye.y)<.14){$('arm-state').textContent='Move the whole arm away from your face so the fist and skin stay unobstructed.';return;}
    const image=await createImageBitmap(cap.image),w=image.width,h=image.height;
    const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);image.close();
    const raw=ctx.getImageData(0,0,w,h),mask=document.createElement('canvas');mask.width=w;mask.height=h;const mctx=mask.getContext('2d');
    const span=Math.hypot((hand[5].x-hand[17].x)*w,(hand[5].y-hand[17].y)*h);
    mctx.fillStyle='white';mctx.strokeStyle='white';mctx.lineCap='round';mctx.lineJoin='round';mctx.lineWidth=Math.max(span*1.8,40);
    mctx.beginPath();joints.forEach((p,i)=>i?mctx.lineTo(p.x*w,p.y*h):mctx.moveTo(p.x*w,p.y*h));mctx.stroke();
    const xs=hand.map(p=>p.x*w),ys=hand.map(p=>p.y*h),pad=span*.35;
    mctx.fillRect(Math.min(...xs)-pad,Math.min(...ys)-pad,Math.max(...xs)-Math.min(...xs)+pad*2,Math.max(...ys)-Math.min(...ys)+pad*2);
    const region=mctx.getImageData(0,0,w,h),maskPixels=mctx.createImageData(w,h);let pixels=0,sharpness=0;
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      const i=(y*w+x)*4,mi=Math.min(cap.maskHeight-1,Math.floor(y/h*cap.maskHeight))*cap.maskWidth+Math.min(cap.maskWidth-1,Math.floor(x/w*cap.maskWidth));
      const keep=region.data[i+3]>0&&cap.mask[mi]>.65;raw.data[i+3]=keep?255:0;
      if(keep){pixels++;if(x>0&&x<w-1)sharpness+=Math.abs(raw.data[i-4]-2*raw.data[i]+raw.data[i+4]);}
      maskPixels.data[i]=maskPixels.data[i+1]=maskPixels.data[i+2]=keep?255:0;maskPixels.data[i+3]=255;
    }
    ctx.putImageData(raw,0,0);mctx.putImageData(maskPixels,0,0);
    const preview=$('arm-preview'),pc=preview.getContext('2d');pc.clearRect(0,0,640,360);pc.drawImage(canvas,0,0,640,360);
    if(pixels<w*h*.015||sharpness/Math.max(pixels,1)<1.4){$('arm-state').textContent='The arm is too small or blurred. Move closer and pause between angles.';return;}
    const angle=Math.atan2(signature.view[0],signature.view[2]),bin=Math.floor((angle+Math.PI)/(Math.PI*2)*24)%24;
    const png=await new Promise(r=>canvas.toBlob(r,'image/png')),maskPNG=await new Promise(r=>mask.toBlob(r,'image/png'));
    const full=await blobURL(cap.image);
    const frame={image:await blobURL(png),original:full,mask:await blobURL(maskPNG),pose:pose.map(p=>({x:p.x,y:p.y,z:p.z,visibility:p.visibility})),hand,side:this.side,timestamp:data.timestamp,orientationBin:bin,capturePose:signature,sharpness:sharpness/pixels};
    if(!this.running||token!==this.token)return;
    this.frames.push(frame);this.signatures.push(signature);this.sectors.add(bin);$('arm-state').textContent=`${this.frames.length} views · ${armViewCoverage(this.signatures).toFixed(0)}° estimated arm-view span · ${this.side} arm`;
  }
  async train(){
    $('arm-train').disabled=true;$('arm-result').textContent='Saving the masked photographs locally…';
    try{
      await this.stop();const draft=this.saved.get(this.side);if(!draft)throw new Error('Capture and save this arm first.');
      const response=await fetch('/api/arm-train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:draft.id})});const result=await response.json();if(!response.ok)throw new Error(result.error);draft.status='running';this.job=result.id;
      $('arm-result').textContent='Verifying camera poses, then training a Gaussian splat. This can take several minutes.';
      this.watch(result.id);
    }catch(e){$('arm-result').textContent=e.message;$('arm-train').disabled=false;}
  }
  watch(id){const poll=async()=>{try{const r=await fetch(`/api/arm-status?id=${id}`);if(!r.ok)throw new Error('Status unavailable.');const state=await r.json();$('arm-result').textContent=state.message;if(state.status==='complete'||state.status==='failed'){for(const draft of this.saved.values())if(draft.id===id){draft.status=state.status;draft.message=state.message;}if(state.status==='complete'){await this.onArmReady(state.bundle);if(!this.running)$('arm-dialog').close();}else if(!this.running)this.showSaved();}else setTimeout(poll,2500);}catch(e){$('arm-result').textContent='Status connection interrupted. Retrying…';setTimeout(poll,5000);}};poll();}
}
