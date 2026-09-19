// Cornerman: the OMNI Live coach. It sees (webcam keyframes), hears (your voice, hands-free),
// and speaks (streamed audio you can interrupt). Perception stays on this device at 30 Hz; only
// a spoken question, up to four small keyframes and a few numbers leave it, once per turn, and
// only while the coach is switched on. The key never reaches this page: the loopback relay holds it.
import {downsample,encodeWav,bytesToBase64,base64ToBytes,pcm16ToFloat32,rms,VoiceGate} from './audio.js';
import {EventStream} from './sse.js';
import {obs} from './sentry.js';

const TAP="class Tap extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('punching-face-tap',Tap)";
const FRAME_MS=20,PREROLL_FRAMES=15,KEYFRAME_MS=700,KEYFRAMES=4,QUIET_AFTER_TURN_MS=7000,PUNCHES_PER_CUE=8;

export function createCornerman({api,panel,config,stats,refreshConfig}){
  panel.innerHTML=`
    <div class="sd-row" style="margin-top:0"><button class="primary" data-k="toggle" style="flex:1">Start coach</button><span class="sd-badge" data-k="model"></span></div>
    <div class="sd-meter" data-k="meter"><i></i></div>
    <div class="sd-status" data-k="status">Hands-free: just talk. Your fists are busy, so there is nothing to press.</div>
    <div class="sd-log" data-k="log" aria-live="polite"></div>
    <div class="sd-row"><input type="text" data-k="ask" placeholder="…or type a question" maxlength="300"><button data-k="send">Ask</button></div>
    <label class="sd-check"><input type="checkbox" data-k="vision" checked><span>Let the coach see me <span class="sd-badge leaving" data-k="leaving"></span></span></label>
    <label class="sd-check"><input type="checkbox" data-k="voice" checked><span>Spoken replies (interrupt by talking)</span></label>
    <label class="sd-check"><input type="checkbox" data-k="proactive" checked><span>Speak up on combos and personal bests</span></label>
    <details><summary>OMNI key</summary>
      <div class="sd-status">Stored only on this computer (<code>.local/secrets/omni.json</code>, mode 0600). Get one from the Huawei form; the gateway is <span data-k="gateway"></span>.</div>
      <div class="sd-row"><input type="password" data-k="key" placeholder="API key" autocomplete="off"><button data-k="save">Save</button></div>
    </details>`;
  const el=Object.fromEntries([...panel.querySelectorAll('[data-k]')].map(n=>[n.dataset.k,n]));
  let ctx=null,mic=null,node=null,out=null,streamOut=null,gate=null,enabled=false,busy=false,controller=null,nextTime=0,lastTurnAt=-Infinity,sinceTurn=0,keyTimer=null;
  let pending=new Float32Array(0),preroll=[],recording=null,frames=[],history=[];const playing=new Set(),grab=document.createElement('canvas');

  const status=(text,error=false)=>{el.status.textContent=text;el.status.classList.toggle('error',error);};
  const say=(who,text)=>{const p=document.createElement('p');p.className=who;p.textContent=(who==='you'?'You: ':'Coach: ')+text;el.log.append(p);el.log.scrollTop=el.log.scrollHeight;return p;};
  function paint(){
    const omni=config().omni;el.model.textContent=omni.configured?omni.model:'MOCK · no key';el.model.classList.toggle('mock',!omni.configured);el.gateway.textContent=omni.gateway;
    el.leaving.textContent=!enabled?'':el.vision.checked?`≤${KEYFRAMES} keyframes + voice per turn → ${omni.gateway}`:'voice + numbers only';
    el.toggle.textContent=enabled?'Stop coach':'Start coach';el.toggle.classList.toggle('danger',enabled);el.toggle.classList.toggle('primary',!enabled);
  }

  function stopSpeaking(){for(const source of playing){try{source.stop();}catch{/* already ended */}}playing.clear();nextTime=0;speechSynthesis?.cancel();if(gate)gate.ratio=3.2;}
  function play(samples,rate){
    const buffer=ctx.createBuffer(1,samples.length,rate);buffer.copyToChannel(samples,0);const source=ctx.createBufferSource();source.buffer=buffer;source.connect(out);
    const at=Math.max(ctx.currentTime+.04,nextTime);source.start(at);nextTime=at+buffer.duration;playing.add(source);
    // The mic hears the speakers. Echo cancellation does most of the work; a stiffer gate does the rest,
    // so the coach cannot interrupt itself but a person talking over it still can.
    gate.ratio=8;source.onended=()=>{playing.delete(source);if(!playing.size)gate.ratio=3.2;};
  }

  function keyframe(){
    const video=document.getElementById('webcam');
    if(!enabled||!el.vision.checked||!video?.videoWidth||video.readyState<2)return;
    grab.width=320;grab.height=Math.round(320*video.videoHeight/video.videoWidth);grab.getContext('2d').drawImage(video,0,0,grab.width,grab.height);
    frames.push(grab.toDataURL('image/jpeg',.6).split(',')[1]);if(frames.length>KEYFRAMES)frames.shift();
  }

  async function turn({audio=null,text=null,trigger=null}){
    if(busy)return;busy=true;controller=new AbortController();sinceTurn=0;lastTurnAt=performance.now();
    const sent=el.vision.checked?frames.slice():[];const body={frames:sent,telemetry:stats.snapshot(performance.now(),trigger),history:history.slice(-6),voice:el.voice.checked};
    if(audio)body.audioWav=bytesToBase64(encodeWav(downsample(audio,ctx.sampleRate,16000),16000));else if(text)body.text=text;
    if(text)say('you',text);else if(audio)say('you','(spoke)');
    const line=say('coach','…');let said='',mock=false,started=performance.now(),firstAt=null;
    try{
      await obs.span('coach.turn',{'coach.trigger':trigger||(audio?'voice':'text'),'coach.frames':sent.length,'coach.audio_ms':audio?Math.round(audio.length/ctx.sampleRate*1000):0},async span=>{
        const response=await fetch(api+'/sponsors/coach/turn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:controller.signal});
        if(!response.ok)throw new Error((await response.json().catch(()=>({}))).error||'Coach relay refused the request.');
        const reader=response.body.getReader(),decoder=new TextDecoder(),stream=new EventStream();
        for(;;){
          const {value,done}=await reader.read();if(done)break;
          for(const {event,data} of stream.feed(decoder.decode(value,{stream:true}))){
            if(event==='meta'){mock=data.mock;span.setAttribute('coach.model',data.model);span.setAttribute('coach.mock',!!data.mock);}
            else if(event==='text'){firstAt??=performance.now();said+=data.delta;line.textContent='Coach: '+said;el.log.scrollTop=el.log.scrollHeight;}
            else if(event==='audio'){firstAt??=performance.now();if(el.voice.checked&&ctx)play(pcm16ToFloat32(base64ToBytes(data.pcm16)),data.rate||24000);}
            else if(event==='error')throw new Error(data.message+(data.detail?' '+String(data.detail).slice(0,160):''));
          }
        }
        span.setAttribute('coach.first_response_ms',Math.round((firstAt??performance.now())-started));
      });
      if(!said)line.textContent='Coach: (no reply)';
      // The stand-in has no voice of its own; the browser reads it aloud so the loop can be rehearsed. It is labelled.
      if(mock&&said&&el.voice.checked&&'speechSynthesis' in window)speechSynthesis.speak(new SpeechSynthesisUtterance(said));
      if(said)history.push({role:'user',content:text||(audio?'[spoken question]':'[asked for a cue]')},{role:'assistant',content:said});
      status((mock?'Mock reply (add an OMNI key for the real model). ':'')+(firstAt?`First response in ${Math.round(firstAt-started)} ms.`:''));
    }catch(error){
      if(error.name==='AbortError'){line.textContent='Coach: '+(said||'…')+' (interrupted)';}
      else{line.remove();status(error.message,true);obs.error(error,{feature:'coach'});}
    }finally{busy=false;controller=null;}
  }

  function onAudio(chunk){
    const joined=new Float32Array(pending.length+chunk.length);joined.set(pending);joined.set(chunk,pending.length);pending=joined;
    const size=Math.round(ctx.sampleRate*FRAME_MS/1000);
    while(pending.length>=size){
      const frame=pending.slice(0,size);pending=pending.slice(size);const level=rms(frame),event=gate.push(level,FRAME_MS);
      el.meter.firstElementChild.style.width=Math.min(100,level*600)+'%';el.meter.classList.toggle('open',gate.speaking);
      if(recording)recording.push(frame);else{preroll.push(frame);if(preroll.length>PREROLL_FRAMES)preroll.shift();}
      if(event==='start'){
        // Barge-in: talking over the coach cuts it off mid-sentence, like a real corner.
        if(playing.size||busy){stopSpeaking();controller?.abort();}
        recording=preroll.slice();preroll=[];status('Listening…');
      }else if(event==='end'||event==='discard'){
        const spoken=recording;recording=null;
        if(event==='discard'||!spoken){status('Heard a noise, not a question.');continue;}
        const audio=new Float32Array(spoken.reduce((n,f)=>n+f.length,0));let offset=0;for(const f of spoken){audio.set(f,offset);offset+=f.length;}
        status('Thinking…');turn({audio});
      }
    }
  }

  async function start(){
    try{
      ctx=new AudioContext();await ctx.resume();out=ctx.createGain();out.connect(ctx.destination);streamOut=ctx.createMediaStreamDestination();out.connect(streamOut);
      gate=new VoiceGate();enabled=true;keyTimer=setInterval(keyframe,KEYFRAME_MS);paint();
      try{
        mic=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
        await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP],{type:'application/javascript'})));
        node=new AudioWorkletNode(ctx,'punching-face-tap');node.port.onmessage=e=>onAudio(e.data);ctx.createMediaStreamSource(mic).connect(node);
        status('Learning the room noise… then just talk.');setTimeout(()=>enabled&&!busy&&status('Listening. Ask “how is my guard?” or throw a combo.'),900);
      }catch(error){status('No microphone ('+error.name+'). Typed questions still work.',true);obs.warn('coach.mic_unavailable',{reason:error.name});}
      obs.crumb('coach','started',{vision:el.vision.checked,voice:el.voice.checked});window.dispatchEvent(new CustomEvent('cornerman:audio',{detail:streamOut.stream}));
    }catch(error){status(error.message,true);obs.error(error,{feature:'coach'});}
  }
  function stop(){
    enabled=false;clearInterval(keyTimer);controller?.abort();stopSpeaking();node?.disconnect();mic?.getTracks().forEach(t=>t.stop());ctx?.close();
    ctx=mic=node=null;frames=[];recording=null;preroll=[];pending=new Float32Array(0);el.meter.firstElementChild.style.width='0';paint();status('Coach is off. Nothing is being sent.');
  }

  el.toggle.onclick=()=>enabled?stop():start();
  el.vision.onchange=()=>{if(!el.vision.checked)frames=[];paint();};
  const ask=()=>{const text=el.ask.value.trim();if(!text)return;el.ask.value='';turn({text});};
  el.send.onclick=ask;el.ask.onkeydown=e=>{e.stopPropagation();if(e.key==='Enter')ask();};el.ask.onkeyup=e=>e.stopPropagation();el.key.onkeydown=e=>e.stopPropagation();
  el.save.onclick=async()=>{
    const apiKey=el.key.value.trim();if(!apiKey)return;el.key.value='';
    try{const r=await fetch(api+'/sponsors/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({group:'omni',apiKey})});if(!r.ok)throw new Error((await r.json()).error);await refreshConfig();paint();status('Key saved on this computer.');}
    catch(error){status(error.message,true);}
  };
  paint();

  return {
    // Called for every landed punch, local or remote. The coach volunteers a cue at natural beats,
    // never over a person who is speaking, and never back-to-back.
    onPunch(triggers){
      sinceTurn++;
      if(!enabled||!el.proactive.checked||busy||gate?.speaking||playing.size||performance.now()-lastTurnAt<QUIET_AFTER_TURN_MS)return;
      if(triggers.length||sinceTurn>=PUNCHES_PER_CUE)turn({trigger:triggers[0]||`${PUNCHES_PER_CUE} punches since the last cue`});
    },
    get outputStream(){return streamOut?.stream||null;},
    repaint:paint,
  };
}
