// Mouth shape derived from the speech that is actually playing.
//
// Analysed rather than predicted from text. OMNI streams PCM with no word or
// phoneme timestamps, so a text-scheduled viseme track would have nothing to
// align to and would drift. Reading the audible signal is inherently in sync,
// is language-agnostic, and handles barge-in for free: when the panel cancels
// its sources the level falls and the mouth closes on its own.
//
// An AnalyserNode is a tap. It is connected as a leaf, so the existing
// `out → destination` and `out → streamOut` paths are untouched and the LiveKit
// guest stream is unaffected.

import {rms} from '../sponsors/audio.js';

const FLOOR=.006,CEIL=.13;      // speech RMS lives between these
const ATTACK=.020,RELEASE=.085; // a mouth opens faster than it closes
// Voiced energy below this is body/vowel; above it is the fricative and
// formant range that separates a wide "ee" from a rounded "oo".
const SPLIT_HZ=1100,TOP_HZ=5000;
const SILENT={open:0,spread:0,round:0};

/**
 * @param {AudioContext} ctx
 * @param {AudioNode} source the speech output bus to listen to
 * @returns {{read:(dt:number)=>{open:number,spread:number,round:number},dispose:()=>void}}
 */
export function createMouthSignal(ctx,source){
  const analyser=ctx.createAnalyser();
  analyser.fftSize=512;
  analyser.smoothingTimeConstant=0; // we do our own, asymmetrically
  try{source.connect(analyser);}catch{return {read:()=>SILENT,dispose(){}};}
  const time=new Float32Array(analyser.fftSize);
  const bins=new Uint8Array(analyser.frequencyBinCount);
  const binHz=ctx.sampleRate/analyser.fftSize;
  const lowFrom=Math.max(1,Math.round(180/binHz)),lowTo=Math.round(SPLIT_HZ/binHz);
  const highTo=Math.min(bins.length,Math.round(TOP_HZ/binHz));
  let open=0,spread=0,round=0,disposed=false;

  return {
    read(dt){
      if(disposed)return SILENT;
      const step=clamp(Number(dt)||0,0,.1);
      analyser.getFloatTimeDomainData(time);
      const level=rms(time);
      const loud=Math.pow(clamp((level-FLOOR)/(CEIL-FLOOR),0,1),.62);

      let wantSpread=0,wantRound=0;
      if(loud>.01){
        analyser.getByteFrequencyData(bins);
        let low=0,high=0;
        for(let i=lowFrom;i<lowTo;i++)low+=bins[i];
        for(let i=lowTo;i<highTo;i++)high+=bins[i];
        const total=low+high;
        if(total>0){
          // Balance of the two bands, recentred so a neutral vowel sits at 0.
          const tilt=(high/total-.42)*2.6;
          // Both are gated by loudness: a closed mouth must not purse.
          wantSpread=clamp(tilt,0,1)*loud;
          wantRound=clamp(-tilt,0,1)*loud;
        }
      }

      open=follow(open,loud,step);
      spread=follow(spread,wantSpread,step);
      round=follow(round,wantRound,step);
      return {open,spread,round};
    },
    dispose(){
      if(disposed)return;
      disposed=true;
      try{source.disconnect(analyser);}catch{/* graph already torn down */}
    },
  };
}

/**
 * A stand-in envelope for mock mode. Without an OMNI key the panel speaks
 * through `speechSynthesis`, which cannot be routed into WebAudio, so the
 * analyser would see silence and the face would sit there mute. This drives a
 * syllable-rate mouth for the duration of an utterance instead. It is a
 * stand-in, like the mock reply it accompanies — not a claim of lip-sync.
 */
export function createSyntheticSignal(){
  let until=0,open=0,spread=0,round=0,phase=0;
  return {
    speakFor(seconds){until=now()+Math.max(0,seconds);},
    stop(){until=0;},
    read(dt){
      const step=clamp(Number(dt)||0,0,.1);
      let target=0,tilt=0;
      if(now()<until){
        phase+=step;
        // ~4.2 syllables a second, the rate of unhurried speech, with a second
        // slower term so it does not pulse like a metronome.
        const syllable=.5-.5*Math.cos(phase*2*Math.PI*4.2);
        target=.25+.65*syllable*(.72+.28*Math.sin(phase*2.1));
        tilt=Math.sin(phase*1.7);
      }
      open=follow(open,target,step);
      spread=follow(spread,clamp(tilt,0,1)*target*.8,step);
      round=follow(round,clamp(-tilt,0,1)*target*.8,step);
      return {open,spread,round};
    },
    dispose(){until=0;},
  };
}

const now=()=>(typeof performance!=='undefined'?performance.now():Date.now())/1000;
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
function follow(value,target,dt){
  const tau=target>value?ATTACK:RELEASE;
  return value+(target-value)*(1-Math.exp(-dt/tau));
}
