// Speech shapes for the head, layered over the contact solver exactly the way
// `FaceImpactRig` is (src/impact-rig.js): a separate additive offset applied
// after the spring integration, never fed into it.
//
// It must NOT drive `FaceDynamics.rig.jaw`. That is a pose control, and
// `NewtonFaceDynamics.step` (src/newton-dynamics.js) treats any change to `rig`
// as a new pose: it bumps a version, zeroes the accumulated offsets and
// re-uploads to the physics server. At 60 fps that would thrash the solver.
//
// Heads arrive watertight — `build_photo_face.py` rejects anything else — so the
// lips start as a sealed surface. `src/mouth-aperture.js` cuts the sealing band
// away at load; this rig then drives three things into the hole it leaves: the
// jaw swinging on a hinge, the lip rim parting, and the corners spreading or
// gathering. On a head whose mouth could not be located nothing is cut, and the
// same shapes simply read as stretching lips.
//
// Shapes are precomputed once per anchor set as three basis fields, because a
// speaking mouth needs a new pose every frame and impact fields do not. `step`
// is then one multiply-add pass, not a field rebuild.

const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const smooth=(a,b,v)=>{const t=clamp((v-a)/(b-a),0,1);return t*t*(3-2*t);};

// Same reference-frame table as FaceImpactRig, so a head that never had anchors
// detected still gets a plausible mouth instead of nothing.
const DEFAULT_ANCHORS={
  13:[0,-.04,.07],14:[0,-.042,.07],152:[0,-.105,.045],
  50:[-.05,-.005,.06],280:[.05,-.005,.06],
  61:[-.03,-.04,.065],291:[.03,-.04,.065],
};

// Peak travel per shape, in metres on a head normalised to ~0.2 m hairline-to-chin.
// The jaw limit is deliberately under what the manual Jaw slider allows: sealed
// lips stretch rather than part, and past this the lower face reads as rubber.
const OPEN_LIMIT=.0145,SPREAD_LIMIT=.0060,ROUND_LIMIT=.0050;
// Share of the jaw's travel spent parting the lip rim rather than swinging the
// whole lower face. Once src/mouth-aperture.js has removed the sealing band this
// is what actually opens the hole; on a still-sealed head it just reads as the
// lips stretching, which is what they did before.
const LIP_PART=.055;
// Shapes chase the signal with a short constant so an abrupt analyser reading
// cannot snap a vertex. The signal is already smoothed; this is a safety net.
const FOLLOW_SECONDS=.025;

/** Scale a basis field so its largest vertex displacement is exactly `limit`. */
function normalize(field,limit){
  let maximum=0;
  for(let i=0;i<field.length;i+=3){
    const m=Math.hypot(field[i],field[i+1],field[i+2]);
    if(m>maximum)maximum=m;
  }
  if(maximum>1e-9){const k=limit/maximum;for(let i=0;i<field.length;i++)field[i]*=k;}
  return field;
}

export class FaceSpeechRig{
  constructor(rest,anchors){
    this.rest=rest;
    this.offset=new Float32Array(rest.length);
    this.open=0;this.spread=0;this.round=0;
    this._want={open:0,spread:0,round:0};
    this._written=false;
    this.setAnchors(anchors);
  }

  setAnchors(anchors){this.anchors=anchors??DEFAULT_ANCHORS;this._build();}

  /** Target shape, each 0..1. Called once per frame from the render loop. */
  set(signal){
    const s=signal||{};
    this._want.open=clamp(Number(s.open)||0,0,1);
    this._want.spread=clamp(Number(s.spread)||0,0,1);
    this._want.round=clamp(Number(s.round)||0,0,1);
  }

  /**
   * `duck` scales the whole rig down, so a landed punch reads over the top of a
   * sentence instead of fighting it.
   */
  step(dt,duck=1){
    const k=1-Math.exp(-Math.max(dt,0)/FOLLOW_SECONDS);
    this.open+=(this._want.open-this.open)*k;
    this.spread+=(this._want.spread-this.spread)*k;
    this.round+=(this._want.round-this.round)*k;
    const scale=clamp(duck,0,1);
    const open=this.open*scale,spread=this.spread*scale,round=this.round*scale;
    // A silent face must be bit-for-bit untouched, and must not pay for a full
    // buffer write on every frame it stays silent.
    if(!this.openBasis||(open<1e-4&&spread<1e-4&&round<1e-4)){
      if(this._written){this.offset.fill(0);this._written=false;}
      return;
    }
    for(let i=0;i<this.offset.length;i++){
      this.offset[i]=open*this.openBasis[i]+spread*this.spreadBasis[i]+round*this.roundBasis[i];
    }
    this._written=true;
  }

  reset(){
    this.open=this.spread=this.round=0;
    this._want.open=this._want.spread=this._want.round=0;
    this.offset.fill(0);this._written=false;
  }

  _build(){
    const rest=this.rest,a=this.anchors;
    const upper=a[13],lower=a[14],chin=a[152];
    // Without a mouth line there is no defensible place to put a mouth. Go inert
    // rather than deform the middle of someone's face on a guess.
    if(!upper||!lower||!chin){this.openBasis=this.spreadBasis=this.roundBasis=null;return;}
    const mouth=upper.map((v,i)=>(v+lower[i])*.5),mx=mouth[0],my=mouth[1],mz=mouth[2];
    // Same normalisation the impact rig uses, so both rigs agree on head size.
    const scale=clamp((my-chin[1])/.058,.65,1.5);
    const cornerL=a[61]??[mx-.030*scale,my,mz],cornerR=a[291]??[mx+.030*scale,my,mz];
    const open=new Float32Array(rest.length);
    const spread=new Float32Array(rest.length);
    const round=new Float32Array(rest.length);
    const gaussian=(x,y,c,rx,ry)=>Math.exp(-(((x-c[0])/(rx*scale))**2)-(((y-c[1])/(ry*scale))**2));
    // Hinge placement lifted from the tuned hook field in src/impact-rig.js.
    const hy=my+.064*scale,hz=mz-.085*scale;
    for(let i=0;i<rest.length;i+=3){
      const x=rest[i],y=rest[i+1],z=rest[i+2];
      // Speech moves the lower front of the face only: never the skull, never
      // the neck, never above the nose. Blend out rather than cut, or the jaw
      // tears at the mask border.
      const front=smooth(mz-.14*scale,mz-.035*scale,z);
      const neck=smooth(chin[1]-.045*scale,chin[1]-.008*scale,y);
      // Fade out by the base of the nose. A wider band reaches eye level on a
      // real head (anchors 159/386 sit around y+.037), and lips that spread must
      // not stir an eyelid — however faintly.
      const underNose=1-smooth(my+.030*scale,my+.055*scale,y);
      const mask=front*neck*underNose;
      if(mask<1e-5)continue;

      const mouthWeight=gaussian(x,y,mouth,.053,.030);
      const lowerFace=1-smooth(my-.015*scale,my+.035*scale,y);
      const jaw=lowerFace*Math.exp(-(((x-mx)/(.115*scale))**4));
      const lowerLip=smooth(my+.003*scale,my-.013*scale,y);
      const jawWeight=(jaw*(1-mouthWeight)+mouthWeight*lowerLip)*mask;
      // Linearised hinge rotation. For small angles dy≈-(z-hz)θ and dz≈(y-hy)θ,
      // which is linear in θ — so `open` scales this basis directly instead of
      // the rig having to re-derive a rotation every frame.
      open[i+1]=-(z-hz)*jawWeight;
      open[i+2]=(y-hy)*jawWeight;
      // Explicit rim separation: upper lip up, lower lip down, in a tight band
      // along the lip line. The hinge above drags both lips the same way, so
      // without this the aperture barely changes shape.
      const lipBand=Math.exp(-(((x-mx)/(.034*scale))**2)-(((y-my)/(.011*scale))**2))*mask;
      open[i+1]+=(y>my?1:-1)*lipBand*LIP_PART;

      const corners=(gaussian(x,y,cornerL,.030,.026)+gaussian(x,y,cornerR,.030,.026))*mask;
      // No `||1` fallback here, unlike the directional hook field in impact-rig.js:
      // spreading and rounding are symmetric, so a vertex exactly on the midline
      // must get no lateral motion at all rather than an arbitrary sideways nudge.
      const side=Math.sign(x-mx);
      // Wide vowels: corners travel outward, lips flatten toward the mouth line.
      spread[i]=side*corners;
      spread[i+1]=-(y-my)*mouthWeight*mask*.55;
      // Rounded vowels: corners gather in, lips purse forward and bunch.
      round[i]=-side*corners*.85;
      round[i+1]=(y-my)*mouthWeight*mask*.35;
      round[i+2]=mouthWeight*mask;
    }
    this.openBasis=normalize(open,OPEN_LIMIT);
    this.spreadBasis=normalize(spread,SPREAD_LIMIT);
    this.roundBasis=normalize(round,ROUND_LIMIT);
  }
}
