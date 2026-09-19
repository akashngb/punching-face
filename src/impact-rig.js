const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const smooth=(a,b,v)=>{const t=clamp((v-a)/(b-a),0,1);return t*t*(3-2*t);};

// Art-directed impact correctives, layered over the contact solver. These are
// facial animation fields, not a claim that the FEM predicts this much motion.
// All fields share the same rest coordinates, including UV seam duplicates.
export class FaceImpactRig {
  constructor(rest,anchors){
    this.offset=new Float32Array(rest.length);this.events=[];this.setAnchors(anchors);
  }
  setAnchors(anchors){
    this.anchors=anchors??{
      13:[0,-.04,.07],14:[0,-.042,.07],152:[0,-.105,.045],
      50:[-.05,-.005,.06],280:[.05,-.005,.06],
      61:[-.03,-.04,.065],291:[.03,-.04,.065],
      159:[-.035,.037,.065],386:[.035,.037,.065],
    };
  }
  // Smooth onset, a readable compression crest, then a damped recovery.
  envelope(age){
    if(age<.12)return smooth(0,.12,age);
    const t=age-.12;return (1+16*t)*Math.exp(-16*t)*(1-smooth(.65,.9,t));
  }
  get hasPeaked(){return this.events.length>0&&this.events.every(e=>e.age>=.12);}
  // Public entry: dispatch to a mode-specific field. hook is the historical pose.
  trigger(rest,point,direction,speed,softness,mode='hook'){
    let field;
    if(mode==='uppercut')field=this._uppercutField(rest,point,speed,softness);
    else if(mode==='jab')field=this._jabField(rest,point,speed,softness);
    else field=this._hookField(rest,point,direction,speed,softness);
    // A short queue supports alternating hooks without unbounded accumulation.
    this.events.push({age:0,field});if(this.events.length>4)this.events.shift();
  }
  _hookField(rest,point,direction,speed,softness){
    const a=this.anchors,upper=a[13],lower=a[14],chin=a[152];
    const mouth=upper.map((v,i)=>(v+lower[i])*.5),mx=mouth[0],my=mouth[1],mz=mouth[2];
    const scale=clamp((my-chin[1])/.058,.65,1.5);
    const side=Math.sign(point.x-mx)||-Math.sign(direction.x)||1,push=-side;
    const eye=a[side<0?159:386],otherEye=a[side<0?386:159];
    const cheek=a[side<0?50:280]??[mx+side*.05*scale,my+.035*scale,mz-.012*scale];
    const corner=a[side<0?61:291];
    const amount=clamp(speed/1.05,.25,1.35)*(.78+clamp(softness,0,1)*.37)*scale;
    const field=new Float32Array(rest.length);
    const gaussian=(x,y,c,rx,ry)=>Math.exp(-(((x-c[0])/(rx*scale))**2)-((y-c[1])/(ry*scale))**2);
    let maximum=0;
    for(let i=0;i<rest.length;i+=3){
      const x=rest[i],y=rest[i+1],z=rest[i+2];
      // Blend to the skull/neck continuously. Do not use the binary Newton
      // binding mask here: that would tear the jaw at the observed-face border.
      const front=smooth(mz-.14*scale,mz-.035*scale,z);
      const neck=smooth(chin[1]-.04*scale,chin[1]-.005*scale,y);
      const brow=1-smooth(eye[1]+.004*scale,eye[1]+.035*scale,y);
      const mask=front*neck*brow;if(mask<1e-5)continue;
      const struck=smooth(-.055*scale,.04*scale,side*(x-mx));
      const cheekWeight=gaussian(x,y,cheek,.05,.047);
      const contact=gaussian(x,y,[point.x,point.y],.032,.037);
      const mouthWeight=gaussian(x,y,mouth,.053,.030);
      const cornerWeight=gaussian(x,y,corner,.03,.03);
      const lowerFace=1-smooth(my-.015*scale,my+.035*scale,y);
      const jaw=lowerFace*Math.exp(-(((x-mx)/(.115*scale))**4));
      const eyeWeight=gaussian(x,y,eye,.028,.024);
      const farEyeWeight=gaussian(x,y,otherEye,.028,.024);
      const cheekLift=gaussian(x,y,[eye[0],eye[1]-.023*scale],.033,.026);

      // Broad tissue transport carries the mouth and jaw with the hook. The
      // cheek also flattens inward, with a lifted/bulging rim below the eye.
      let dx=push*(.015*cheekWeight+.014*mouthWeight+.010*jaw);
      let dy=.008*cheekLift+.004*cornerWeight*struck;
      let dz=-.009*contact-.006*cheekWeight+.004*cheekLift;

      // Rotate the lower face around a jaw hinge, with smooth lip separation.
      // This acts on the full head, so the chin and jaw silhouette move too.
      const lowerLip=smooth(my+.003*scale,my-.013*scale,y);
      const jawWeight=jaw*(1-mouthWeight)+mouthWeight*lowerLip;
      const angle=.17*jawWeight,hy=my+.064*scale,hz=mz-.085*scale;
      dy+=(y-hy)*(Math.cos(angle)-1)-(z-hz)*Math.sin(angle);
      dz+=(y-hy)*Math.sin(angle)+(z-hz)*(Math.cos(angle)-1);
      dy-=.005*mouthWeight*(1-struck); // asymmetric mouth stretch

      // Squeeze each eyelid toward its own eye line instead of translating
      // both lids down. The struck eye reacts more strongly than the far eye.
      const eyeY=eye[1]-.003*scale,farEyeY=otherEye[1]-.003*scale;
      dy-=.65*(y-eyeY)*eyeWeight+.30*(y-farEyeY)*farEyeWeight;
      dx+=push*.004*eyeWeight;
      dz-=.0025*eyeWeight;
      field[i]=dx*mask*amount;field[i+1]=dy*mask*amount;field[i+2]=dz*mask*amount;
      maximum=Math.max(maximum,Math.hypot(field[i],field[i+1],field[i+2]));
    }
    // Scale the complete pose together rather than clipping vertices, which
    // would flatten the silhouette and introduce creases under repeated hits.
    const limit=.032*scale;if(maximum>limit)for(let i=0;i<field.length;i++)field[i]*=limit/maximum;
    return field;
  }
  // Uppercut: symmetric chin lift. Chin+jaw rotate around a hinge just above/behind
  // the mouth so the whole jaw silhouette swings up and slightly forward. Lower lip
  // compresses toward the upper lip. Eyes and forehead stay put. No left/right bias.
  _uppercutField(rest,point,speed,softness){
    const a=this.anchors,upper=a[13],lower=a[14],chin=a[152];
    const mouth=upper.map((v,i)=>(v+lower[i])*.5),mx=mouth[0],my=mouth[1],mz=mouth[2];
    const scale=clamp((my-chin[1])/.058,.65,1.5);
    const amount=clamp(speed/1.05,.25,1.35)*(.78+clamp(softness,0,1)*.37)*scale;
    const field=new Float32Array(rest.length);
    const gaussian=(x,y,c,rx,ry)=>Math.exp(-(((x-c[0])/(rx*scale))**2)-((y-c[1])/(ry*scale))**2);
    // Hinge sits above and behind the chin so a small negative X-rotation angle
    // sweeps the chin up and forward — the arc of a real jaw swing.
    const hy=my+.020*scale,hz=mz-.050*scale;
    const cheek50=a[50],cheek280=a[280];
    let maximum=0;
    for(let i=0;i<rest.length;i+=3){
      const x=rest[i],y=rest[i+1],z=rest[i+2];
      const front=smooth(mz-.14*scale,mz-.030*scale,z);
      // Include the underside of the jaw and the top of the neck; a hook mask
      // truncates too early because a hook doesn't lift the throat.
      const neck=smooth(chin[1]-.055*scale,chin[1]+.005*scale,y);
      const brow=1-smooth(my+.028*scale,my+.058*scale,y);
      const mask=front*neck*brow;if(mask<1e-5)continue;
      // Region weights along the vertical axis: 1 at chin, decays to 0 at brow.
      const chinRegion=smooth(my+.008*scale,chin[1]-.008*scale,-y);
      const lowerLip=gaussian(x,y,[mx,my+.006*scale],.038,.014);
      const lowerCheekL=gaussian(x,y,[cheek50[0],my+.010*scale],.035,.028);
      const lowerCheekR=gaussian(x,y,[cheek280[0],my+.010*scale],.035,.028);
      // Jaw hinge rotation. Negative angle swings a point below the hinge up-and-forward.
      const jawSwing=chinRegion*0.75+Math.max(lowerLip,lowerCheekL+lowerCheekR)*0.25;
      const angle=-0.22*jawSwing;
      const yr=y-hy,zr=z-hz;
      let dy=yr*(Math.cos(angle)-1)-zr*Math.sin(angle);
      let dz=yr*Math.sin(angle)+zr*(Math.cos(angle)-1);
      // Explicit lift and slight forward compression on top of the swing,
      // symmetric around the mouth centerline.
      dy+=.008*lowerLip+.006*(lowerCheekL+lowerCheekR);
      dz+=-.004*lowerLip-.003*(lowerCheekL+lowerCheekR);
      // dx is deliberately zero — an uppercut is symmetric.
      field[i]=0;field[i+1]=dy*mask*amount;field[i+2]=dz*mask*amount;
      const m=Math.hypot(field[i],field[i+1],field[i+2]);if(m>maximum)maximum=m;
    }
    const limit=.038*scale;if(maximum>limit)for(let i=0;i<field.length;i++)field[i]*=limit/maximum;
    return field;
  }
  // Jab: symmetric nose-and-upper-lip inward push. Nose flattens slightly, upper
  // lip compresses inward. No lateral bias.
  _jabField(rest,point,speed,softness){
    const a=this.anchors,upper=a[13],lower=a[14],chin=a[152];
    const mouth=upper.map((v,i)=>(v+lower[i])*.5),mx=mouth[0],my=mouth[1],mz=mouth[2];
    const scale=clamp((my-chin[1])/.058,.65,1.5);
    const amount=clamp(speed/1.05,.25,1.35)*(.78+clamp(softness,0,1)*.37)*scale;
    const field=new Float32Array(rest.length);
    const gaussian=(x,y,c,rx,ry)=>Math.exp(-(((x-c[0])/(rx*scale))**2)-((y-c[1])/(ry*scale))**2);
    // Nose tip roughly halfway between mouth and brow line.
    const nose=[mx,my+.035*scale,mz+.010*scale];
    let maximum=0;
    for(let i=0;i<rest.length;i+=3){
      const x=rest[i],y=rest[i+1],z=rest[i+2];
      const front=smooth(mz-.12*scale,mz-.020*scale,z);
      const brow=1-smooth(my+.055*scale,my+.080*scale,y);
      const chinCut=smooth(chin[1]-.015*scale,chin[1]+.020*scale,y);
      const mask=front*brow*chinCut;if(mask<1e-5)continue;
      const noseWeight=gaussian(x,y,nose,.028,.033);
      const upperLipWeight=gaussian(x,y,[mx,my+.005*scale],.036,.014);
      const contact=gaussian(x,y,[point.x,point.y],.028,.032);
      // Symmetric inward push; a hair of downward on nose ridge to sell the flatten.
      const dx=0;
      const dy=-.003*noseWeight;
      const dz=-.014*noseWeight-.010*upperLipWeight-.008*contact;
      field[i]=dx*mask*amount;field[i+1]=dy*mask*amount;field[i+2]=dz*mask*amount;
      const m=Math.hypot(field[i],field[i+1],field[i+2]);if(m>maximum)maximum=m;
    }
    const limit=.028*scale;if(maximum>limit)for(let i=0;i<field.length;i++)field[i]*=limit/maximum;
    return field;
  }
  step(dt){
    this.offset.fill(0);
    for(const e of this.events)e.age+=Math.max(0,dt);
    this.events=this.events.filter(e=>e.age<1.02);
    const weights=this.events.map(e=>this.envelope(e.age));
    const normalization=Math.max(1,weights.reduce((sum,v)=>sum+v,0));
    for(let e=0;e<this.events.length;e++){
      const weight=weights[e]/normalization,field=this.events[e].field;
      for(let i=0;i<this.offset.length;i++)this.offset[i]+=field[i]*weight;
    }
  }
  reset(){this.events=[];this.offset.fill(0);}
}
