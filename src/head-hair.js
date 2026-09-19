import * as THREE from 'three';
import { photoStrandGeometry, validatePhotoCurves } from './photo-hair-strands.js';

const clamp=THREE.MathUtils.clamp;
const types=['straight','wavy','curly','coily','braided','locs'];
function randomGenerator(seed){let s=seed>>>0;return ()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};}

// Independent tapered fibers. Roots follow the welded surface, so UV seams,
// expression edits and sculpting cannot detach hair from the head.
export class HeadHair extends THREE.Mesh {
  constructor(source,spec){
    const geometry=new THREE.BufferGeometry();
    const material=new THREE.MeshPhysicalMaterial({vertexColors:true,roughness:.78,metalness:0,specularIntensity:.06,anisotropy:.5,side:THREE.DoubleSide});
    super(geometry,material);this.name='Editable strand hair';this.userData={accessory:'hair',estimated:true};
    this.spec=structuredClone(spec);this.frustumCulled=false;
    const ids=spec?.rootTriangles,weights=spec?.rootWeights,count=source.attributes.position.count;
    if(!ids?.length||ids.length%3||ids.length>45000||weights?.length!==ids.length||ids.some(i=>!Number.isInteger(i)||i<0||i>=count)||weights.some(w=>!Number.isFinite(w)||w<0||w>1)||!Number.isFinite(spec.hairlineY))throw new Error('Invalid hair root binding.');
    for(let i=0;i<weights.length;i+=3)if(Math.abs(weights[i]+weights[i+1]+weights[i+2]-1)>1e-4)throw new Error('Invalid hair root weights.');
    this.rebuild(source);
  }
  rebuild(source,overrides={}){
    this.curveBinding=null;
    if(overrides.type&&overrides.type!==this.spec.parameters.type)this.spec.mode='editable-prior';
    const p={...this.spec.parameters,...overrides};this.spec.parameters=p;
    const guides=['photo-detail','photo-strands'].includes(this.spec.mode)?this.spec.photoGuides:null;
    if(guides&&(!guides.directions?.every(Number.isFinite)||!guides.colors?.every(Number.isFinite)||!guides.confidence?.every(Number.isFinite)||guides.directions.length!==this.spec.rootTriangles.length||guides.colors.length!==guides.directions.length||guides.confidence.length*3!==guides.directions.length))throw new Error('Invalid photographic hair guides.');
    if(this.spec.mode==='photo-strands'){
      validatePhotoCurves(guides,this.spec.rootTriangles.length/3);
      // The source colors already include illumination. Very weak added
      // specularity avoids a silver sheen under the bright studio lights.
      this.material.dispose();this.material=guides.observedOnly?new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide,toneMapped:false}):new THREE.MeshPhysicalMaterial({vertexColors:true,color:0xaaaaaa,roughness:.9,metalness:0,specularIntensity:.0008,anisotropy:.8,anisotropyRotation:Math.PI/2,side:THREE.DoubleSide});
      const result=photoStrandGeometry(source,this.spec,p);
      this.geometry.dispose();this.geometry=result.geometry;this.roots=result.roots;this.strandCount=result.strandCount;
      this.base=this.geometry.attributes.position.array.slice();this.baseNormals=this.geometry.attributes.normal.array.slice();
      this.bindCurveStations(source);
      this.lastRoots=new Float32Array(this.roots.length*6);this.lastRoots.fill(Infinity);return;
    }
    if(guides&&!this.material.isMeshBasicMaterial){this.material.dispose();this.material=new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide,toneMapped:false});}
    if(!guides&&this.material.isMeshBasicMaterial){this.material.dispose();this.material=new THREE.MeshPhysicalMaterial({vertexColors:true,roughness:.78,specularIntensity:.06,anisotropy:.5,side:THREE.DoubleSide});}
    const finite=(key,fallback,lo,hi)=>Number.isFinite(p[key])?clamp(p[key],lo,hi):fallback;
    const type=types.includes(p.type)?p.type:'wavy';
    const density=finite('density',.8,0,1),length=finite('lengthMm',55,1,450)*.001,sideLength=finite('sideLengthMm',15,0,300)*.001;
    const curl=finite('curlTightness',.3,0,1),frizz=finite('frizz',.15,0,1),lift=finite('rootLiftMm',3,0,12)*.001;
    const wave=finite('waveLengthMm',20,3,80)*.001,curlRadius=finite('curlRadiusMm',3,.3,25)*.001;
    const flow=finite('flowDegrees',0,-90,90)*Math.PI/180,part=finite('partOffset',0,-1,1)*.07;
    const rand=randomGenerator(this.spec.seed??42),positions=[],normals=[],colors=[],uv=[],indices=[];
    const ids=this.spec.rootTriangles,weights=this.spec.rootWeights,src=source.attributes.position.array,sn=source.attributes.normal.array;
    const strandCount=Math.floor(ids.length/3*density),segments=['curly','coily','braided','locs'].includes(type)?24:16,sides=3;
    this.roots=[];const color=new THREE.Color();
    const rgb=p.colorSrgb?.length===3&&p.colorSrgb.every(Number.isFinite)?p.colorSrgb:[.06,.05,.04];
    const root=new THREE.Vector3(),normal=new THREE.Vector3(),tangent=new THREE.Vector3(),binormal=new THREE.Vector3();
    for(let i=0;i<strandCount;i++){
      if(guides&&guides.confidence[i]<.08)continue;
      root.set(0,0,0);normal.set(0,0,0);
      for(let k=0;k<3;k++){const id=ids[i*3+k]*3,w=weights[i*3+k];root.addScaledVector(new THREE.Vector3().fromArray(src,id),w);normal.addScaledVector(new THREE.Vector3().fromArray(sn,id),w);}
      normal.normalize();let top=clamp((root.y-this.spec.hairlineY+.025)/.055,0,1);
      // A high occiput is still the back of a tapered cut, not a long crown
      // lock. Blend by normal as well as height to avoid dangling nape strips.
      const posterior=clamp((-normal.z-.15)/.65,0,1)*(1-clamp(normal.y,0,1));
      top*=1-posterior;
      // Parting affects crown flow; side strands follow gravity around the ears.
      tangent.set(Math.sin(flow)+(root.x-part)*3,-(1-top)*1.8,-Math.cos(flow)*top);
      tangent.addScaledVector(normal,-tangent.dot(normal));
      if(tangent.lengthSq()<.01)tangent.set(1,0,0).addScaledVector(normal,-normal.x);
      tangent.normalize();binormal.crossVectors(normal,tangent).normalize();
      if(guides){tangent.fromArray(guides.directions,i*3).normalize();binormal.crossVectors(normal,tangent).normalize();}
      const clump=finite('clumpSizeMm',5,.3,14)*.001;
      const group=Math.round(root.x/clump)*2.4+Math.round(root.z/clump)*1.7;
      const strandLength=(sideLength*(1-top)+length*top)*(.8+.15*Math.sin(group)+.1*rand());
      const travel=Math.min(strandLength,.115),tail=Math.max(0,strandLength-.115);
      const phase=group+(rand()-.5)*.4,fly=rand()<frizz*.12;
      const radius=(.000075+rand()*.000055)*(type==='locs'?1.5:1);
      const variation=guides?.96+rand()*.08:.75+rand()*.5;
      const strandColor=guides?guides.colors.slice(i*3,i*3+3):rgb;
      color.setRGB(...strandColor.map(x=>clamp(x*variation,0,1)),THREE.SRGBColorSpace);
      const start=positions.length/3;
      const baseRoot=root.clone();this.roots.push({root:baseRoot,normal:normal.clone(),sourceIndex:i,start,count:(segments+1)*sides});
      // Curvature follows the fitted outer envelope, never an upright grass tuft.
      const bendRadius=.095,curve=[];
      for(let j=0;j<=segments;j++){
        const t=j/segments,d=travel*t,arch=Math.sin(Math.PI*t),fade=Math.min(1,t*8);
        if(guides){
          // Sub-centimeter, photo-aligned detail follows the existing locks.
          // It cannot replace the photographed hairstyle with a generic cap.
          const detailLength=Math.min(.007,.002+length*.05),along=(t-.5)*detailLength;
          const q=root.clone().addScaledVector(tangent,along).addScaledVector(normal,.00008+.00016*arch);
          q.addScaledVector(binormal,Math.sin(t*Math.PI+phase)*Math.min(.0002,curlRadius*.01)*curl);
          curve.push(q);continue;
        }
        const q=root.clone().addScaledVector(tangent,bendRadius*Math.sin(d/bendRadius)).addScaledVector(normal,bendRadius*(Math.cos(d/bendRadius)-1));
        let lateral=0,raised=0;
        if(type==='wavy'){lateral=Math.sin(d/wave*Math.PI*2+phase)*curlRadius*.45*curl*fade;raised=(1+Math.sin(d/wave*Math.PI*2+phase))*.0015*curl*fade;}
        if(type==='curly'||type==='coily'){
          const r=Math.min(curlRadius,type==='coily'?.004:.014),theta=d/Math.max(.001,wave*(1-.7*curl))*Math.PI*2+phase;
          lateral=Math.sin(theta)*r*fade;raised=(1+Math.cos(theta))*r*.7*fade;
        }
        if(type==='braided'||type==='locs'){
          // Shared spatial phase clusters neighboring fibers into twisted locks.
          lateral=Math.sin(d/.012*Math.PI*2+group)*clump*.3*fade;
          raised=(1+Math.cos(d/.012*Math.PI*2+group))*clump*.3*fade;
        }
        const fuzz=(Math.sin(t*31+phase)+Math.sin(t*57+phase*2))*.0004*frizz*fade;
        q.addScaledVector(binormal,lateral+fuzz);
        q.addScaledVector(normal,.00035+(lift*arch+raised)*(top*.7+.3)+(fly?.0035*arch:0));
        if(tail>0)q.y-=tail*t*t;
        curve.push(q);
      }
      for(let j=0;j<=segments;j++){
        const t=j/segments,axis=curve[Math.min(segments,j+1)].clone().sub(curve[Math.max(0,j-1)]).normalize();
        const across=new THREE.Vector3().crossVectors(axis,normal).normalize(),up=new THREE.Vector3().crossVectors(across,axis).normalize();
        for(let s=0;s<sides;s++){
          const theta=s/sides*Math.PI*2,n=across.clone().multiplyScalar(Math.cos(theta)).addScaledVector(up,Math.sin(theta));
          const point=curve[j].clone().addScaledVector(n,radius*(guides?.45:1)*(1-t*.9));positions.push(...point);normals.push(...n);colors.push(color.r,color.g,color.b);uv.push(s/sides,t);
          if(j<segments){const a=start+j*sides+s,b=start+j*sides+(s+1)%sides;indices.push(a,b,a+sides,b,b+sides,a+sides);}
        }
      }
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('normal',new THREE.Float32BufferAttribute(normals,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));geometry.setIndex(indices);geometry.computeBoundingSphere();
    this.geometry.dispose();this.geometry=geometry;this.base=geometry.attributes.position.array.slice();this.baseNormals=geometry.attributes.normal.array.slice();this.strandCount=this.roots.length;
    this.lastRoots=new Float32Array(this.roots.length*6);this.lastRoots.fill(Infinity);
  }
  updateSurface(source){
    if(this.curveBinding)return this.updateCurveStations(source);
    const src=source.attributes.position.array,sn=source.attributes.normal.array,ids=this.spec.rootTriangles,weights=this.spec.rootWeights,out=this.geometry.attributes.position.array,on=this.geometry.attributes.normal.array;
    const root=new THREE.Vector3(),normal=new THREE.Vector3(),point=new THREE.Vector3(),q=new THREE.Quaternion();let changed=false;
    for(let i=0;i<this.roots.length;i++){
      root.set(0,0,0);normal.set(0,0,0);
      const rootIndex=this.roots[i].sourceIndex;
      for(let k=0;k<3;k++){const id=ids[rootIndex*3+k]*3,w=weights[rootIndex*3+k];root.x+=src[id]*w;root.y+=src[id+1]*w;root.z+=src[id+2]*w;normal.x+=sn[id]*w;normal.y+=sn[id+1]*w;normal.z+=sn[id+2]*w;}
      normal.normalize();const values=[root.x,root.y,root.z,normal.x,normal.y,normal.z];
      if(values.every((v,k)=>Math.abs(v-this.lastRoots[i*6+k])<1e-6))continue;
      this.lastRoots.set(values,i*6);const bind=this.roots[i];q.setFromUnitVectors(bind.normal,normal);
      for(let j=bind.start;j<bind.start+bind.count;j++){
        point.fromArray(this.base,j*3).sub(bind.root).applyQuaternion(q).add(root).toArray(out,j*3);
        point.fromArray(this.baseNormals,j*3).applyQuaternion(q).toArray(on,j*3);
      }changed=true;
    }
    if(changed){this.geometry.attributes.position.needsUpdate=true;this.geometry.attributes.normal.needsUpdate=true;}
  }
  bindCurveStations(source){
    this.curveBinding=null;
    const binding=this.spec.photoGuides?.curveBindings;if(!binding)return;
    const stations=this.spec.photoGuides.segments+1,expected=this.spec.rootCount*stations*3;
    const ids=binding.triangles,w=binding.weights,n=source.attributes.position.count;
    if(ids?.length!==expected||w?.length!==expected||ids.some(i=>!Number.isInteger(i)||i<0||i>=n)||w.some(v=>!Number.isFinite(v)||v<0||v>1))throw new Error('Invalid hair curve bindings.');
    for(let i=0;i<w.length;i+=3)if(Math.abs(w[i]+w[i+1]+w[i+2]-1)>1e-4)throw new Error('Invalid hair curve weights.');
    const count=this.roots.length*stations,rest=new Float32Array(count*6);
    const p=source.attributes.position.array,sn=source.attributes.normal.array;
    for(let i=0;i<count;i++){
      for(let k=0;k<3;k++){const id=ids[i*3+k]*3,weight=w[i*3+k];for(let a=0;a<3;a++){rest[i*6+a]+=p[id+a]*weight;rest[i*6+a+3]+=sn[id+a]*weight;}}
      const len=Math.hypot(...rest.subarray(i*6+3,i*6+6))||1;for(let a=3;a<6;a++)rest[i*6+a]/=len;
    }
    this.curveBinding={ids,w,stations,rest,last:new Float32Array(count*6).fill(Infinity),count};
  }
  updateCurveStations(source){
    const {ids,w,stations,rest,last,count}=this.curveBinding,p=source.attributes.position.array,sn=source.attributes.normal.array;
    const out=this.geometry.attributes.position.array,on=this.geometry.attributes.normal.array;
    const root=new THREE.Vector3(),normal=new THREE.Vector3(),point=new THREE.Vector3(),baseRoot=new THREE.Vector3(),baseNormal=new THREE.Vector3(),q=new THREE.Quaternion();let changed=false;
    for(let i=0;i<count;i++){
      root.set(0,0,0);normal.set(0,0,0);
      for(let k=0;k<3;k++){const id=ids[i*3+k]*3,weight=w[i*3+k];point.fromArray(p,id);root.addScaledVector(point,weight);point.fromArray(sn,id);normal.addScaledVector(point,weight);}
      normal.normalize();const at=i*6;
      if(Math.abs(root.x-last[at])<1e-6&&Math.abs(root.y-last[at+1])<1e-6&&Math.abs(root.z-last[at+2])<1e-6&&Math.abs(normal.x-last[at+3])<1e-6&&Math.abs(normal.y-last[at+4])<1e-6&&Math.abs(normal.z-last[at+5])<1e-6)continue;
      root.toArray(last,at);normal.toArray(last,at+3);baseRoot.fromArray(rest,at);baseNormal.fromArray(rest,at+3);q.setFromUnitVectors(baseNormal,normal);
      const group=this.roots[Math.floor(i/stations)],ring=i%stations;
      for(let child=0;child<3;child++)for(let side=0;side<3;side++){
        const vertex=(group.start+child*stations*3+ring*3+side)*3;
        point.fromArray(this.base,vertex).sub(baseRoot).applyQuaternion(q).add(root).toArray(out,vertex);
        point.fromArray(this.baseNormals,vertex).applyQuaternion(q).toArray(on,vertex);
      }changed=true;
    }
    if(changed){this.geometry.attributes.position.needsUpdate=true;this.geometry.attributes.normal.needsUpdate=true;}
  }
  dispose(){this.removeFromParent();this.geometry.dispose();this.material.dispose();}
}

// Export welding reorders vertex indices. Rebind using original rest positions,
// rather than assuming a GLB importer preserves the source vertex order.
export function remapHairRoots(spec,oldPositions,newPositions){
  if(!spec)return null;const result=structuredClone(spec),lookup=new Map();
  const key=(a,i)=>[a[i],a[i+1],a[i+2]].map(x=>Math.round(x/1e-7)).join(',');
  for(let i=0;i<newPositions.length;i+=3)lookup.set(key(newPositions,i),i/3);
  result.rootTriangles=result.rootTriangles.map(i=>{const mapped=lookup.get(key(oldPositions,i*3));if(mapped===undefined)throw new Error('Hair roots do not match the imported head.');return mapped;});
  if(result.photoGuides?.curveBindings)result.photoGuides.curveBindings.triangles=result.photoGuides.curveBindings.triangles.map(i=>{const mapped=lookup.get(key(oldPositions,i*3));if(mapped===undefined)throw new Error('Hair curve points do not match the imported head.');return mapped;});
  result.sourceVertexCount=newPositions.length/3;return result;
}
