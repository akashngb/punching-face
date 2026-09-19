import * as THREE from 'three';

const curve=points=>new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p)),false,'centripetal');
// Sweep a bevelled rectangular section. Acetate frames have flat faces and
// small edge radii; circular TubeGeometry makes them look like bent wire.
function sweep(path,closed,section,segments=96){
  const positions=[],indices=[],sides=16;
  for(let i=0;i<=segments;i++){
    const t=i/segments,p=path.getPoint(t),tangent=path.getTangent(t).normalize();
    const {axis,width,depth}=section(t,p,tangent);
    const u=axis.clone().addScaledVector(tangent,-axis.dot(tangent)).normalize(),v=new THREE.Vector3().crossVectors(tangent,u).normalize();
    for(let j=0;j<sides;j++){
      const a=j/sides*Math.PI*2,c=Math.cos(a),s=Math.sin(a);
      positions.push(...p.clone().addScaledVector(u,Math.sign(c)*Math.pow(Math.abs(c),.45)*width/2).addScaledVector(v,Math.sign(s)*Math.pow(Math.abs(s),.45)*depth/2).toArray());
    }
    if(i)for(let j=0;j<sides;j++){const a=(i-1)*sides+j,b=(i-1)*sides+(j+1)%sides,c=i*sides+j,d=i*sides+(j+1)%sides;indices.push(a,b,c,b,d,c);}
  }
  if(!closed)for(const end of [0,segments]){const c=positions.length/3;positions.push(...path.getPoint(end/segments).toArray());for(let j=0;j<sides;j++){const a=end*sides+j,b=end*sides+(j+1)%sides;indices.push(...(end?[c,a,b]:[c,b,a]));}}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setIndex(indices);g.computeVertexNormals();return g;
}
function lensGeometry(path,radius){
  const ring=path.getPoints(96).slice(0,-1),center=ring.reduce((a,p)=>a.add(p),new THREE.Vector3()).multiplyScalar(1/ring.length),positions=[],indices=[],levels=8;
  for(let k=0;k<=levels;k++)for(const p of ring){
    const offset=p.clone().sub(center),inset=Math.max(0,1-radius*.65/Math.hypot(offset.x,offset.y)),r=k/levels;
    const point=center.clone().addScaledVector(offset,r*inset);point.z+=.00065*(1-r*r);positions.push(...point.toArray());
  }
  for(let k=0;k<levels;k++)for(let j=0;j<ring.length;j++){const a=k*ring.length+j,b=k*ring.length+(j+1)%ring.length,c=a+ring.length,d=b+ring.length;indices.push(a,c,b,b,c,d);}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setIndex(indices);g.computeVertexNormals();return g;
}

// Rigid eyewear remains independent of deformable skin and serializes from spec.
export class HeadGlasses extends THREE.Group {
  constructor(spec){
    super();this.name='Photo-fitted 3D glasses';this.spec=structuredClone(spec);this.userData={accessory:'eyeglasses',estimated:true};
    const paths=[...(spec?.rims??[]),spec?.bridge,...(spec?.temples??[])];
    if(spec?.rims?.length!==2||paths.some(p=>!Array.isArray(p)||p.length<2||p.length>64||p.some(v=>v.length!==3||!v.every(x=>Number.isFinite(x)&&Math.abs(x)<1))))throw new Error('Invalid reconstructed glasses paths.');
    const color=new THREE.Color().setRGB(...(spec.frameColor??[.04,.04,.04]),THREE.SRGBColorSpace);
    const material=new THREE.MeshPhysicalMaterial({color,roughness:.3,metalness:0,clearcoat:.65,clearcoatRoughness:.2});
    const tint=THREE.MathUtils.clamp(spec.lensTint??0,0,.6);
    // Preserve the measured eyes: screen-space refraction bends the already
    // photographed lens distortion a second time and pulls in the backdrop.
    const lensMaterial=new THREE.MeshPhysicalMaterial({color:new THREE.Color().setRGB(1-tint*.25,1-tint*.22,1-tint*.2),roughness:.055,metalness:0,ior:1.5,transparent:true,opacity:.035+tint*.12,depthWrite:false,side:THREE.DoubleSide,clearcoat:1,clearcoatRoughness:.08});
    const metal=new THREE.MeshStandardMaterial({color:0x959995,roughness:.3,metalness:.8});
    const radius=THREE.MathUtils.clamp(spec.radius??.0015,.0008,.003);
    const add=(geometry,mat,name)=>{const m=new THREE.Mesh(geometry,mat);m.name=name;m.userData.accessory='eyeglasses';this.add(m);return m;};
    spec.rims.forEach((points,i)=>{
      const path=curve(points);path.closed=true;const ys=points.map(p=>p[1]),bottom=Math.min(...ys),height=Math.max(...ys)-bottom;
      add(sweep(path,true,(_,p,t)=>({axis:new THREE.Vector3(-t.y,t.x,0),width:radius*(1.25+.75*(p.y-bottom)/Math.max(.001,height)),depth:radius*1.65}),128),material,`Eyeglass rim ${i+1}`);
      const lens=add(lensGeometry(path,radius),lensMaterial,`Eyeglass lens ${i+1}`);lens.renderOrder=2;
    });
    add(sweep(curve(spec.bridge),false,()=>({axis:new THREE.Vector3(0,1,0),width:radius*1.8,depth:radius*1.7}),48),material,'Eyeglass bridge');
    spec.temples.forEach((points,i)=>{
      const path=curve(points),width=THREE.MathUtils.clamp(spec.templeWidth??.005,.002,.007);
      add(sweep(path,false,t=>({axis:new THREE.Vector3(0,1,0),width:width*(1-.52*THREE.MathUtils.smoothstep(t,.08,.95)),depth:.0022*(1-.25*t)})),material,`Eyeglass temple ${i+1}`);
      // Small hinge plates on the outside, aligned with the start of each arm.
      const t=.035,p=path.getPoint(t),tangent=path.getTangent(t).normalize(),sign=Math.sign(p.x)||1;
      const plate=add(new THREE.BoxGeometry(.0005,width*.45,.004),metal,`Eyeglass hinge ${i+1}`);plate.position.copy(p);plate.position.x+=sign*.00125;plate.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),tangent);
    });
  }
  dispose(){this.removeFromParent();const materials=new Set();this.traverse(o=>{if(o.isMesh){o.geometry.dispose();materials.add(o.material);}});materials.forEach(m=>m.dispose());}
}
