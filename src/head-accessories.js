import * as THREE from 'three';

// A rigid accessory attached to the skull, independent of deformable skin.
// Photographs constrain front contours; thickness/depth/temples are estimates.
export class HeadGlasses extends THREE.Group {
  constructor(spec){
    super();this.name='Photo-fitted 3D glasses';this.spec=structuredClone(spec);this.userData={accessory:'eyeglasses',estimated:true};
    const paths=[...(spec?.rims??[]),spec?.bridge,...(spec?.temples??[])];
    if(spec?.rims?.length!==2||paths.some(p=>!Array.isArray(p)||p.length<2||p.length>64||p.some(v=>v.length!==3||!v.every(x=>Number.isFinite(x)&&Math.abs(x)<1))))throw new Error('Invalid reconstructed glasses paths.');
    const rgb=spec.frameColor??[.04,.04,.04];const color=new THREE.Color().setRGB(...rgb,THREE.SRGBColorSpace);
    const material=new THREE.MeshStandardMaterial({color,roughness:.27,metalness:.12});
    const lensMaterial=new THREE.MeshPhysicalMaterial({color:0xd2e1e1,roughness:.06,metalness:0,transparent:true,opacity:.055+Math.min(.6,Math.max(0,spec.lensTint??0)),depthWrite:false,side:THREE.DoubleSide,clearcoat:1,clearcoatRoughness:.08});
    const radius=THREE.MathUtils.clamp(spec.radius??.0015,.0008,.003);
    const curve=points=>new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p)),false,'centripetal');
    const tube=(points,closed,r,name)=>{const path=curve(points);path.closed=closed;const g=new THREE.TubeGeometry(path,closed?128:48,r,10,closed);const m=new THREE.Mesh(g,material);m.name=name;m.userData.accessory='eyeglasses';this.add(m);return path;};
    spec.rims.forEach((points,i)=>{
      const path=tube(points,true,radius,`Eyeglass rim ${i+1}`);const ring=path.getPoints(96).slice(0,-1);const center=ring.reduce((a,p)=>a.add(p),new THREE.Vector3()).multiplyScalar(1/ring.length);center.z+=.0008;
      const positions=[...center.toArray(),...ring.flatMap(v=>v.toArray())],indices=[];
      for(let j=0;j<ring.length;j++)indices.push(0,j+1,(j+1)%ring.length+1);
      const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setIndex(indices);g.computeVertexNormals();
      const lens=new THREE.Mesh(g,lensMaterial);lens.name=`Eyeglass lens ${i+1}`;lens.userData.accessory='eyeglasses';lens.renderOrder=2;this.add(lens);
    });
    tube(spec.bridge,false,radius*.85,'Eyeglass bridge');
    spec.temples.forEach((points,i)=>tube(points,false,Math.min(.003,(spec.templeWidth??.004)*.45),`Eyeglass temple ${i+1}`));
  }
  dispose(){this.removeFromParent();const materials=new Set();this.traverse(o=>{if(o.isMesh){o.geometry.dispose();materials.add(o.material);}});materials.forEach(m=>m.dispose());}
}
