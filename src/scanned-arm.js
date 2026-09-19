import * as THREE from 'three';
import {ARM_PAIRS as PAIRS,capturedArmProfile,armBoneRotations} from './arm-pose.js';

// Joints: shoulder, elbow, wrist, then hand landmarks 0..20 (wrist repeated).
// Mesh geometry and appearance MUST come from a reconstructed capture bundle.
export class ScannedArm extends THREE.Group {
  constructor(bundle){
    super();
    if(bundle.format!=='punching-face-arm'||!bundle.mesh?.positions||bundle.joints?.length!==24)throw new Error('Choose a Punching Face arm reconstruction bundle with 24 measured joint anchors.');
    this.profile=capturedArmProfile(bundle);this.side=bundle.side;this.rest=this.profile.rest;
    const positions=bundle.mesh.positions,indices=bundle.mesh.indices,colorsIn=bundle.mesh.colors;
    if(!Array.isArray(positions)||positions.length%3||positions.length<9||positions.length>900000||!positions.every(Number.isFinite)||!Array.isArray(indices)||indices.length%3||!indices.every(i=>Number.isInteger(i)&&i>=0&&i<positions.length/3)||colorsIn?.length!==positions.length||!colorsIn.every(Number.isFinite))throw new Error('Arm bundle has invalid surface geometry or color data.');
    const data=bundle.mesh,g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));g.setIndex(data.indices);
    const colors=new Float32Array(data.colors),c=new THREE.Color();for(let i=0;i<colors.length;i+=3){c.setRGB(colors[i],colors[i+1],colors[i+2],THREE.SRGBColorSpace);colors.set([c.r,c.g,c.b],i);}g.setAttribute('color',new THREE.BufferAttribute(colors,3));g.computeVertexNormals();
    const boneIndices=[],weights=[];
    for(let i=0;i<data.positions.length;i+=3){
      const p=new THREE.Vector3(...data.positions.slice(i,i+3));const nearest=[];
      for(let b=0;b<this.rest.length;b++){
        const next=this.rest[PAIRS[b][1]],d=next.clone().sub(this.rest[b]),t=THREE.MathUtils.clamp(p.clone().sub(this.rest[b]).dot(d)/Math.max(d.lengthSq(),1e-8),0,1);
        const dist=p.distanceToSquared(this.rest[b].clone().addScaledVector(d,t));
        nearest.push({b,w:1/Math.max(dist,.00005)**2});
      }
      nearest.sort((a,b)=>b.w-a.w);const top=nearest.slice(0,4),sum=top.reduce((s,x)=>s+x.w,0);for(const e of top){boneIndices.push(e.b);weights.push(e.w/sum);}
    }
    g.setAttribute('skinIndex',new THREE.Uint16BufferAttribute(boneIndices,4));g.setAttribute('skinWeight',new THREE.Float32BufferAttribute(weights,4));
    this.mesh=new THREE.SkinnedMesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.82,side:THREE.DoubleSide}));
    this.mesh.frustumCulled=false;
    // Independent world-space bones make landmark-driven affine updates explicit.
    this.bones=this.rest.map(p=>{const b=new THREE.Bone();b.position.copy(p);this.mesh.add(b);return b;});
    this.add(this.mesh);this.mesh.bind(new THREE.Skeleton(this.bones));this.visible=false;this.userData.evidence=bundle.evidence;
  }
  updateFromHand(hand){
    const pose=hand.armPose;this.visible=!!pose&&hand.visible&&hand.tracked;
    if(!this.visible)return;
    const rotations=armBoneRotations(this.profile,pose);
    this.bones.forEach((bone,i)=>{bone.position.copy(pose.joints[i]);bone.quaternion.copy(rotations[i]);bone.updateMatrix();});
  }
  dispose(){this.removeFromParent();this.mesh.geometry.dispose();this.mesh.material.dispose();this.mesh.skeleton.dispose();}
}
