import * as THREE from 'three';

// UV seams duplicate render vertices only. Physics remains on the welded source
// surface; every render copy receives exactly the same simulated displacement.
export class SurfaceAppearance extends THREE.Mesh {
  constructor(source,atlas,texture){
    const count=source.attributes.position.count;
    if(!atlas.mapping?.length||atlas.mapping.some(i=>!Number.isInteger(i)||i<0||i>=count)||atlas.uv?.length!==atlas.mapping.length*2||!atlas.uv.every(Number.isFinite)||!atlas.indices?.length||atlas.indices.length%3||atlas.indices.some(i=>!Number.isInteger(i)||i<0||i>=atlas.mapping.length))throw new Error('Texture atlas does not match the simulated surface.');
    const geometry=new THREE.BufferGeometry(),mapping=new Uint32Array(atlas.mapping);
    geometry.setIndex(atlas.indices);geometry.setAttribute('uv',new THREE.Float32BufferAttribute(atlas.uv,2));
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(mapping.length*3),3));
    geometry.setAttribute('normal',new THREE.Float32BufferAttribute(new Float32Array(mapping.length*3),3));
    const physical=atlas.stats?.material==='physical-photo',lit=physical||atlas.stats?.material==='lit';
    texture.anisotropy=8;
    super(geometry,physical?new THREE.MeshPhysicalMaterial({map:texture,bumpMap:texture,bumpScale:.00009,color:0xbebebe,roughness:.67,specularIntensity:.22,metalness:0,side:THREE.DoubleSide}):lit?new THREE.MeshStandardMaterial({map:texture,color:0xbebebe,roughness:.78,metalness:0,side:THREE.DoubleSide}):new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide,toneMapped:false}));
    this.name='Textured editable surface';this.mapping=mapping;this.atlas=atlas;this.frustumCulled=false;
    this.updateSurface(source.attributes.position.array,source.attributes.normal.array);
    geometry.setAttribute('restNormal',geometry.attributes.normal.clone());
    // Photographs retain their original lighting. Add only the change caused
    // by deformed normals, so contact shading does not bleach the likeness.
    if(!lit)this.material.onBeforeCompile=shader=>{
      shader.vertexShader=shader.vertexShader.replace('void main() {','attribute vec3 restNormal; varying vec3 contactNormal; varying vec3 captureNormal; void main() {').replace('#include <begin_vertex>','#include <begin_vertex>\n contactNormal=normalize(normalMatrix*normal); captureNormal=normalize(normalMatrix*restNormal);');
      shader.fragmentShader=shader.fragmentShader.replace('void main() {','varying vec3 contactNormal; varying vec3 captureNormal; void main() {').replace('#include <opaque_fragment>','vec3 lightDirection=normalize(vec3(-0.4,0.65,1.)); float deformedLight=.55+.45*max(dot(normalize(contactNormal),lightDirection),0.); float capturedLight=.55+.45*max(dot(normalize(captureNormal),lightDirection),0.); outgoingLight*=clamp(deformedLight/capturedLight,.55,1.35);\n #include <opaque_fragment>');
    };
    this.material.customProgramCacheKey=()=> 'photographic-contact-shading-v1';
  }
  remap(array){const result=new Float32Array(this.mapping.length*3);for(let i=0;i<this.mapping.length;i++){const from=this.mapping[i]*3,to=i*3;result[to]=array[from];result[to+1]=array[from+1];result[to+2]=array[from+2];}return result;}
  updateSurface(position,normal){
    for(const [name,source] of [['position',position],['normal',normal]]){const attribute=this.geometry.attributes[name],out=attribute.array;
      for(let i=0;i<this.mapping.length;i++){const a=i*3,b=this.mapping[i]*3;out[a]=source[b];out[a+1]=source[b+1];out[a+2]=source[b+2];}attribute.needsUpdate=true;
    }
  }
  exportGeometry(rest,morphs){
    const g=this.geometry.clone();g.setAttribute('position',new THREE.BufferAttribute(this.remap(rest),3));
    g.morphAttributes.position=morphs.map(m=>{const a=new THREE.BufferAttribute(this.remap(m.array),3);a.name=m.name;return a;});g.morphTargetsRelative=true;g.computeVertexNormals();return g;
  }
  dispose(){this.removeFromParent();this.geometry.dispose();this.material.map?.dispose();this.material.dispose();}
}

// glTF stores separate vertices across UV seams. Restore a welded simulation
// surface while retaining the original render indices and UVs on re-import.
export function weldTexturedSurface(source){
  const input=source.attributes.position.array,positions=[],mapping=[],lookup=new Map();
  for(let i=0;i<input.length;i+=3){
    const key=[input[i],input[i+1],input[i+2]].map(x=>Math.round(x/1e-7)).join(',');
    let index=lookup.get(key);if(index===undefined){index=positions.length/3;lookup.set(key,index);positions.push(input[i],input[i+1],input[i+2]);}mapping.push(index);
  }
  const indices=source.index?Array.from(source.index.array):mapping.map((_,i)=>i),geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setIndex(indices.map(i=>mapping[i]));geometry.computeVertexNormals();
  return {geometry,atlas:{mapping,indices,uv:Array.from(source.attributes.uv.array)}};
}
