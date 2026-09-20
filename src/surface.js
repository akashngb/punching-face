import * as THREE from 'three';

export function normalizeHead(geometry,normalizedHeight=.28){
  geometry.computeBoundingBox();
  const box=geometry.boundingBox,size=new THREE.Vector3();box.getSize(size);
  if(size.y<1e-6)return geometry;
  // A generated asset may be a bust. Measure the upper silhouette so the head
  // stays target-sized while the retained shoulders extend below the view.
  const cutoff=box.max.y-size.y*.34,position=geometry.attributes.position;
  let minX=Infinity,maxX=-Infinity,minZ=Infinity,maxZ=-Infinity;
  for(let i=0;i<position.count;i++)if(position.getY(i)>=cutoff){const x=position.getX(i),z=position.getZ(i);minX=Math.min(minX,x);maxX=Math.max(maxX,x);minZ=Math.min(minZ,z);maxZ=Math.max(maxZ,z);}
  const topWidth=maxX-minX,isBust=Number.isFinite(topWidth)&&size.y>topWidth*1.8;
  const headHeight=isBust?THREE.MathUtils.clamp(topWidth*1.35,size.y*.25,size.y*.55):size.y;
  const center=isBust?new THREE.Vector3((minX+maxX)/2,box.max.y-headHeight/2,(minZ+maxZ)/2):box.getCenter(new THREE.Vector3());
  const scale=normalizedHeight/headHeight;
  geometry.translate(-center.x,-center.y,-center.z);geometry.scale(scale,scale,scale);geometry.computeVertexNormals();geometry.computeBoundingBox();
  return geometry;
}

// Add real surface degrees of freedom while preserving the captured texture and
// boundary. Subdivision adds simulation resolution, not new scan evidence.
export function refineSurface(input,passes=2){
  let g=input;
  for(let pass=0;pass<passes;pass++){
    const attrs=Object.fromEntries(Object.entries(g.attributes).filter(([k])=>k!=='normal').map(([k,a])=>[k,{size:a.itemSize,values:Array.from(a.array)}]));
    const source=g.index?Array.from(g.index.array):Array.from({length:g.attributes.position.count},(_,i)=>i),edges=new Map(),indices=[];
    const mid=(a,b)=>{
      const key=a<b?`${a}:${b}`:`${b}:${a}`;
      if(edges.has(key))return edges.get(key);
      const id=attrs.position.values.length/3;
      for(const attr of Object.values(attrs))for(let k=0;k<attr.size;k++)attr.values.push((attr.values[a*attr.size+k]+attr.values[b*attr.size+k])*.5);
      edges.set(key,id);return id;
    };
    for(let i=0;i<source.length;i+=3){const [a,b,c]=source.slice(i,i+3),ab=mid(a,b),bc=mid(b,c),ca=mid(c,a);indices.push(a,ab,ca,ab,b,bc,ca,bc,c,ab,bc,ca);}
    const next=new THREE.BufferGeometry();for(const [name,attr] of Object.entries(attrs))next.setAttribute(name,new THREE.Float32BufferAttribute(attr.values,attr.size));next.setIndex(indices);next.computeVertexNormals();
    if(g!==input)g.dispose();g=next;
  }
  return g;
}
