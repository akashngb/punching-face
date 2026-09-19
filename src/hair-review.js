import * as THREE from 'three';
import {SurfaceAppearance} from './surface-appearance.js';
import {HeadHair} from './head-hair.js';
import {HeadGlasses} from './head-accessories.js';
const $=id=>document.getElementById(id),base='/generated/hair-review/',version=Date.now(),asset=name=>base+name+'?v='+version;
async function start(){
  const [review,data,atlas]=await Promise.all(['review.json','mesh.json','texture-atlas.json'].map(async name=>{const r=await fetch(asset(name));if(!r.ok)throw new Error('Prepare this capture with scripts/prepare_hair_review.py first.');return r.json();}));
  const scene=new THREE.Scene();scene.background=new THREE.Color('#202923');
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;
  $('render').prepend(renderer.domElement);
  const camera=new THREE.PerspectiveCamera();camera.matrixAutoUpdate=false;
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.positions,3));geometry.setAttribute('normal',new THREE.Float32BufferAttribute(data.normals,3));geometry.setIndex(data.indices);
  const loader=new THREE.TextureLoader(),texture=await loader.loadAsync(asset('appearance.png'));texture.colorSpace=THREE.SRGBColorSpace;
  const appearance=new SurfaceAppearance(geometry,atlas,texture);scene.add(appearance);
  const clay=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0x8c9c88,roughness:.95,side:THREE.DoubleSide}));clay.visible=false;scene.add(clay);
  const hair=data.accessories?.hair?new HeadHair(geometry,data.accessories.hair):null;if(hair)scene.add(hair);
  const glasses=data.accessories?.glasses?new HeadGlasses(data.accessories.glasses):null;if(glasses)scene.add(glasses);
  scene.add(new THREE.HemisphereLight(0xf0f2e9,0x414a39,2.3));for(const [p,power] of [[[-.8,1,1],3.4],[[.8,.3,-1],2.2],[[.7,.3,1],.75]]){const light=new THREE.DirectionalLight(0xffffff,power);light.position.set(...p);scene.add(light);}
  let active;
  function render(){if(active)renderer.render(scene,camera);}
  function select(view,index){
    active=view;const {width:w,height:h,fx,fy,cx,cy}=view,n=.001;
    camera.projectionMatrix.makePerspective(-cx/fx*n,(w-cx)/fx*n,cy/fy*n,-(h-cy)/fy*n,n,10);camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.matrixWorldInverse.set(...view.viewMatrix);camera.matrixWorld.copy(camera.matrixWorldInverse).invert();camera.matrix.copy(camera.matrixWorld);
    const height=Math.min(540,$('render').clientWidth*h/w);renderer.setSize(height*w/h,height);renderer.domElement.style.height='100%';renderer.domElement.style.width='auto';
    $('photo').src=$('overlay-photo').src=asset(view.photo);
    $('photo-label').textContent=`Video ${view.time.toFixed(2)}s · recovered angle ${Math.round(view.yaw)}°`;
    [...$('angles').children].forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
    $('status').textContent='Look for agreement in the outline and hairline. Use the overlay to expose alignment differences.';render();
  }
  review.views.forEach((v,i)=>{const b=document.createElement('button');b.textContent=`${Math.round(v.yaw)}°`;b.onclick=()=>select(v,i);$('angles').append(b);});
  $('capture-info').textContent=`${review.extraction.frames} saved frames · ${review.extraction.registeredViews} recovered camera views · ${review.extraction.detailSize.join(' × ')} original video detail`;
  $('fibers').onchange=()=>{if(hair)hair.visible=$('fibers').checked&&!clay.visible;render();};
  $('overlay').onchange=()=>{$('overlay-photo').hidden=!$('overlay').checked;$('overlay-photo').style.opacity='.5';};
  $('clay').onchange=()=>{clay.visible=$('clay').checked;appearance.visible=!clay.visible;if(hair)hair.visible=$('fibers').checked&&!clay.visible;if(glasses)glasses.visible=!clay.visible;render();};
  select(review.views[0],0);
}
start().catch(e=>$('status').textContent=e.message);
