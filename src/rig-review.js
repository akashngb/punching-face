import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadHeadBundle } from './head-bundle.js';
import { SurfaceAppearance } from './surface-appearance.js';
import { FaceImpactRig } from './impact-rig.js';

// Development review surface: uses the production impact path on both sides,
// bypassing the pain reaction on the left. No camera or remote physics.
const $ = (id) => document.getElementById(id);
const views = [];
let playback = 0;
try {
  const captures = await fetch('/api/face-captures').then((r) => r.json());
  const id = captures.captures.find((c) => c.photoModel && !c.testFixture)?.id;
  if (!id)
    throw new Error(
      'A reconstructed photo head is required. Open CONTACT to load one.',
    );
  const { data, atlas, cage, textureBytes } = await loadHeadBundle(id);
  const url = URL.createObjectURL(new Blob([textureBytes], { type: 'image/png' }));
  let texture;
  try {
    texture = await new THREE.TextureLoader().loadAsync(url);
  } finally {
    URL.revokeObjectURL(url);
  }
  texture.colorSpace = THREE.SRGBColorSpace;
  for (const name of ['before', 'after']) {
    const container = $(name),
      renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121820);
    scene.add(new THREE.HemisphereLight(0xdbefff, 0x62616b, 2.4));
    const light = new THREE.DirectionalLight(0xfff1e5, 2.4);
    light.position.set(-1, 2, 3);
    scene.add(light);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(data.positions, 3),
    );
    geometry.setIndex(data.indices);
    geometry.computeVertexNormals();
    const rest = geometry.attributes.position.array.slice();
    const rig = new FaceImpactRig(rest, cage.rigAnchors, {
      indices: geometry.index.array,
      normals: geometry.attributes.normal.array,
    });
    if (name === 'before') rig.reactionEnabled = false;
    const skin = new SurfaceAppearance(geometry, atlas, texture);
    const clay = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        color: 0xb3b9c1,
        roughness: 0.62,
        side: THREE.DoubleSide,
      }),
    );
    const wire = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color: 0x559aca,
        wireframe: true,
        transparent: true,
        opacity: 0.18,
        depthWrite: false,
      }),
    );
    clay.visible = false;
    wire.visible = false;
    scene.add(skin, clay, wire);
    geometry.computeBoundingBox();
    const center = geometry.boundingBox.getCenter(new THREE.Vector3());
    const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 10);
    camera.position.set(center.x, center.y + 0.008, center.z + 0.68);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.update();
    const view = {
      container,
      renderer,
      scene,
      camera,
      controls,
      geometry,
      rest,
      rig,
      skin,
      clay,
      wire,
    };
    views.push(view);
    new ResizeObserver(() => {
      renderer.setSize(container.clientWidth, container.clientHeight);
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      render();
    }).observe(container);
    controls.addEventListener('change', () => {
      for (const other of views)
        if (other !== view) {
          other.camera.position.copy(camera.position);
          other.camera.quaternion.copy(camera.quaternion);
          other.controls.target.copy(controls.target);
        }
      render();
    });
  }
  function render() {
    for (const v of views) v.renderer.render(v.scene, v.camera);
  }
  function pose(age) {
    for (const v of views) {
      v.rig.events = [v.event];
      v.event.age = age;
      v.rig.permanent.fill(0);
      v.event.committed = 0;
      v.rig.step(0);
      const p = v.geometry.attributes.position.array;
      for (let i = 0; i < p.length; i++) p[i] = v.rest[i] + v.rig.offset[i];
      v.geometry.attributes.position.needsUpdate = true;
      v.geometry.computeVertexNormals();
      v.skin.updateSurface(p, v.geometry.attributes.normal.array);
    }
    render();
    $('time').value = `${age.toFixed(2)} s`;
  }
  function prepare() {
    cancelAnimationFrame(playback);
    const magnitude = Number($('magnitude').value),
      strike = $('strike').value;
    $('strength').value = magnitude.toFixed(2);
    for (const v of views) {
      v.rig.reset();
      const anchor =
        cage.rigAnchors[strike === 'chin' ? 152 : strike === 'right' ? 280 : 50];
      const hit = v.rig.tissue.nearest(anchor);
      const direction =
        strike === 'chin'
          ? [0, 0.85, -0.5]
          : strike === 'front'
            ? [0, 0, -1]
            : [strike === 'left' ? 0.75 : -0.75, -0.05, -0.6];
      v.rig.impact(
        v.rest,
        { location: v.rig.tissue.vertices[hit.node].p, direction, magnitude },
        0.75,
      );
      v.event = v.rig.events[0];
    }
    $('pose').value = '0.35';
    pose(0.35);
    const quality = views[1].rig.tissue.measure(
      views[1].geometry.attributes.position.array,
    );
    $('metrics').textContent =
      `The right-hand head squeezes its eyelids, lowers its brows and grimaces after contact. The expression lingers as the dent recovers.\n${quality.reversedTriangles} reversed triangles at the reaction pose · same impact on both sides · authored expression, not a pain measurement.`;
  }
  $('strike').onchange = prepare;
  $('magnitude').onchange = prepare;
  $('pose').oninput = () => {
    cancelAnimationFrame(playback);
    pose(Number($('pose').value));
  };
  $('wire').onchange = () => {
    for (const v of views) v.wire.visible = $('wire').checked;
    render();
  };
  $('texture').onchange = () => {
    for (const v of views) {
      v.skin.visible = $('texture').checked;
      v.clay.visible = !$('texture').checked;
    }
    render();
  };
  $('replay').disabled = false;
  $('replay').onclick = () => {
    cancelAnimationFrame(playback);
    const start = performance.now();
    const tick = () => {
      const age = Math.min(1.8, (performance.now() - start) / 1600);
      $('pose').value = age;
      pose(age);
      if (age < 1.8) playback = requestAnimationFrame(tick);
    };
    tick();
  };
  prepare();
} catch (error) {
  $('metrics').textContent = error.message;
}
