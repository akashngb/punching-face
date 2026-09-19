import * as THREE from 'three';
import { impactParameters, DEFAULT_IMPACT_MAGNITUDE } from './tissue-field.js';

// UI and CV boundary are separate from the capture / sponsor entry points.
export function installImpactControls({
  getDynamics,
  getMesh,
  headPivot,
  contact,
  setView,
  release,
  toast,
  rigMarkers,
}) {
  const $ = (id) => document.getElementById(id);
  const section = $('softness').closest('section');
  const panel = document.createElement('div');
  panel.className = 'impact-controls';
  panel.innerHTML = /* HTML */ `<label class="controls-label" for="head-mode"
      >Head response</label
    >
    <select id="head-mode">
      <option value="live">Live head · elastic</option>
      <option value="clay">Clay head · permanent dents</option>
    </select>
    <p id="head-mode-note" class="muted">
      Recovers after impact. Above 0.90, bone regions retain damage until Reset head.
    </p>
    <label class="controls-label" for="impact-strength"
      >Impact magnitude
      <output id="impact-strength-value"
        >${DEFAULT_IMPACT_MAGNITUDE.toFixed(2)}</output
      ></label
    >
    <input
      id="impact-strength"
      aria-label="Impact magnitude"
      type="range"
      min="0"
      max="1"
      step=".01"
      value="${DEFAULT_IMPACT_MAGNITUDE}"
    />
    <details id="impact-lab">
      <summary>Impact lab · location & direction</summary>
      <label class="controls-label" for="impact-region">Target region</label>
      <select id="impact-region">
        <option value="left-cheek">Left cheek</option>
        <option value="right-cheek">Right cheek</option>
        <option value="mouth">Mouth</option>
        <option value="nose">Nose bridge</option>
        <option value="chin">Chin / uppercut</option>
        <option value="forehead">Forehead</option>
        <option value="left-temple">Left temple</option>
        <option value="right-temple">Right temple</option>
        <option value="crown">Crown</option>
        <option value="back">Back of head</option>
      </select>
      <label class="controls-label" for="impact-angle"
        >Strike angle <output id="impact-angle-value">0°</output></label
      >
      <input
        id="impact-angle"
        aria-label="Strike angle"
        type="range"
        min="-75"
        max="75"
        step="5"
        value="0"
      />
      <label class="controls-label" for="impact-elevation"
        >Strike elevation <output id="impact-elevation-value">0°</output></label
      >
      <input
        id="impact-elevation"
        aria-label="Strike elevation"
        type="range"
        min="-75"
        max="75"
        step="5"
        value="0"
      />
      <button id="test-impact" class="small primary full">Apply test impact</button>
      <label class="check"
        >Click head to punch <input id="click-impact" type="checkbox"
      /></label>
      <p class="muted">
        Drag to orbit, then click any surface. Angles tilt the incoming vector from the
        surface normal.
      </p>
      <label class="check"
        >Tissue attachment map <input id="tissue-map" type="checkbox"
      /></label>
      <p class="muted">
        Blue: firm support · coral: mobile tissue. Estimated from facial landmarks.
      </p>
    </details>
    <p id="impact-state" class="muted" role="status">Ready for impact</p>`;
  section.querySelector('h2').after(panel);
  $('reset').textContent = 'Reset head';
  let overlay = null,
    overlaySource = null,
    markerBinding = [];
  function setMode(mode) {
    getDynamics()?.setHeadMode(mode);
    $('head-mode').value = mode;
    $('head-mode-note').textContent =
      mode === 'clay'
        ? 'Dents accumulate and stay until Reset head.'
        : 'Recovers after impact. Above 0.90, bone regions retain damage until Reset head.';
    setView(mode === 'clay' ? 'clay' : 'mesh');
    release();
  }
  $('head-mode').onchange = () => setMode($('head-mode').value);
  for (const [id, suffix] of [
    ['impact-strength', ''],
    ['impact-angle', '°'],
    ['impact-elevation', '°'],
  ])
    $(id).oninput = () =>
      ($(id + '-value').textContent =
        (id === 'impact-strength' ? Number($(id).value).toFixed(2) : $(id).value) +
        suffix);
  function applyImpact(input) {
    const p = impactParameters(input),
      d = getDynamics();
    if (!d) return false;
    if (input.space !== undefined && !['head', 'world'].includes(input.space))
      throw new RangeError('Impact space must be head or world.');
    const point = new THREE.Vector3(...p.location),
      direction = new THREE.Vector3(...p.direction);
    if (input.space === 'world') {
      headPivot.updateWorldMatrix(true, false);
      headPivot.worldToLocal(point);
      direction.transformDirection(headPivot.matrixWorld.clone().invert());
    }
    const landed = contact(
      point,
      direction,
      p.magnitude * 1.4,
      input.source ?? 'cv',
      'directional',
      { magnitude: p.magnitude },
    );
    if (landed)
      $('impact-state').textContent =
        `${d.headMode === 'clay' ? 'Clay impression' : p.magnitude > 0.9 ? 'Live impact · damage enabled' : 'Live impact'} · magnitude ${p.magnitude.toFixed(2)}`;
    return landed;
  }
  function angledDirection(normal) {
    const n = normal.clone().normalize(),
      u = new THREE.Vector3(1, 0, 0).addScaledVector(n, -n.x);
    if (u.lengthSq() < 0.01) u.set(0, 0, 1).addScaledVector(n, -n.z);
    u.normalize();
    const v = new THREE.Vector3().crossVectors(u, n).normalize();
    return n
      .negate()
      .addScaledVector(u, Math.tan((Number($('impact-angle').value) * Math.PI) / 180))
      .addScaledVector(
        v,
        Math.tan((Number($('impact-elevation').value) * Math.PI) / 180),
      )
      .normalize();
  }
  function impactAt(point, normal) {
    return applyImpact({
      location: point,
      direction: angledDirection(normal),
      magnitude: Number($('impact-strength').value),
      source: 'lab',
    });
  }
  $('test-impact').onclick = () => {
    const d = getDynamics();
    if (!d) return;
    const a = d.impactRig.anchors,
      t = d.impactRig.tissue,
      m = a[13].map((v, j) => (v + a[14][j]) * 0.5),
      s = t.anatomy(m, a).scale;
    const targets = {
      'left-cheek': a[50],
      'right-cheek': a[280],
      mouth: m,
      nose: a[1] ?? [m[0], m[1] + 0.047 * s, m[2] + 0.014 * s],
      chin: a[152],
      forehead: [m[0], m[1] + 0.135 * s, m[2] - 0.012 * s],
      'left-temple': [a[50][0] * 1.4, a[159][1] + 0.016 * s, a[50][2] - 0.04 * s],
      'right-temple': [a[280][0] * 1.4, a[386][1] + 0.016 * s, a[280][2] - 0.04 * s],
      crown: [m[0], m[1] + 0.24 * s, m[2] - 0.075 * s],
      back: [m[0], m[1] + 0.09 * s, m[2] - 0.23 * s],
    };
    const hit = t.nearest(targets[$('impact-region').value]),
      node = t.vertices[hit.node],
      positions = d.geometry.attributes.position;
    const point = new THREE.Vector3().fromBufferAttribute(positions, hit.index);
    let normal = new THREE.Vector3(...node.n);
    if ($('impact-region').value === 'chin') normal.set(0, -0.9, 0.4).normalize();
    if (!impactAt(point, normal))
      toast('Impact not accepted. Check that the model and physics are ready.');
  };
  function onModel() {
    const d = getDynamics(),
      mesh = getMesh();
    if (!d || !mesh) return;
    d.setHeadMode($('head-mode').value);
    if (overlay) {
      overlay.removeFromParent();
      overlay.geometry.dispose();
      overlay.material.dispose();
    }
    const g = new THREE.BufferGeometry();
    g.setIndex(mesh.geometry.index);
    g.setAttribute('position', mesh.geometry.attributes.position);
    g.setAttribute('normal', mesh.geometry.attributes.normal);
    const colors = new Float32Array(d.rest.length),
      blue = new THREE.Color('#315dc2'),
      coral = new THREE.Color('#ed795b'),
      color = new THREE.Color();
    const t = d.impactRig.tissue;
    for (let v = 0; v < t.map.length; v++) {
      const anatomy = t.anatomy(t.vertices[t.map[v]].p, d.impactRig.anchors);
      color.copy(blue).lerp(coral, anatomy.compliance);
      colors.set(color.toArray(), v * 3);
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    overlay = new THREE.Mesh(
      g,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        side: THREE.DoubleSide,
      }),
    );
    overlay.visible = $('tissue-map').checked;
    overlay.renderOrder = 3;
    headPivot.add(overlay);
    overlaySource = d;
    const ids = [159, 386, 159, 386, 61, 291, 152];
    markerBinding = rigMarkers.children.map(
      (marker, i) => t.nearest(d.impactRig.anchors[ids[i]] ?? [0, 0, 0]).index,
    );
    if (d.headMode === 'clay') setView('clay');
  }
  $('tissue-map').onchange = () => {
    if (overlay) overlay.visible = $('tissue-map').checked;
  };
  function update() {
    const d = getDynamics();
    if (!d) return;
    if (overlaySource !== d) onModel();
    if (rigMarkers.visible)
      rigMarkers.children.forEach((marker, i) => {
        marker.position.fromBufferAttribute(
          d.geometry.attributes.position,
          markerBinding[i],
        );
      });
  }
  return {
    applyImpact,
    setMode,
    onModel,
    update,
    impactAt,
    get clickEnabled() {
      return $('click-impact').checked;
    },
  };
}
