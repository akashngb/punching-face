import * as THREE from 'three';
import { SkinConstraints } from './skin-constraints.js';
import { FaceImpactRig } from './impact-rig.js';
import { impactParameters, DEFAULT_SOFTNESS } from './tissue-field.js';
import { FaceSpeechRig } from './speech-rig.js';
import { ImpactPreparation } from './impact-preparation.js';

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

// Broad phase only: a swept sphere against an expanded head ellipsoid.
export function sweptEllipsoid(from, to, center, radii, radius = 0.04) {
  const a = from.map((v, i) => (v - center[i]) / (radii[i] + radius));
  const d = to.map((v, i) => (v - from[i]) / (radii[i] + radius));
  const A = d.reduce((s, v) => s + v * v, 0),
    B = 2 * a.reduce((s, v, i) => s + v * d[i], 0),
    C = a.reduce((s, v) => s + v * v, 0) - 1;
  if (C <= 0) return 0;
  if (A < 1e-12) return null;
  const D = B * B - 4 * A * C;
  if (D < 0) return null;
  const t = (-B - Math.sqrt(D)) / (2 * A);
  return t >= 0 && t <= 1 ? t : null;
}

export function fistScore(lm) {
  const dist = (a, b) =>
    Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y, lm[a].z - lm[b].z);
  const span = Math.max(dist(0, 9), 0.001);
  // Fingertips curl back toward the wrist, relative to their PIP joints.
  return (
    [
      [8, 6],
      [12, 10],
      [16, 14],
      [20, 18],
    ].filter(
      ([tip, pip]) =>
        dist(0, tip) < dist(0, pip) * 1.22 && dist(tip, pip) < span * 0.72,
    ).length / 4
  );
}

export class FaceDynamics {
  // The optional membrane experiment has not passed recovery validation yet.
  constructor(geometry, { membrane = false, asyncImpacts = true } = {}) {
    this.geometry = geometry;
    this.original = geometry.attributes.position.array.slice();
    this.rest = this.original.slice();
    this.impactRig = new FaceImpactRig(this.rest, undefined, {
      indices: geometry.index?.array,
      normals: geometry.attributes.normal?.array,
    });
    this.speechRig = new FaceSpeechRig(this.rest);
    this.offset = new Float32Array(this.rest.length);
    this.velocity = new Float32Array(this.rest.length);
    this.posedRest = new Float32Array(this.rest);
    this.membrane = membrane ? new SkinConstraints(geometry) : null;
    this.recoil = new THREE.Vector3();
    this.recoilVelocity = new THREE.Vector3();
    this.rig = { jaw: 0, smile: 0, brow: 0, squint: 0 };
    this.softness = DEFAULT_SOFTNESS;
    this.shear = 0.35;
    this.bulge = 0.3;
    this.maxDisplacement = 0;
    this.history = [];
    this.future = [];
    const count = this.rest.length / 3,
      neighbors = Array.from({ length: count }, () => new Set());
    const indices = geometry.index?.array ?? Array.from({ length: count }, (_, i) => i);
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i], indices[i + 1], indices[i + 2]];
      neighbors[a].add(b).add(c);
      neighbors[b].add(a).add(c);
      neighbors[c].add(a).add(b);
    }
    this.neighborOffsets = new Uint32Array(count + 1);
    const packed = [];
    neighbors.forEach((list, i) => {
      this.neighborOffsets[i] = packed.length;
      packed.push(...list);
    });
    this.neighborOffsets[count] = packed.length;
    this.neighbors = new Uint32Array(packed);
    this.stiffness = new Float32Array(count);
    this.acceleration = new Float32Array(this.rest.length);
    for (let v = 0; v < count; v++) {
      const [x, y, z] = this.rest.slice(v * 3, v * 3 + 3),
        front = clamp((z + 0.01) / 0.07, 0, 1);
      const cheek =
        Math.exp(-(((Math.abs(x) - 0.044) / 0.027) ** 2) - ((y + 0.008) / 0.04) ** 2) *
        front;
      const lips = Math.exp(-((x / 0.029) ** 2) - ((y + 0.045) / 0.024) ** 2) * front;
      const nose = Math.exp(-((x / 0.02) ** 2) - ((y - 0.017) / 0.037) ** 2) * front;
      const forehead = clamp((y - 0.06) / 0.045, 0, 1);
      this.stiffness[v] = 320 - 190 * cheek - 170 * lips + 700 * nose + 450 * forehead;
    }
    this.lastImpact = null;
    this.regionPeaks = { cheeks: 0, nose: 0, lips: 0, forehead: 0, jaw: 0 };
    if (asyncImpacts) this.enableAsyncImpacts();
  }

  enableAsyncImpacts() {
    this.impactRig.enableAsync((model) => new ImpactPreparation(model));
  }
  // A landed punch reads over the top of a sentence rather than fighting it:
  // the speech rig is scaled down while an impact is at its loudest, then
  // returns as the impact decays. Cheap — the event queue is at most four.
  get speechDuck() {
    let peak = 0;
    for (const event of this.impactRig.events)
      peak = Math.max(peak, this.impactRig.envelope(event.age));
    return 1 - clamp(peak, 0, 1) * 0.85;
  }

  get headMode() {
    return this.impactRig.mode;
  }

  setHeadMode(mode) {
    this.impactRig.setMode(mode);
    this.offset.fill(0);
    this.velocity.fill(0);
  }

  applyImpact(input) {
    const p = impactParameters(input);
    return this.impulse(
      new THREE.Vector3(...p.location),
      new THREE.Vector3(...p.direction),
      p.magnitude * 1.4,
      'directional',
      { magnitude: p.magnitude },
    );
  }

  impulse(point, direction, speed, mode = 'hook', options = {}) {
    if (mode === 'body') {
      this.lastImpact = { point: point.clone(), direction: direction.clone() };
      this.applyRecoil(point, direction, speed);
      return 1;
    }
    const magnitude = options.magnitude ?? clamp(speed / 1.4, 0, 0.9);
    const affectedRig = this.impactRig.impact(
      this.rest,
      { location: point, direction, magnitude },
      this.softness,
      this.geometry.attributes.position.array,
    );
    if (!affectedRig) return 0;
    if (this.headMode === 'clay') {
      this.lastImpact = { point: point.clone(), direction: direction.clone() };
      return affectedRig;
    }
    let affected = 0;
    const radius = 0.025 + this.softness * 0.021;
    this.lastImpact = { point: point.clone(), direction: direction.clone() };
    let nearest = 0,
      best = Infinity;
    for (let i = 0; i < this.rest.length; i += 3) {
      const d =
        (this.rest[i] - point.x) ** 2 +
        (this.rest[i + 1] - point.y) ** 2 +
        (this.rest[i + 2] - point.z) ** 2;
      if (d < best) {
        best = d;
        nearest = i;
      }
    }
    const ns = this.geometry.attributes.normal.array;
    const outward = new THREE.Vector3(
      ns[nearest],
      ns[nearest + 1],
      ns[nearest + 2],
    ).normalize();
    if (outward.dot(direction) > 0) outward.negate();
    const tangent = direction.clone().addScaledVector(outward, -direction.dot(outward));
    const movement = outward
      .clone()
      .multiplyScalar(-0.85)
      .addScaledVector(tangent, this.shear * 2);
    for (let i = 0; i < this.rest.length; i += 3) {
      const d2 =
        (this.rest[i] - point.x) ** 2 +
        (this.rest[i + 1] - point.y) ** 2 +
        (this.rest[i + 2] - point.z) ** 2;
      const weight = Math.exp(-d2 / (radius * radius * 0.42));
      const ring = Math.exp(-(((Math.sqrt(d2) - radius * 0.95) / (radius * 0.3)) ** 2));
      if (weight < 0.005 && ring < 0.005) continue;
      affected++;
      const soft = (0.7 + this.softness * 0.45) * clamp(speed, 0.3, 3.5) * 0.35;
      // Compression beneath contact and outward displacement around its rim.
      // This is a visual volume-preservation approximation, not measured FEM.
      for (let j = 0; j < 3; j++)
        this.velocity[i + j] +=
          soft *
          (movement.getComponent(j) * weight +
            outward.getComponent(j) * ring * this.bulge * 0.36);
    }
    this.applyRecoil(point, direction, speed);
    return Math.max(affected, affectedRig);
  }

  applyRecoil(point, direction, speed) {
    const a = this.impactRig.anchors,
      center = new THREE.Vector3(a[13][0], a[159][1] - 0.025, a[13][2] - 0.075);
    const torque = point
      .clone()
      .sub(center)
      .cross(direction)
      .multiplyScalar(clamp(speed, 0, 4) * 16);
    this.recoilVelocity.add(torque);
  }
  rigDelta(x, y, z) {
    const front = clamp((z + 0.015) / 0.075, 0, 1);
    const jaw =
      clamp((-y + 0.005) / 0.075, 0, 1) * Math.exp(-((x / 0.09) ** 4)) * front;
    const mouth =
      Math.exp(-(((Math.abs(x) - 0.028) / 0.025) ** 2) - ((y + 0.028) / 0.022) ** 2) *
      front;
    const eyes =
      Math.exp(-(((Math.abs(x) - 0.035) / 0.028) ** 2) - ((y - 0.035) / 0.021) ** 2) *
      front;
    const brow =
      Math.exp(-(((Math.abs(x) - 0.035) / 0.035) ** 2) - ((y - 0.065) / 0.022) ** 2) *
      front;
    return [
      Math.sign(x) * mouth * this.rig.smile * 0.009,
      -jaw * this.rig.jaw * 0.022 +
        mouth * this.rig.smile * 0.009 +
        brow * this.rig.brow * 0.01 +
        eyes * this.rig.squint * 0.003,
      -jaw * this.rig.jaw * 0.006 - eyes * this.rig.squint * 0.003,
    ];
  }

  // Manual expressions are constant between edits. Keep double precision here
  // so the final Float32 position rounding matches the uncached calculation.
  poseOffsets() {
    const key = JSON.stringify(this.rig);
    if (
      this._poseRest !== this.rest ||
      this._poseKey !== key ||
      this._poseAnchors !== this.anchors
    ) {
      this._poseOffsets = new Float64Array(this.rest.length);
      if (Object.values(this.rig).some((value) => value !== 0))
        for (let i = 0; i < this.rest.length; i += 3)
          this._poseOffsets.set(
            this.rigDelta(this.rest[i], this.rest[i + 1], this.rest[i + 2]),
            i,
          );
      this._poseRest = this.rest;
      this._poseKey = key;
      this._poseAnchors = this.anchors;
    }
    return this._poseOffsets;
  }

  step(dt) {
    dt = clamp(dt, 0, 1 / 30);
    this.impactRig.step(dt);
    this.speechRig.step(dt, this.speechDuck);
    const pose = this.poseOffsets();
    for (let i = 0; i < this.rest.length; i += 3) {
      for (let j = 0; j < 3; j++)
        this.posedRest[i + j] = this.rest[i + j] + pose[i + j];
    }
    this.membrane?.refreshReference(this.posedRest);
    const n = Math.max(1, Math.ceil(dt / (1 / 120))),
      h = dt / n;
    const damping = 20 - this.softness * 5,
      coupling = this.membrane ? 0 : 750;
    for (let s = 0; s < n; s++) {
      for (let v = 0; v < this.stiffness.length; v++) {
        const i = v * 3,
          start = this.neighborOffsets[v],
          end = this.neighborOffsets[v + 1],
          degree = Math.max(end - start, 1),
          k = this.stiffness[v] * (1.2 - this.softness * 0.5);
        for (let j = 0; j < 3; j++) {
          let sum = 0;
          if (coupling)
            for (let edge = start; edge < end; edge++)
              sum += this.offset[this.neighbors[edge] * 3 + j];
          this.acceleration[i + j] =
            -k * this.offset[i + j] -
            damping * this.velocity[i + j] +
            coupling * (sum / degree - this.offset[i + j]);
        }
      }
      for (let i = 0; i < this.offset.length; i++) {
        this.velocity[i] += this.acceleration[i] * h;
        this.offset[i] += this.velocity[i] * h;
      }
      this.membrane?.solve(this.offset, this.velocity, h, this.softness);
      // Rotationally invariant displacement bound. Clip outward velocity too,
      // rather than accumulating energy against a per-axis position clamp.
      for (let i = 0; i < this.offset.length; i += 3) {
        const length = Math.hypot(
          this.offset[i],
          this.offset[i + 1],
          this.offset[i + 2],
        );
        if (length > 0.03) {
          const scale = 0.03 / length;
          let outward = 0;
          for (let j = 0; j < 3; j++) {
            this.offset[i + j] *= scale;
            outward += (this.velocity[i + j] * this.offset[i + j]) / 0.03;
          }
          if (outward > 0)
            for (let j = 0; j < 3; j++)
              this.velocity[i + j] -= (outward * this.offset[i + j]) / 0.03;
        }
      }
      this.recoilVelocity
        .addScaledVector(this.recoil, -40 * h)
        .multiplyScalar(Math.exp(-9 * h));
      this.recoil.addScaledVector(this.recoilVelocity, h);
    }
    const p = this.geometry.attributes.position.array;
    this.maxDisplacement = 0;
    for (const key in this.regionPeaks) this.regionPeaks[key] = 0;
    for (let i = 0; i < p.length; i += 3) {
      for (let j = 0; j < 3; j++)
        p[i + j] =
          this.posedRest[i + j] +
          this.offset[i + j] +
          this.impactRig.offset[i + j] +
          this.speechRig.offset[i + j];
      const amount = Math.hypot(
        this.offset[i] + this.impactRig.offset[i],
        this.offset[i + 1] + this.impactRig.offset[i + 1],
        this.offset[i + 2] + this.impactRig.offset[i + 2],
      );
      this.maxDisplacement = Math.max(this.maxDisplacement, amount);
      const x = this.rest[i],
        y = this.rest[i + 1],
        region =
          y > 0.065
            ? 'forehead'
            : y < -0.075
              ? 'jaw'
              : Math.abs(x) < 0.023
                ? y < -0.02
                  ? 'lips'
                  : 'nose'
                : 'cheeks';
      this.regionPeaks[region] = Math.max(this.regionPeaks[region], amount);
    }
    this.geometry.attributes.position.needsUpdate = true;
  }

  remember() {
    this.history.push(this.rest.slice());
    if (this.history.length > 24) this.history.shift();
    this.future = [];
  }

  sculpt(point, amount, radius = 0.034) {
    this.impactRig.restEdited();
    this._poseRest = null;
    const ns = this.geometry.attributes.normal.array;
    for (let i = 0; i < this.rest.length; i += 3) {
      const d2 =
        (this.rest[i] - point.x) ** 2 +
        (this.rest[i + 1] - point.y) ** 2 +
        (this.rest[i + 2] - point.z) ** 2;
      const weight = Math.exp(-d2 / (radius * radius * 0.4));
      for (let j = 0; j < 3; j++) this.rest[i + j] += ns[i + j] * amount * weight;
    }
  }

  undo() {
    if (!this.history.length) return false;
    this.impactRig.restEdited();
    this.future.push(this.rest.slice());
    this.rest = this.history.pop();
    this.resetMotion();
    return true;
  }

  redo() {
    if (!this.future.length) return false;
    this.impactRig.restEdited();
    this.history.push(this.rest.slice());
    this.rest = this.future.pop();
    this.resetMotion();
    return true;
  }

  resetMotion(clearPermanent = false) {
    this.impactRig.cancelPreparation();
    this._poseRest = null;
    this.offset.fill(0);
    this.velocity.fill(0);
    if (clearPermanent) this.impactRig.reset();
    else {
      this.impactRig.events = [];
      this.impactRig.offset.set(this.impactRig.permanent);
    }
    this.speechRig.reset();
    this.recoil.set(0, 0, 0);
    this.recoilVelocity.set(0, 0, 0);
  }

  reset() {
    this.remember();
    this.rest = this.original.slice();
    for (const k in this.rig) this.rig[k] = 0;
    this.resetMotion(true);
    this.step(0);
  }

  dispose() {
    this.impactRig.dispose();
  }

  // Portable editable expression controls in exported glTF. These are heuristic
  // deformation fields, not measured FACS poses or anatomically fitted muscles.
  exportMorphs() {
    const saved = { ...this.rig },
      names = Object.keys(saved),
      arrays = [];
    for (const name of names) {
      this.rig = { jaw: 0, smile: 0, brow: 0, squint: 0 };
      this.rig[name] = 1;
      const values = new Float32Array(this.rest.length);
      for (let i = 0; i < values.length; i += 3)
        values.set(this.rigDelta(...this.rest.slice(i, i + 3)), i);
      const attr = new THREE.BufferAttribute(values, 3);
      attr.name = name;
      arrays.push(attr);
    }
    this.rig = saved;
    return arrays;
  }
}
