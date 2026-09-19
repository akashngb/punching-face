import * as THREE from 'three';
import { FaceDynamics, clamp } from './physics.js';
const request = async (path, data) => {
  const r = await fetch('/physics/' + path, {
    method: 'POST',
    keepalive: path === 'close',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  const v = await r.json();
  if (!r.ok) throw new Error(v.error || 'Newton physics unavailable.');
  return v;
};

// A page reload can interrupt its close request. A tab-scoped lease lets the
// server replace that abandoned simulation without evicting other open tabs.
function physicsClient() {
  try {
    let id = sessionStorage.getItem('punching-face-newton-client');
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem('punching-face-newton-client', id);
    }
    return id;
  } catch {
    return undefined;
  }
}

// Newton supplies local tissue contact. The browser layers expressive impact
// correctives over it, as well as the independent manual expression controls.
export class NewtonFaceDynamics extends FaceDynamics {
  constructor(geometry, binding, cage, onStatus) {
    super(geometry);
    this.binding = binding;
    this.cage = cage;
    this.onStatus = onStatus;
    this.pending = [];
    this.ready = false;
    this.disposed = false;
    this.accumulated = 0;
    this.remote = new Float32Array(cage.positions.length);
    this.targetOffset = new Float32Array(this.rest.length);
    this.physicsInfo = null;
    this.lastPose = '';
    this.poseDirty = true;
    this.lastMetrics = null;
    this.lastRig = { ...this.rig };
    this.needsReset = false;
    if (
      binding.active.length !== geometry.attributes.position.count ||
      binding.indices.length !== this.rest.length ||
      binding.weights.length !== this.rest.length
    )
      throw new Error('Physics binding does not match the face mesh.');
    if (
      binding.indices.some((v) => !Number.isInteger(v) || v < 0 || v >= 468) ||
      binding.weights.some((v) => !Number.isFinite(v) || v < 0 || v > 1)
    )
      throw new Error('Invalid face cage weights.');
    this.anchors = cage.rigAnchors;
    this.impactRig.setAnchors(this.anchors);
  }

  async connect(id, generation = 'legacy') {
    try {
      this.generation = generation;
      const positions = new Float32Array(this.cage.positions);
      const indices = new Uint32Array(this.cage.indices);
      const bytes = new Uint8Array(positions.byteLength + indices.byteLength);
      bytes.set(new Uint8Array(positions.buffer));
      bytes.set(new Uint8Array(indices.buffer), positions.byteLength);
      const cageSha256 = Array.from(
        new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
        (v) => v.toString(16).padStart(2, '0'),
      ).join('');
      const result = await request('open', {
        id,
        generation,
        cageSha256,
        softness: this.softness,
        client: physicsClient(),
      });
      this.session = result.session;
      this.physicsInfo = result;
      if (this.disposed) {
        await this.dispose();
        return;
      }
      this.ready = true;
      this.onStatus?.(
        `Newton ${result.version} · CPU · ${result.tetrahedra.toLocaleString()} tetrahedra`,
      );
    } catch (e) {
      this.fail(e);
      throw e;
    }
  }

  fail(e) {
    this.ready = false;
    this.error = e.message;
    this.pending = [];
    this.onStatus?.('Newton stopped: ' + e.message);
  }

  rigDelta(x, y, z) {
    if (!this.anchors) return [0, 0, 0];
    const a = this.anchors,
      p = [x, y, z],
      fall = (index, r) =>
        Math.exp(-p.reduce((s, v, j) => s + (v - a[index][j]) ** 2, 0) / (r * r));
    const mouthL = fall('61', 0.024),
      mouthR = fall('291', 0.024),
      brow = fall('70', 0.024) + fall('300', 0.024),
      lid = fall('159', 0.013) + fall('386', 0.013);
    const chin = a['152'],
      lower = a['14'],
      upper = a['13'],
      midY = (lower[1] + upper[1]) * 0.5;
    const lowerLip = clamp(
        0.5 + (midY - y) / Math.max(Math.abs(upper[1] - lower[1]), 0.0002),
        0,
        1,
      ),
      mouth = Math.exp(
        -((x / 0.04) ** 4) - ((y - midY) / 0.018) ** 2 - ((z - lower[2]) / 0.03) ** 2,
      );
    const base = clamp(
        (lower[1] + 0.008 - y) / Math.max(0.02, lower[1] - chin[1]),
        0,
        1,
      ),
      jaw =
        (base * (1 - mouth) + lowerLip * 0.58 * mouth) *
        Math.exp(-((x / 0.11) ** 4)) *
        clamp((z + 0.06) / 0.07, 0, 1);
    const angle = this.rig.jaw * 0.42 * jaw,
      hingeY = lower[1] + 0.065,
      hingeZ = lower[2] - 0.09,
      dy = y - hingeY,
      dz = z - hingeZ;
    return [
      (mouthR - mouthL) * this.rig.smile * 0.006,
      dy * (Math.cos(angle) - 1) -
        dz * Math.sin(angle) +
        (mouthL + mouthR) * this.rig.smile * 0.006 +
        brow * this.rig.brow * 0.006 -
        lid * this.rig.squint * 0.0015,
      dy * Math.sin(angle) + dz * (Math.cos(angle) - 1) - lid * this.rig.squint * 0.001,
    ];
  }

  setHeadMode(mode) {
    super.setHeadMode(mode);
    this.pending = [];
    this.targetOffset.fill(0);
    this.remote.fill(0);
    this.version = (this.version || 0) + 1;
    this.poseDirty = true;
    this.needsReset = true;
  }

  impulse(point, direction, speed, mode = 'hook', options = {}) {
    if ((!this.ready && this.headMode !== 'clay') || this.pending.length >= 2) return 0;
    const magnitude = options.magnitude ?? clamp(speed / 1.4, 0, 0.9);
    const affected = this.impactRig.impact(
      this.rest,
      { location: point, direction, magnitude },
      this.softness,
      this.geometry.attributes.position.array,
    );
    if (!affected) return 0;
    if (this.headMode === 'live')
      this.pending.push({
        point: point.toArray(),
        direction: direction.toArray(),
        speed: clamp(speed, 0, 4),
        mode,
      });
    this.lastImpact = { point: point.clone(), direction: direction.clone() };
    this.applyRecoil(point, direction, speed);
    return affected;
  }

  pose() {
    const p = new Float32Array(1404);
    for (let i = 0; i < 1404; i += 3) {
      const d = this.rigDelta(...this.rest.slice(i, i + 3));
      for (let j = 0; j < 3; j++) p[i + j] = this.rest[i + j] + d[j];
    }
    return Array.from(p);
  }

  async advance(dt) {
    this.inflight = true;
    const reset = this.needsReset;
    this.needsReset = false;
    const version = this.version || 0;
    const payload = {
      session: this.session,
      dt: clamp(dt, 1 / 1000, 1 / 30),
      impacts: this.pending.splice(0),
      softness: this.softness,
    };
    if (this.poseDirty || reset) {
      payload.pose = this.pose();
      this.poseDirty = false;
    }
    try {
      const result = await request('step', payload);
      if (this.disposed || version !== (this.version || 0)) return;
      if (result.offsets.length !== 1404 || !result.offsets.every(Number.isFinite))
        throw new Error('Invalid Newton displacement field.');
      if (this.headMode === 'clay') return;
      this.remote.set(result.offsets);
      this.lastMetrics = { ...result, offsets: undefined };
      const b = this.binding;
      for (let v = 0; v < b.active.length; v++)
        for (let j = 0; j < 3; j++) {
          let d = 0;
          for (let k = 0; k < 3; k++)
            d += this.remote[b.indices[v * 3 + k] * 3 + j] * b.weights[v * 3 + k];
          this.targetOffset[v * 3 + j] = d * b.active[v];
        }
    } catch (e) {
      if (!this.disposed) this.fail(e);
    } finally {
      this.inflight = false;
    }
  }

  step(dt) {
    this.impactRig.step(dt);
    const rigKey = JSON.stringify(this.rig) + ':' + this.softness;
    if (rigKey !== this.lastPose) {
      this.lastPose = rigKey;
      this.version = (this.version || 0) + 1;
      this.poseDirty = true;
      this.offset.fill(0);
      this.targetOffset.fill(0);
    }
    this.accumulated += dt;
    if (
      this.headMode === 'live' &&
      this.ready &&
      !this.inflight &&
      this.accumulated >= 1 / 60
    ) {
      const elapsed = Math.min(this.accumulated, 1 / 30);
      this.accumulated = 0;
      void this.advance(elapsed);
    }
    const p = this.geometry.attributes.position.array,
      blend = 1 - Math.exp(-dt * 65);
    this.maxDisplacement = 0;
    for (const k in this.regionPeaks) this.regionPeaks[k] = 0;
    for (let i = 0; i < p.length; i += 3) {
      const r = this.rigDelta(this.rest[i], this.rest[i + 1], this.rest[i + 2]);
      for (let j = 0; j < 3; j++) {
        this.offset[i + j] += (this.targetOffset[i + j] - this.offset[i + j]) * blend;
        p[i + j] =
          this.rest[i + j] + r[j] + this.offset[i + j] + this.impactRig.offset[i + j];
      }
      const d = Math.hypot(
        this.offset[i] + this.impactRig.offset[i],
        this.offset[i + 1] + this.impactRig.offset[i + 1],
        this.offset[i + 2] + this.impactRig.offset[i + 2],
      );
      this.maxDisplacement = Math.max(this.maxDisplacement, d);
      const x = this.rest[i],
        y = this.rest[i + 1],
        region =
          y > 0.05
            ? 'forehead'
            : y < -0.075
              ? 'jaw'
              : Math.abs(x) < 0.024
                ? y < -0.02
                  ? 'lips'
                  : 'nose'
                : 'cheeks';
      this.regionPeaks[region] = Math.max(this.regionPeaks[region], d);
    }
    this.recoilVelocity
      .addScaledVector(this.recoil, -40 * dt)
      .multiplyScalar(Math.exp(-9 * dt));
    this.recoil.addScaledVector(this.recoilVelocity, dt);
    this.geometry.attributes.position.needsUpdate = true;
  }

  resetMotion(clearPermanent = false) {
    super.resetMotion(clearPermanent);
    if (!this.targetOffset) return;
    this.targetOffset.fill(0);
    this.pending = [];
    this.needsReset = true;
    this.version = (this.version || 0) + 1;
  }

  sculpt(...args) {
    super.sculpt(...args);
    this.poseDirty = true;
  }

  async dispose() {
    this.disposed = true;
    this.ready = false;
    if (this.session) {
      const session = this.session;
      this.session = null;
      try {
        await request('close', { session });
      } catch {}
    }
  }
}
