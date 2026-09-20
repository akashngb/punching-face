import * as THREE from 'three';

export const STRIKE_PHASE = {
  GUARD: 'guard',
  EXTENDING: 'extending',
  CONTACT: 'contact',
  RECOVERING: 'recovering',
};
export const impactIntensity = (speed) =>
  Math.min(100, Math.max(0, (Number(speed) || 0) * 32));

const finite3 = (value) =>
  Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const length = (v) => Math.hypot(...v);
const normalise = (v) => {
  const n = length(v) || 1;
  return v.map((x) => x / n);
};
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const distance = (a, b) => length(sub(a, b));
const clamp01 = (value) => Math.min(1, Math.max(0, Number(value) || 0));

// Attack type is the relationship between where the knuckles point and where the fist is going,
// not the travel direction alone. Direction alone cannot separate a hook from a wide jab -- both
// have a large lateral component; only the fist's orientation tells them apart. `knuckleNormal`
// (unit, world space) is supplied by the rigid pose estimator; without it this degrades to the
// original direction-only rule so legacy callers keep their old behaviour.
export function strikeMode(direction, knuckleNormal = null) {
  if (!finite3(knuckleNormal)) {
    if (direction[1] > 0.45 && direction[1] > Math.abs(direction[0]) * 0.9)
      return 'uppercut';
    if (Math.abs(direction[0]) > 0.45) return 'hook';
    return 'jab';
  }
  const v = normalise(direction),
    n = normalise(knuckleNormal);
  // Where the knuckles point is a steadier signal than instantaneous travel, which swings through
  // an arc and is noisiest exactly at contact. Travel only has to agree the fist is driving that
  // way, which is also what rejects a hand merely sweeping past the target.
  if (n[1] > 0.45 && v[1] > 0.25) return 'uppercut';
  if (n[1] < -0.5 && v[1] < -0.2) return 'overhand';
  if (Math.abs(n[0]) > 0.5 && Math.abs(v[0]) > 0.3) return 'hook';
  return 'jab';
}

// One lifecycle per tracked hand. Geometry is supplied as a callback so the same
// state machine can drive the local mesh, the CV debugger and a remote guest.
export class StrikeTracker {
  constructor({
    startSpeed = 0.45,
    minInward = 0.16,
    stopSpeed = 0.2,
    minFist = 0.25,
    maxGapMs = 180,
    rearmDistance = 0.07,
    rearmMs = 110,
    maxRecoverMs = 700,
    radius = 0.042,
    blindMs = 200,
  } = {}) {
    Object.assign(this, {
      startSpeed,
      minInward,
      stopSpeed,
      minFist,
      maxGapMs,
      rearmDistance,
      rearmMs,
      maxRecoverMs,
      radius,
      blindMs,
    });
    this.hands = new Map();
  }
  reset(hand) {
    if (hand === undefined) this.hands.clear();
    else this.hands.delete(hand);
  }
  // A first-person fist that drops out of tracking mid-extension has usually occluded itself
  // behind its own forearm — which happens precisely at full punch, so losing the hand is
  // evidence of a strike, not the absence of one. Carry the last velocity through a short
  // blind window and sweep that path once before discarding the lifecycle.
  release(hand, timestamp, collide) {
    const state = this.hands.get(hand);
    this.hands.delete(hand);
    if (
      !state ||
      state.phase !== STRIKE_PHASE.EXTENDING ||
      typeof collide !== 'function' ||
      !Number.isFinite(timestamp)
    )
      return null;
    const speed = length(state.velocity);
    if (speed < this.stopSpeed) return null;
    const horizon = Math.min(Math.max(timestamp - state.time, 40), this.blindMs) / 1000;
    const to = state.position.map(
      (value, axis) => value + state.velocity[axis] * horizon,
    );
    const hit = collide({
      from: state.position,
      to,
      radius: this.radius * 1.2,
      hand,
      velocity: state.velocity,
      axis: state.axis,
      halfLength: state.halfLength,
      knuckleNormal: state.knuckleNormal,
    });
    if (!hit) return null;
    const direction = finite3(hit.direction)
      ? normalise(hit.direction)
      : normalise(state.velocity);
    return {
      type: 'impact',
      hand,
      point: hit.point,
      direction,
      speed,
      intensity: impactIntensity(speed),
      confidence: 0.6,
      region: hit.region ?? 'face',
      mode: strikeMode(
        finite3(state.knuckleNormal) ? state.velocity : direction,
        state.knuckleNormal,
      ),
      timestamp,
      blind: true,
      normalSpeed: hit.normalSpeed,
      tangentSpeed: hit.tangentSpeed,
      obliquity: hit.obliquity,
    };
  }
  state(hand) {
    return this.hands.get(hand)?.phase ?? STRIKE_PHASE.GUARD;
  }
  update(
    {
      hand,
      position,
      target,
      timestamp,
      closed = 1,
      confidence = 1,
      velocity: measured = null,
      axis = null,
      halfLength = 0,
      knuckleNormal = null,
    },
    collide,
  ) {
    if (
      hand === undefined ||
      !finite3(position) ||
      !finite3(target) ||
      !Number.isFinite(timestamp)
    ) {
      this.reset(hand);
      return null;
    }
    let state = this.hands.get(hand);
    if (!state) {
      this.hands.set(hand, {
        position: [...position],
        velocity: [0, 0, 0],
        time: timestamp,
        phase: STRIKE_PHASE.GUARD,
        phaseSince: timestamp,
        origin: [...position],
      });
      return null;
    }
    const elapsed = timestamp - state.time;
    if (elapsed <= 0) return null;
    if (elapsed > this.maxGapMs) {
      this.reset(hand);
      this.update(
        {
          hand,
          position,
          target,
          timestamp,
          closed,
          confidence,
          velocity: measured,
          axis,
          halfLength,
          knuckleNormal,
        },
        collide,
      );
      return null;
    }

    const dt = elapsed / 1000;
    // A filtered velocity from the pose estimator is a state estimate, not a difference of two
    // noisy samples, so it neither lags nor amplifies noise. Fall back to differencing when the
    // caller has no filter (the legacy path and the sponsor detector).
    const velocity = finite3(measured)
      ? [...measured]
      : mix(
          state.velocity,
          sub(position, state.position).map((v) => v / dt),
          0.68,
        );
    const speed = length(velocity);
    const inward = dot(velocity, normalise(sub(target, position)));
    const from = state.position;
    state.position = [...position];
    state.velocity = velocity;
    state.time = timestamp;
    state.axis = axis;
    state.halfLength = halfLength;
    state.knuckleNormal = knuckleNormal;

    if (state.phase === STRIKE_PHASE.CONTACT) {
      if (timestamp - state.phaseSince >= 45) {
        state.phase = STRIKE_PHASE.RECOVERING;
        state.phaseSince = timestamp;
      }
      return null;
    }
    if (state.phase === STRIKE_PHASE.RECOVERING) {
      const recovered =
        distance(position, state.origin) >= this.rearmDistance &&
        timestamp - state.phaseSince >= this.rearmMs;
      if (recovered || timestamp - state.phaseSince >= this.maxRecoverMs) {
        state.phase = STRIKE_PHASE.GUARD;
        state.phaseSince = timestamp;
        state.origin = [...position];
      }
      return null;
    }
    if (state.phase === STRIKE_PHASE.GUARD) {
      state.origin = [...position];
      if (closed < this.minFist || speed < this.startSpeed || inward < this.minInward)
        return null;
      state.phase = STRIKE_PHASE.EXTENDING;
      state.phaseSince = timestamp;
    }

    const hit =
      typeof collide === 'function'
        ? collide({
            from,
            to: position,
            radius: this.radius,
            hand,
            velocity,
            axis,
            halfLength,
            knuckleNormal,
            fromVelocity: state.velocity,
            toVelocity: velocity,
          })
        : null;
    if (hit) {
      const direction = finite3(hit.direction)
        ? normalise(hit.direction)
        : normalise(velocity);
      state.phase = STRIKE_PHASE.CONTACT;
      state.phaseSince = timestamp;
      state.origin = [...position];
      // contactTime is the fraction along this sweep where the capsule first touched. The sweep used
      // to compute it and throw it away; kept, it dates the impact to sub-frame precision.
      const contactTime = Number.isFinite(hit.t)
        ? timestamp - elapsed * (1 - hit.t)
        : timestamp;
      return {
        type: 'impact',
        hand,
        point: hit.point,
        direction,
        speed,
        intensity: impactIntensity(speed),
        confidence: clamp01(confidence) * clamp01(closed / 0.7),
        region: hit.region ?? 'face',
        mode: strikeMode(finite3(knuckleNormal) ? velocity : direction, knuckleNormal),
        timestamp,
        contactTime,
        normalSpeed: hit.normalSpeed,
        tangentSpeed: hit.tangentSpeed,
        obliquity: hit.obliquity,
      };
    }
    if (speed < this.stopSpeed || inward < -0.05) {
      state.phase = STRIKE_PHASE.RECOVERING;
      state.phaseSince = timestamp;
      state.origin = [...position];
    }
    return null;
  }
}

function point(positions, index) {
  return positions?.length > index * 3 + 2
    ? [positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]]
    : null;
}

export function faceTargetFrom(dynamics) {
  const cage = dynamics?.cage?.positions,
    a = dynamics?.impactRig?.anchors ?? {};
  const at = (index) => point(cage, index) ?? a[index];
  const left = at(234) ?? at(50) ?? [-0.05, 0, 0.06],
    right = at(454) ?? at(280) ?? [0.05, 0, 0.06];
  const chin = at(152) ?? [0, -0.105, 0.045],
    eyeL = at(159) ?? [-0.035, 0.037, 0.065],
    eyeR = at(386) ?? [0.035, 0.037, 0.065],
    nose = at(1);
  const cx = (left[0] + right[0]) / 2,
    eyeY = (eyeL[1] + eyeR[1]) / 2;
  const bottom = chin[1] - 0.012,
    top = eyeY + Math.max(0.035, (eyeY - chin[1]) * 0.48);
  const rx = Math.max(0.055, Math.abs(right[0] - left[0]) * 0.62),
    ry = Math.max(0.075, (top - bottom) / 2);
  const front = Math.max(nose?.[2] ?? -Infinity, left[2], right[2], eyeL[2], eyeR[2]);
  return {
    center: [cx, (top + bottom) / 2, front - rx * 0.18],
    radii: [rx, ry, Math.max(0.045, rx * 0.72)],
    bottom,
    front,
  };
}

function segmentDistanceSq(pointValue, from, to) {
  const travel = to.clone().sub(from),
    denom = Math.max(travel.lengthSq(), 1e-12);
  const t = THREE.MathUtils.clamp(
    pointValue.clone().sub(from).dot(travel) / denom,
    0,
    1,
  );
  return {
    distanceSq: pointValue.distanceToSquared(from.clone().addScaledVector(travel, t)),
    t,
  };
}

// The face is a preferred region of the real render mesh, not a separate fake
// target. If no face surface is swept, the same pass falls back to head/torso.
export class HeadCollider {
  constructor(meshes = [], root = null, getDynamics = () => null) {
    this._caches = new Map();
    this.set(meshes, root, getDynamics);
    this._vertex = new THREE.Vector3();
    this._normal = new THREE.Vector3();
    this._from = new THREE.Vector3();
    this._to = new THREE.Vector3();
    this.stats = { sweeps: 0, broadRejects: 0, vertexTests: 0 };
  }
  set(meshes, root, getDynamics = () => null) {
    this.meshes = (Array.isArray(meshes) ? meshes : [meshes]).filter(Boolean);
    this.root = root;
    this.getDynamics = getDynamics;
    this._caches = new Map();
  }
  // Positions baked into root-local space, with a bounding box for the broad phase. Keyed on the
  // transform and the attribute version, so a static debug head builds this once and a head
  // deforming under FaceDynamics rebuilds each frame -- which still costs no more than the old
  // full scan did, because normals are deliberately NOT baked here. Only the winning vertex needs
  // its normal, so transforming all of them every frame was pure waste.
  _cache(mesh, relative) {
    const positions = mesh.geometry.attributes.position;
    const key = `${relative.elements.join(',')}|${positions.version}|${positions.count}`;
    const existing = this._caches.get(mesh);
    if (existing && existing.key === key) return existing;
    const count = positions.count;
    const points =
      existing && existing.points.length === count * 3
        ? existing.points
        : new Float32Array(count * 3);
    const v = this._vertex;
    const min = [Infinity, Infinity, Infinity],
      max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i++) {
      v.fromBufferAttribute(positions, i).applyMatrix4(relative);
      points[i * 3] = v.x;
      points[i * 3 + 1] = v.y;
      points[i * 3 + 2] = v.z;
      for (let k = 0; k < 3; k++) {
        const value = v.getComponent(k);
        if (value < min[k]) min[k] = value;
        if (value > max[k]) max[k] = value;
      }
    }
    const entry = { key, count, points, min, max };
    this._caches.set(mesh, entry);
    return entry;
  }
  targetWorld() {
    const target = faceTargetFrom(this.getDynamics?.());
    return (
      this.root?.localToWorld(new THREE.Vector3(...target.center)) ??
      new THREE.Vector3(...target.center)
    );
  }
  regionAt(pointValue, target = faceTargetFrom(this.getDynamics?.())) {
    const x = (pointValue.x - target.center[0]) / target.radii[0],
      y = (pointValue.y - target.center[1]) / target.radii[1];
    if (x * x + y * y <= 1.15 && pointValue.z >= target.front - target.radii[2] * 1.35)
      return 'face';
    if (pointValue.y >= target.bottom) return 'head';
    return Math.abs(pointValue.x - target.center[0]) > target.radii[0] * 0.72
      ? 'shoulder'
      : 'chest';
  }
  // Swept fist against the target.
  //
  // Three things changed from the point-sphere version. The fist is an oriented capsule along the
  // knuckle bar, so a glancing contact is distinguishable from a square one. The path is
  // sub-stepped along a Hermite curve built from the velocity at both ends, because a hook is an
  // arc and its chord can sit ~25 mm inside the real path at speed. And a bounding-box broad phase
  // runs first, so the full vertex scan happens only on frames where contact is possible instead of
  // every frame for every hand.
  //
  // Narrow phase still tests vertices rather than triangles: mean vertex spacing on this head is
  // ~8 mm against a capsule radius of tens of mm, so triangles would cost more and find the same
  // contacts. That stops being true for a coarse mesh.
  sweep({
    from,
    to,
    radius = 0.042,
    axis = null,
    halfLength = 0,
    velocity = null,
    knuckleNormal = null,
    fromVelocity = null,
    toVelocity = null,
    steps = 0,
  }) {
    if (!this.root || !this.meshes.length) return null;
    this.stats.sweeps++;
    this.root.updateWorldMatrix(true, true);
    this._from.fromArray(from);
    this._to.fromArray(to);
    this.root.worldToLocal(this._from);
    this.root.worldToLocal(this._to);
    const worldScale = this.root.getWorldScale(new THREE.Vector3());
    const uniform = Math.max((worldScale.x + worldScale.y + worldScale.z) / 3, 1e-6);
    const localRadius = radius / uniform,
      localHalf = Math.max(0, halfLength) / uniform;
    const target = faceTargetFrom(this.getDynamics?.()),
      inverseRoot = this.root.matrixWorld.clone().invert();

    // Capsule axis in local space (direction only), constant across the sweep.
    let localAxis = null;
    if (finite3(axis) && localHalf > 1e-6) {
      const a = new THREE.Vector3().fromArray(axis).transformDirection(inverseRoot);
      if (a.lengthSq() > 1e-12) localAxis = a.normalize();
    }
    const half = localAxis ? localHalf : 0;

    // Path samples. With velocities at both ends this is a cubic Hermite, which follows a hook's
    // arc instead of cutting across it; without them it degrades to the straight segment.
    const travel = this._from.distanceTo(this._to);
    const count = Math.max(
      1,
      Math.min(12, steps || Math.ceil(travel / Math.max(localRadius + half, 1e-4))),
    );
    const useHermite = finite3(fromVelocity) && finite3(toVelocity);
    // Hermite tangents are dp/ds across the segment. Scaling each end's direction by the segment
    // length weighted by its own speed reproduces a straight line at constant speed and bows
    // correctly toward the faster end when the fist is still accelerating.
    const m0 = new THREE.Vector3(),
      m1 = new THREE.Vector3();
    if (useHermite) {
      const f = new THREE.Vector3().fromArray(fromVelocity),
        t2 = new THREE.Vector3().fromArray(toVelocity);
      const fs = f.length() || 1,
        ts = t2.length() || 1,
        mean = (fs + ts) / 2 || 1;
      m0.copy(f)
        .normalize()
        .transformDirection(inverseRoot)
        .multiplyScalar((travel * fs) / mean);
      m1.copy(t2)
        .normalize()
        .transformDirection(inverseRoot)
        .multiplyScalar((travel * ts) / mean);
    }
    const path = [];
    for (let i = 0; i <= count; i++) {
      const s = i / count,
        p = new THREE.Vector3();
      if (useHermite) {
        const s2 = s * s,
          s3 = s2 * s;
        p.copy(this._from)
          .multiplyScalar(2 * s3 - 3 * s2 + 1)
          .addScaledVector(m0, s3 - 2 * s2 + s)
          .addScaledVector(this._to, -2 * s3 + 3 * s2)
          .addScaledVector(m1, s3 - s2);
      } else p.copy(this._from).lerp(this._to, s);
      path.push(p);
    }

    // Broad phase: one AABB overlap per mesh instead of a full vertex scan.
    const pad = localRadius + half;
    const lo = [Infinity, Infinity, Infinity],
      hi = [-Infinity, -Infinity, -Infinity];
    for (const p of path)
      for (let k = 0; k < 3; k++) {
        const value = p.getComponent(k);
        if (value < lo[k]) lo[k] = value;
        if (value > hi[k]) hi[k] = value;
      }
    for (let k = 0; k < 3; k++) {
      lo[k] -= pad;
      hi[k] += pad;
    }

    const limit = localRadius * localRadius;
    let face = null,
      other = null;
    for (const mesh of this.meshes) {
      if (!mesh?.geometry?.attributes?.position) continue;
      mesh.updateWorldMatrix(true, false);
      const relative = new THREE.Matrix4().multiplyMatrices(
        inverseRoot,
        mesh.matrixWorld,
      );
      const cache = this._cache(mesh, relative);
      if (
        cache.min[0] > hi[0] ||
        cache.max[0] < lo[0] ||
        cache.min[1] > hi[1] ||
        cache.max[1] < lo[1] ||
        cache.min[2] > hi[2] ||
        cache.max[2] < lo[2]
      ) {
        this.stats.broadRejects++;
        continue;
      }
      const { points, count: n } = cache;
      for (let i = 0; i < n; i++) {
        const x = points[i * 3],
          y = points[i * 3 + 1],
          z = points[i * 3 + 2];
        if (x < lo[0] || x > hi[0] || y < lo[1] || y > hi[1] || z < lo[2] || z > hi[2])
          continue;
        this.stats.vertexTests++;
        // Nearest approach of this vertex to the capsule, over the sub-stepped path.
        let bestStep = -1,
          bestDistanceSq = Infinity;
        for (let step = 0; step <= count; step++) {
          const c = path[step];
          let dx, dy, dz;
          if (half > 0) {
            // distance to the capsule's axis segment, centred on the path point
            const ax = c.x - localAxis.x * half,
              ay = c.y - localAxis.y * half,
              az = c.z - localAxis.z * half;
            const ex = localAxis.x * 2 * half,
              ey = localAxis.y * 2 * half,
              ez = localAxis.z * 2 * half;
            const denom = ex * ex + ey * ey + ez * ez;
            const u =
              denom > 1e-12
                ? Math.min(
                    1,
                    Math.max(
                      0,
                      ((x - ax) * ex + (y - ay) * ey + (z - az) * ez) / denom,
                    ),
                  )
                : 0;
            dx = x - (ax + ex * u);
            dy = y - (ay + ey * u);
            dz = z - (az + ez * u);
          } else {
            dx = x - c.x;
            dy = y - c.y;
            dz = z - c.z;
          }
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 <= limit && d2 < bestDistanceSq) {
            bestDistanceSq = d2;
            bestStep = step;
            break;
          }
        }
        if (bestStep < 0) continue;
        this._vertex.set(x, y, z);
        const t = bestStep / count;
        const region = this.regionAt(this._vertex, target),
          bucket = region === 'face' ? 'face' : 'other';
        const current = bucket === 'face' ? face : other;
        if (
          current &&
          (current.t < t || (current.t === t && current.distanceSq <= bestDistanceSq))
        )
          continue;
        const candidate = {
          point: [x, y, z],
          mesh,
          index: i,
          relative,
          direction: new THREE.Vector3()
            .subVectors(
              path[Math.min(bestStep + 1, count)],
              path[Math.max(bestStep - 1, 0)],
            )
            .normalize()
            .toArray(),
          region,
          t,
          distanceSq: bestDistanceSq,
        };
        if (bucket === 'face') face = candidate;
        else other = candidate;
      }
    }
    const hit = face ?? other;
    if (!hit) return null;
    // Resolve the surface normal for the one vertex that won, not for all of them.
    const attribute = hit.mesh.geometry.attributes.normal;
    hit.normal = attribute
      ? this._normal
          .fromBufferAttribute(attribute, hit.index)
          .applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(hit.relative))
          .normalize()
          .toArray()
      : [0, 0, 1];
    delete hit.distanceSq;
    delete hit.mesh;
    delete hit.index;
    delete hit.relative;
    // Contact quality, in world space. normalSpeed is the compression that drives deformation
    // depth; tangentSpeed is the rake that drives shear; obliquity separates a clean knuckle
    // landing from slapping with the side of the hand.
    if (finite3(velocity)) {
      const worldNormal = new THREE.Vector3()
        .fromArray(hit.normal)
        .transformDirection(this.root.matrixWorld)
        .normalize();
      const v = new THREE.Vector3().fromArray(velocity);
      const vn = v.dot(worldNormal);
      hit.normalSpeed = Math.abs(vn);
      hit.tangentSpeed = v.clone().addScaledVector(worldNormal, -vn).length();
      if (finite3(knuckleNormal)) {
        const k = new THREE.Vector3().fromArray(knuckleNormal).normalize();
        hit.obliquity =
          (Math.acos(Math.min(1, Math.max(-1, k.dot(worldNormal.clone().negate())))) *
            180) /
          Math.PI;
      }
    }
    return hit;
  }
}
