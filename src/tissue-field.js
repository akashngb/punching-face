import {SurfaceValidity} from './surface-validity.js';
// Surface-connected, seam-welded tissue field. Parameters are animation controls,
// not material measurements. See docs/IMPACT_RIG.md for sources and limits.
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const smooth = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const vector = (v) =>
  Array.isArray(v) || ArrayBuffer.isView(v) ? Array.from(v) : [v?.x, v?.y, v?.z];
export const DEFAULT_IMPACT_MAGNITUDE = 0.85;
export const DEFAULT_SOFTNESS = 0.75;
export const MAX_PERMANENT_DISPLACEMENT = 0.065;
export function impactParameters(input = {}) {
  const location = vector(input.location ?? input.point),
    direction = vector(input.direction),
    magnitude = input.magnitude;
  if (
    location.length !== 3 ||
    direction.length !== 3 ||
    ![...location, ...direction].every(Number.isFinite) ||
    !Number.isFinite(magnitude) ||
    magnitude < 0 ||
    magnitude > 1
  )
    throw new RangeError(
      'Impact requires finite location[3], direction[3], and magnitude from 0 to 1.',
    );
  const length = Math.hypot(...direction);
  if (length < 1e-8) throw new RangeError('Impact direction must be nonzero.');
  return { location, direction: direction.map((v) => v / length), magnitude };
}
class Heap {
  data = [];
  push(item) {
    let i = this.data.length;
    this.data.push(item);
    while (i) {
      const p = (i - 1) >> 1;
      if (this.data[p][0] <= item[0]) break;
      this.data[i] = this.data[p];
      i = p;
    }
    this.data[i] = item;
  }
  pop() {
    const first = this.data[0],
      last = this.data.pop();
    if (this.data.length) {
      let i = 0;
      while (i * 2 + 1 < this.data.length) {
        let c = i * 2 + 1;
        if (c + 1 < this.data.length && this.data[c + 1][0] < this.data[c][0]) c++;
        if (this.data[c][0] >= last[0]) break;
        this.data[i] = this.data[c];
        i = c;
      }
      this.data[i] = last;
    }
    return first;
  }
}
export class TissueField {
  constructor(rest, indices, normals) {
    this.rest = rest;
    this.map = new Uint32Array(rest.length / 3);
    this.vertices = [];
    const weld = new Map();
    for (let v = 0; v < this.map.length; v++) {
      const p = Array.from(rest.slice(v * 3, v * 3 + 3)),
        key = p.map((x) => Math.round(x * 1e6)).join(',');
      let node = weld.get(key);
      if (node === undefined) {
        node = this.vertices.length;
        weld.set(key, node);
        this.vertices.push({ p, n: [0, 0, 0], links: new Set(), copies: [] });
      }
      this.map[v] = node;
      const q = this.vertices[node];
      q.copies.push(v);
      for (let j = 0; j < 3; j++) q.n[j] += normals?.[v * 3 + j] ?? 0;
    }
    const ids = indices ?? Array.from(this.map, (_, i) => i);
    for (let i = 0; i < ids.length; i += 3)
      for (const [a, b] of [
        [ids[i], ids[i + 1]],
        [ids[i + 1], ids[i + 2]],
        [ids[i + 2], ids[i]],
      ]) {
        const u = this.map[a],
          v = this.map[b];
        if (u !== v) {
          this.vertices[u].links.add(v);
          this.vertices[v].links.add(u);
        }
      }
    this.indices = Array.from(ids);
    this.edges = [];
    this.vertices.forEach((v, i) => {
      v.links = Array.from(v.links).map((j) => [
        j,
        Math.hypot(...v.p.map((x, k) => x - this.vertices[j].p[k])),
      ]);
      for (const [j, l] of v.links) if (i < j && l > 1e-7) this.edges.push([i, j, l]);
      const length = Math.hypot(...v.n);
      v.n = length > 1e-8 ? v.n.map((x) => x / length) : [0, 0, 1];
    });
  }
  get validity(){return this._validity??=new SurfaceValidity(this);}
  nearest(location, positions = this.rest) {
    let index = 0,
      best = Infinity;
    for (let v = 0; v < this.map.length; v++) {
      // Photo models retain 468 unrendered cage points before the skin vertices.
      // Only a node connected to triangles can receive a visible skin impact.
      if (!this.vertices[this.map[v]].links.length) continue;
      let d = 0;
      for (let j = 0; j < 3; j++) d += (positions[v * 3 + j] - location[j]) ** 2;
      if (d < best) {
        best = d;
        index = v;
      }
    }
    return { index, node: this.map[index], distance: Math.sqrt(best) };
  }
  distances(seed, radius) {
    const d = new Float32Array(this.vertices.length).fill(Infinity),
      heap = new Heap();
    d[seed] = 0;
    heap.push([0, seed]);
    while (heap.data.length) {
      const [distance, i] = heap.pop();
      if (distance > d[i] + 1e-7) continue;
      for (const [j, length] of this.vertices[i].links) {
        const next = distance + length;
        if (next < d[j] && next < radius) {
          d[j] = next;
          heap.push([d[j], j]);
        }
      }
    }
    return d;
  }
  anatomy(p, a) {
    const mouth = a[13].map((v, j) => (v + a[14][j]) * 0.5),
      s = clamp((mouth[1] - a[152][1]) / 0.06, 0.6, 1.6);
    const g = (c, rx, ry, rz) =>
      Math.exp(
        -p.reduce((sum, x, j) => sum + ((x - c[j]) / ([rx, ry, rz][j] * s)) ** 2, 0),
      );
    const cheek = Math.max(g(a[50], 0.041, 0.037, 0.04), g(a[280], 0.041, 0.037, 0.04));
    const lips = g(mouth, 0.036, 0.016, 0.022),
      nose = a[1] ?? [mouth[0], mouth[1] + 0.043 * s, mouth[2] + 0.014 * s];
    const bridge = g(
      [nose[0], nose[1] + 0.015 * s, nose[2] - 0.008 * s],
      0.018,
      0.031,
      0.025,
    );
    const zygoma = Math.max(
      ...[50, 280].map((id) =>
        g([a[id][0], a[id][1] + 0.023 * s, a[id][2] - 0.004 * s], 0.03, 0.02, 0.03),
      ),
    );
    const mandible = g(a[152], 0.065, 0.027, 0.06);
    const temples = Math.max(
      ...[50, 280].map((id) =>
        g(
          [
            a[id][0] * 1.4,
            a[id === 50 ? 159 : 386][1] + 0.016 * s,
            a[id][2] - 0.04 * s,
          ],
          0.03,
          0.04,
          0.045,
        ),
      ),
    );
    const upper = smooth(mouth[1] + 0.085 * s, mouth[1] + 0.12 * s, p[1]);
    const back = 1 - smooth(mouth[2] - 0.075 * s, mouth[2] - 0.035 * s, p[2]);
    const bone = clamp(
      Math.max(bridge, zygoma, mandible, temples, upper, back) * (1 - lips * 0.98),
      0,
      1,
    );
    return {
      scale: s,
      cheek,
      lips,
      bone,
      compliance: clamp(0.35 + 0.55 * cheek + 0.5 * lips - 0.22 * bone, 0.18, 1),
    };
  }
  build(seed, direction, magnitude, softness, anchors) {
    const source = this.vertices[seed],
      material = this.anatomy(source.p, anchors),
      s = material.scale;
    const normal = source.n.slice();
    if (normal.reduce((v, x, j) => v + x * direction[j], 0) > 0)
      normal.forEach((x, j) => (normal[j] = -x));
    const incidence = clamp(-normal.reduce((v, x, j) => v + x * direction[j], 0), 0, 1);
    const tangent = direction.map((x, j) => x + normal[j] * incidence);
    const radius = (0.026 + 0.019 * magnitude) * s,
      distance = this.distances(seed, radius * 3.3);
    const values = new Float32Array(this.vertices.length * 3),
      damage = new Float32Array(values.length);
    const amplitude = 1.65 * Math.pow(magnitude, 1.12) * (0.72 + 0.46 * softness) * s;
    let affected = 0;
    this.vertices.forEach((v, i) => {
      const d = distance[i];
      if (!Number.isFinite(d)) return;
      const local = this.anatomy(v.p, anchors);
      const core = Math.exp(-((d / (radius * 0.7)) ** 2)),
        broad =
          Math.exp(-((d / (radius * 1.6)) ** 2)) *
          (1 - smooth(radius * 2.3, radius * 3.3, d));
      const rim =
        Math.exp(-(((d - radius * 0.98) / (radius * 0.36)) ** 2)) * (1 - core);
      const mobility = 0.5 + 0.5 * local.compliance;
      // Skin transport spreads farther than indentation; the rim displaces tissue
      // outwards instead of shrinking the complete cheek into a spherical crater.
      const depth = (0.008 + 0.017 * local.compliance) * incidence * core;
      const shear = (0.023 * broad + 0.008 * core) * mobility;
      const bulge = 0.006 * incidence * rim * local.compliance;
      for (let j = 0; j < 3; j++)
        values[i * 3 + j] =
          amplitude * (-normal[j] * depth + tangent[j] * shear + v.n[j] * bulge);
      if (magnitude > 0.9) {
        const severity = (magnitude - 0.9) / 0.1,
          patch = Math.exp(-((d / (radius * 0.88)) ** 2));
        const amount = 0.016 * s * severity * patch * local.bone * material.bone;
        for (let j = 0; j < 3; j++)
          damage[i * 3 + j] = amount * (-normal[j] * incidence + tangent[j] * 0.45);
      }
      if (broad > 0.02) affected += v.copies.length;
    });
    // The complete field is constrained after facial correctives are added.
    if (magnitude > 0.9) this.limitGradient(damage, 0.45);
    return {
      field: this.expand(values),
      damage: this.expand(damage),
      affected,
      material,
      normal,
      incidence,
      seed,
    };
  }
  limitGradient(values, limit = 0.78) {
    // Local Lipschitz envelopes bound each component along surface edges. Taking
    // their midpoint smooths only over-steep regions instead of shrinking every
    // dent to satisfy the worst edge. Dijkstra propagation converges in one solve
    // even on dense scans with very short edges.
    const slope = limit / Math.sqrt(3),
      count = this.vertices.length;
    for (let axis = 0; axis < 3; axis++) {
      const envelopes = [];
      for (const sign of [1, -1]) {
        const envelope = new Float64Array(count),
          heap = new Heap();
        for (let i = 0; i < count; i++) envelope[i] = sign * values[i * 3 + axis];
        for (let i = 0; i < count; i++)
          if (
            this.vertices[i].links.some(
              ([j, length]) => envelope[j] > envelope[i] + slope * length + 1e-10,
            )
          )
            heap.push([envelope[i], i]);
        while (heap.data.length) {
          const [value, i] = heap.pop();
          if (value > envelope[i] + 1e-10) continue;
          for (const [j, length] of this.vertices[i].links) {
            const next = value + slope * length;
            if (next < envelope[j] - 1e-10) {
              envelope[j] = next;
              heap.push([next, j]);
            }
          }
        }
        envelopes.push(envelope);
      }
      for (let i = 0; i < count; i++)
        values[i * 3 + axis] = (envelopes[0][i] - envelopes[1][i]) * 0.5;
    }
  }

  expand(values) {
    const out = new Float32Array(this.rest.length);
    for (let v = 0; v < this.map.length; v++)
      for (let j = 0; j < 3; j++) out[v * 3 + j] = values[this.map[v] * 3 + j];
    return out;
  }
  measure(positions, rest = this.rest) {
    let reversed = 0,
      minAreaRatio = Infinity,
      minNormalAgreement = 1,
      triangles = 0;
    const normal = (p, a, b, c) => {
      const u = [0, 1, 2].map((j) => p[b * 3 + j] - p[a * 3 + j]),
        v = [0, 1, 2].map((j) => p[c * 3 + j] - p[a * 3 + j]);
      return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
      ];
    };
    for (let i = 0; i < this.indices.length; i += 3) {
      const [a, b, c] = this.indices.slice(i, i + 3),
        before = normal(rest, a, b, c),
        after = normal(positions, a, b, c),
        area = Math.hypot(...before),
        next = Math.hypot(...after);
      if (area < 1e-12) continue;
      const dot =
        before.reduce((sum, v, j) => sum + v * after[j], 0) /
        (area * Math.max(next, 1e-15));
      triangles++;
      if (dot <= 0) reversed++;
      minAreaRatio = Math.min(minAreaRatio, next / area);
      minNormalAgreement = Math.min(minNormalAgreement, dot);
    }
    return {
      triangles,
      reversedTriangles: reversed,
      minAreaRatio,
      minNormalAgreement,
      finite: positions.every(Number.isFinite),
    };
  }
  fitIncrement(base, increment) {
    if(!increment.some(v=>v!==0))return;
    if(!base.some(v=>v!==0)){this.validity.constrain(increment);return;} // also protect the first retained damage field
    const total = new Float32Array(base.length);
    for (let i = 0; i < total.length; i++) total[i] = base[i] + increment[i];
    this.constrainAccumulation(total, MAX_PERMANENT_DISPLACEMENT, 0.86, false);
    this.validity.constrainRelative(total,base,MAX_PERMANENT_DISPLACEMENT);
    for (let i = 0; i < increment.length; i++) increment[i] = total[i] - base[i];
  }
  constrainAccumulation(field, limit = 0.055, gradient = 0.88, protectArea = true) {
    const values = new Float32Array(this.vertices.length * 3);
    this.vertices.forEach((v, i) => {
      for (let j = 0; j < 3; j++) values[i * 3 + j] = field[v.copies[0] * 3 + j];
    });
    // Projection onto the displacement ball is local and non-expansive.
    for (let i = 0; i < values.length; i += 3) {
      const length = Math.hypot(values[i], values[i + 1], values[i + 2]);
      if (length > limit) for (let j = 0; j < 3; j++) values[i + j] *= limit / length;
    }
    this.limitGradient(values, gradient);
    for (let i = 0; i < values.length; i += 3) {
      const length = Math.hypot(values[i], values[i + 1], values[i + 2]);
      if (length > limit) for (let j = 0; j < 3; j++) values[i + j] *= limit / length;
    }
    field.set(this.expand(values));
    if(protectArea)this.validity.constrain(field);
  }
}
