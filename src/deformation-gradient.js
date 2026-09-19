// NFR-inspired gradient-domain deformation, implemented for browser CPU use.
// Reference: Dafei Qin et al., NFR (2023), deformation_transfer.py::Transfer.
// See docs/NFR_RIG.md and third_party/licenses/NFR-MIT.txt. No neural weights.
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

// A 3x2 tangent Jacobian has the same nonzero singular values as its 2x2
// Gram matrix. Clamp those stretches without damping its rigid rotation.
export function boundedTangents(u, v, minimum = 0.6, maximum = 1.45) {
  const a = dot(u, u),
    b = dot(u, v),
    c = dot(v, v);
  const spread = Math.hypot(a - c, 2 * b);
  const l1 = Math.sqrt(Math.max(0, (a + c + spread) * 0.5));
  const l2 = Math.sqrt(Math.max(0, (a + c - spread) * 0.5));
  if (l1 <= maximum + 1e-6 && l2 >= minimum - 1e-6) return { u, v, changed: false };
  const angle = 0.5 * Math.atan2(2 * b, a - c);
  const cs = Math.cos(angle),
    sn = Math.sin(angle);
  // Collapsed input has no well-defined rotation. It is handled by the final
  // area/orientation projection; never divide by a zero singular value.
  const s1 = clamp(l1, minimum, maximum) / Math.max(l1, 1e-10);
  const s2 = clamp(l2, minimum, maximum) / Math.max(l2, 1e-10);
  const aa = cs * cs * s1 + sn * sn * s2;
  const ab = cs * sn * (s1 - s2);
  const bb = sn * sn * s1 + cs * cs * s2;
  return {
    u: u.map((x, j) => aa * x + ab * v[j]),
    v: v.map((x, j) => ab * u[j] + bb * x),
    changed: true,
  };
}

export class DeformationGradientRig {
  constructor(tissue, { attachmentLength = 0.012 } = {}) {
    this.tissue = tissue;
    this.count = tissue.vertices.length;
    this.triangles = [];
    this.mass = new Float64Array(this.count);
    const rows = Array.from({ length: this.count }, () => new Map());
    const { vertices, indices, map } = tissue;
    for (let f = 0; f < indices.length; f += 3) {
      const ids = indices.slice(f, f + 3).map((v) => map[v]);
      const [a, b, c] = ids.map((i) => vertices[i].p);
      const e1 = b.map((x, j) => x - a[j]),
        e2 = c.map((x, j) => x - a[j]);
      const length = Math.hypot(...e1),
        normal = cross(e1, e2);
      const doubleArea = Math.hypot(...normal);
      // Exclude only degenerate/sliver faces from the differential solve. The
      // complete mesh still participates in the existing validity projection.
      if (length < 1e-8 || doubleArea < 1e-12) continue;
      const t1 = e1.map((x) => x / length);
      const t2 = cross(
        normal.map((x) => x / doubleArea),
        t1,
      );
      const x = dot(e2, t1),
        y = doubleArea / length;
      if (y / Math.max(length, Math.hypot(...e2)) < 1e-5) continue;
      const gx = [-1 / length, 1 / length, 0];
      const gy = [(x / length - 1) / y, -x / (length * y), 1 / y];
      const area = doubleArea * 0.5;
      this.triangles.push({ ids, gx, gy, area, t1, t2 });
      for (let i = 0; i < 3; i++) {
        this.mass[ids[i]] += area / 3;
        for (let j = 0; j < 3; j++) {
          const row = rows[ids[i]],
            col = ids[j];
          row.set(col, (row.get(col) ?? 0) + area * (gx[i] * gx[j] + gy[i] * gy[j]));
        }
      }
    }
    this.diagonal = new Float64Array(this.count);
    this.attachments = new Float64Array(this.count);
    this.starts = new Uint32Array(this.count + 1);
    const columns = [],
      values = [];
    rows.forEach((row, i) => {
      // Positive mass anchors every component, including isolated cage points.
      const attachment = Math.max(this.mass[i] / attachmentLength ** 2, 1e-8);
      this.attachments[i] = attachment;
      row.set(i, (row.get(i) ?? 0) + attachment);
      this.diagonal[i] = row.get(i);
      this.starts[i] = columns.length;
      for (const [j, value] of row) {
        columns.push(j);
        values.push(value);
      }
    });
    this.starts[this.count] = columns.length;
    this.columns = new Uint32Array(columns);
    this.values = new Float64Array(values);
    this.lastSolve = null;
  }

  multiply(x, out) {
    for (let i = 0; i < this.count; i++) {
      let v = 0;
      for (let k = this.starts[i]; k < this.starts[i + 1]; k++)
        v += this.values[k] * x[this.columns[k]];
      out[i] = v;
    }
  }

  solve(rhs, x) {
    const r = new Float64Array(this.count),
      p = new Float64Array(this.count);
    const ap = new Float64Array(this.count);
    this.multiply(x, ap);
    let rz = 0,
      initial = 0;
    for (let i = 0; i < this.count; i++) {
      r[i] = rhs[i] - ap[i];
      p[i] = r[i] / this.diagonal[i];
      rz += r[i] * p[i];
      initial += r[i] * r[i];
    }
    if (initial < 1e-24) return { iterations: 0, relativeResidual: 0 };
    let iterations = 0,
      residual = initial;
    for (; iterations < 100 && residual > initial * 1e-6; iterations++) {
      this.multiply(p, ap);
      let denominator = 0;
      for (let i = 0; i < this.count; i++) denominator += p[i] * ap[i];
      if (!(denominator > 1e-30)) break;
      const alpha = rz / denominator;
      let next = 0;
      residual = 0;
      for (let i = 0; i < this.count; i++) {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
        next += (r[i] * r[i]) / this.diagonal[i];
        residual += r[i] * r[i];
      }
      const beta = next / rz;
      for (let i = 0; i < this.count; i++) p[i] = r[i] / this.diagonal[i] + beta * p[i];
      rz = next;
    }
    return { iterations, relativeResidual: Math.sqrt(residual / initial) };
  }

  refine(field, { minimum = 0.6, maximum = 1.45 } = {}) {
    const start = performance.now();
    const target = new Float64Array(this.count * 3);
    this.tissue.vertices.forEach((v, i) => {
      for (const copy of v.copies)
        for (let j = 0; j < 3; j++)
          target[i * 3 + j] += field[copy * 3 + j] / v.copies.length;
    });
    const rhs = new Float64Array(target.length);
    for (let i = 0; i < this.count; i++)
      for (let j = 0; j < 3; j++)
        rhs[i * 3 + j] = this.attachments[i] * target[i * 3 + j];
    let limitedTriangles = 0;
    for (const { ids, gx, gy, area, t1, t2 } of this.triangles) {
      const u = t1.slice(),
        v = t2.slice();
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) {
          u[j] += gx[i] * target[ids[i] * 3 + j];
          v[j] += gy[i] * target[ids[i] * 3 + j];
        }
      const bounded = boundedTangents(u, v, minimum, maximum);
      limitedTriangles += Number(bounded.changed);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          rhs[ids[i] * 3 + j] +=
            area * (gx[i] * (bounded.u[j] - t1[j]) + gy[i] * (bounded.v[j] - t2[j]));
    }
    let iterations = 0,
      relativeResidual = 0;
    const solved = target.slice();
    if (limitedTriangles)
      for (let axis = 0; axis < 3; axis++) {
        const b = new Float64Array(this.count),
          x = new Float64Array(this.count);
        for (let i = 0; i < this.count; i++) {
          b[i] = rhs[i * 3 + axis];
          x[i] = target[i * 3 + axis];
        }
        const result = this.solve(b, x);
        iterations = Math.max(iterations, result.iterations);
        relativeResidual = Math.max(relativeResidual, result.relativeResidual);
        for (let i = 0; i < this.count; i++) solved[i * 3 + axis] = x[i];
      }
    // Fail closed on ill-conditioned inputs. Keep the authored field available
    // for the downstream area/orientation guard rather than returning NaNs.
    const accepted = solved.every(Number.isFinite) && relativeResidual < 0.02;
    const result = accepted ? solved : target;
    let correction = 0;
    for (let i = 0; i < this.count; i++) {
      correction = Math.max(
        correction,
        Math.hypot(
          result[i * 3] - target[i * 3],
          result[i * 3 + 1] - target[i * 3 + 1],
          result[i * 3 + 2] - target[i * 3 + 2],
        ),
      );
      for (const copy of this.tissue.vertices[i].copies)
        for (let j = 0; j < 3; j++) field[copy * 3 + j] = result[i * 3 + j];
    }
    this.lastSolve = {
      method: 'NFR-inspired gradient solve',
      triangles: this.triangles.length,
      limitedTriangles,
      iterations,
      relativeResidual,
      accepted,
      correctionMm: correction * 1000,
      milliseconds: performance.now() - start,
    };
    return field;
  }
}
