// Compliant edge constraints following Macklin et al., XPBD (2016), Eq. 18.
// This is a surface membrane with positional tissue attachments, not a fitted
// anatomical volume model. The material values below are prototype parameters.
export class SkinConstraints {
  constructor(geometry) {
    const rest = geometry.attributes.position.array,
      count = rest.length / 3,
      indices = geometry.index?.array ?? Array.from({ length: count }, (_, i) => i);
    this.reference = new Float32Array(rest);
    this.inverseMass = new Float32Array(count);
    this.areas = new Float32Array(count);
    const pairs = new Map();
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i], indices[i + 1], indices[i + 2]],
        ai = a * 3,
        bi = b * 3,
        ci = c * 3;
      const ux = rest[bi] - rest[ai],
        uy = rest[bi + 1] - rest[ai + 1],
        uz = rest[bi + 2] - rest[ai + 2],
        vx = rest[ci] - rest[ai],
        vy = rest[ci + 1] - rest[ai + 1],
        vz = rest[ci + 2] - rest[ai + 2];
      const area =
        Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
      if (area < 1e-12) continue;
      for (const v of [a, b, c]) this.areas[v] += area / 3;
      for (let [u, v] of [
        [a, b],
        [b, c],
        [c, a],
      ]) {
        if (u > v) [u, v] = [v, u];
        const key = u * count + v,
          edge = pairs.get(key);
        if (edge) edge.area += area / 3;
        else pairs.set(key, { a: u, b: v, area: area / 3 });
      }
    }
    const valid = [...pairs.values()].filter(
      (e) =>
        Math.hypot(
          rest[e.a * 3] - rest[e.b * 3],
          rest[e.a * 3 + 1] - rest[e.b * 3 + 1],
          rest[e.a * 3 + 2] - rest[e.b * 3 + 2],
        ) > 1e-5,
    );
    this.edges = new Uint32Array(valid.length * 2);
    this.edgeArea = new Float32Array(valid.length);
    this.lengths = new Float32Array(valid.length);
    this.compliance = new Float32Array(valid.length);
    this.lambda = new Float32Array(valid.length);
    valid.forEach((e, i) => {
      this.edges[i * 2] = e.a;
      this.edges[i * 2 + 1] = e.b;
      this.edgeArea[i] = e.area;
    });
    // Areal mass = density * nominal layer thickness. A small floor prevents
    // nearly degenerate triangles around the eyes dominating the solve.
    for (let i = 0; i < count; i++)
      this.inverseMass[i] = 1 / Math.max(this.areas[i] * 1000 * 0.003, 1e-6);
    this.positions = new Float32Array(rest.length);
    this.predicted = new Float32Array(rest.length);
    this.iterations = 4;
    this.refreshReference(rest, true);
  }

  refreshReference(reference, force = false) {
    let changed = force;
    for (let i = 0; !changed && i < reference.length; i++)
      if (reference[i] !== this.reference[i]) changed = true;
    if (!changed) return;
    this.reference.set(reference);
    for (let e = 0; e < this.lengths.length; e++) {
      const a = this.edges[e * 2] * 3,
        b = this.edges[e * 2 + 1] * 3,
        l = Math.hypot(
          reference[a] - reference[b],
          reference[a + 1] - reference[b + 1],
          reference[a + 2] - reference[b + 2],
        );
      this.lengths[e] = Math.max(l, 1e-5);
      // Edge energy approximates a membrane's strain energy over its dual area.
      const stiffness = (5000 * 0.003 * this.edgeArea[e]) / Math.max(l * l, 1e-10);
      this.compliance[e] = 1 / Math.max(stiffness, 0.1);
    }
  }

  solve(offset, velocity, dt, softness) {
    if (dt <= 0) return;
    const p = this.positions,
      base = this.reference,
      edges = this.edges,
      inv = this.inverseMass,
      lambda = this.lambda;
    for (let i = 0; i < p.length; i++) p[i] = base[i] + offset[i];
    this.predicted.set(p);
    lambda.fill(0);
    const factor = (0.45 + softness * 1.4) / (dt * dt);
    for (let iteration = 0; iteration < this.iterations; iteration++)
      for (let e = 0; e < this.lengths.length; e++) {
        const av = edges[e * 2],
          bv = edges[e * 2 + 1],
          a = av * 3,
          b = bv * 3;
        const dx = p[a] - p[b],
          dy = p[a + 1] - p[b + 1],
          dz = p[a + 2] - p[b + 2],
          length = Math.hypot(dx, dy, dz);
        if (length < 1e-10) continue;
        const alpha = this.compliance[e] * factor,
          change =
            (-(length - this.lengths[e]) - alpha * lambda[e]) /
            (inv[av] + inv[bv] + alpha);
        lambda[e] += change;
        const sa = (inv[av] * change) / length,
          sb = (inv[bv] * change) / length;
        p[a] += dx * sa;
        p[a + 1] += dy * sa;
        p[a + 2] += dz * sa;
        p[b] -= dx * sb;
        p[b + 1] -= dy * sb;
        p[b + 2] -= dz * sb;
      }
    for (let i = 0; i < p.length; i++) {
      offset[i] = p[i] - base[i];
      velocity[i] += (p[i] - this.predicted[i]) / dt;
    }
  }

  measure(offset) {
    let sum = 0,
      area = 0,
      maxStretch = 0,
      maxCompression = 0;
    const base = this.reference;
    for (let e = 0; e < this.lengths.length; e++) {
      const a = this.edges[e * 2] * 3,
        b = this.edges[e * 2 + 1] * 3;
      const length = Math.hypot(
        base[a] + offset[a] - base[b] - offset[b],
        base[a + 1] + offset[a + 1] - base[b + 1] - offset[b + 1],
        base[a + 2] + offset[a + 2] - base[b + 2] - offset[b + 2],
      );
      const strain = length / this.lengths[e] - 1,
        weight = this.edgeArea[e];
      sum += strain * strain * weight;
      area += weight;
      maxStretch = Math.max(maxStretch, strain);
      maxCompression = Math.max(maxCompression, -strain);
    }
    return { rms: Math.sqrt(sum / Math.max(area, 1e-12)), maxStretch, maxCompression };
  }
}
