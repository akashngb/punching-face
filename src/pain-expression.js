// Authored pain-expression pose: AU4 brow lowering, AU6/7 orbital tightening,
// AU9/10 upper-lip/levator lift, and jaw release. See docs/NFR_RIG.md.
// An animation response to an impact, not an estimate of a person's pain.
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const smooth = (a, b, x) => {
  const t = clamp((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};

export function painEnvelope(age) {
  return smooth(0.025, 0.17, age) * (1 - smooth(0.55, 1.65, age));
}

export class PainExpression {
  constructor(tissue) {
    this.tissue = tissue;
    // The head template has separate eyeball surfaces. Move the eyelids over
    // those rigid spheres instead of squashing the eyes with the skin.
    this.skin = new Uint8Array(tissue.vertices.length);
    const visited = new Uint8Array(tissue.vertices.length);
    let largest = [];
    tissue.vertices.forEach((v, seed) => {
      if (visited[seed] || !v.links.length) return;
      const component = [seed];
      visited[seed] = 1;
      for (let k = 0; k < component.length; k++)
        for (const [j] of tissue.vertices[component[k]].links)
          if (!visited[j]) {
            visited[j] = 1;
            component.push(j);
          }
      if (component.length > largest.length) largest = component;
    });
    for (const i of largest) this.skin[i] = 1;
  }

  build(anchors, point, magnitude) {
    this.prepare(anchors);
    const { mouth, scale, left, right } = this.basis;
    const side = clamp((point[0] - mouth[0]) / (0.035 * scale), -1, 1);
    const front = smooth(mouth[2] - 0.085 * scale, mouth[2] - 0.025 * scale, point[2]);
    const strength = Math.pow(magnitude, 0.8) * front;
    const blend = (side + 1) * 0.5;
    const field = new Float32Array(left.length);
    for (let i = 0; i < field.length; i++)
      field[i] = (left[i] * (1 - blend) + right[i] * blend) * strength;
    return field;
  }

  prepare(anchors) {
    const key = JSON.stringify(anchors);
    if (this.basis?.key === key) return;
    const mouth = anchors[13].map((x, j) => (x + anchors[14][j]) * 0.5);
    const scale = clamp((mouth[1] - anchors[152][1]) / 0.06, 0.65, 1.6);
    // The authored expression is affine in contact side and linear in strength.
    // Two double-precision endpoints reproduce it without per-hit Gaussian work.
    const left = this.buildBasis(
      anchors,
      [mouth[0] - 0.035 * scale, mouth[1], mouth[2]],
      1,
    );
    const right = this.buildBasis(
      anchors,
      [mouth[0] + 0.035 * scale, mouth[1], mouth[2]],
      1,
    );
    this.basis = { key, mouth, scale, left, right };
  }

  buildBasis(anchors, point, magnitude) {
    const a = { ...anchors },
      { rest, map, vertices } = this.tissue;
    // Photo heads reserve the first 468 *unrendered* semantic cage landmarks.
    // Use their measured lid/brow positions only when this layout is present.
    if (
      map.length >= 468 &&
      [145, 374, 107, 336].every((i) => !vertices[map[i]].links.length)
    )
      for (const id of [145, 374, 107, 336, 105, 334])
        a[id] = Array.from(rest.slice(id * 3, id * 3 + 3));
    const mouth = a[13].map((x, j) => (x + a[14][j]) * 0.5);
    const scale = clamp((mouth[1] - a[152][1]) / 0.06, 0.65, 1.6);
    const side = clamp((point[0] - mouth[0]) / (0.035 * scale), -1, 1);
    const frontContact = smooth(
      mouth[2] - 0.085 * scale,
      mouth[2] - 0.025 * scale,
      point[2],
    );
    const strength = Math.pow(magnitude, 0.8) * frontContact;
    const field = new Float64Array(rest.length);
    const gaussian = (p, c, rx, ry, rz) =>
      Math.exp(
        -p.reduce((s, x, j) => s + ((x - c[j]) / ([rx, ry, rz][j] * scale)) ** 2, 0),
      );
    for (let i = 0; i < vertices.length; i++) {
      if (!this.skin[i]) continue;
      const p = vertices[i].p,
        [x, y, z] = p;
      const front = smooth(mouth[2] - 0.08 * scale, mouth[2] - 0.025 * scale, z);
      const neck = smooth(a[152][1] - 0.025 * scale, a[152][1] + 0.008 * scale, y);
      if (front * neck < 1e-6) continue;
      let dx = 0,
        dy = 0,
        dz = 0;
      for (const s of [-1, 1]) {
        const upper = a[s < 0 ? 159 : 386],
          lower = a[s < 0 ? 145 : 374] ?? [
            upper[0],
            upper[1] - 0.009 * scale,
            upper[2],
          ];
        const center = upper.map((v, j) => (v + lower[j]) * 0.5);
        const local = 0.9 + 0.1 * s * side;
        // Close along the lid gap rather than translate the whole socket.
        // A broad support avoids a crease immediately above the upper eyelid.
        const eye = gaussian(p, center, 0.026, 0.026, 0.024);
        // Leave clearance for the cheek lift and physical contact layer.
        dy += -(y - center[1]) * 0.74 * eye * local;
        const brow = a[s < 0 ? 107 : 336] ?? [
          center[0] * 0.65,
          center[1] + 0.024 * scale,
          center[2],
        ];
        const bw = gaussian(p, brow, 0.031, 0.022, 0.036);
        dx -= s * 0.0045 * scale * bw * local;
        dy -= 0.0105 * scale * bw * local;
        dz += 0.002 * scale * bw;
        const cheek = [center[0], center[1] - 0.02 * scale, center[2] - 0.004 * scale];
        dy += 0.004 * scale * gaussian(p, cheek, 0.03, 0.018, 0.03) * local;
        const corner = a[s < 0 ? 61 : 291],
          cw = gaussian(p, corner, 0.023, 0.022, 0.026);
        dx += s * 0.0045 * scale * cw;
        dy -= (0.008 + 0.003 * s * side) * scale * cw;
      }
      const upperLip = gaussian(
        p,
        [mouth[0], mouth[1] + 0.009 * scale, mouth[2]],
        0.03,
        0.018,
        0.026,
      );
      dy += 0.005 * scale * upperLip;
      dz += 0.002 * scale * upperLip;
      // Broad lower-face hinge motion makes the response visible in silhouette.
      // The sealed reconstructed lip seam limits true mouth opening; do not
      // invent an interior or pull adjacent seam vertices in opposite directions.
      const jaw =
        (1 - smooth(mouth[1] - 0.015 * scale, mouth[1] + 0.015 * scale, y)) *
        Math.exp(-(((x - mouth[0]) / (0.085 * scale)) ** 4));
      const angle = 0.19 * jaw,
        hy = mouth[1] + 0.058 * scale,
        hz = mouth[2] - 0.072 * scale;
      dy += (y - hy) * (Math.cos(angle) - 1) - (z - hz) * Math.sin(angle);
      dz += (y - hy) * Math.sin(angle) + (z - hz) * (Math.cos(angle) - 1);
      const weight = strength * front * neck;
      for (const copy of vertices[i].copies) {
        field[copy * 3] = dx * weight;
        field[copy * 3 + 1] = dy * weight;
        field[copy * 3 + 2] = dz * weight;
      }
    }
    return field;
  }
}
