// A local, topology-aware appearance pass. It estimates broad colour across
// connected triangles; it does not invent anatomy, sharpen pixels or move vertices.
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const smooth = (v) => {
  v = clamp(v);
  return v * v * (3 - 2 * v);
};

function validate({ positions, mapping, indices, uv, pixels, width, height }) {
  const n = positions.length / 3;
  if (
    !Number.isInteger(n) ||
    n < 3 ||
    !positions.every(Number.isFinite) ||
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width * height > 16777216 ||
    pixels.length !== width * height * 4 ||
    uv.length !== mapping.length * 2 ||
    !uv.every((v) => Number.isFinite(v) && v >= 0 && v <= 1) ||
    !mapping.every((v) => Number.isInteger(v) && v >= 0 && v < n) ||
    !indices.length ||
    indices.length % 3 ||
    !indices.every((v) => Number.isInteger(v) && v >= 0 && v < mapping.length)
  )
    throw new Error(
      'Realism needs a finite mesh and a valid, non-repeating UV texture up to 4096² pixels.',
    );
}

// Use the same UV orientation as the actual texture, including imported GLBs.
function raster(indices, uv, width, height, flipY, visit) {
  for (let t = 0; t < indices.length; t += 3) {
    const ids = [indices[t], indices[t + 1], indices[t + 2]];
    const x = ids.map((i) => uv[i * 2] * width - 0.5);
    const y = ids.map(
      (i) => (flipY ? 1 - uv[i * 2 + 1] : uv[i * 2 + 1]) * height - 0.5,
    );
    const d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2]);
    if (Math.abs(d) < 1e-10) continue;
    const x0 = Math.max(0, Math.ceil(Math.min(...x))),
      x1 = Math.min(width - 1, Math.floor(Math.max(...x)));
    const y0 = Math.max(0, Math.ceil(Math.min(...y))),
      y1 = Math.min(height - 1, Math.floor(Math.max(...y)));
    for (let py = y0; py <= y1; py++)
      for (let px = x0; px <= x1; px++) {
        const a = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / d;
        const b = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / d;
        const c = 1 - a - b;
        if (a >= -1e-6 && b >= -1e-6 && c >= -1e-6) visit(px, py, ids, [a, b, c]);
      }
  }
}

export function featureProtection(positions, landmarks, protectedVertices = []) {
  const weight = new Float32Array(positions.length / 3).fill(1);
  if (landmarks?.length === 468 * 3) {
    const point = (i) => landmarks.slice(i * 3, i * 3 + 3);
    const mid = (a, b) => point(a).map((v, k) => (v + point(b)[k]) / 2);
    const span = Math.hypot(...point(33).map((v, k) => v - point(263)[k]));
    // Protect eyeballs, eyelids, eyebrows, nose, lips and moustache. The outer
    // transition is continuous, so protection cannot stamp another crop edge.
    const regions = [
      [mid(33, 133), [0.3, 0.27, 0.35]],
      [mid(263, 362), [0.3, 0.27, 0.35]],
      [point(70), [0.29, 0.18, 0.3]],
      [point(300), [0.29, 0.18, 0.3]],
      [point(4), [0.27, 0.38, 0.35]],
      [mid(61, 291), [0.43, 0.28, 0.35]],
    ];
    for (let i = 0; i < weight.length; i++) {
      weight[i] *= smooth(
        (point(10)[1] - span * 0.03 - positions[i * 3 + 1]) / (span * 0.14),
      );
      for (const [center, radii] of regions) {
        const r = Math.hypot(
          ...center.map(
            (v, k) => (positions[i * 3 + k] - v) / Math.max(span * radii[k], 1e-6),
          ),
        );
        weight[i] = Math.min(weight[i], smooth((r - 1) / 0.65));
      }
    }
  }
  for (const i of protectedVertices)
    if (Number.isInteger(i) && i >= 0 && i < weight.length) weight[i] = 0;
  return weight;
}

export function enhanceSurface(input, progress = () => {}) {
  validate(input);
  const {
    positions,
    mapping,
    indices,
    uv,
    pixels,
    width,
    height,
    landmarks,
    protectedVertices,
    flipY = true,
  } = input;
  const n = positions.length / 3,
    r = mapping.length;
  const sums = new Float64Array(r * 3),
    mass = new Float64Array(r);
  const scale = Math.min(1, 768 / Math.max(width, height));
  const sw = Math.max(1, Math.round(width * scale)),
    sh = Math.max(1, Math.round(height * scale));
  progress('Reading the texture on the 3D surface…');
  raster(indices, uv, sw, sh, flipY, (x, y, ids, bary) => {
    const source =
      (Math.min(height - 1, Math.floor(((y + 0.5) * height) / sh)) * width +
        Math.min(width - 1, Math.floor(((x + 0.5) * width) / sw))) *
      4;
    if (pixels[source + 3] < 250) return;
    for (let k = 0; k < 3; k++) {
      const v = ids[k],
        w = Math.max(0, bary[k]);
      mass[v] += w;
      for (let c = 0; c < 3; c++) sums[v * 3 + c] += (pixels[source + c] / 255) * w;
    }
  });
  const field = new Float32Array(n * 3),
    support = new Float64Array(n);
  for (let i = 0; i < r; i++)
    if (mass[i]) {
      support[mapping[i]] += mass[i];
      for (let c = 0; c < 3; c++) {
        field[mapping[i] * 3 + c] += sums[i * 3 + c];
      }
    }
  for (let i = 0; i < n; i++)
    if (support[i]) for (let c = 0; c < 3; c++) field[i * 3 + c] /= support[i];
  const allowed = featureProtection(positions, landmarks, protectedVertices);
  let minY = Infinity,
    maxY = -Infinity;
  for (let i = 1; i < positions.length; i += 3) {
    minY = Math.min(minY, positions[i]);
    maxY = Math.max(maxY, positions[i]);
  }
  const headHeight = Math.max(1e-6, maxY - minY),
    radius = headHeight * 0.035;
  const neighbors = Array.from({ length: n }, () => []),
    seen = new Set();
  for (let t = 0; t < indices.length; t += 3) {
    const face = [
      mapping[indices[t]],
      mapping[indices[t + 1]],
      mapping[indices[t + 2]],
    ];
    // Neck closure must never become a shortcut between opposite sides.
    if (face.every((i) => positions[i * 3 + 1] <= minY + headHeight * 0.001)) continue;
    for (let k = 0; k < 3; k++) {
      const a = Math.min(face[k], face[(k + 1) % 3]),
        b = Math.max(face[k], face[(k + 1) % 3]),
        key = a * n + b;
      if (a === b || seen.has(key) || !support[a] || !support[b]) continue;
      seen.add(key);
      let d2 = 0,
        colour2 = 0;
      for (let c = 0; c < 3; c++) {
        d2 += (positions[a * 3 + c] - positions[b * 3 + c]) ** 2;
        colour2 += (field[a * 3 + c] - field[b * 3 + c]) ** 2;
      }
      if (d2 > radius * radius * 4) continue;
      const w = Math.exp(-d2 / (radius * radius) - colour2 / 0.045);
      neighbors[a].push([b, w]);
      neighbors[b].push([a, w]);
    }
  }
  progress('Balancing crop boundaries across connected triangles…');
  let current = field.slice();
  for (let step = 0; step < 18; step++) {
    const next = current.slice();
    for (let i = 0; i < n; i++) {
      if (!allowed[i] || !support[i] || !neighbors[i].length) continue;
      let total = 0;
      const mean = [0, 0, 0];
      for (const [j, w] of neighbors[i]) {
        total += w;
        for (let c = 0; c < 3; c++) mean[c] += current[j * 3 + c] * w;
      }
      if (total < 1e-8) continue;
      for (let c = 0; c < 3; c++) {
        const target = 0.06 * field[i * 3 + c] + (0.94 * mean[c]) / total;
        next[i * 3 + c] =
          current[i * 3 + c] + 0.7 * allowed[i] * (target - current[i * 3 + c]);
      }
    }
    current = next;
  }
  progress('Baking the repaired texture at its original resolution…');
  // Identify UV islands independently of the welded surface. Texture filtering
  // may use one island only; diffusion may cross a seam only via a mesh edge.
  const parent = Int32Array.from({ length: r }, (_, i) => i);
  const root = (i) => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  for (let t = 0; t < indices.length; t += 3) {
    parent[root(indices[t + 1])] = root(indices[t]);
    parent[root(indices[t + 2])] = root(indices[t]);
  }
  const island = Int32Array.from(parent, (_, i) => root(i) + 1);
  const owner = new Int32Array(width * height);
  let overlap = false;
  raster(indices, uv, width, height, flipY, (x, y, ids) => {
    const i = y * width + x,
      group = island[ids[0]];
    if (owner[i] && owner[i] !== group) overlap = true;
    owner[i] = group;
  });
  if (overlap)
    throw new Error(
      'This texture has overlapping UV islands. Unwrap it before applying Realism.',
    );
  const stride = Math.max(1, Math.round(Math.max(width, height) / 768));
  const kernel = [1, 4, 6, 4, 1];
  const result = new Uint8ClampedArray(pixels),
    coverage = new Uint8Array(width * height);
  let changed = 0,
    covered = 0,
    totalChange = 0,
    maxChange = 0;
  raster(indices, uv, width, height, flipY, (x, y, ids, bary) => {
    const i = y * width + x,
      offset = i * 4;
    if (coverage[i] || pixels[offset + 3] < 250) return;
    coverage[i] = 1;
    covered++;
    const amount = bary.reduce(
      (sum, w, k) => sum + w * (mass[ids[k]] ? allowed[mapping[ids[k]]] : 0),
      0,
    );
    if (amount < 0.001) return;
    const low = [0, 0, 0];
    let lowMass = 0;
    for (let dy = -2; dy <= 2; dy++)
      for (let dx = -2; dx <= 2; dx++) {
        const sx = x + dx * stride,
          sy = y + dy * stride;
        if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
        const sample = sy * width + sx;
        if (owner[sample] !== owner[i] || pixels[sample * 4 + 3] < 250) continue;
        const w = kernel[dx + 2] * kernel[dy + 2];
        lowMass += w;
        for (let c = 0; c < 3; c++) low[c] += pixels[sample * 4 + c] * w;
      }
    let change = 0;
    for (let c = 0; c < 3; c++) {
      const target =
        bary.reduce((sum, w, k) => sum + w * current[mapping[ids[k]] * 3 + c], 0) * 255;
      // Transfer broad colour only; the original's high-frequency residual
      // stays in place, including pores, stubble and photographed texture.
      const d = amount * clamp(target - low[c] / lowMass, -35.7, 35.7);
      result[offset + c] = pixels[offset + c] + d;
      change = Math.max(change, Math.abs(result[offset + c] - pixels[offset + c]));
    }
    if (change > 1) changed++;
    totalChange += change;
    maxChange = Math.max(maxChange, change);
  });
  // Extend only the colour *correction* into UV gutters, retaining the original
  // padded texture. This prevents dark seams in bilinear filtering and mipmaps.
  let frontier = [];
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      if (coverage[i] && [i - 1, i + 1, i - width, i + width].some((j) => !coverage[j]))
        frontier.push(i);
    }
  for (let step = 0; step < 5; step++) {
    const next = [];
    for (const i of frontier)
      for (const j of [
        i % width ? i - 1 : -1,
        i % width < width - 1 ? i + 1 : -1,
        i - width,
        i + width,
      ]) {
        if (j < 0 || j >= coverage.length || coverage[j] || owner[j]) continue;
        coverage[j] = 2;
        next.push(j);
        for (let c = 0; c < 3; c++)
          result[j * 4 + c] = pixels[j * 4 + c] + result[i * 4 + c] - pixels[i * 4 + c];
      }
    frontier = next;
  }
  return {
    pixels: result,
    report: {
      version: 1,
      method: 'Connected-surface colour balancing with protected facial features',
      coveredTexels: covered,
      adjustedTexels: changed,
      meanChange: totalChange / Math.max(covered, 1),
      maxChannelChange: maxChange,
      protectedVertices: allowed.filter((v) => v === 0).length,
      geometryUnchanged: true,
      textureResolution: [width, height],
      limitation:
        'Appearance cleanup only. Missing anatomy and unobserved detail are not recovered.',
    },
  };
}
