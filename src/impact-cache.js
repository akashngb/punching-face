// Ignore only normalization roundoff in directions/controls. Contact locations
// are canonical Float32 mesh vertices and must match exactly.
export function impactCacheKey(input, softness, reactionEnabled) {
  const round = (value) => Math.round(value * 1e9);
  return JSON.stringify([
    input.location,
    input.direction.map(round),
    round(input.magnitude),
    round(softness),
    reactionEnabled,
  ]);
}

// The live detector raycasts from each semantic target along its incoming
// direction. Repeat that rest-pose intersection when warming exact contacts.
export function restContact(tissue, point, direction) {
  const p = tissue.rest,
    ids = tissue.indices;
  const [dx, dy, dz] = direction;
  const ox = point[0] - dx * 0.12,
    oy = point[1] - dy * 0.12,
    oz = point[2] - dz * 0.12;
  let nearest = 0.3,
    hit = point;
  for (let f = 0; f < ids.length; f += 3) {
    const a = ids[f] * 3,
      b = ids[f + 1] * 3,
      c = ids[f + 2] * 3;
    const ex = p[b] - p[a],
      ey = p[b + 1] - p[a + 1],
      ez = p[b + 2] - p[a + 2];
    const fx = p[c] - p[a],
      fy = p[c + 1] - p[a + 1],
      fz = p[c + 2] - p[a + 2];
    const hx = dy * fz - dz * fy,
      hy = dz * fx - dx * fz,
      hz = dx * fy - dy * fx;
    const det = ex * hx + ey * hy + ez * hz;
    if (det <= 1e-12) continue;
    const tx = ox - p[a],
      ty = oy - p[a + 1],
      tz = oz - p[a + 2];
    const u = (tx * hx + ty * hy + tz * hz) / det;
    if (u < 0 || u > 1) continue;
    const qx = ty * ez - tz * ey,
      qy = tz * ex - tx * ez,
      qz = tx * ey - ty * ex;
    const v = (dx * qx + dy * qy + dz * qz) / det;
    if (v < 0 || u + v > 1) continue;
    const distance = (fx * qx + fy * qy + fz * qz) / det;
    if (distance >= 0 && distance < nearest) {
      nearest = distance;
      hit = [ox + dx * distance, oy + dy * distance, oz + dz * distance];
    }
  }
  return tissue.vertices[tissue.nearest(hit).node].p;
}
