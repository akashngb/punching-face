import * as THREE from 'three';

// Add real surface degrees of freedom while preserving the captured texture and
// boundary. Subdivision adds simulation resolution, not new scan evidence.
export function refineSurface(input, passes = 2) {
  let g = input;
  for (let pass = 0; pass < passes; pass++) {
    const attrs = Object.fromEntries(
      Object.entries(g.attributes)
        .filter(([k]) => k !== 'normal')
        .map(([k, a]) => [k, { size: a.itemSize, values: Array.from(a.array) }]),
    );
    const source = g.index
        ? Array.from(g.index.array)
        : Array.from({ length: g.attributes.position.count }, (_, i) => i),
      edges = new Map(),
      indices = [];
    const mid = (a, b) => {
      const key = a < b ? `${a}:${b}` : `${b}:${a}`;
      if (edges.has(key)) return edges.get(key);
      const id = attrs.position.values.length / 3;
      for (const attr of Object.values(attrs))
        for (let k = 0; k < attr.size; k++)
          attr.values.push(
            (attr.values[a * attr.size + k] + attr.values[b * attr.size + k]) * 0.5,
          );
      edges.set(key, id);
      return id;
    };
    for (let i = 0; i < source.length; i += 3) {
      const [a, b, c] = source.slice(i, i + 3),
        ab = mid(a, b),
        bc = mid(b, c),
        ca = mid(c, a);
      indices.push(a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca);
    }
    const next = new THREE.BufferGeometry();
    for (const [name, attr] of Object.entries(attrs))
      next.setAttribute(name, new THREE.Float32BufferAttribute(attr.values, attr.size));
    next.setIndex(indices);
    next.computeVertexNormals();
    if (g !== input) g.dispose();
    g = next;
  }
  return g;
}
