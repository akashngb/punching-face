import { Float32BufferAttribute } from 'three';

// The editable surface reads packed floating-point arrays directly. GLB also
// permits interleaved and normalized integer attributes, plus unindexed meshes.
export function importedHeadGeometry(mesh) {
  const geometry = mesh.geometry.clone();
  for (const [name, attribute] of Object.entries(geometry.attributes)) {
    if (
      !attribute.isInterleavedBufferAttribute &&
      !attribute.normalized &&
      attribute.array instanceof Float32Array
    )
      continue;
    const values = new Float32Array(attribute.count * attribute.itemSize);
    for (let vertex = 0; vertex < attribute.count; vertex++)
      for (let component = 0; component < attribute.itemSize; component++)
        values[vertex * attribute.itemSize + component] = attribute.getComponent(
          vertex,
          component,
        );
    geometry.setAttribute(name, new Float32BufferAttribute(values, attribute.itemSize));
  }
  if (!geometry.index)
    geometry.setIndex(
      Array.from({ length: geometry.attributes.position.count }, (_, i) => i),
    );
  geometry.applyMatrix4(mesh.matrixWorld);
  return geometry;
}
