// Pin every asset and the Newton session to one accepted reconstruction.
export async function loadHeadBundle(id, fetcher = fetch) {
  const base = `/api/face-asset?id=${encodeURIComponent(id)}`;
  const releaseResponse = await fetcher(`${base}&asset=model-release.json`, {
    cache: 'no-store',
  });
  if (!releaseResponse.ok)
    throw new Error('The saved model release could not be read.');
  const release = await releaseResponse.json();
  const generation = release.generation;
  if (generation !== 'legacy' && !/^[a-f0-9]{32}$/.test(generation || ''))
    throw new Error('The saved model has an invalid release identifier.');
  const asset = (name) => `${base}&asset=${name}&generation=${generation}`;
  const digest = async (bytes) =>
    Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (v) =>
      v.toString(16).padStart(2, '0'),
    ).join('');
  const read = async (name) => {
    const response = await fetcher(asset(name), { cache: 'no-store' });
    if (!response.ok) throw new Error('The saved model is missing ' + name);
    const bytes = await response.arrayBuffer();
    if (
      generation !== 'legacy' &&
      (!release.files?.[name] || (await digest(bytes)) !== release.files[name])
    )
      throw new Error('The saved model file failed verification: ' + name);
    return bytes;
  };
  const json = async (name) => JSON.parse(new TextDecoder().decode(await read(name)));
  const [data, atlas, binding, cage, textureBytes] = await Promise.all([
    json('mesh.json'),
    json('texture-atlas.json'),
    json('physics-binding.json'),
    json('physics-cage.json'),
    read('appearance.png'),
  ]);
  if (
    (atlas.textureSha256 && (await digest(textureBytes)) !== atlas.textureSha256) ||
    (atlas.positionsSha256 &&
      (await digest(new Float32Array(data.positions).buffer)) !== atlas.positionsSha256)
  )
    throw new Error(
      'The saved geometry and texture do not match. The existing model has been retained.',
    );
  if (
    cage.positions.length !== 468 * 3 ||
    data.positions.length < cage.positions.length ||
    cage.positions.some(
      (v, i) =>
        !Number.isFinite(v) || Math.fround(v) !== Math.fround(data.positions[i]),
    )
  )
    throw new Error('The saved physics cage does not match the head.');
  const roughnessBytes = atlas.roughnessTexture
    ? await read('appearance-roughness.png')
    : null;
  return { data, atlas, binding, cage, textureBytes, roughnessBytes, generation };
}
