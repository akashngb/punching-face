// Saved local and Meshy heads share a capture, but have independent build states.
// Concurrent library/job lookups share one read burst; later calls get fresh state.
let inFlight = null;

async function readJson(path, options = {}) {
  const response = await fetch(`/api/${path}`, { cache: 'no-store', ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error || `Saved heads request failed (${response.status}).`);
  }
  return response.json();
}

async function loadSnapshot() {
  const data = await readJson('face-captures');
  if (!Array.isArray(data.captures))
    throw new Error('The saved heads response is invalid.');
  const captures = data.captures.filter(
    (capture) => !capture.testFixture && typeof capture.id === 'string',
  );
  const snapshot = new Array(captures.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(4, captures.length) }, async () => {
      while (next < captures.length) {
        const index = next++;
        const capture = captures[index];
        let meshy = null;
        try {
          meshy = await readJson(`meshy-job?id=${encodeURIComponent(capture.id)}`, {
            signal: AbortSignal.timeout(5000),
          });
        } catch {
          // An unavailable Meshy endpoint must not hide saved local heads.
        }
        snapshot[index] = { capture, meshy };
      }
    }),
  );
  return snapshot;
}

function readSnapshot() {
  if (!inFlight)
    inFlight = loadSnapshot().finally(() => {
      inFlight = null;
    });
  return inFlight;
}

function describe(capture, engine, job = null) {
  return {
    ...capture,
    ...job,
    id: capture.id,
    engine,
    key: `${engine}:${capture.id}`,
    name: capture.name || (engine === 'meshy' ? 'Meshy head' : 'Scanned head'),
    frames: capture.frames,
    savedAt: capture.savedAt,
  };
}

export async function listSavedHeads() {
  const snapshot = await readSnapshot();
  return snapshot.flatMap(({ capture, meshy }) => [
    ...(capture.photoModel ? [describe(capture, 'local')] : []),
    ...(meshy?.model ? [describe(capture, 'meshy', meshy)] : []),
  ]);
}

export async function discoverScanJobs() {
  const snapshot = await readSnapshot();
  return snapshot.flatMap(({ capture, meshy }) => [
    ...(capture.status === 'running' ? [describe(capture, 'local')] : []),
    ...(meshy?.status === 'running' ? [describe(capture, 'meshy', meshy)] : []),
  ]);
}
