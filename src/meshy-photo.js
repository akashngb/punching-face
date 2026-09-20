import qualityURL from './face-quality.js?url';
import { loadMeshyModel } from './meshy-engine.js';
import { acknowledgeScanJob, trackScanJob } from './scan-jobs.js';

async function api(path, data) {
  const response = await fetch(
    '/api/' + path,
    data === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        },
  );
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'The Meshy request failed.');
  return result;
}

function workerReply(worker, message, transfer = []) {
  return new Promise((resolve, reject) => {
    const finish = (error, data) => {
      clearTimeout(timer);
      worker.onmessage = null;
      worker.onerror = null;
      error ? reject(error) : resolve(data);
    };
    const timer = setTimeout(
      () => finish(new Error('Photo capture timed out. Try again.')),
      25000,
    );
    worker.onerror = () => finish(new Error('Could not prepare the head photo.'));
    worker.onmessage = ({ data }) =>
      finish(data.type === 'error' ? new Error(data.message) : null, data);
    worker.postMessage(message, transfer);
  });
}

async function savedFrame(photo) {
  // One capture uses the same validation and masking as a recorded scan. No worker
  // remains alive while Meshy builds or while the visitor punches another head.
  const worker = new Worker('/face-worker.js');
  try {
    await workerReply(worker, { type: 'init', qualityURL, origin: location.origin });
    const bitmap = await createImageBitmap(photo);
    const frame = await workerReply(worker, { type: 'frame', bitmap }, [bitmap]);
    if (!frame.ok || !frame.landmarks || Math.abs(frame.yaw) > 20)
      throw new Error(frame.message || 'Face the camera directly and try again.');
    const image = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not save the head photo.'));
      reader.readAsDataURL(frame.blob);
    });
    return {
      image,
      yaw: frame.yaw,
      landmarks: frame.landmarks,
      irisLandmarks: frame.irisLandmarks,
    };
  } finally {
    worker.terminate();
  }
}

export async function startMeshyPhoto(photo) {
  const frame = await savedFrame(photo);
  const capture = await api('face-captures', { captureRegion: 'head' });
  await api('face-frames', { id: capture.id, frames: [frame] });
  // The server stores the task and final GLB beside this photo, just as it does
  // for a multiview scan. Closing the page does not lose the model or the job.
  await api('meshy-train', { id: capture.id });
  trackScanJob({ id: capture.id, engine: 'meshy' });
  return capture.id;
}

export async function loadReadyMeshyPhoto(id) {
  const job = await api('meshy-job?id=' + encodeURIComponent(id));
  if (job.status === 'failed')
    throw new Error(
      job.message || 'The Meshy build failed. Open the saved scan to retry.',
    );
  if (job.status !== 'complete' || !job.model) {
    trackScanJob({ id, engine: 'meshy' });
    return false;
  }
  await loadMeshyModel(id);
  acknowledgeScanJob({ id, engine: 'meshy' });
  return true;
}
