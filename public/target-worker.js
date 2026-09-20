/* Target-perspective worker. Classic worker for the same reason as pov-worker: MediaPipe's WASM
   loader uses importScripts. */
self.exports = {};
importScripts('/vendor/vision_bundle.cjs');
const { FilesetResolver, HandLandmarker } = self.exports;

importScripts('/target-motion.js');
let hands,
  lastStamp = 0;
async function create(task, files, options) {
  try {
    return await task.createFromOptions(files, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: 'GPU' },
    });
  } catch {
    return task.createFromOptions(files, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: 'CPU' },
    });
  }
}
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const files = await FilesetResolver.forVisionTasks(`${data.origin}/wasm`);
      // Detection confidence sits low because a blurred incoming fist scores poorly and dropping
      // it is worse than tracking a marginal one.
      hands = await create(HandLandmarker, files, {
        baseOptions: { modelAssetPath: `${data.origin}/models/hand_landmarker.task` },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minTrackingConfidence: 0.3,
      });
      self.postMessage({ type: 'ready' });
      return;
    }
    if (data.type !== 'frame') return;
    try {
      // detectForVideo demands strictly increasing timestamps. Two frames are in flight at once and
      // createImageBitmap can resolve out of order, so the stamps can arrive inverted -- which threw
      // and killed the whole camera. Clamp instead: a frame is never worth losing the feed over.
      lastStamp = Math.max(lastStamp + 1, Math.round(data.timestamp));
      let result = null,
        detectError = '';
      try {
        result = hands.detectForVideo(data.bitmap, lastStamp);
      } catch (error) {
        detectError = error?.message ?? String(error);
      }
      const motion = self.TargetMotion.motionOf(data.bitmap, data.timestamp);
      self.postMessage({
        type: 'result',
        landmarks: result?.landmarks ?? [],
        worldLandmarks: result?.worldLandmarks ?? [],
        handedness: result?.handedness ?? result?.handednesses ?? [],
        motion,
        detectError,
        timestamp: data.timestamp,
      });
    } finally {
      data.bitmap.close();
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message });
  }
};
