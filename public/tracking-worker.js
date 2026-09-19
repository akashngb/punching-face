/* MediaPipe's WASM loader uses importScripts; keep this a classic worker. */
self.exports = {};
importScripts('/vendor/vision_bundle.cjs');
const { FilesetResolver, HandLandmarker, PoseLandmarker } = self.exports;
let detector, poseDetector, origin, lastPose;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      origin = data.origin;
      const files = await FilesetResolver.forVisionTasks(`${data.origin}/wasm`);
      // GPU delegate cuts inference from ~30-60ms to ~5-10ms on Apple Silicon; falls back to CPU
      // automatically when the browser has no WebGL2 in this worker context.
      try {
        detector = await HandLandmarker.createFromOptions(files, {
          baseOptions: {
            modelAssetPath: `${data.origin}/models/hand_landmarker.task`,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minTrackingConfidence: 0.4,
        });
      } catch {
        detector = await HandLandmarker.createFromOptions(files, {
          baseOptions: {
            modelAssetPath: `${data.origin}/models/hand_landmarker.task`,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minTrackingConfidence: 0.4,
        });
      }
      self.postMessage({ type: 'ready' });
    } else if (data.type === 'enablePose') {
      if (poseDetector) {
        self.postMessage({ type: 'poseReady' });
        return;
      }
      const files = await FilesetResolver.forVisionTasks(`${origin}/wasm`);
      // Segmentation masks are only used by the arm-capture flow. Turning them on unconditionally
      // added ~15 ms per pose frame; leave them off unless the caller explicitly asked for them.
      const wantSegmentation = !!data.wantSegmentation;
      try {
        poseDetector = await PoseLandmarker.createFromOptions(files, {
          baseOptions: {
            modelAssetPath: `${origin}/models/pose_landmarker.task`,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          outputSegmentationMasks: wantSegmentation,
        });
      } catch {
        poseDetector = await PoseLandmarker.createFromOptions(files, {
          baseOptions: {
            modelAssetPath: `${origin}/models/pose_landmarker.task`,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          outputSegmentationMasks: wantSegmentation,
        });
      }
      self.postMessage({ type: 'poseReady' });
    } else if (data.type === 'frame') {
      try {
        const result = detector.detectForVideo(data.bitmap, data.timestamp);
        let capture;
        // Pose inference only runs when the caller actively needs it. The old "every 4th frame"
        // heartbeat was stalling the default punch flow with a CPU model that nothing consumed.
        if (poseDetector && (data.capture || data.trackBody)) {
          poseDetector.detectForVideo(data.bitmap, data.timestamp, (pose) => {
            lastPose = {
              landmarks: pose.landmarks,
              worldLandmarks: pose.worldLandmarks,
              timestamp: data.timestamp,
            };
            if (data.capture && pose.segmentationMasks?.[0]) {
              const mask = pose.segmentationMasks[0];
              capture = {
                mask: new Float32Array(mask.getAsFloat32Array()),
                maskWidth: mask.width,
                maskHeight: mask.height,
                pose: lastPose,
                side: data.capture,
              };
            }
          });
        }
        if (capture) {
          const canvas = new OffscreenCanvas(data.bitmap.width, data.bitmap.height);
          canvas.getContext('2d').drawImage(data.bitmap, 0, 0);
          capture.image = await canvas.convertToBlob({ type: 'image/png' });
        }
        self.postMessage({
          type: 'result',
          landmarks: result.landmarks,
          worldLandmarks: result.worldLandmarks,
          handedness: result.handedness,
          pose: lastPose,
          capture,
          timestamp: data.timestamp,
        });
      } finally {
        data.bitmap.close();
      }
    }
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
  }
};
