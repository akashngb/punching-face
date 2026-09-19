import test from 'node:test';
import assert from 'node:assert/strict';
import { faceQuality, captureCoverage, FACE_OVAL } from '../src/face-quality.js';

function result(yaw = 0) {
  const lm = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5 }));
  FACE_OVAL.forEach(
    (id, i) =>
      (lm[id] = {
        x: 0.5 + 0.2 * Math.sin((i / FACE_OVAL.length) * Math.PI * 2),
        y: 0.5 - 0.35 * Math.cos((i / FACE_OVAL.length) * Math.PI * 2),
      }),
  );
  lm[33] = { x: 0.4, y: 0.45 };
  lm[263] = { x: 0.6, y: 0.45 };
  lm[13] = { x: 0.5, y: 0.65 };
  lm[14] = { x: 0.5, y: 0.66 };
  const a = (yaw * Math.PI) / 180,
    m = [1, 0, 0, 0, 0, 1, 0, 0, Math.sin(a), 0, Math.cos(a), 0, 0, 0, 0, 1];
  return { faceLandmarks: [lm], facialTransformationMatrixes: [{ data: m }] };
}

test('face capture requires one visible neutral face and skips identical view angles', () => {
  const r = result();
  assert.ok(faceQuality(r, 1280, 720).ok);
  assert.equal(faceQuality(r, 1280, 720, { yaw: 0, pitch: 0 }).ok, false);
  r.faceLandmarks[0][14].y = 0.8;
  assert.equal(faceQuality(r, 1280, 720).ok, false);
  assert.equal(faceQuality({ faceLandmarks: [] }, 1280, 720).ok, false);
});
test('capture coverage requires front and both sides; frontal repeats cannot satisfy it', () => {
  assert.deepEqual(captureCoverage([{ yaw: 0 }, { yaw: -30 }, { yaw: 30 }]), {
    front: true,
    left: true,
    right: true,
    headOnly: 0,
    count: 3,
  });
  assert.equal(
    captureCoverage(Array.from({ length: 60 }, () => ({ yaw: 0 }))).left,
    false,
  );
  assert.ok(faceQuality(result(30), 1280, 720, { yaw: 0, pitch: 0 }).ok);
});
test('iris landmarks are retained separately without rejecting old or occluded captures', () => {
  const r = result();
  assert.equal(faceQuality(r, 1280, 720).irisLandmarks, null);
  r.faceLandmarks[0].push(
    ...Array.from({ length: 10 }, () => ({ x: 0.45, y: 0.45, z: 0.01 })),
  );
  const quality = faceQuality(r, 1280, 720);
  assert.equal(quality.landmarks.length, 468);
  assert.equal(quality.irisLandmarks.length, 10);
  assert.deepEqual(quality.irisLandmarks[0], { x: 0.45, y: 0.45 });
  r.faceLandmarks[0][468].x = NaN;
  const occluded = faceQuality(r, 1280, 720);
  assert.ok(occluded.ok);
  assert.equal(occluded.irisLandmarks, null);
});
