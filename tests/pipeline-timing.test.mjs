import test from 'node:test';
import assert from 'node:assert/strict';
import { duration, timingRows } from '../src/pipeline-timing.js';
test('video processing totals exclude recording duration and user idle time', () => {
  const source = {
    durationSeconds: 24.3,
    extractionSeconds: 15.6,
    extractionComplete: true,
  };
  const timing = {
    status: 'complete',
    reconstructionSeconds: 121.2,
    loadSeconds: 1.1,
    stages: [{ stage: 'cameras', seconds: 31 }],
  };
  assert.deepEqual(timingRows(source, timing).at(-1), [
    'Video → interactive model',
    '2m 17.9s',
  ]);
  assert.equal(duration(119.99), '2m 0.0s');
});
test('partial import or failed reconstruction cannot claim a complete end-to-end time', () => {
  const source = {
    durationSeconds: 24.3,
    extractionSeconds: 5,
    extractionComplete: false,
  };
  assert.ok(
    !timingRows(source, { status: 'complete', reconstructionSeconds: 30 }).some(
      ([label]) => label.startsWith('Video →'),
    ),
  );
  assert.ok(
    !timingRows(
      { ...source, extractionComplete: true },
      { status: 'failed', reconstructionSeconds: 30 },
    ).some(([label]) => label.startsWith('Video →')),
  );
  assert.deepEqual(
    timingRows(
      null,
      { status: 'running', activeStage: 'astra', stageStartedAt: 10 },
      14,
    ).at(-1),
    ['AI hair & glasses analysis · running', '4.0 s'],
  );
});
