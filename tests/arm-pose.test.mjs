import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  ARM_PAIRS,
  capturedArmProfile,
  calibrateBodyFrame,
  matchHandsToBody,
  retargetCapturedArm,
  armBoneRotations,
} from '../src/arm-pose.js';
import { ScannedArm } from '../src/scanned-arm.js';
import { Tracking, VirtualHand } from '../src/hands.js';

// Mathematical rig fixture only: never loaded in the user app or presented as
// a scan. It makes coordinate, bone-length, twist and stale-tracking errors visible.
const v = (x, y, z) => new THREE.Vector3(x, y, z);
const object = (p) => ({ x: p.x, y: p.y, z: p.z, visibility: 1, presence: 1 });

function fixture() {
  const shoulder = v(-0.19, -0.21, 0.04),
    elbow = v(-0.2, -0.4, -0.15),
    wrist = v(-0.14, -0.24, -0.36),
    hand = [wrist.clone()];
  for (let f = 0; f < 5; f++)
    for (let j = 0; j < 4; j++)
      hand.push(
        wrist
          .clone()
          .add(v((2 - f) * 0.015, 0.03 + j * 0.025, -0.012 - (f === 0 ? 0.01 : 0))),
      );
  const joints = [shoulder, elbow, wrist, ...hand];
  const bundle = {
    format: 'punching-face-arm',
    version: 1,
    side: 'left',
    joints: joints.map((p) => p.toArray()),
    mesh: {
      positions: [-0.2, -0.3, -0.1, -0.18, -0.3, -0.1, -0.19, -0.29, -0.1],
      indices: [0, 1, 2],
      colors: [0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3],
    },
    evidence: { testFixture: true },
  };
  const profile = capturedArmProfile(bundle);
  const world = Array.from({ length: 33 }, () => object(v(0, 0, 0))),
    image = Array.from({ length: 33 }, () => ({
      x: 0.5,
      y: 0.5,
      z: 0,
      visibility: 1,
      presence: 1,
    }));
  const camera = (p) => object(v(-p.x, -p.y, p.z));
  for (const [index, p] of [
    [2, v(-0.03, 0, 0)],
    [5, v(0.03, 0, 0)],
    [11, shoulder],
    [12, v(0.19, -0.21, 0.04)],
    [13, elbow],
    [15, wrist],
    [14, v(0.2, -0.4, -0.15)],
    [16, v(0.14, -0.24, -0.36)],
    [23, v(-0.12, -0.6, 0.04)],
    [24, v(0.12, -0.6, 0.04)],
  ])
    world[index] = camera(p);
  image[15].x = 0.7;
  image[16].x = 0.3;
  const handWorld = hand.map(camera),
    frame = calibrateBodyFrame(world, image, [profile]);
  return { bundle, profile, world, image, handWorld, frame };
}

const near = (a, b, tolerance = 1e-6) =>
  assert.ok(a.distanceTo(b) < tolerance, `${a.toArray()} != ${b.toArray()}`);

test('body coordinates place each captured arm on the anatomical side and in front of the eye origin', () => {
  const f = fixture(),
    p = retargetCapturedArm(f.profile, f.frame, f.world, f.image, f.handWorld);
  assert.ok(p.joints[0].x < 0);
  assert.ok(p.joints[2].z < -0.25);
  assert.ok(p.joints[0].y < -0.15);
  near(p.joints[0], f.profile.rest[0], 0.04);
  assert.ok(
    Math.abs(p.joints[0].distanceTo(p.joints[1]) - f.profile.upperLength) < 1e-9,
  );
  assert.ok(
    Math.abs(p.joints[1].distanceTo(p.joints[2]) - f.profile.forearmLength) < 1e-9,
  );
  for (const root of [4, 8, 12, 16, 20])
    for (let j = root + 1; j <= root + 3; j++)
      assert.ok(
        Math.abs(
          p.joints[j].distanceTo(p.joints[j - 1]) -
            f.profile.rest[j].distanceTo(f.profile.rest[j - 1]),
        ) < 1e-9,
      );
});
test('joint directions change with observed elbow movement without stretching the captured bones', () => {
  const f = fixture(),
    before = retargetCapturedArm(f.profile, f.frame, f.world, f.image, f.handWorld);
  f.world[13].x += 0.13;
  f.world[13].z -= 0.09;
  const after = retargetCapturedArm(f.profile, f.frame, f.world, f.image, f.handWorld);
  assert.ok(before.joints[1].distanceTo(after.joints[1]) > 0.05);
  assert.ok(
    Math.abs(after.joints[0].distanceTo(after.joints[1]) - f.profile.upperLength) <
      1e-9,
  );
  assert.ok(
    Math.abs(after.joints[1].distanceTo(after.joints[2]) - f.profile.forearmLength) <
      1e-9,
  );
});
test('palm roll rotates the captured skin even when the forearm endpoints stay fixed', () => {
  const f = fixture(),
    before = retargetCapturedArm(f.profile, f.frame, f.world, f.image, f.handWorld),
    rotations = armBoneRotations(f.profile, before);
  const origin = new THREE.Vector3().copy(f.handWorld[0]),
    axis = new THREE.Vector3().copy(f.world[15]).sub(f.world[13]).normalize(),
    turn = new THREE.Quaternion().setFromAxisAngle(axis, Math.PI / 2);
  f.handWorld = f.handWorld.map((p) =>
    object(new THREE.Vector3().copy(p).sub(origin).applyQuaternion(turn).add(origin)),
  );
  const after = retargetCapturedArm(f.profile, f.frame, f.world, f.image, f.handWorld),
    next = armBoneRotations(f.profile, after);
  near(before.joints[2], after.joints[2]);
  assert.ok(rotations[1].angleTo(next[1]) > 0.8);
  assert.ok(rotations[2].angleTo(next[2]) > 1.4);
  const original = f.profile.rest[2].clone().sub(f.profile.rest[1]).normalize(),
    expected = after.joints[2].clone().sub(after.joints[1]).normalize();
  near(original.applyQuaternion(next[1]), expected);
});
test('crossed hands match anatomical wrists without trusting handedness order', () => {
  const f = fixture();
  f.image[15].x = 0.3;
  f.image[16].x = 0.7;
  const h = [
    Array.from({ length: 21 }, () => ({ x: 0.69, y: 0.5, z: 0 })),
    Array.from({ length: 21 }, () => ({ x: 0.31, y: 0.5, z: 0 })),
  ];
  assert.deepEqual(
    [...matchHandsToBody(h, f.image)],
    [
      [0, 'right'],
      [1, 'left'],
    ],
  );
  f.image[15].visibility = 0.1;
  assert.equal(matchHandsToBody(h, f.image).size, 1);
});
test('missing joints hide a reconstructed mesh rather than using fixed proxy elbows', () => {
  const f = fixture(),
    arm = new ScannedArm(f.bundle);
  f.image[13].visibility = 0.2;
  assert.equal(
    retargetCapturedArm(f.profile, f.frame, f.world, f.image, f.handWorld),
    null,
  );
  assert.equal(calibrateBodyFrame(f.world, f.image, [f.profile]), null);
  arm.updateFromHand({ visible: true, tracked: true, armPose: null });
  assert.equal(arm.visible, false);
  arm.dispose();
});
test('bone skinning preserves bind geometry, then responds to tracked pose and twist', () => {
  const f = fixture(),
    arm = new ScannedArm(f.bundle);
  arm.updateMatrixWorld(true);
  arm.mesh.skeleton.update();
  const g = arm.mesh.geometry;
  for (let i = 0; i < g.attributes.position.count; i++) {
    const original = new THREE.Vector3().fromBufferAttribute(g.attributes.position, i);
    near(arm.mesh.applyBoneTransform(i, original.clone()), original);
  }
  const pose = retargetCapturedArm(f.profile, f.frame, f.world, f.image, f.handWorld);
  arm.updateFromHand({ visible: true, tracked: true, armPose: pose });
  arm.updateMatrixWorld(true);
  arm.mesh.skeleton.update();
  assert.equal(arm.visible, true);
  assert.ok(arm.mesh.skeleton.boneMatrices.every(Number.isFinite));
  for (const [a, b] of ARM_PAIRS)
    assert.ok(Number.isFinite(pose.joints[a].distanceTo(pose.joints[b])));
  arm.dispose();
});
test('stale webcam samples and reacquisition cannot produce a phantom punch', async () => {
  const f = fixture(),
    tracking = new Tracking(
      { videoWidth: 1280, videoHeight: 720, readyState: 0 },
      () => {},
    ),
    hands = [new VirtualHand(-1), new VirtualHand(1)];
  tracking.active = true;
  tracking.armProfiles.set('left', f.profile);
  tracking.bodyFrame = f.frame;
  tracking.results = {
    timestamp: 1000,
    landmarks: [
      Array.from({ length: 21 }, (_, i) => ({ x: 0.7 + i * 0.001, y: 0.5, z: 0 })),
    ],
    handedness: [[{ categoryName: 'Left' }]],
    worldLandmarks: [f.handWorld],
    pose: { timestamp: 1000, landmarks: [f.image], worldLandmarks: [f.world] },
  };
  await tracking.tick(1005, hands);
  assert.equal(hands[0].tracked, true);
  assert.equal(hands[0].updated, false);
  // Brief gaps retain the drawing but must revoke contact eligibility.
  await tracking.tick(1400, hands);
  assert.equal(hands[0].tracked, false);
  assert.equal(hands[0].armPose, null);
  assert.equal(hands[0].visible, true);
  // The silence window in hands.js is 1 s, so the drop has to be probed past it.
  await tracking.tick(2100, hands);
  assert.equal(hands[0].tracked, false);
  assert.equal(hands[0].armPose, null);
  assert.equal(hands[0].visible, false);
  tracking.results.timestamp = 2150;
  tracking.results.pose.timestamp = 2150;
  await tracking.tick(2160, hands);
  assert.equal(hands[0].tracked, true);
  assert.equal(hands[0].updated, false);
  near(hands[0].previous, hands[0].center);
});
