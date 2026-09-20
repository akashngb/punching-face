// Ground-truth motions for the estimator A/B. Each returns the true hand state at a time in
// seconds, so velocity and contact are differentiated from the motion itself rather than inferred
// from either estimator under test.
const smooth = (x) => x * x * (3 - 2 * x);
const clamp01 = (x) => Math.min(1, Math.max(0, x));

export const GUARD_DEPTH = 0.25;
// The synthetic hand's +y axis runs wrist->knuckles, so at zero rotation the knuckles point up.
// pitch = -90deg turns them down-range, which is the resting orientation for a straight punch.
export const FORWARD = -Math.PI / 2;

// Head geometry mirrors the debug scene: LeePerrySmith normalised to 0.28 m tall, scaled 2.47x by
// fitTarget and pushed to 0.65 m. Its front surface therefore sits at 0.42 m of depth.
export const HEAD = { distance: 0.65, scale: 2.47, radii: [0.15, 0.14, 0.0915] };

export const SCENARIOS = [
  {
    name: 'guard hold',
    detail: 'fist still at guard for 6 s — nothing should ever fire',
    duration: 6,
    expectHit: false,
    at: (t) => ({
      position: [
        0.06 + 0.002 * Math.sin(t * 1.7),
        -0.05 + 0.002 * Math.sin(t * 1.1),
        -(GUARD_DEPTH + 0.002 * Math.sin(t * 0.9)),
      ],
      rotation: { pitch: FORWARD + 0.25, yaw: -0.18, roll: 0.05 },
      closure: 1,
    }),
  },
  {
    name: 'wrist rotation only',
    detail:
      'held at guard, wrist pitches through 55deg and back; the hand never translates',
    duration: 4,
    expectHit: false,
    at: (t) => ({
      position: [0.06, -0.05, -GUARD_DEPTH],
      rotation: {
        pitch: FORWARD + 0.96 * Math.sin(Math.PI * clamp01(t / 4) * 2) ** 2,
        yaw: 0,
        roll: 0,
      },
      closure: 1,
    }),
  },
  {
    name: 'jab',
    detail: 'straight to 0.50 m in 120 ms, wrist turning over 50deg, then retract',
    duration: 0.62,
    expectHit: true,
    expectMode: 'jab',
    at: (t) => {
      const out = clamp01(t / 0.12),
        back = clamp01((t - 0.24) / 0.22),
        reach = smooth(out) - smooth(back) * 0.92;
      return {
        position: [
          0.06 - 0.02 * reach,
          -0.05 + 0.02 * reach,
          -(GUARD_DEPTH + 0.25 * reach),
        ],
        rotation: {
          pitch: FORWARD + 0.3 - 0.3 * smooth(out),
          yaw: -0.18 + 0.18 * smooth(out),
          roll: 0.45 * smooth(out),
        },
        closure: 1,
      };
    },
  },
  {
    name: 'hook',
    detail:
      'quadratic arc around the shoulder into the right cheek, knuckles turning to face across',
    duration: 0.62,
    expectHit: true,
    expectMode: 'hook',
    at: (t) => {
      const out = clamp01(t / 0.15),
        back = clamp01((t - 0.26) / 0.24),
        s = smooth(out) - smooth(back) * 0.92;
      // Bezier: wide out to the right, then in. The chord between samples cuts well inside this.
      const S = [0.34, -0.05, -0.22],
        C = [0.46, -0.03, -0.52],
        E = [0.1, -0.02, -0.6];
      const u = 1 - s,
        position = [0, 1, 2].map((k) => u * u * S[k] + 2 * s * u * C[k] + s * s * E[k]);
      // Knuckles swing from down-range to facing across the body (-x) as the hook comes round.
      return {
        position,
        rotation: { pitch: FORWARD, yaw: 1.35 * s, roll: 0.25 },
        closure: 1,
      };
    },
  },
  {
    name: 'uppercut',
    detail: 'rises from below the frame with the knuckles up, 0.32 m of lift',
    duration: 0.62,
    expectHit: true,
    expectMode: 'uppercut',
    at: (t) => {
      const out = clamp01(t / 0.14),
        back = clamp01((t - 0.25) / 0.23),
        s = smooth(out) - smooth(back) * 0.92;
      return {
        position: [0.05, -0.3 + 0.32 * s, -(GUARD_DEPTH + 0.23 * s)],
        rotation: { pitch: -0.3 * s, yaw: 0, roll: 0 },
        closure: 1,
      };
    },
  },
];

export function trueVelocity(scenario, t, h = 0.002) {
  const a = scenario.at(Math.max(0, t - h)).position,
    b = scenario.at(t + h).position;
  return [0, 1, 2].map((k) => (b[k] - a[k]) / (t + h - Math.max(0, t - h)));
}
