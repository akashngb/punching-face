import {
  TissueField,
  impactParameters,
  MAX_PERMANENT_DISPLACEMENT,
} from './tissue-field.js';
import { DeformationGradientRig } from './deformation-gradient.js';
import { PainExpression, painEnvelope } from './pain-expression.js';
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = (a, b, v) => {
  const t = clamp((v - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

// Art-directed impact correctives, layered over the contact solver. These are
// facial animation fields, not a claim that the FEM predicts this much motion.
// All fields share the same rest coordinates, including UV seam duplicates.
export class FaceImpactRig {
  constructor(rest, anchors, topology) {
    this.offset = new Float32Array(rest.length);
    this.permanent = new Float32Array(rest.length);
    this.events = [];
    this.mode = 'live';
    this.reactionEnabled = true;
    this.setAnchors(anchors);
    if (topology) {
      this.topology = topology;
      this.tissue = new TissueField(rest, topology.indices, topology.normals);
      this.prepare();
    }
  }

  setAnchors(anchors) {
    this.anchors = {
      13: [0, -0.04, 0.07],
      14: [0, -0.042, 0.07],
      152: [0, -0.105, 0.045],
      50: [-0.05, -0.005, 0.06],
      280: [0.05, -0.005, 0.06],
      61: [-0.03, -0.04, 0.065],
      291: [0.03, -0.04, 0.065],
      159: [-0.035, 0.037, 0.065],
      386: [0.035, 0.037, 0.065],
      ...anchors,
    };
    if (this.tissue) this.prepare();
    this.preparer?.configure(this.preparationModel());
  }

  preparationModel() {
    return {
      rest: this.tissue.rest,
      anchors: this.anchors,
      contactAnchors: this.contactAnchors,
      topology: this.topology,
    };
  }

  enableAsync(makePreparation) {
    if (typeof Worker === 'undefined' || !this.tissue || this.preparer) return;
    try {
      this.preparer = makePreparation(this.preparationModel());
    } catch {
      /* Synchronous fallback. */
    }
  }

  cancelPreparation() {
    this.preparer?.invalidate();
  }

  restEdited() {
    // A sculpted/restored rest buffer no longer matches the worker snapshot.
    // Preserve the existing ordered sculpt path until a new head is loaded.
    this.asyncRestEdited = true;
    this.cancelPreparation();
  }

  dispose() {
    this.preparer?.dispose();
  }

  prepare() {
    this.gradientRig ??= new DeformationGradientRig(this.tissue);
    this.painExpression ??= new PainExpression(this.tissue);
    this.tissue.prepareAnatomy(this.anchors);
    this.painExpression.prepare(this.anchors);
    // Build the triangle guard while the model is loading, not on its first hit.
    void this.tissue.validity;
  }

  // Smooth onset, a readable compression crest, then a damped recovery.
  envelope(age) {
    if (age < 0.12) return smooth(0, 0.12, age);
    const t = age - 0.12;
    return (1 + 16 * t) * Math.exp(-16 * t) * (1 - smooth(0.65, 0.9, t));
  }

  get hasPeaked() {
    return (
      this.events.length > 0 &&
      this.events.every((e) => e.age + 1e-8 >= (e.reaction ? 0.17 : 0.12))
    );
  }

  // Public entry: dispatch to a mode-specific field. hook is the historical pose.
  trigger(rest, point, direction, speed, softness, mode = 'hook') {
    if (this.tissue)
      return this.impact(
        rest,
        { location: point, direction, magnitude: clamp(speed / 1.4, 0, 0.9) },
        softness,
      );
    let field;
    if (mode === 'uppercut') field = this._uppercutField(rest, point, speed, softness);
    else if (mode === 'jab') field = this._jabField(rest, point, speed, softness);
    else field = this._hookField(rest, point, direction, speed, softness);
    // A short queue supports alternating hooks without unbounded accumulation.
    this.events.push({ age: 0, field });
    if (this.events.length > 4) this.events.shift();
  }

  setMode(mode) {
    if (!['clay', 'live'].includes(mode))
      throw new RangeError('Head mode must be clay or live.');
    if (mode !== this.mode) this.cancelPreparation();
    this.mode = mode;
  }

  impact(rest, input, softness, positions = rest, { refine = true } = {}) {
    const p = impactParameters(input);
    if (p.magnitude === 0) return 0;
    const hit = this.tissue.nearest(p.location, positions);
    if (hit.distance > 0.045) return 0;
    if (
      this.preparer &&
      !this.asyncRestEdited &&
      !this.preparer.failed &&
      this.mode === 'live' &&
      p.magnitude <= 0.9 &&
      !this.permanent.some((v) => v !== 0) &&
      !this.events.some((e) => e.plastic?.some((v) => v !== 0))
    ) {
      // A live contact without retained damage is independent of other live
      // endpoints. Prepare the identical fields off-thread; recoil/Newton and
      // camera processing can respond immediately. Capture the selected rest
      // node so moving skin cannot change the contact while work is queued.
      const input = { ...p, location: this.tissue.vertices[hit.node].p };
      const queuedAt = performance.now();
      let previewEvent = null,
        previewMs = null;
      const queued = this.preparer.request(
        input,
        softness,
        this.reactionEnabled,
        (result) => {
          if (result.error) {
            if (previewEvent)
              this.events = this.events.filter((event) => event !== previewEvent);
            this.preparer.fail();
            this.impact(rest, input, softness);
            return;
          }
          if (!result.event || !result.affected) return;
          if (result.stage === 'preview') {
            previewMs = performance.now() - queuedAt;
            previewEvent = result.event;
            this.events.push(previewEvent);
          } else if (previewEvent && this.events.includes(previewEvent)) {
            const from = {
              target: previewEvent.target,
              reactionTarget: previewEvent.reactionTarget,
              combinedTarget: previewEvent.combinedTarget,
            };
            const { age, committed } = previewEvent;
            Object.assign(previewEvent, result.event, {
              age,
              committed,
              refinement: { from, elapsed: 0 },
            });
          } else if (!previewEvent) this.events.push(result.event);
          else return; // Never revive an expired preview after a delayed solve.
          if (this.events.length > 12) this.events.shift();
          this.lastImpact = {
            ...result.impact,
            preparationMs: result.milliseconds,
            readyMs: performance.now() - queuedAt,
            previewMs,
            stage: result.stage,
            cached: !!result.cached,
            worker: true,
          };
        },
      );
      return queued ? this.tissue.vertices[hit.node].copies.length : 0;
    }
    // A damaging contact changes the base for every pending endpoint. Cancel
    // those jobs before applying it so an old response cannot erase a dent.
    this.cancelPreparation();
    const result = this.tissue.build(
      hit.node,
      p.direction,
      p.magnitude,
      softness,
      this.anchors,
    );
    // Expressive coupling carries a lateral cheek strike through the lips/jaw.
    // It is weighted by the actual contact region and incoming tangent, never
    // selected solely from the caller's punch label.
    const point = this.tissue.vertices[hit.node].p;
    const coupling = result.material.cheek * Math.abs(p.direction[0]);
    if (coupling > 0.05) {
      const side = Math.sign(point[0] - (this.anchors[13][0] || 0)) || 1;
      if (p.direction[0] * side < 0) {
        const corrective = this._hookField(
          rest,
          { x: point[0], y: point[1], z: point[2] },
          { x: p.direction[0], y: p.direction[1], z: p.direction[2] },
          p.magnitude * 1.4,
          softness,
        );
        for (let i = 0; i < result.field.length; i++)
          result.field[i] += corrective[i] * coupling * 0.65;
      }
    }
    const chin = this.anchors[152],
      scale = result.material.scale;
    const chinContact = Math.exp(
      -point.reduce((sum, v, j) => sum + ((v - chin[j]) / (0.042 * scale)) ** 2, 0),
    );
    if (p.direction[1] > 0.1 && chinContact > 0.05) {
      const corrective = this._uppercutField(
        rest,
        { x: point[0], y: point[1], z: point[2] },
        p.magnitude * 1.4,
        softness,
      );
      for (let i = 0; i < result.field.length; i++)
        result.field[i] += corrective[i] * chinContact * p.direction[1] * 0.7;
    }
    // Integrate bounded local triangle transforms into one connected face.
    // Built lazily, shared by all contact directions, with no per-frame solve.
    this.gradientRig ??= new DeformationGradientRig(this.tissue);
    if (refine) this.gradientRig.refine(result.field);
    this.tissue.constrainAccumulation(result.field);
    const permanent = (this.mode === 'clay' ? result.field : result.damage).slice();
    const reserved = this.permanent.slice();
    for (const e of this.events)
      for (let i = 0; i < reserved.length; i++)
        reserved[i] += e.plastic[i] * (1 - e.committed);
    this.tissue.fitIncrement(reserved, permanent);
    let reaction = null;
    if (this.mode === 'live' && this.reactionEnabled) {
      this.painExpression ??= new PainExpression(this.tissue);
      reaction = this.painExpression.build(this.anchors, point, p.magnitude);
    }
    this.events.push({
      age: 0,
      field: result.field,
      plastic: permanent,
      mode: this.mode,
      committed: 0,
      reaction,
    });
    // Precompute bounded live endpoints once per impact. Every animation frame
    // then blends valid endpoints; no surface solve is needed during playback.
    for (const e of this.events) {
      if (e.mode !== 'live') continue;
      e.target = new Float32Array(rest.length);
      for (let i = 0; i < rest.length; i++)
        e.target[i] = reserved[i] + permanent[i] + e.field[i] - e.plastic[i];
      if (reserved.some((v) => v !== 0) || permanent.some((v) => v !== 0))
        this.tissue.constrainAccumulation(e.target, MAX_PERMANENT_DISPLACEMENT);
      if (e.reaction) {
        e.reactionTarget = new Float32Array(rest.length);
        e.combinedTarget = new Float32Array(rest.length);
        for (let i = 0; i < rest.length; i++) {
          e.reactionTarget[i] = reserved[i] + permanent[i] + e.reaction[i];
          e.combinedTarget[i] = e.target[i] + e.reaction[i];
        }
        for (const field of [e.reactionTarget, e.combinedTarget]) {
          // Eyelid contraction must not be suppressed by the impact field's
          // isotropic stretch clamp. Protect orientation and displacement here.
          for (let i = 0; i < field.length; i += 3) {
            const length = Math.hypot(field[i], field[i + 1], field[i + 2]);
            if (length > MAX_PERMANENT_DISPLACEMENT)
              for (let j = 0; j < 3; j++)
                field[i + j] *= MAX_PERMANENT_DISPLACEMENT / length;
          }
          this.tissue.validity.constrain(field, 0.12);
        }
      }
    }
    // Finish plastic commitments before retiring an event under sustained input.
    if (this.events.length > 12) {
      const e = this.events.shift();
      for (let i = 0; i < this.permanent.length; i++)
        this.permanent[i] += e.plastic[i] * (1 - e.committed);
    }
    this.lastImpact = {
      ...p,
      regionBoneWeight: result.material.bone,
      gradientSolve: refine ? this.gradientRig.lastSolve : null,
      painReaction: !!reaction,
      affected: result.affected,
    };
    return result.affected;
  }

  get permanentPeak() {
    let peak = 0;
    for (let i = 0; i < this.permanent.length; i += 3)
      peak = Math.max(
        peak,
        Math.hypot(this.permanent[i], this.permanent[i + 1], this.permanent[i + 2]),
      );
    return peak;
  }
  snapshot() {
    return { mode: this.mode, permanent: Array.from(this.permanent) };
  }

  restore(data) {
    if (!data) return;
    this.cancelPreparation();
    this.setMode(data.mode ?? 'live');
    if (
      !Array.isArray(data.permanent) ||
      data.permanent.length !== this.permanent.length ||
      !data.permanent.every(
        (v) => Number.isFinite(v) && Math.abs(v) <= MAX_PERMANENT_DISPLACEMENT + 1e-6,
      )
    )
      throw new RangeError('Invalid saved impact deformation.');
    this.permanent.set(data.permanent);
    this.events = [];
    this.offset.set(this.permanent);
  }

  _hookField(rest, point, direction, speed, softness) {
    const a = this.anchors,
      upper = a[13],
      lower = a[14],
      chin = a[152];
    const mouth = upper.map((v, i) => (v + lower[i]) * 0.5),
      mx = mouth[0],
      my = mouth[1],
      mz = mouth[2];
    const scale = clamp((my - chin[1]) / 0.058, 0.65, 1.5);
    const side = Math.sign(point.x - mx) || -Math.sign(direction.x) || 1,
      push = -side;
    const eye = a[side < 0 ? 159 : 386],
      otherEye = a[side < 0 ? 386 : 159];
    const cheek = a[side < 0 ? 50 : 280] ?? [
      mx + side * 0.05 * scale,
      my + 0.035 * scale,
      mz - 0.012 * scale,
    ];
    const corner = a[side < 0 ? 61 : 291];
    const amount =
      clamp(speed / 1.05, 0.25, 1.35) * (0.78 + clamp(softness, 0, 1) * 0.37) * scale;
    const field = new Float32Array(rest.length);
    const gaussian = (x, y, c, rx, ry) =>
      Math.exp(-(((x - c[0]) / (rx * scale)) ** 2) - ((y - c[1]) / (ry * scale)) ** 2);
    let maximum = 0;
    for (let i = 0; i < rest.length; i += 3) {
      const x = rest[i],
        y = rest[i + 1],
        z = rest[i + 2];
      // Blend to the skull/neck continuously. Do not use the binary Newton
      // binding mask here: that would tear the jaw at the observed-face border.
      const front = smooth(mz - 0.14 * scale, mz - 0.035 * scale, z);
      const neck = smooth(chin[1] - 0.04 * scale, chin[1] - 0.005 * scale, y);
      const brow = 1 - smooth(eye[1] + 0.004 * scale, eye[1] + 0.035 * scale, y);
      const mask = front * neck * brow;
      if (mask < 1e-5) continue;
      const struck = smooth(-0.055 * scale, 0.04 * scale, side * (x - mx));
      const cheekWeight = gaussian(x, y, cheek, 0.05, 0.047);
      const contact = gaussian(x, y, [point.x, point.y], 0.032, 0.037);
      const mouthWeight = gaussian(x, y, mouth, 0.053, 0.03);
      const cornerWeight = gaussian(x, y, corner, 0.03, 0.03);
      const lowerFace = 1 - smooth(my - 0.015 * scale, my + 0.035 * scale, y);
      const jaw = lowerFace * Math.exp(-(((x - mx) / (0.115 * scale)) ** 4));
      const eyeWeight = gaussian(x, y, eye, 0.028, 0.024);
      const farEyeWeight = gaussian(x, y, otherEye, 0.028, 0.024);
      const cheekLift = gaussian(x, y, [eye[0], eye[1] - 0.023 * scale], 0.033, 0.026);

      // Broad tissue transport carries the mouth and jaw with the hook. The
      // cheek also flattens inward, with a lifted/bulging rim below the eye.
      let dx = push * (0.015 * cheekWeight + 0.014 * mouthWeight + 0.01 * jaw);
      let dy = 0.008 * cheekLift + 0.004 * cornerWeight * struck;
      let dz = -0.009 * contact - 0.006 * cheekWeight + 0.004 * cheekLift;

      // Rotate the lower face around a jaw hinge, with smooth lip separation.
      // This acts on the full head, so the chin and jaw silhouette move too.
      const lowerLip = smooth(my + 0.003 * scale, my - 0.013 * scale, y);
      const jawWeight = jaw * (1 - mouthWeight) + mouthWeight * lowerLip;
      const angle = 0.17 * jawWeight,
        hy = my + 0.064 * scale,
        hz = mz - 0.085 * scale;
      dy += (y - hy) * (Math.cos(angle) - 1) - (z - hz) * Math.sin(angle);
      dz += (y - hy) * Math.sin(angle) + (z - hz) * (Math.cos(angle) - 1);
      dy -= 0.005 * mouthWeight * (1 - struck); // asymmetric mouth stretch

      // Squeeze each eyelid toward its own eye line instead of translating
      // both lids down. The struck eye reacts more strongly than the far eye.
      const eyeY = eye[1] - 0.003 * scale,
        farEyeY = otherEye[1] - 0.003 * scale;
      dy -= 0.65 * (y - eyeY) * eyeWeight + 0.3 * (y - farEyeY) * farEyeWeight;
      dx += push * 0.004 * eyeWeight;
      dz -= 0.0025 * eyeWeight;
      field[i] = dx * mask * amount;
      field[i + 1] = dy * mask * amount;
      field[i + 2] = dz * mask * amount;
      maximum = Math.max(maximum, Math.hypot(field[i], field[i + 1], field[i + 2]));
    }
    // Scale the complete pose together rather than clipping vertices, which
    // would flatten the silhouette and introduce creases under repeated hits.
    const limit = 0.032 * scale;
    if (maximum > limit)
      for (let i = 0; i < field.length; i++) field[i] *= limit / maximum;
    return field;
  }

  // Uppercut: symmetric chin lift. Chin+jaw rotate around a hinge just above/behind
  // the mouth so the whole jaw silhouette swings up and slightly forward. Lower lip
  // compresses toward the upper lip. Eyes and forehead stay put. No left/right bias.
  _uppercutField(rest, point, speed, softness) {
    const a = this.anchors,
      upper = a[13],
      lower = a[14],
      chin = a[152];
    const mouth = upper.map((v, i) => (v + lower[i]) * 0.5),
      mx = mouth[0],
      my = mouth[1],
      mz = mouth[2];
    const scale = clamp((my - chin[1]) / 0.058, 0.65, 1.5);
    const amount =
      clamp(speed / 1.05, 0.25, 1.35) * (0.78 + clamp(softness, 0, 1) * 0.37) * scale;
    const field = new Float32Array(rest.length);
    const gaussian = (x, y, c, rx, ry) =>
      Math.exp(-(((x - c[0]) / (rx * scale)) ** 2) - ((y - c[1]) / (ry * scale)) ** 2);
    // Hinge sits above and behind the chin so a small negative X-rotation angle
    // sweeps the chin up and forward — the arc of a real jaw swing.
    const hy = my + 0.02 * scale,
      hz = mz - 0.05 * scale;
    const cheek50 = a[50],
      cheek280 = a[280];
    let maximum = 0;
    for (let i = 0; i < rest.length; i += 3) {
      const x = rest[i],
        y = rest[i + 1],
        z = rest[i + 2];
      const front = smooth(mz - 0.14 * scale, mz - 0.03 * scale, z);
      // Include the underside of the jaw and the top of the neck; a hook mask
      // truncates too early because a hook doesn't lift the throat.
      const neck = smooth(chin[1] - 0.055 * scale, chin[1] + 0.005 * scale, y);
      const brow = 1 - smooth(my + 0.028 * scale, my + 0.058 * scale, y);
      const mask = front * neck * brow;
      if (mask < 1e-5) continue;
      // Region weights along the vertical axis: 1 at chin, decays to 0 at brow.
      const chinRegion = 1 - smooth(chin[1] - 0.008 * scale, my + 0.008 * scale, y);
      const lowerLip = gaussian(x, y, [mx, my + 0.006 * scale], 0.038, 0.014);
      const lowerCheekL = gaussian(x, y, [cheek50[0], my + 0.01 * scale], 0.035, 0.028);
      const lowerCheekR = gaussian(
        x,
        y,
        [cheek280[0], my + 0.01 * scale],
        0.035,
        0.028,
      );
      // Jaw hinge rotation. Negative angle swings a point below the hinge up-and-forward.
      const jawSwing =
        chinRegion * 0.75 + Math.max(lowerLip, lowerCheekL + lowerCheekR) * 0.25;
      const angle = -0.22 * jawSwing;
      const yr = y - hy,
        zr = z - hz;
      let dy = yr * (Math.cos(angle) - 1) - zr * Math.sin(angle);
      let dz = yr * Math.sin(angle) + zr * (Math.cos(angle) - 1);
      // Explicit lift and slight forward compression on top of the swing,
      // symmetric around the mouth centerline.
      dy += 0.008 * lowerLip + 0.006 * (lowerCheekL + lowerCheekR);
      dz += -0.004 * lowerLip - 0.003 * (lowerCheekL + lowerCheekR);
      // dx is deliberately zero — an uppercut is symmetric.
      field[i] = 0;
      field[i + 1] = dy * mask * amount;
      field[i + 2] = dz * mask * amount;
      const m = Math.hypot(field[i], field[i + 1], field[i + 2]);
      if (m > maximum) maximum = m;
    }
    const limit = 0.038 * scale;
    if (maximum > limit)
      for (let i = 0; i < field.length; i++) field[i] *= limit / maximum;
    return field;
  }

  // Jab: symmetric nose-and-upper-lip inward push. Nose flattens slightly, upper
  // lip compresses inward. No lateral bias.
  _jabField(rest, point, speed, softness) {
    const a = this.anchors,
      upper = a[13],
      lower = a[14],
      chin = a[152];
    const mouth = upper.map((v, i) => (v + lower[i]) * 0.5),
      mx = mouth[0],
      my = mouth[1],
      mz = mouth[2];
    const scale = clamp((my - chin[1]) / 0.058, 0.65, 1.5);
    const amount =
      clamp(speed / 1.05, 0.25, 1.35) * (0.78 + clamp(softness, 0, 1) * 0.37) * scale;
    const field = new Float32Array(rest.length);
    const gaussian = (x, y, c, rx, ry) =>
      Math.exp(-(((x - c[0]) / (rx * scale)) ** 2) - ((y - c[1]) / (ry * scale)) ** 2);
    // Nose tip roughly halfway between mouth and brow line.
    const nose = [mx, my + 0.035 * scale, mz + 0.01 * scale];
    let maximum = 0;
    for (let i = 0; i < rest.length; i += 3) {
      const x = rest[i],
        y = rest[i + 1],
        z = rest[i + 2];
      const front = smooth(mz - 0.12 * scale, mz - 0.02 * scale, z);
      const brow = 1 - smooth(my + 0.055 * scale, my + 0.08 * scale, y);
      const chinCut = smooth(chin[1] - 0.015 * scale, chin[1] + 0.02 * scale, y);
      const mask = front * brow * chinCut;
      if (mask < 1e-5) continue;
      const noseWeight = gaussian(x, y, nose, 0.028, 0.033);
      const upperLipWeight = gaussian(x, y, [mx, my + 0.005 * scale], 0.036, 0.014);
      const contact = gaussian(x, y, [point.x, point.y], 0.028, 0.032);
      // Symmetric inward push; a hair of downward on nose ridge to sell the flatten.
      const dx = 0;
      const dy = -0.003 * noseWeight;
      const dz = -0.014 * noseWeight - 0.01 * upperLipWeight - 0.008 * contact;
      field[i] = dx * mask * amount;
      field[i + 1] = dy * mask * amount;
      field[i + 2] = dz * mask * amount;
      const m = Math.hypot(field[i], field[i + 1], field[i + 2]);
      if (m > maximum) maximum = m;
    }
    const limit = 0.028 * scale;
    if (maximum > limit)
      for (let i = 0; i < field.length; i++) field[i] *= limit / maximum;
    return field;
  }

  step(dt) {
    if (this.tissue) {
      const elapsed = Number.isFinite(dt) ? Math.max(0, dt) : 0;
      for (const e of this.events) {
        e.age += elapsed;
        if (e.refinement) {
          e.refinement.elapsed += elapsed;
          // A held peak still adopts the finished shape even though simulation
          // time is paused; otherwise it would remain on the preview forever.
          if (elapsed === 0 || e.refinement.elapsed >= 0.04) e.refinement = null;
        }
        const onset = smooth(0, 0.12, e.age),
          delta = onset - e.committed;
        for (let i = 0; i < this.permanent.length; i++)
          this.permanent[i] += e.plastic[i] * delta;
        e.committed = onset;
      }
      this.offset.set(this.permanent);
      const live = this.events.filter((e) => e.mode === 'live'),
        weights = live.map((e) => {
          const contact = this.envelope(e.age),
            reaction = e.reaction ? painEnvelope(e.age) : 0;
          return [
            contact * (1 - reaction),
            contact * reaction,
            (1 - contact) * reaction,
          ];
        });
      const normalization = Math.max(
        1,
        weights.reduce((a, b) => a + b[0] + b[1] + b[2], 0),
      );
      live.forEach((e, index) => {
        const from = e.refinement?.from;
        const mix = from ? smooth(0, 0.04, e.refinement.elapsed) : 1;
        for (let i = 0; i < this.offset.length; i++)
          this.offset[i] +=
            (((from
              ? from.target[i] + (e.target[i] - from.target[i]) * mix
              : e.target[i]) -
              this.permanent[i]) *
              weights[index][0] +
              (e.reaction
                ? ((from
                    ? from.combinedTarget[i] +
                      (e.combinedTarget[i] - from.combinedTarget[i]) * mix
                    : e.combinedTarget[i]) -
                    this.permanent[i]) *
                    weights[index][1] +
                  ((from
                    ? from.reactionTarget[i] +
                      (e.reactionTarget[i] - from.reactionTarget[i]) * mix
                    : e.reactionTarget[i]) -
                    this.permanent[i]) *
                    weights[index][2]
                : 0)) /
            normalization;
      });
      this.events = this.events.filter(
        (e) => e.age < (e.mode === 'clay' ? 0.12 : e.reaction ? 1.65 : 1.02),
      );
      return;
    }
    this.offset.fill(0);
    for (const e of this.events) e.age += Math.max(0, dt);
    this.events = this.events.filter((e) => e.age < 1.02);
    const weights = this.events.map((e) => this.envelope(e.age));
    const normalization = Math.max(
      1,
      weights.reduce((sum, v) => sum + v, 0),
    );
    for (let e = 0; e < this.events.length; e++) {
      const weight = weights[e] / normalization,
        field = this.events[e].field;
      for (let i = 0; i < this.offset.length; i++) this.offset[i] += field[i] * weight;
    }
  }

  reset() {
    this.cancelPreparation();
    this.events = [];
    this.offset.fill(0);
    this.permanent.fill(0);
    this.lastImpact = null;
  }
}
