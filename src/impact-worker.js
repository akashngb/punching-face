import { FaceImpactRig } from './impact-rig.js';
import { impactCacheKey, restContact } from './impact-cache.js';

let rig = null;
function emit(payload) {
  const buffers = payload.event
    ? [
        ...new Set(
          Object.values(payload.event)
            .filter((value) => ArrayBuffer.isView(value))
            .map((value) => value.buffer),
        ),
      ]
    : [];
  self.postMessage(payload, buffers);
  rig.events = [];
}
self.onmessage = ({ data }) => {
  if (data.type === 'init') {
    try {
      rig = new FaceImpactRig(data.rest, data.anchors, data.topology);
      // Exact standard camera/button contacts are ready before the head becomes
      // interactive. No quantized strengths, reduced mesh, or weaker solve.
      const keys = new Set();
      for (const [anchor, incoming] of [
        [50, [0.75, -0.05, -0.6]],
        [280, [-0.75, -0.05, -0.6]],
        [152, [0, 0.85, -0.5]],
        [1, [0, 0, -1]],
      ]) {
        const point =
          data.contactAnchors?.[anchor] ??
          rig.anchors[anchor] ??
          (anchor === 1 ? [0, 0.005, 0.062] : null);
        if (!point) continue;
        const length = Math.hypot(...incoming),
          direction = incoming.map((value) => value / length);
        const direct = rig.tissue.vertices[rig.tissue.nearest(point).node].p;
        for (const location of [direct, restContact(rig.tissue, point, direction)])
          for (const magnitude of [0.85, 0.9]) {
            const input = { location, direction, magnitude };
            const key = impactCacheKey(input, 0.75, true);
            if (keys.has(key)) continue;
            keys.add(key);
            rig.reset();
            const affected = rig.impact(rig.tissue.rest, input, 0.75);
            if (affected)
              emit({
                type: 'cache',
                epoch: data.epoch,
                key,
                affected,
                event: rig.events[0],
                impact: rig.lastImpact,
                milliseconds: 0,
              });
          }
      }
      rig.reset();
      self.postMessage({ type: 'ready', epoch: data.epoch });
    } catch (error) {
      rig = null;
      self.postMessage({ type: 'ready', epoch: data.epoch, error: error.message });
    }
    return;
  }
  if (data.type !== 'impact') return;
  try {
    if (!rig) throw new Error('Impact geometry unavailable');
    rig.reactionEnabled = data.reactionEnabled;
    const started = performance.now();
    // The first pose retains displacement/gradient/triangle guards. Only the
    // expensive differential refinement waits for the second message.
    for (const stage of ['preview', 'complete']) {
      rig.reset();
      const affected = rig.impact(
        rig.tissue.rest,
        data.input,
        data.softness,
        rig.tissue.rest,
        { refine: stage === 'complete' },
      );
      const event = rig.events[0];
      emit({
        epoch: data.epoch,
        id: data.id,
        stage,
        affected,
        event,
        impact: rig.lastImpact,
        milliseconds: performance.now() - started,
      });
    }
  } catch (error) {
    self.postMessage({ epoch: data.epoch, id: data.id, error: error.message });
  }
};
