// Button punches have a known target and direction. Their contact must not depend
// on whether a rendered hand happened to cross the skin in one animation frame.
export function createDemoPunch(dynamics, { side, type = 'hook', magnitude }) {
  const anchor =
    dynamics.impactRig.anchors[type === 'uppercut' ? 152 : side < 0 ? 50 : 280];
  const surface = dynamics.impactRig.tissue.nearest(anchor);
  const direction = type === 'uppercut' ? [0, 0.85, -0.5] : [-side * 0.75, -0.05, -0.6];
  const length = Math.hypot(...direction);
  return {
    side,
    type,
    magnitude,
    vertex: surface.index,
    direction: direction.map((v) => v / length),
    contacted: false,
  };
}

export function takeDemoContact(dynamics, punch, progress) {
  // Also fires if a slow frame skips the whole contact interval. The return
  // motion cannot produce a second punch.
  if (punch.contacted || progress < 0.52) return null;
  punch.contacted = true;
  return {
    location: Array.from(
      dynamics.geometry.attributes.position.array.slice(
        punch.vertex * 3,
        punch.vertex * 3 + 3,
      ),
    ),
    direction: punch.direction,
    magnitude: punch.magnitude,
  };
}
