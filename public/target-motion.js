// Motion evidence ported from jace/cv d0c1763; shared by the existing webcam worker.
(() => {
  // Two evidence streams leave this worker every frame:
  //
  //   1. Hand landmarks (MediaPipe) — precise but fragile: a 5 m/s fist smears ~83 mm at 1/60 s
  //      exposure and the landmarker degrades hard on that blur, precisely in the frames before
  //      contact, which are the ones that matter.
  //   2. Motion blobs — coarse but blur-proof: frame differencing at 96 px wide, connected
  //      components, and a similarity-flow fit (translation + expansion) per blob. Blur *feeds*
  //      this stream. The main thread uses it to corroborate approaches, date reversals when
  //      landmarks are gone, and bridge dropout gaps with low-weight samples.
  const MOTION_W = 96;
  const DIFF_FLOOR = 18; // sensor noise floor on 8-bit grey differences
  const MIN_BLOB_CELLS = 14; // smaller connected components are noise, not fists
  const FLOW_STEP = 3; // sample the flow on every 3rd pixel of a blob
  const FLOW_PATCH = 3; // +/- pixels of context per flow sample (7x7 patch)
  const FLOW_SEARCH = 6; // +/- pixels of three-step search range
  let motionCanvas,
    motionContext,
    previousGrey = null,
    previousStamp = 0;

  function greyOf(bitmap, height) {
    if (!motionCanvas || motionCanvas.height !== height) {
      motionCanvas = new OffscreenCanvas(MOTION_W, height);
      motionContext = motionCanvas.getContext('2d', { willReadFrequently: true });
      previousGrey = null;
    }
    motionContext.drawImage(bitmap, 0, 0, MOTION_W, height);
    const frame = motionContext.getImageData(0, 0, MOTION_W, height).data;
    const grey = new Uint8Array(MOTION_W * height);
    for (let i = 0; i < grey.length; i++)
      grey[i] =
        (frame[i * 4] * 77 + frame[i * 4 + 1] * 150 + frame[i * 4 + 2] * 29) >> 8;
    return grey;
  }

  // Connected components (4-neighbour) over the thresholded difference image, largest two kept.
  function componentsOf(diff, width, height) {
    const labels = new Int32Array(diff.length),
      stack = [];
    const blobs = [];
    for (let start = 0; start < diff.length; start++) {
      if (!diff[start] || labels[start]) continue;
      const id = blobs.length + 1;
      let mass = 0,
        cells = 0,
        sx = 0,
        sy = 0,
        sxx = 0,
        syy = 0,
        minX = width,
        maxX = 0,
        minY = height,
        maxY = 0;
      labels[start] = id;
      stack.push(start);
      while (stack.length) {
        const index = stack.pop(),
          weight = diff[index];
        const x = index % width,
          y = (index - x) / width;
        mass += weight;
        cells++;
        sx += x * weight;
        sy += y * weight;
        sxx += x * x * weight;
        syy += y * y * weight;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (x > 0 && diff[index - 1] && !labels[index - 1]) {
          labels[index - 1] = id;
          stack.push(index - 1);
        }
        if (x < width - 1 && diff[index + 1] && !labels[index + 1]) {
          labels[index + 1] = id;
          stack.push(index + 1);
        }
        if (y > 0 && diff[index - width] && !labels[index - width]) {
          labels[index - width] = id;
          stack.push(index - width);
        }
        if (y < height - 1 && diff[index + width] && !labels[index + width]) {
          labels[index + width] = id;
          stack.push(index + width);
        }
      }
      if (cells >= MIN_BLOB_CELLS)
        blobs.push({ id, mass, cells, sx, sy, sxx, syy, minX, maxX, minY, maxY });
    }
    blobs.sort((a, b) => b.mass - a.mass);
    return { labels, blobs: blobs.slice(0, 2) };
  }

  // Three-step search block matching for one point: the offset in `cur` that best matches the
  // patch around (x, y) in `prev`.
  function matchPoint(prev, cur, width, height, x, y) {
    const sad = (dx, dy) => {
      let total = 0;
      for (let py = -FLOW_PATCH; py <= FLOW_PATCH; py++)
        for (let px = -FLOW_PATCH; px <= FLOW_PATCH; px++) {
          const ax = x + px,
            ay = y + py,
            bx = x + px + dx,
            by = y + py + dy;
          if (
            ax < 0 ||
            ax >= width ||
            ay < 0 ||
            ay >= height ||
            bx < 0 ||
            bx >= width ||
            by < 0 ||
            by >= height
          )
            return Infinity;
          total += Math.abs(prev[ay * width + ax] - cur[by * width + bx]);
        }
      return total;
    };
    let bestX = 0,
      bestY = 0,
      bestCost = sad(0, 0);
    for (
      let step = Math.ceil(FLOW_SEARCH / 2);
      step >= 1;
      step = step > 1 ? Math.ceil(step / 2) : 0
    ) {
      let improved = true;
      while (improved) {
        improved = false;
        for (const [dx, dy] of [
          [step, 0],
          [-step, 0],
          [0, step],
          [0, -step],
        ]) {
          const cost = sad(bestX + dx, bestY + dy);
          if (cost < bestCost) {
            bestCost = cost;
            bestX += dx;
            bestY += dy;
            improved = true;
          }
        }
      }
      if (step === 1) break;
    }
    return { dx: bestX, dy: bestY, cost: bestCost };
  }

  /**
   * Per-blob similarity flow: fit [dx; dy] ~ [a + s(x-cx); b + s(y-cy)] over sparse block matches
   * inside the blob. `s` is relative expansion per frame — the looming signal: positive while the
   * fist approaches, negative the moment it reverses. That sign flip dates a reversal even when
   * every landmark is motion-blurred away.
   */
  function blobFlow(prev, cur, width, height, labels, blob) {
    const cx = blob.sx / blob.mass,
      cy = blob.sy / blob.mass;
    let n = 0,
      sumDx = 0,
      sumDy = 0,
      sumRD = 0,
      sumRR = 0,
      sumX = 0,
      sumY = 0;
    const points = [];
    for (let y = blob.minY; y <= blob.maxY; y += FLOW_STEP)
      for (let x = blob.minX; x <= blob.maxX; x += FLOW_STEP) {
        if (labels[y * width + x] !== blob.id) continue;
        points.push([x, y]);
      }
    for (const [x, y] of points) {
      const { dx, dy, cost } = matchPoint(prev, cur, width, height, x, y);
      if (!Number.isFinite(cost)) continue;
      const rx = x - cx,
        ry = y - cy;
      n++;
      sumDx += dx;
      sumDy += dy;
      sumX += rx;
      sumY += ry;
      sumRD += rx * dx + ry * dy;
      sumRR += rx * rx + ry * ry;
    }
    if (n < 4 || sumRR < 1e-6) return null;
    // Least squares with the radial coordinates re-centred on the sampled points.
    const meanDx = sumDx / n,
      meanDy = sumDy / n,
      meanX = sumX / n,
      meanY = sumY / n;
    const s =
      (sumRD - n * meanX * meanDx - n * meanY * meanDy + 0) /
      Math.max(sumRR - n * (meanX * meanX + meanY * meanY), 1e-6);
    return {
      du: meanDx - s * meanX,
      dv: meanDy - s * meanY,
      scalePerFrame: s,
      samples: n,
    };
  }

  function motionOf(bitmap, timestamp) {
    const height = Math.max(16, Math.round((MOTION_W * bitmap.height) / bitmap.width));
    const grey = greyOf(bitmap, height);
    if (!previousGrey || previousGrey.length !== grey.length) {
      previousGrey = grey;
      previousStamp = timestamp;
      return null;
    }
    const diff = new Uint8Array(grey.length);
    let total = 0,
      sx = 0,
      sy = 0,
      peak = 0;
    for (let i = 0; i < grey.length; i++) {
      const difference = Math.abs(grey[i] - previousGrey[i]);
      if (difference < DIFF_FLOOR) continue;
      diff[i] = difference;
      total += difference;
      const x = i % MOTION_W;
      sx += x * difference;
      sy += ((i - x) / MOTION_W) * difference;
      if (difference > peak) peak = difference;
    }
    const dtSeconds = Math.max((timestamp - previousStamp) / 1000, 1e-3);
    const { labels, blobs } =
      total > 0 ? componentsOf(diff, MOTION_W, height) : { labels: null, blobs: [] };
    const out = [];
    for (const blob of blobs) {
      const cx = blob.sx / blob.mass,
        cy = blob.sy / blob.mass;
      const varX = Math.max(blob.sxx / blob.mass - cx * cx, 0),
        varY = Math.max(blob.syy / blob.mass - cy * cy, 0);
      const flow = blobFlow(previousGrey, grey, MOTION_W, height, labels, blob);
      out.push({
        u: (cx + 0.5) / MOTION_W,
        v: (cy + 0.5) / height,
        mass: blob.mass / (grey.length * 255),
        area: blob.cells / grey.length,
        spread: Math.sqrt(varX + varY) / MOTION_W,
        du: flow ? flow.du / MOTION_W / dtSeconds : 0, // normalised image units per second
        dv: flow ? flow.dv / height / dtSeconds : 0,
        expand: flow ? flow.scalePerFrame / dtSeconds : 0, // relative expansion per second; + = approaching
        flowSamples: flow?.samples ?? 0,
      });
    }
    previousGrey = grey;
    previousStamp = timestamp;
    if (total <= 0) return { energy: 0, x: 0.5, y: 0.5, peak: 0, blobs: out };
    return {
      energy: total / (grey.length * 255),
      x: (sx / total + 0.5) / MOTION_W,
      y: (sy / total + 0.5) / height,
      peak: peak / 255,
      blobs: out,
    };
  }

  self.TargetMotion = { motionOf, componentsOf, blobFlow, matchPoint, MOTION_W };
})();
