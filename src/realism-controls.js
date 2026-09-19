import * as THREE from 'three';
import './realism.css';

function canvasFrom(image) {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext('2d').drawImage(image, 0, 0);
  return canvas;
}

export function installRealismControls({ getTarget, changed, setView, toast }) {
  const panel = document.createElement('section');
  panel.className = 'panel-section realism-panel';
  panel.innerHTML = /* HTML */ ` <div class="realism-heading">
      <h2>Realism</h2>
      <span>3D CLEANUP</span>
    </div>
    <p class="muted">
      Soften photo crop seams and patchy colour while keeping facial detail.
    </p>
    <button id="realism-apply" class="full primary" disabled>✧ Enhance realism</button>
    <div id="realism-adjustments" hidden>
      <label class="controls-label" for="realism-strength"
        >Strength <output id="realism-strength-value">75%</output></label
      >
      <input
        id="realism-strength"
        aria-label="Realism strength"
        type="range"
        min="0"
        max="100"
        step="5"
        value="75"
      />
      <div class="realism-comparison" role="group" aria-label="Realism comparison">
        <button id="realism-before" class="small" aria-pressed="false">Original</button>
        <button id="realism-after" class="small active" aria-pressed="true">
          Enhanced
        </button>
      </div>
      <button id="realism-reset" class="small full">↶ Remove realism pass</button>
    </div>
    <p id="realism-status" class="muted" role="status" aria-live="polite">
      Load a textured head to begin.
    </p>
    <p class="realism-note">
      Local texture repair. Unseen facial detail remains an estimate.
    </p>`;
  document.querySelector('.panel.left .panel-section').after(panel);
  const $ = (id) => panel.querySelector('#' + id);
  let target = null,
    baseline = null,
    original = null,
    enhanced = null,
    output = null,
    texture = null,
    report = null,
    worker = null,
    pendingReject = null,
    before = false,
    originalReport = null,
    renderFrame = null;

  function writeReport(value) {
    if (!target) return;
    for (const stats of [target.stats, target.atlas?.stats])
      if (stats) {
        if (value) stats.realism = value;
        else delete stats.realism;
      }
    changed();
  }
  function cancel() {
    worker?.terminate();
    worker = null;
    pendingReject?.(new Error('Realism cancelled because the model changed.'));
    pendingReject = null;
    if (renderFrame !== null) cancelAnimationFrame(renderFrame);
    renderFrame = null;
  }
  function reset() {
    cancel();
    if (target && baseline) {
      if (target.material.bumpMap === texture) target.material.bumpMap = baseline;
      target.material.map = baseline;
      target.material.needsUpdate = true;
      writeReport(originalReport);
    }
    texture?.dispose();
    target = baseline = original = enhanced = output = texture = report = null;
    originalReport = null;
    before = false;
    $('realism-adjustments').hidden = true;
    $('realism-apply').textContent = '✧ Enhance realism';
    $('realism-apply').disabled = true;
  }
  function update() {
    const next = getTarget(target);
    if (next?.material !== target?.material) {
      reset();
      target = next;
      baseline = target?.material.map ?? null;
      originalReport = target?.stats?.realism ?? target?.atlas?.stats?.realism ?? null;
      $('realism-status').textContent = baseline
        ? 'Ready · original texture is kept for comparison.'
        : 'Load a textured head to begin.';
      $('realism-apply').disabled = !baseline;
    }
  }
  function render() {
    if (!enhanced || !target) return;
    const strength = Number($('realism-strength').value) / 100;
    const ctx = output.getContext('2d');
    ctx.globalAlpha = 1;
    ctx.drawImage(original, 0, 0);
    if (!before && strength > 0) {
      ctx.globalAlpha = strength;
      ctx.drawImage(enhanced, 0, 0);
      ctx.globalAlpha = 1;
    }
    texture.needsUpdate = true;
    // Keep the untouched map attached in Original mode for exact comparison.
    const active = before || strength === 0 ? baseline : texture;
    if (target.material.bumpMap === baseline || target.material.bumpMap === texture)
      target.material.bumpMap = active;
    target.material.map = active;
    target.material.needsUpdate = true;
    $('realism-strength-value').textContent = `${Math.round(strength * 100)}%`;
    $('realism-before').setAttribute('aria-pressed', String(before));
    $('realism-after').setAttribute('aria-pressed', String(!before));
    $('realism-before').classList.toggle('active', before);
    $('realism-after').classList.toggle('active', !before);
    writeReport(before || strength === 0 ? originalReport : { ...report, strength });
    $('realism-status').textContent = before
      ? 'Viewing original · select Enhanced to return.'
      : `${Math.round(strength * 100)}% strength · shape and fine texture retained.`;
  }
  async function apply() {
    update();
    if (!target || !baseline || worker || enhanced) return;
    const active = target;
    $('realism-apply').disabled = true;
    $('realism-apply').textContent = 'Enhancing…';
    try {
      original = canvasFrom(baseline.image);
      const ctx = original.getContext('2d');
      const pixels = ctx.getImageData(0, 0, original.width, original.height).data;
      const result = await new Promise((resolve, reject) => {
        pendingReject = reject;
        worker = new Worker(new URL('./realism-worker.js', import.meta.url), {
          type: 'module',
        });
        worker.onmessage = ({ data }) => {
          if (data.progress) {
            $('realism-status').textContent = data.progress;
            return;
          }
          data.error ? reject(new Error(data.error)) : resolve(data);
        };
        worker.onerror = () =>
          reject(
            new Error(
              'The realism pass could not finish. Your original texture is retained.',
            ),
          );
        worker.postMessage(
          {
            positions: Array.from(active.positions),
            mapping: active.atlas.mapping,
            indices: active.atlas.indices,
            uv: active.atlas.uv,
            landmarks: active.landmarks,
            protectedVertices: active.protectedVertices,
            flipY: baseline.flipY,
            pixels,
            width: original.width,
            height: original.height,
          },
          [pixels.buffer],
        );
      });
      if (target !== active) return;
      report = result.report;
      enhanced = document.createElement('canvas');
      enhanced.width = original.width;
      enhanced.height = original.height;
      enhanced
        .getContext('2d')
        .putImageData(
          new ImageData(result.pixels, original.width, original.height),
          0,
          0,
        );
      output = canvasFrom(original);
      texture = baseline.clone();
      // A Texture clone shares Source with its parent: replacing Source avoids
      // accidentally overwriting the supposedly untouched original.
      texture.source = new THREE.Source(output);
      texture.needsUpdate = true;
      before = false;
      $('realism-adjustments').hidden = false;
      $('realism-apply').textContent = '✓ Realism pass ready';
      setView('mesh');
      render();
    } catch (error) {
      if (target === active) {
        $('realism-status').textContent = error.message;
        $('realism-apply').textContent = 'Retry realism pass';
        $('realism-apply').disabled = false;
        toast(error.message);
      }
    } finally {
      if (target === active) {
        worker?.terminate();
        worker = null;
        pendingReject = null;
      }
    }
  }
  $('realism-apply').onclick = apply;
  $('realism-strength').oninput = () => {
    before = false;
    if (renderFrame !== null) cancelAnimationFrame(renderFrame);
    renderFrame = requestAnimationFrame(() => {
      renderFrame = null;
      render();
    });
  };
  $('realism-before').onclick = () => {
    before = true;
    render();
  };
  $('realism-after').onclick = () => {
    before = false;
    render();
  };
  $('realism-reset').onclick = () => {
    reset();
    update();
    toast('Original texture restored.');
  };
  return {
    update,
    reset,
    apply,
    snapshot() {
      if (!enhanced) return null;
      return {
        version: 1,
        original: original.toDataURL('image/png'),
        originalReport,
        strength: Number($('realism-strength').value),
        before,
      };
    },
    async restore(saved) {
      if (!saved || saved.version !== 1) return;
      // The physics connection yields while accessories are being restored.
      // Refresh after that work so cached protection includes the saved hair.
      reset();
      update();
      if (!target) return;
      const active = target;
      const restored = await new THREE.TextureLoader().loadAsync(saved.original);
      if (target !== active) {
        restored.dispose();
        return;
      }
      baseline.source = restored.source;
      baseline.needsUpdate = true;
      originalReport = saved.originalReport ?? null;
      $('realism-strength').value = Math.max(
        0,
        Math.min(100, Number(saved.strength) || 0),
      );
      await apply();
      before = saved.before === true;
      render();
    },
  };
}
