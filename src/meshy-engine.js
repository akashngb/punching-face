// Reconstruction engine choice for the head-scan dialog: the PunchingFace pipeline on this computer, or
// Meshy's cloud multi-image-to-3D (server side: meshy_backend.py). One scan can hold a model from each, so
// switching engines on a finished scan compares the two; it never rebuilds or spends credits on its own.
import './meshy-engine.css';

const $ = (id) => document.getElementById(id);
const ENGINE_KEY = 'punching-face-engine';
const ACTIVE_KEY = 'punching-face-active-meshy';
const MODEL_NAME = 'Your Meshy head';
const ROLES = {
  front: 'Front',
  'side-a': 'Side A',
  'side-b': 'Side B',
  far: 'Far side',
};

async function api(path, data) {
  const response = await fetch(
    '/api/' + path,
    data === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        },
  );
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      result.error || 'The local server could not complete this request.',
    );
  return result;
}

const stored = (storage, key) => {
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
};
const store = (storage, key, value) => {
  try {
    value === null ? storage.removeItem(key) : storage.setItem(key, value);
  } catch {}
};

// The scene already knows how to take a third-party head: main.js normalises an uploaded GLB and finds its
// facial anchors. Handing the Meshy model to that same input keeps one import path instead of two.
async function importGLB(bytes) {
  const input = $('face-file'),
    overlay = $('busy'),
    file = new File([bytes], MODEL_NAME + '.glb', { type: 'model/gltf-binary' });
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  input.dispatchEvent(new Event('change'));
  if (overlay?.classList.contains('active'))
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        observer.disconnect();
        reject(new Error('Loading the Meshy model timed out.'));
      }, 60000);
      const observer = new MutationObserver(() => {
        if (overlay.classList.contains('active')) return;
        clearTimeout(timer);
        observer.disconnect();
        resolve();
      });
      observer.observe(overlay, { attributes: true, attributeFilter: ['class'] });
    });
  if ($('model-name').textContent !== file.name)
    throw new Error('The Meshy model could not be loaded into the scene.');
  $('model-name').textContent = MODEL_NAME;
  const scene = $('scene-name').firstChild;
  if (scene) scene.textContent = MODEL_NAME;
  $('model-kind').textContent = 'Meshy AI head · textured';
  $('physics-engine').textContent =
    'Meshy import · preview springs + facial impact rig';
  $('photo-count').textContent = 'Meshy cloud reconstruction';
}

async function fetchModel(id) {
  const response = await fetch(`/api/meshy-asset?id=${id}&asset=model.glb`);
  if (!response.ok) {
    let message = 'The Meshy model is not available.';
    try {
      message = (await response.json()).error || message;
    } catch {}
    throw new Error(message);
  }
  return response.arrayBuffer();
}

// A reload (Vite reloads on every save) would otherwise fall back to the local model. Wait until the scene's
// own recovery has finished, then put the Meshy head back.
async function restoreActive() {
  const id = stored(sessionStorage, ACTIVE_KEY);
  if (!id) return;
  try {
    await new Promise((resolve) => {
      const started = performance.now();
      const check = () =>
        (window.__labReady && !$('busy')?.classList.contains('active')) ||
        performance.now() - started > 60000
          ? resolve()
          : setTimeout(check, 250);
      check();
    });
    if (stored(sessionStorage, ACTIVE_KEY) !== id) return;
    await importGLB(await fetchModel(id));
    watchActive();
  } catch (e) {
    store(sessionStorage, ACTIVE_KEY, null);
    console.info('Meshy head not restored:', e.message);
  }
}

let nameObserver;
// Any other model taking the scene (a local scan, an upload, the reference head) ends the Meshy session.
function watchActive() {
  nameObserver?.disconnect();
  nameObserver = new MutationObserver(() => {
    if ($('model-name').textContent === MODEL_NAME) return;
    nameObserver.disconnect();
    store(sessionStorage, ACTIVE_KEY, null);
  });
  nameObserver.observe($('model-name'), {
    childList: true,
    characterData: true,
    subtree: true,
  });
}

export class EngineChoice {
  constructor(capture) {
    this.capture = capture; // the FaceCapture that owns the dialog and its buttons
    this.engine = stored(localStorage, ENGINE_KEY) === 'meshy' ? 'meshy' : 'local';
    this.status = null;
    this.job = null;
    this.awaiting = null;
    $('face-cloud-review')
      .closest('label')
      .insertAdjacentHTML(
        'beforebegin',
        /* HTML */ `<fieldset id="engine-choice" class="engine-choice">
            <legend>Reconstruction engine</legend>
            <label class="engine-option"
              ><input type="radio" name="face-engine" value="local" /><span
                ><strong>PunchingFace pipeline</strong
                ><small
                  >Runs on this computer: fitted head template, hair strands, glasses
                  and a Newton physics cage. Needs 24 or more views.</small
                ></span
              ></label
            ><label class="engine-option"
              ><input type="radio" name="face-engine" value="meshy" /><span
                ><strong>Meshy cloud</strong
                ><small id="engine-meshy-note"
                  >Sends up to four cropped views of this scan (front, both sides, far
                  side) to Meshy's image-to-3D service. Works from a single front
                  view.</small
                ></span
              ></label
            >
          </fieldset>
          <div id="engine-meshy-panel" hidden>
            <p class="note" id="engine-meshy-state" role="status">Checking Meshy…</p>
            <div
              id="engine-meshy-views"
              class="engine-views"
              aria-label="Views sent to Meshy"
              hidden
            ></div>
            <details id="engine-meshy-settings">
              <summary>Meshy API settings</summary>
              <input
                id="engine-meshy-key"
                aria-label="Meshy API key"
                type="password"
                autocomplete="off"
                placeholder="Meshy API key (msy_…)"
              /><button id="engine-meshy-save">Save key on server</button
              ><button id="engine-meshy-test">Test connection</button>
              <p class="muted">
                Create a key at meshy.ai under Settings, API. It stays on this computer
                (MESHY_API_KEY in .env, or .local/secrets) and is never sent back to the
                browser or written into exports.
              </p>
            </details>
          </div>`,
      );
    for (const radio of document.querySelectorAll('input[name="face-engine"]'))
      radio.onchange = () => this.choose(radio.value).catch((e) => capture.fail(e));
    $('engine-meshy-save').onclick = () => this.saveKey();
    $('engine-meshy-test').onclick = () => this.refresh(true);
    this.render();
    restoreActive();
  }

  get meshy() {
    return this.engine === 'meshy';
  }

  get minimumViews() {
    return this.meshy ? 1 : 24;
  }

  async choose(engine) {
    this.engine = engine === 'meshy' ? 'meshy' : 'local';
    store(localStorage, ENGINE_KEY, this.engine);
    this.job = null;
    this.awaiting = null;
    this.render();
    const c = this.capture;
    c.ready = false;
    c.jobRunning = false;
    c.controls();
    if (this.meshy && !this.status) await this.refresh();
    if (c.id && !c.running) await c.poll();
    else if (!this.meshy)
      $('face-job-state').textContent = 'No reconstruction started.';
  }

  async refresh(balance = false) {
    const state = $('engine-meshy-state');
    if (balance) $('engine-meshy-test').disabled = true;
    try {
      this.status = await api('meshy-status' + (balance ? '?balance=1' : ''));
    } catch (e) {
      // A page newer than the running API server: server.py is not reloaded by the dev runner.
      this.status = {
        configured: false,
        error:
          e.message === 'Unknown endpoint'
            ? 'The local API server started before the Meshy engine was added. Restart npm run dev to enable it.'
            : e.message,
      };
    } finally {
      $('engine-meshy-test').disabled = false;
    }
    this.render();
    if (balance && this.status.configured && this.meshy && !this.capture.jobRunning)
      state.textContent = this.status.connected
        ? `Meshy connection verified${Number.isFinite(this.status.balance) ? ` · ${this.status.balance.toLocaleString()} credits available` : ''}.`
        : this.status.error || 'Meshy connection not verified.';
  }

  async saveKey() {
    const field = $('engine-meshy-key');
    try {
      this.status = await api('meshy-config', { apiKey: field.value.trim() });
      field.value = '';
      this.render();
      $('engine-meshy-state').textContent =
        this.status.source === 'environment'
          ? 'Key saved, but MESHY_API_KEY from .env or the environment takes precedence. Edit .env to change the key in use.'
          : 'Meshy key saved on this server. Test connection shows the credit balance.';
    } catch (e) {
      $('engine-meshy-state').textContent = e.message;
    }
  }

  render() {
    const s = this.status,
      c = this.capture;
    for (const radio of document.querySelectorAll('input[name="face-engine"]'))
      radio.checked = radio.value === this.engine;
    $('face-scan-dialog').classList.toggle('engine-meshy', this.meshy);
    $('engine-meshy-panel').hidden = !this.meshy;
    const cost = `about ${s?.estimatedCredits ?? 30} credits`;
    $('face-scan-build').textContent = !this.meshy
      ? 'Create 3D face'
      : c.ready
        ? `Rebuild with Meshy (${cost})`
        : this.job?.resumable
          ? 'Resume Meshy build (no new credits)'
          : `Create 3D face with Meshy (${cost})`;
    $('face-scan-load').textContent = this.meshy ? 'Load Meshy head' : 'Load face';
    if (!this.meshy || !s || c.jobRunning) return;
    if (!s.configured) $('engine-meshy-settings').open = true;
    $('engine-meshy-state').textContent = s.error
      ? s.error
      : !s.configured
        ? 'No Meshy API key on this server yet. Paste one under Meshy API settings, or add MESHY_API_KEY to .env and restart.'
        : `Meshy key configured (${s.source === 'environment' ? 'from .env' : 'saved on this server'}). Each build costs ${cost} and takes one to three minutes.`;
  }

  showViews(id, job) {
    const strip = $('engine-meshy-views'),
      views = job.views?.filter((v) => v.filename) ?? [];
    strip.hidden = !views.length;
    strip.replaceChildren(
      ...views.map((view, n) => {
        const figure = document.createElement('figure'),
          image = document.createElement('img'),
          caption = document.createElement('figcaption');
        image.src = `/api/meshy-asset?id=${id}&asset=view-${n}.png&task=${job.taskId ?? ''}`;
        image.alt = `${ROLES[view.role] ?? 'View'} sent to Meshy`;
        image.loading = 'lazy';
        caption.textContent = ROLES[view.role] ?? 'View';
        figure.append(image, caption);
        return figure;
      }),
    );
  }

  async build() {
    const c = this.capture;
    await c.stop();
    if (
      c.ready &&
      !confirm(
        'Rebuild with Meshy? This spends credits again. The current Meshy head is replaced only when the new one finishes.',
      )
    )
      return;
    const started = await api('meshy-train', { id: c.id, rebuild: c.ready });
    this.awaiting = started.reused ? null : c.id;
    c.jobRunning = started.status === 'running';
    c.autoLoaded = null;
    c.controls();
    await this.poll(c.id);
  }

  async poll(id) {
    const c = this.capture;
    clearTimeout(c.pollTimer);
    const job = await api('meshy-job?id=' + id);
    if (id !== c.id || !this.meshy) return;
    this.job = job;
    c.jobRunning = job.status === 'running';
    c.ready = job.model === true;
    c.showTiming(null, null);
    this.showViews(id, job);
    $('face-job-state').textContent =
      job.status === 'idle'
        ? c.ready
          ? 'A Meshy head is saved for this scan.'
          : 'No Meshy build for this scan yet.'
        : job.status === 'failed' && c.ready
          ? job.message + ' The earlier Meshy head is still saved.'
          : job.message;
    this.render();
    c.controls();
    if (c.jobRunning)
      c.pollTimer = setTimeout(
        () =>
          c.poll(id).catch((e) => {
            $('face-job-state').textContent = e.message;
            c.pollTimer = setTimeout(() => c.poll(id).catch((e) => c.fail(e)), 4000);
          }),
        2000,
      );
    else if (this.awaiting === id) {
      this.awaiting = null;
      if (job.status === 'complete' && c.ready) await c.load();
    }
  }

  async load() {
    const c = this.capture,
      id = c.id;
    c.loading = true;
    c.controls();
    try {
      await importGLB(await fetchModel(id));
      c.autoLoaded = id;
      store(sessionStorage, ACTIVE_KEY, id);
      watchActive();
      $('face-job-state').textContent =
        'Meshy head loaded into the interaction scene. Close this panel to inspect it and try a hook.';
    } finally {
      c.loading = false;
      c.controls();
    }
  }
}
