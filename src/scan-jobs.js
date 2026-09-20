import { discoverScanJobs } from './model-library.js';
import './scan-jobs.css';

const STORAGE_KEY = 'punching-face-background-jobs';
const jobs = new Map();
let notify = () => {};
const keyOf = ({ id, engine }) => `${engine}:${id}`;
const pending = (job) => job.status === 'running';

function persist() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([...jobs.values()].map(({ timer, polling, ...job }) => job)),
    );
  } catch {}
}

function changed() {
  persist();
  document.body.classList.toggle('scan-processing', [...jobs.values()].some(pending));
  notify();
}

// One lightweight status request per active job, independent of the open dialog or head.
async function poll(job) {
  if (job.polling || !pending(job) || jobs.get(keyOf(job)) !== job) return;
  clearTimeout(job.timer);
  job.polling = true;
  let delay = 2000;
  try {
    const endpoint = job.engine === 'meshy' ? 'meshy-job' : 'face-status';
    const response = await fetch(`/api/${endpoint}?id=${encodeURIComponent(job.id)}`, {
      signal: AbortSignal.timeout(20000),
    });
    const state = await response.json();
    if (!response.ok)
      throw new Error(state.error || 'Could not reach the scan server.');
    if (jobs.get(keyOf(job)) !== job) return;
    job.progress = Number.isFinite(state.progress)
      ? Math.min(100, Math.max(0, state.progress))
      : null;
    job.message = state.message || state.stage || 'Building your head…';
    job.stage = state.stage || '';
    job.retries = 0;
    const saved = job.engine === 'meshy' ? state.model : state.photoModel;
    if (state.status === 'complete' && saved) {
      job.status = 'complete';
      job.progress = 100;
      window.dispatchEvent(new CustomEvent('punching-face-library-changed'));
    } else if (state.status === 'failed' || state.status === 'idle') {
      job.status = 'failed';
      if (state.status === 'idle')
        job.message = 'This build is no longer running. Open the scan to retry.';
    }
  } catch (error) {
    if (jobs.get(keyOf(job)) !== job) return;
    job.retries = (job.retries || 0) + 1;
    job.message = 'Reconnecting to the scan server…';
    delay = Math.min(20000, 4000 * job.retries);
  } finally {
    job.polling = false;
    if (jobs.get(keyOf(job)) === job) {
      changed();
      if (pending(job)) job.timer = setTimeout(() => void poll(job), delay);
    }
  }
}

export function trackScanJob({ id, engine = 'local', resume = false }) {
  if (!id || !['local', 'meshy'].includes(engine)) return;
  const key = keyOf({ id, engine });
  const existing = jobs.get(key);
  if (existing && pending(existing)) return;
  if (existing) clearTimeout(existing.timer);
  const job = {
    id,
    engine,
    status: 'running',
    progress: null,
    message: 'Building your head…',
    acknowledged: false,
  };
  jobs.set(key, job);
  changed();
  if (!resume)
    window.dispatchEvent(
      new CustomEvent('punching-face-scan-started', { detail: { id, engine } }),
    );
  void poll(job);
}

export function acknowledgeScanJob(model) {
  const job = jobs.get(keyOf(model));
  if (!job) return;
  job.acknowledged = true;
  changed();
}

export function forgetScanJobs(id) {
  for (const [key, job] of jobs) {
    if (job.id !== id) continue;
    clearTimeout(job.timer);
    jobs.delete(key);
  }
  changed();
}

export function installScanJobs({
  onContinue,
  onOpenScan,
  onSwap,
  onPrompt,
  canPrompt,
}) {
  const panel = document.createElement('aside');
  panel.id = 'scan-jobs';
  panel.setAttribute('aria-label', 'Background scans');
  panel.hidden = true;
  const dialog = document.createElement('dialog');
  dialog.id = 'scan-ready-dialog';
  dialog.setAttribute('aria-labelledby', 'scan-ready-title');
  dialog.innerHTML = /* HTML */ `
    <span class="scan-ready-mark" aria-hidden="true">✓</span>
    <h1 id="scan-ready-title">Your head is ready.</h1>
    <p>Saved to your preloaded models.</p>
    <p id="scan-ready-status" role="status"></p>
    <div class="scan-ready-actions">
      <button data-scan-action="swap">Swap head</button>
      <button data-scan-action="later">Keep playing</button>
    </div>
  `;
  document.body.append(panel, dialog);
  let shown = null;
  let swapping = false;
  const status = dialog.querySelector('#scan-ready-status');
  const rows = new Map();

  function prompt(job) {
    if (dialog.open || swapping || document.hidden || !canPrompt()) return;
    shown = job;
    status.textContent = job.engine === 'meshy' ? 'Meshy head' : 'Scanned head';
    onPrompt(true);
    dialog.showModal();
  }

  function checkReady() {
    const ready = [...jobs.values()].find(
      (job) => job.status === 'complete' && !job.acknowledged,
    );
    if (ready) prompt(ready);
  }

  function render() {
    panel.hidden = !jobs.size;
    for (const [key, row] of rows) {
      if (!jobs.has(key)) {
        row.remove();
        rows.delete(key);
      }
    }
    for (const [key, job] of jobs) {
      let row = rows.get(key);
      if (!row) {
        row = document.createElement('section');
        row.className = 'scan-job';
        row.innerHTML = /* HTML */ `<div class="scan-job-heading">
            <strong></strong
            ><button class="scan-job-dismiss" aria-label="Hide scan status">×</button>
          </div>
          <p role="status"></p>
          <progress max="100" aria-label="Scan progress"></progress
          ><button class="scan-job-action"></button>`;
        row.querySelector('.scan-job-dismiss').onclick = () => {
          if (pending(jobs.get(key))) return;
          jobs.delete(key);
          changed();
        };
        row.querySelector('.scan-job-action').onclick = () => {
          const current = jobs.get(key);
          if (current.status === 'complete') prompt(current);
          else if (pending(current)) onContinue();
          else onOpenScan(current);
        };
        rows.set(key, row);
        panel.append(row);
      }
      const label = job.engine === 'meshy' ? 'Meshy' : 'Scan';
      row.querySelector('strong').textContent =
        job.status === 'complete'
          ? `${label} saved ✓`
          : job.status === 'failed'
            ? `${label} failed`
            : `${label} processing${job.progress === null ? '' : ` · ${Math.round(job.progress)}%`}`;
      const message = row.querySelector('p');
      message.textContent =
        job.status === 'complete' ? 'Ready whenever you are.' : job.message;
      message.title = message.textContent;
      const progress = row.querySelector('progress');
      progress.hidden = !pending(job);
      if (job.progress === null) progress.removeAttribute('value');
      else progress.value = job.progress;
      row.querySelector('.scan-job-dismiss').hidden = pending(job);
      row.querySelector('.scan-job-action').textContent =
        job.status === 'complete'
          ? 'Swap head'
          : pending(job)
            ? 'Use another head →'
            : 'Open scan';
    }
    checkReady();
  }

  function dismiss() {
    if (swapping) return;
    if (shown) acknowledgeScanJob(shown);
    dialog.close();
  }
  dialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    dismiss();
  });
  dialog.addEventListener('close', () => {
    shown = null;
    onPrompt(false);
    queueMicrotask(checkReady);
  });
  dialog.querySelector('[data-scan-action="later"]').onclick = dismiss;
  dialog.querySelector('[data-scan-action="swap"]').onclick = async () => {
    if (!shown || swapping) return;
    swapping = true;
    const job = shown;
    for (const button of dialog.querySelectorAll('button')) button.disabled = true;
    status.textContent = 'Loading your head…';
    try {
      await onSwap(job);
      acknowledgeScanJob(job);
      dialog.close();
    } catch (error) {
      status.textContent =
        error.message || 'Could not load this head. It is still saved; you can retry.';
    } finally {
      swapping = false;
      for (const button of dialog.querySelectorAll('button')) button.disabled = false;
    }
  };
  notify = render;
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    for (const item of saved) {
      if (
        !item.id ||
        !['local', 'meshy'].includes(item.engine) ||
        !['running', 'complete', 'failed'].includes(item.status)
      )
        continue;
      jobs.set(keyOf(item), { ...item, polling: false });
    }
  } catch {}
  changed();
  for (const job of jobs.values()) if (pending(job)) void poll(job);
  void discoverScanJobs()
    .then((running) => {
      for (const job of running) trackScanJob({ ...job, resume: true });
    })
    .catch(() => {});
  document.addEventListener('visibilitychange', checkReady);
  // A completed job waits until camera capture or another model load is finished.
  document.addEventListener('close', checkReady, true);
  window.addEventListener('punching-face-model-loaded', () =>
    queueMicrotask(checkReady),
  );
  return {
    checkReady,
    get isOpen() {
      return dialog.open;
    },
  };
}
