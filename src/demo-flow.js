import { installDemoHUD } from './demo-hud.js';
import { listSavedHeads } from './model-library.js';
import { installScanJobs, acknowledgeScanJob } from './scan-jobs.js';
import './demo-flow.css';

// The demo is a small UI layer over the same model, camera and physics controls.
export function installDemoFlow(api) {
  const $ = (id) => document.getElementById(id);
  let screen = 'welcome';
  let selection = 'demo';
  let selectedCapture = null;
  let loading = false;
  let awaitingImport = false;
  let calibrationTimer = null;
  let cameraStarting = false;
  let keyboardOnly = false;
  let focused = false;
  let scanPromptOpen = false;
  let view = 'mesh';
  let response = 'clay';
  const dialog = document.createElement('dialog');
  dialog.id = 'demo-onboarding';
  dialog.setAttribute('aria-label', 'Get ready to punch');
  dialog.innerHTML = /* HTML */ `
    <div class="demo-onboarding-top">
      <button class="demo-text-button" data-action="advanced">Advanced mode ↗</button>
    </div>
    <section data-screen="welcome" class="demo-welcome">
      <img class="demo-hero-logo" src="/punching-face-logo.png" alt="Punching Face" />
      <h1 class="demo-hero-wordmark">PUNCHING FACE</h1>
      <p class="demo-tagline">Release yourself.</p>
      <button class="demo-primary" data-action="choose">Start <span>→</span></button>
    </section>
    <section data-screen="models" hidden>
      <h1>Choose your model.</h1>
      <div class="demo-model-options" role="group" aria-label="Model options">
        <button class="demo-model-card" data-model="demo" aria-pressed="true">
          <span class="demo-card-number">01</span><span class="demo-card-icon">◉</span>
          <strong>Demo head</strong>
        </button>
        <button class="demo-model-card" data-model="saved" aria-pressed="false">
          <span class="demo-card-number">02</span><span class="demo-card-icon">▧</span>
          <strong>Preloaded model</strong>
        </button>
        <button class="demo-model-card" data-model="import" aria-pressed="false">
          <span class="demo-card-number">03</span><span class="demo-card-icon">＋</span>
          <strong>Record / upload head</strong>
        </button>
      </div>
      <div id="demo-saved-models" class="demo-model-detail" hidden></div>
      <div id="demo-import-options" class="demo-model-detail" hidden>
        <button class="demo-secondary" data-action="record">
          Record / upload video
        </button>
        <button class="demo-secondary" data-action="upload">
          Upload GLB / session
        </button>
      </div>
      <p id="demo-model-status" class="demo-status" role="status"></p>
      <div class="demo-step-actions">
        <button class="demo-text-button" data-action="welcome">← Back</button>
        <button id="demo-use-model" class="demo-primary" data-action="use-model">
          Use demo head <span>→</span>
        </button>
      </div>
    </section>
    <section data-screen="calibration" hidden>
      <h1 id="demo-calibration-title">Calibrate your guard.</h1>
      <p class="demo-subtitle">
        Face the webcam and hold your open hands up in a comfortable guard.
      </p>
      <div class="demo-camera-frame">
        <video
          id="demo-camera-preview"
          autoplay
          playsinline
          muted
          aria-label="Calibration camera preview"
        ></video>
        <div class="demo-camera-guide" aria-hidden="true">
          <span>LEFT HAND</span><span>RIGHT HAND</span>
        </div>
        <span id="demo-camera-badge">Starting camera…</span>
      </div>
      <p id="demo-calibration-status" class="demo-status" role="status">
        Allow camera access when your browser asks.
      </p>
      <button id="demo-calibrate" class="demo-primary" data-action="calibrate" disabled>
        Calibrate my guard
      </button>
      <button
        id="demo-retry-camera"
        class="demo-secondary"
        data-action="retry-camera"
        hidden
      >
        Retry camera
      </button>
      <button id="demo-begin" class="demo-primary" data-action="begin" hidden>
        Begin punching <span>↗</span>
      </button>
      <p class="demo-footnote">
        60-second round · Allow microphone access for voice replies.
      </p>
      <div class="demo-step-actions">
        <button class="demo-text-button" data-action="choose">← Change model</button>
        <button class="demo-text-button" data-action="keyboard">
          Use keyboard instead →
        </button>
      </div>
    </section>
    <footer class="demo-onboarding-footer">
      <span id="demo-step-label"></span>
    </footer>
  `;
  document.body.append(dialog);

  const hud = installDemoHUD({
    setMode(mode) {
      const selectedView = view;
      response = mode;
      api.setMode(mode);
      api.setView(selectedView);
      hud.sync({ mode: response, view });
    },
    setView(next) {
      view = next;
      api.setView(view);
      hud.sync({ mode: response, view });
    },
    resetHead: api.resetHead,
    onAdvanced: advanced,
    onChooseModel: () => show('models'),
    onCamera: () => document.body.classList.toggle('camera-open'),
  });

  const scanJobs = installScanJobs({
    canPrompt: () => !loading && !cameraStarting && !api.isCapturing(),
    onPrompt(open) {
      scanPromptOpen = open;
      hud.setActive(focused && !dialog.open && !open);
    },
    onContinue: continueWhileScanning,
    onOpenScan: api.openCapture,
    async onSwap(model) {
      awaitingImport = false;
      loading = true;
      const selectedResponse = response;
      const selectedView = view;
      const needsCalibration = dialog.open;
      try {
        await api.closeCapture();
        await api.ready;
        await api.loadSaved(model);
        await api.prepareImpacts();
        api.setMode(selectedResponse);
        api.setView(selectedView);
        api.resetHead();
        if (focused) {
          api.firstPerson();
          hud.startRound();
        }
        if (needsCalibration) await calibrateScreen();
      } finally {
        loading = false;
      }
    },
  });

  async function continueWhileScanning() {
    awaitingImport = false;
    await api.closeCapture();
    show('models');
    $('demo-model-status').textContent =
      'Your scan is building in the background. Choose a head to play.';
  }

  function stopCalibrationPreview() {
    clearInterval(calibrationTimer);
    calibrationTimer = null;
    $('demo-camera-preview').srcObject = null;
  }

  function show(next) {
    stopCalibrationPreview();
    screen = next;
    document.body.classList.add('demo-onboarding-open');
    hud.setActive(false);
    for (const section of dialog.querySelectorAll('[data-screen]'))
      section.hidden = section.dataset.screen !== next;
    $('demo-step-label').textContent = {
      welcome: '',
      models: 'STEP 01 / 02',
      calibration: 'STEP 02 / 02',
    }[next];
    if (!dialog.open) dialog.showModal();
    if (next === 'models' && selection === 'saved') void savedModels();
    dialog
      .querySelector(`[data-screen="${next}"] button:not([disabled]):not([hidden])`)
      ?.focus();
  }

  function close() {
    stopCalibrationPreview();
    document.body.classList.remove('demo-onboarding-open');
    dialog.close();
  }

  function advanced() {
    awaitingImport = false;
    focused = false;
    close();
    document.body.classList.remove('immersive', 'demo-focused', 'camera-open');
    hud.setActive(false);
    $('fullscreen').setAttribute('aria-pressed', 'false');
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  }

  function begin() {
    focused = true;
    close();
    response = 'clay';
    view = 'mesh';
    api.startSession();
    api.setMode(response);
    api.setView(view);
    api.firstPerson();
    hud.sync({ mode: response, view });
    api.resetHead();
    hud.startRound();
    hud.setActive(true);
    document.body.classList.add('immersive', 'demo-focused', 'camera-open');
    $('fullscreen').setAttribute('aria-pressed', 'true');
    // Keep the user gesture for fullscreen and AudioContext/microphone startup.
    if (!document.fullscreenElement)
      document.documentElement.requestFullscreen?.().catch(() => {});
    window.__punchingFaceDemoStarted = true;
    window.dispatchEvent(new CustomEvent('punching-face-demo-start'));
  }

  async function savedModels() {
    const box = $('demo-saved-models');
    box.textContent = 'Finding your saved heads…';
    const previousKey = selectedCapture?.key;
    selectedCapture = null;
    $('demo-use-model').disabled = true;
    try {
      const models = await listSavedHeads();
      box.replaceChildren();
      if (!models.length) {
        box.textContent = 'No saved heads yet. Try the demo head or record your own.';
        return;
      }
      const label = document.createElement('label');
      label.htmlFor = 'demo-saved-select';
      label.textContent = 'Your saved heads';
      const select = document.createElement('select');
      select.id = 'demo-saved-select';
      models.forEach((model, i) => {
        const date = model.savedAt
          ? new Date(model.savedAt * 1000).toLocaleDateString(undefined, {
              month: 'short',
              day: 'numeric',
              hour: 'numeric',
              minute: '2-digit',
            })
          : '';
        select.add(
          new Option(
            `${model.name || `Saved head ${i + 1}`} · ${date || model.id.slice(0, 8)}`,
            model.key,
          ),
        );
      });
      selectedCapture = models.find((model) => model.key === previousKey) || models[0];
      select.value = selectedCapture.key;
      select.onchange = () =>
        (selectedCapture = models.find((model) => model.key === select.value));
      box.append(label, select);
      if (selection === 'saved') $('demo-use-model').disabled = false;
    } catch (error) {
      box.textContent = error.message;
    }
  }

  function selectModel(model) {
    if (loading) return;
    awaitingImport = false;
    selection = model;
    for (const card of dialog.querySelectorAll('[data-model]'))
      card.setAttribute('aria-pressed', String(card.dataset.model === model));
    $('demo-saved-models').hidden = model !== 'saved';
    $('demo-import-options').hidden = model !== 'import';
    $('demo-use-model').hidden = model === 'import';
    $('demo-use-model').disabled = false;
    $('demo-use-model').textContent =
      model === 'saved' ? 'Use selected head →' : 'Use demo head →';
    $('demo-model-status').textContent = '';
    if (model === 'saved') void savedModels();
  }

  function syncCalibration() {
    if (screen !== 'calibration' || !dialog.open) return;
    const state = api.getTracking();
    const video = $('demo-camera-preview');
    if (state.stream && video.srcObject !== state.stream) {
      video.srcObject = state.stream;
      video.play().catch(() => {});
    }
    const calibrated = state.active && state.calibrated;
    $('demo-begin').hidden = !calibrated;
    $('demo-calibrate').hidden = calibrated || !state.active;
    $('demo-calibrate').disabled = !state.handCount;
    $('demo-retry-camera').hidden = state.active || cameraStarting;
    $('demo-camera-badge').textContent = calibrated
      ? 'GUARD CALIBRATED'
      : state.active
        ? `${state.handCount} HAND${state.handCount === 1 ? '' : 'S'} DETECTED`
        : cameraStarting
          ? 'STARTING CAMERA…'
          : 'CAMERA OFF';
    $('demo-calibration-title').textContent = calibrated
      ? 'Ready.'
      : 'Calibrate your guard.';
    $('demo-calibration-status').textContent = calibrated
      ? 'Guard locked in. Throw a left hook, right hook, or uppercut.'
      : state.active
        ? state.handCount
          ? 'Hands found. Hold your guard and press Calibrate.'
          : 'Show an open hand in the frame so we can find your guard.'
        : cameraStarting
          ? 'Allow camera access. Hand tracking may take a moment to get ready.'
          : 'Camera unavailable. Allow access and retry, or continue with your keyboard.';
  }

  async function calibrateScreen() {
    show('calibration');
    keyboardOnly = false;
    cameraStarting = true;
    syncCalibration();
    calibrationTimer = setInterval(syncCalibration, 250);
    try {
      await api.startCamera();
    } finally {
      cameraStarting = false;
      if (keyboardOnly) api.stopCamera();
      syncCalibration();
      scanJobs.checkReady();
    }
  }

  async function useModel() {
    if (loading || (selection === 'saved' && !selectedCapture)) return;
    loading = true;
    $('demo-use-model').disabled = true;
    $('demo-model-status').textContent = 'Preparing your head…';
    dialog.classList.add('demo-loading-model');
    try {
      await api.ready;
      if (selection === 'saved') {
        await api.loadSaved(selectedCapture);
        acknowledgeScanJob(selectedCapture);
      } else await api.loadReference();
      await api.prepareImpacts();
      if (!dialog.open || screen !== 'models') return;
      hud.resetScore();
      loading = false;
      await calibrateScreen();
    } catch (error) {
      $('demo-model-status').textContent =
        error.message || 'Could not load that head. Try another model.';
    } finally {
      loading = false;
      dialog.classList.remove('demo-loading-model');
      $('demo-use-model').disabled = false;
      scanJobs.checkReady();
    }
  }

  dialog.addEventListener('click', (event) => {
    const model = event.target.closest('[data-model]');
    if (model) return selectModel(model.dataset.model);
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (!action || loading) return;
    if (action === 'advanced') advanced();
    if (action === 'welcome') show('welcome');
    if (action === 'choose') show('models');
    if (action === 'use-model') void useModel();
    if (action === 'retry-camera') void calibrateScreen();
    if (action === 'calibrate') {
      try {
        api.calibrate();
        syncCalibration();
        $('demo-begin').focus();
      } catch (error) {
        $('demo-calibration-status').textContent = error.message;
      }
    }
    if (action === 'begin') begin();
    if (action === 'keyboard') {
      keyboardOnly = true;
      api.stopCamera();
      begin();
    }
    if (action === 'record' || action === 'upload') {
      awaitingImport = true;
      $('demo-model-status').textContent =
        'Load your head to continue to camera setup.';
      if (action === 'record') api.openCapture();
      else api.openUpload();
    }
  });
  dialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    advanced();
  });
  window.addEventListener('punching-face-model-loaded', async () => {
    if (!awaitingImport) return;
    awaitingImport = false;
    await api.closeCapture();
    await api.prepareImpacts();
    hud.resetScore();
    await calibrateScreen();
  });
  window.addEventListener('punching-face-scan-started', () => {
    awaitingImport = false;
    $('demo-model-status').textContent =
      'Your head will be saved automatically when it is ready.';
  });
  window.addEventListener(
    'punching-face-scan-background',
    () => void continueWhileScanning(),
  );
  window.addEventListener('punching-face-library-changed', () => {
    if (dialog.open && screen === 'models' && selection === 'saved' && !loading)
      void savedModels();
  });
  window.addEventListener('punching-face-model-error', (event) => {
    if (awaitingImport) $('demo-model-status').textContent = event.detail;
  });
  window.addEventListener('punching-face-view-change', (event) => {
    view = event.detail.view;
    response = api.getMode();
    hud.sync({ mode: response, view });
  });
  window.addEventListener('punching-face-mode-change', (event) => {
    response = event.detail.mode;
    hud.sync({ mode: response, view });
  });
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && focused && !dialog.open && !scanPromptOpen)
      advanced();
  });
  const demoButton = document.createElement('button');
  demoButton.id = 'demo-launch';
  demoButton.className = 'small';
  demoButton.textContent = 'Demo mode ↗';
  demoButton.onclick = () => show('models');
  document.querySelector('.header-right').prepend(demoButton);
  show('welcome');
  return {
    get isOpen() {
      return dialog.open || scanJobs.isOpen;
    },
    get canPunch() {
      return !scanPromptOpen && (!focused || hud.isRoundActive);
    },
    advanced,
    begin,
  };
}
