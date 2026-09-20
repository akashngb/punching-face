import { timingRows } from './pipeline-timing.js';
import { captureCoverage } from './face-quality.js';
import { EngineChoice } from './meshy-engine.js';
import { acknowledgeScanJob, forgetScanJobs, trackScanJob } from './scan-jobs.js';
import qualityURL from './face-quality.js?url';
const $ = (id) => document.getElementById(id);
const asDataURL = (blob) =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });

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

export class FaceCapture {
  constructor(onReady, onDelete, onPreview) {
    this.onReady = onReady;
    this.onDelete = onDelete;
    this.onPreview = onPreview;
    this.samples = [];
    this.running = false;
    this.saving = false;
    this.closing = false;
    this.id = null;
    document.body.insertAdjacentHTML(
      'beforeend',
      /* HTML */ `<dialog id="face-scan-dialog">
        <button class="close" id="face-scan-close" aria-label="Close face scan">
          ×</button
        ><span class="eyebrow">Photographs → 3D mesh → Newton physics</span>
        <h1>Scan your whole head.</h1>
        <p class="muted">
          Glasses are supported as a separate estimated 3D accessory. For the clearest
          eye capture, record without them. Keep a neutral expression and your hair,
          ears, and chin fully visible. Start facing the camera with your eyes open and
          look toward the lens for a brief pause; slowly turn your head and torso
          together through a full circle, then face the camera again. Keep your head
          upright. Use even light and pause briefly at each angle. A phone video filmed
          around a seated, still person also works.
        </p>
        <video
          id="face-scan-video"
          playsinline
          muted
          autoplay
          aria-label="Face capture preview"
        ></video
        ><canvas
          id="face-scan-preview"
          width="480"
          height="270"
          aria-label="Last saved head and hair crop"
        ></canvas>
        <div class="face-coverage" id="face-coverage">
          0 saved views · front ○ · side A ○ · side B ○
        </div>
        <p class="note" id="face-scan-feedback" role="status">
          The camera stays off until you press Record. Imported video and extracted
          frames stay in this local project.
        </p>
        <div class="row">
          <button id="face-scan-record" class="primary">Record 360°</button
          ><button id="face-scan-stop" disabled>Stop & save</button>
        </div>
        <button id="face-scan-video-import" class="full small" style="margin-top:12px">
          Upload video</button
        ><input id="face-scan-video-file" type="file" accept="video/*" hidden /><button
          id="face-scan-import"
          class="full small"
          style="margin-top:12px"
        >
          Import head photos</button
        ><input
          id="face-scan-photos"
          type="file"
          accept="image/png,image/jpeg"
          multiple
          hidden
        /><label class="check"
          >AI head completion (slower)
          <input
            type="checkbox"
            id="face-cloud-review"
            ${import.meta.env.VITE_CONTACT_FAST_CAPTURE === '1' ? '' : 'checked'}
        /></label>
        <p class="muted">
          Create 3D face sends selected cropped views to Astra to estimate missing head
          regions, hair type, hairstyle, strand controls, hair masks, glasses masks,
          visible ear landmarks, and eye material parameters. Eye crops are checked for
          usable iris detail; Astra generates missing detail when it cannot be scanned.
          For glasses, it can send up to three cropped views to the image API to
          estimate hidden skin and remove lens artifacts. The edited region must pass
          source-image alignment checks. If rear views are missing, it can also send
          three views for a labeled rear prediction. Face geometry, hair silhouette
          fitting, texture baking and Newton physics run locally. At least 24
          overlapping views, including 12 with visible facial landmarks, are required.
          Camera recovery verifies angular coverage; a saved rear-facing frame alone
          does not prove a complete 360° reconstruction.
        </p>
        <section id="face-timing" class="pipeline-timing" hidden>
          <h2>Video to model</h2>
          <p id="face-video-source" class="muted"></p>
          <video
            id="face-source-video"
            controls
            playsinline
            preload="metadata"
            hidden
            aria-label="Imported source video"
          ></video>
          <dl id="face-timing-rows"></dl>
          <p class="muted">
            Measured processing time. Recording length and time spent waiting to press
            Create are excluded.
          </p>
        </section>
        <button id="face-scan-build" class="primary full" disabled>
          Create 3D face
        </button>
        <p class="note" id="face-job-state" role="status">No reconstruction started.</p>
        <button id="face-scan-background" class="primary full" hidden>
          Use another head while this builds
        </button>
        <div class="row">
          <button id="face-scan-load" disabled>Load face</button
          ><button id="face-scan-delete" disabled>Delete scan</button>
        </div>
        <label class="controls-label" for="face-scan-saved">Saved scans</label
        ><select id="face-scan-saved">
          <option value="">No saved face scans</option>
        </select>
        <details>
          <summary>Camera calibration (optional)</summary>
          <label class="controls-label" for="face-fov"
            >Horizontal field of view, degrees</label
          ><input
            id="face-fov"
            type="number"
            min="10"
            max="120"
            step=".01"
            placeholder="Blank = estimate from images"
          />
          <p class="muted">
            Enter a known horizontal FOV for this camera and resolution. Leave blank if
            unknown. Use the same lens and zoom for every view.
          </p>
        </details>
        <details>
          <summary>OpenAI API settings</summary>
          <p class="muted" id="face-api-status">Checking server configuration…</p>
          <input
            id="face-api-key"
            aria-label="OpenAI API key"
            type="password"
            autocomplete="off"
            placeholder="Replacement API key"
          /><button id="face-api-save">Save key on server</button
          ><button id="face-api-test">Test connection</button>
          <p class="muted">
            The key is never included in exports or browser storage. Replace any key
            shared in chat.
          </p>
        </details>
        <p class="muted">
          This fits a full-head template to facial measurements and captured head
          silhouettes, then adds editable hairstyle-specific strands and separate 3D
          eyewear. Opaque frame contours receive local cleanup. A registered edited
          reference can replace the glasses-affected area; that skin is labeled
          estimated. Eyes use recorded iris detail where reliable, otherwise clearly
          labeled generated detail. A glasses-free reference is needed to verify skin
          hidden by lens tint or reflections. Visible ear placement is checked across
          views. Inner-ear folds, eye geometry and unobserved surfaces remain estimates.
          Reconstruction and the impact rig are experimental; inspect the result from
          multiple angles.
        </p>
      </dialog>`,
    );
    // Local pipeline or Meshy cloud for this scan (src/meshy-engine.js). build, poll and load defer to it.
    this.engines = new EngineChoice(this);
    $('face-scan-record').onclick = () => this.start().catch((e) => this.fail(e));
    $('face-scan-stop').onclick = () => this.stop().catch((e) => this.fail(e));
    $('face-scan-close').onclick = () => this.close().catch((e) => this.fail(e));
    $('face-scan-background').onclick = async () => {
      try {
        await this.close();
        window.dispatchEvent(new CustomEvent('punching-face-scan-background'));
      } catch (error) {
        this.fail(error);
      }
    };
    $('face-scan-dialog').addEventListener('cancel', (e) => {
      e.preventDefault();
      this.close().catch((error) => this.fail(error));
    });
    $('face-scan-build').onclick = () => this.build().catch((e) => this.fail(e));
    $('face-scan-load').onclick = () => this.load().catch((e) => this.fail(e));
    $('face-scan-delete').onclick = () => this.remove().catch((e) => this.fail(e));
    $('face-scan-video-import').onclick = () => $('face-scan-video-file').click();
    $('face-scan-video-file').onchange = (e) =>
      this.importVideo(e.target.files[0])
        .catch((e) => this.fail(e))
        .finally(() => (e.target.value = ''));
    $('face-scan-import').onclick = () => $('face-scan-photos').click();
    $('face-scan-photos').onchange = (e) =>
      this.importPhotos([...e.target.files])
        .catch((e) => this.fail(e))
        .finally(() => (e.target.value = ''));
    $('face-scan-saved').onchange = () =>
      this.select($('face-scan-saved').value).catch((e) => this.fail(e));
    $('face-api-save').onclick = async () => {
      try {
        await api('openai-config', { apiKey: $('face-api-key').value.trim() });
        $('face-api-key').value = '';
        await this.configuration();
      } catch (e) {
        $('face-api-status').textContent = e.message;
      }
    };
    $('face-api-test').onclick = async () => {
      $('face-api-test').disabled = true;
      try {
        const result = await api('openai-test', {});
        $('face-api-status').textContent = result.connected
          ? 'OpenAI connection verified.'
          : 'Connection not verified.';
      } catch (e) {
        $('face-api-status').textContent = e.message;
      } finally {
        $('face-api-test').disabled = false;
      }
    };
    window.addEventListener('beforeunload', () => {
      this.worker?.terminate();
      this.stream?.getTracks().forEach((t) => t.stop());
    });
  }

  showTiming(source, timing) {
    this.source = source;
    this.timing = timing;
    $('face-timing').hidden = !source && !timing;
    $('face-video-source').textContent =
      source?.filename ?? 'Saved photograph reconstruction';
    $('face-timing-rows').replaceChildren(
      ...timingRows(source, timing).flatMap(([label, value]) => {
        const dt = document.createElement('dt'),
          dd = document.createElement('dd');
        dt.textContent = label;
        dd.textContent = value;
        return [dt, dd];
      }),
    );
    const preview = $('face-source-video'),
      url = source?.videoStored ? '/api/face-video?id=' + this.id : null;
    preview.hidden = !url;
    if (url && preview.getAttribute('src') !== url) preview.src = url;
    if (preview.hidden) preview.pause();
  }

  async open() {
    if (this.closing) return;
    $('face-scan-dialog').showModal();
    await Promise.all([this.configuration(), this.refresh(), this.engines.refresh()]);
    if (this.id && !this.running) await this.poll();
  }

  async configuration() {
    const c = await api('openai-config');
    $('face-api-status').textContent = c.configured
      ? `Server key configured · gpt-6-astra`
      : 'No API key configured. Local reconstruction is available.';
    if (!c.configured) $('face-cloud-review').checked = false;
  }

  async refresh() {
    const data = await api('face-captures');
    this.saved = data.captures;
    const select = $('face-scan-saved');
    select.replaceChildren(new Option('Select a saved face scan', ''));
    for (const scan of this.saved)
      select.add(
        new Option(
          `${scan.testFixture ? 'Public test · ' : ''}${scan.frames} views · ${scan.status} · ${new Date(scan.savedAt * 1000).toLocaleString()}`,
          scan.id,
        ),
      );
    select.value = this.id || '';
    if (!this.id && this.saved[0]) await this.select(this.saved[0].id);
  }

  controls() {
    const busy =
      this.running ||
      this.saving ||
      this.loading ||
      this.submitting ||
      this.deleting ||
      this.closing;
    for (const id of [
      'face-scan-record',
      'face-scan-video-import',
      'face-scan-video-file',
      'face-scan-import',
      'face-scan-photos',
      'face-scan-saved',
      'face-fov',
    ])
      $(id).disabled = !!busy;
    $('face-scan-stop').disabled = !this.running;
    $('face-scan-delete').disabled = !this.id || !!busy || !!this.jobRunning;
    $('face-scan-background').hidden = !this.jobRunning;
    $('face-scan-background').disabled = !!busy;
    $('face-scan-build').disabled =
      !!busy || !this.id || this.count < this.engines.minimumViews || this.jobRunning;
    $('face-scan-load').disabled = !!busy || !this.ready;
  }

  async init() {
    if (this.worker) return;
    this.worker = new Worker('/face-worker.js');
    await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('Face tracking initialization timed out.')),
        25000,
      );
      this.worker.onmessage = ({ data }) => {
        if (data.type === 'ready') {
          clearTimeout(timer);
          resolve();
        } else if (data.type === 'error') {
          clearTimeout(timer);
          reject(new Error(data.message));
        }
      };
      this.worker.onerror = () => {
        clearTimeout(timer);
        reject(new Error('Could not load the face tracking worker.'));
      };
      this.worker.postMessage({ type: 'init', qualityURL, origin: location.origin });
    });
  }

  async start() {
    this.loading = true;
    this.controls();
    try {
      await this.init();
      this.worker.postMessage({ type: 'reset' });
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const video = $('face-scan-video');
      video.srcObject = this.stream;
      await video.play();
      const scan = await api('face-captures', {
        captureRegion: 'head',
        horizontalFovDegrees: $('face-fov').value ? Number($('face-fov').value) : null,
      });
      this.id = scan.id;
      this.samples = [];
      this.count = 0;
      this.previous = null;
      this.lastRejection = null;
      this.ready = false;
      this.jobRunning = false;
      this.running = true;
      this.started = performance.now();
      this.generation = (this.generation || 0) + 1;
      $('face-job-state').textContent =
        'Each accepted frame is saved immediately on this computer.';
      this.updateCoverage();
      this.loop();
    } catch (e) {
      this.stream?.getTracks().forEach((t) => t.stop());
      this.worker?.terminate();
      this.worker = null;
      throw e;
    } finally {
      this.loading = false;
      this.controls();
    }
  }

  process(bitmap, options = {}) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('Face detection timed out.')),
        15000,
      );
      this.pendingReject = reject;
      this.worker.onmessage = ({ data }) => {
        clearTimeout(timer);
        this.pendingReject = null;
        data.type === 'error' ? reject(new Error(data.message)) : resolve(data);
      };
      this.worker.postMessage(
        { type: 'frame', bitmap, previous: this.previous, ...options },
        [bitmap],
      );
    });
  }

  async accept(data, generation) {
    if (generation !== this.generation || !this.running) return;
    if (!data.ok) {
      this.lastRejection = data.message;
      $('face-scan-feedback').textContent = data.message;
      return;
    }
    const image = await asDataURL(data.blob);
    if (generation !== this.generation) return;
    this.saving = true;
    this.controls();
    try {
      const saved = await api('face-frames', {
        id: this.id,
        frames: [
          {
            image,
            yaw: data.yaw,
            landmarks: data.landmarks,
            irisLandmarks: data.irisLandmarks,
            timeSeconds: data.timeSeconds ?? null,
          },
        ],
      });
      this.count = saved.frames;
      this.samples.push({ yaw: data.yaw });
      this.previous = data.landmarks ? { yaw: data.yaw, pitch: data.pitch } : null;
      const bitmap = await createImageBitmap(data.blob);
      const canvas = $('face-scan-preview');
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      bitmap.close();
      this.updateCoverage();
      $('face-scan-feedback').textContent =
        `Saved view ${this.count}${data.landmarks ? ' · facial landmarks tracked' : ' · profile/rear image retained'}. ${this.importingVideo ? 'Extracting video frames…' : 'Continue slowly around the whole head.'}`;
    } finally {
      this.saving = false;
      this.controls();
    }
  }

  async loop() {
    while (
      this.running &&
      this.count < 240 &&
      performance.now() - this.started < 120000
    ) {
      const generation = this.generation;
      this.inflight = (async () => {
        const bitmap = await createImageBitmap($('face-scan-video'));
        const data = await this.process(bitmap);
        data.timeSeconds = (performance.now() - this.started) / 1000;
        await this.accept(data, generation);
      })();
      try {
        await this.inflight;
      } catch (e) {
        this.fail(e);
        break;
      } finally {
        this.inflight = null;
      }
      if (this.running) await new Promise((r) => setTimeout(r, 600));
    }
    if (this.running) await this.stop();
  }

  updateCoverage() {
    const c = captureCoverage(this.samples);
    $('face-coverage').textContent =
      `${this.count || 0} saved views · front ${c.front ? '✓' : '○'} · side A ${c.left ? '✓' : '○'} · side B ${c.right ? '✓' : '○'} · ${c.headOnly} profile/rear views saved (angles pending reconstruction)`;
  }

  async stop() {
    this.running = false;
    this.generation = (this.generation || 0) + 1;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    $('face-scan-video').srcObject = null;
    try {
      await this.inflight;
    } catch {}
    this.worker?.terminate();
    this.worker = null;
    this.controls();
    $('face-scan-feedback').textContent =
      `Capture stopped. ${this.count || 0} views saved locally.${this.lastRejection ? ' Last skipped view: ' + this.lastRejection : ''}`;
    await this.refresh();
  }

  async close() {
    if (this.closing) return;
    this.closing = true;
    clearTimeout(this.pollTimer);
    this.controls();
    try {
      $('face-source-video').pause();
      await this.stop();
    } finally {
      clearTimeout(this.pollTimer);
      $('face-scan-dialog').close();
      this.closing = false;
      this.controls();
    }
  }

  fail(e) {
    $('face-scan-feedback').textContent = e.message;
    if (this.running) {
      this.running = false;
      this.stream?.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    this.controls();
  }

  async importPhotos(files) {
    if (!files.length) return;
    await this.stop();
    if (files.length > 240) throw new Error('Import at most 240 overlapping photos.');
    this.loading = true;
    this.controls();
    try {
      await this.init();
      this.worker.postMessage({ type: 'reset' });
      const scan = await api('face-captures', {
        captureRegion: 'head',
        horizontalFovDegrees: $('face-fov').value ? Number($('face-fov').value) : null,
      });
      this.id = scan.id;
      this.samples = [];
      this.count = 0;
      this.previous = null;
      this.lastRejection = null;
      this.ready = false;
      this.running = true;
      this.generation = (this.generation || 0) + 1;
      for (const file of files) {
        if (!this.running) break;
        const original = await createImageBitmap(file);
        const scale = Math.min(1, 1280 / original.width, 960 / original.height),
          canvas = document.createElement('canvas');
        canvas.width = Math.round(original.width * scale);
        canvas.height = Math.round(original.height * scale);
        canvas.getContext('2d').drawImage(original, 0, 0, canvas.width, canvas.height);
        original.close();
        const generation = this.generation;
        this.inflight = (async () => {
          const data = await this.process(await createImageBitmap(canvas));
          await this.accept(data, generation);
        })();
        try {
          await this.inflight;
        } finally {
          this.inflight = null;
        }
      }
    } finally {
      this.loading = false;
      await this.stop();
    }
  }

  async importVideo(file) {
    if (!file) return;
    if (file.size > 500 * 1024 * 1024)
      throw new Error('Use a video smaller than 500 MB.');
    await this.stop();
    const extractionStarted = performance.now();
    this.loading = true;
    this.importingVideo = true;
    this.controls();
    const video = document.createElement('video'),
      url = URL.createObjectURL(file);
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    const waitFor = (event, action) =>
      new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => finish(new Error('Video decoding timed out. Try MP4 or WebM.')),
          20000,
        );
        const done = () => finish(),
          error = () =>
            finish(new Error('This browser cannot decode the video. Try MP4 or WebM.'));
        function finish(e) {
          clearTimeout(timeout);
          video.removeEventListener(event, done);
          video.removeEventListener('error', error);
          e ? reject(e) : resolve();
        }
        video.addEventListener(event, done, { once: true });
        video.addEventListener('error', error, { once: true });
        action();
      });
    try {
      await waitFor('loadeddata', () => {
        video.src = url;
        video.load();
      });
      if (
        !Number.isFinite(video.duration) ||
        video.duration < 3 ||
        video.duration > 300
      )
        throw new Error('Use a head rotation video between 3 seconds and 5 minutes.');
      await this.init();
      this.worker.postMessage({ type: 'reset' });
      const scan = await api('face-captures', {
        captureRegion: 'head',
        horizontalFovDegrees: $('face-fov').value ? Number($('face-fov').value) : null,
      });
      this.id = scan.id;
      this.samples = [];
      this.count = 0;
      this.previous = null;
      this.ready = false;
      this.jobRunning = false;
      this.lastRejection = null;
      this.running = true;
      this.generation = (this.generation || 0) + 1;
      this.controls();
      const scale = Math.min(1, 1280 / video.videoWidth, 960 / video.videoHeight),
        canvas = document.createElement('canvas');
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      this.showTiming(
        {
          filename: file.name,
          durationSeconds: video.duration,
          extractionSeconds: 0,
          extractionComplete: false,
        },
        null,
      );
      await api('face-timing', {
        id: this.id,
        kind: 'video',
        filename: file.name,
        durationSeconds: video.duration,
        extractionSeconds: 0,
        extractionComplete: false,
      });
      const upload = await fetch('/api/face-video?id=' + this.id, {
        method: 'POST',
        headers: { 'Content-Type': file.type || 'application/octet-stream' },
        body: file,
      });
      if (!upload.ok)
        throw new Error('The original video could not be retained locally.');
      let nativeFrames = null;
      try {
        $('face-job-state').textContent = 'Decoding video locally…';
        nativeFrames = (await api('face-video-frames', { id: this.id })).frames;
      } catch (error) {
        console.info('Using browser video decoding:', error.message);
      }
      const nativeBitmap = async (i) =>
        createImageBitmap(await (await fetch(nativeFrames[i].image)).blob());
      // Imported clips may start behind the head. Locate a frontal view first
      // so the chronological pass can keep the earlier profile/rear frames.
      for (const fraction of [0, 0.8, 0.6, 0.4, 0.2]) {
        let bitmap;
        if (nativeFrames) {
          bitmap = await nativeBitmap(
            Math.min(
              nativeFrames.length - 1,
              Math.round(fraction * nativeFrames.length),
            ),
          );
        } else {
          const time = Math.min(video.duration - 0.05, video.duration * fraction);
          if (Math.abs(video.currentTime - time) > 0.001)
            await waitFor('seeked', () => {
              video.currentTime = time;
            });
          canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
          bitmap = await createImageBitmap(canvas);
        }
        const seed = await this.process(bitmap, { calibrateOnly: true });
        if (seed.ok && Math.abs(seed.yaw) < 35) break;
      }
      const steps =
        nativeFrames?.length ??
        Math.min(220, Math.max(48, Math.ceil(video.duration / 0.3)));
      const sampleTime = (i) =>
        nativeFrames?.[i].timeSeconds ??
        Math.min(video.duration - 0.05, (video.duration * i) / steps);
      const seek = (time) =>
        !nativeFrames && Math.abs(video.currentTime - time) > 0.001
          ? waitFor('seeked', () => {
              video.currentTime = time;
            })
          : Promise.resolve();
      let nextSeek = seek(sampleTime(0));
      let pendingSave = Promise.resolve();
      let processed = 0;
      for (let i = 0; i < steps && this.running; i++) {
        const time = sampleTime(i);
        await nextSeek;
        if (!this.running) break;
        if (!nativeFrames)
          canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const bitmap = nativeFrames
          ? await nativeBitmap(i)
          : await createImageBitmap(canvas);
        // Decode the next frame while the worker analyzes this private bitmap.
        nextSeek = i + 1 < steps ? seek(sampleTime(i + 1)) : Promise.resolve();
        nextSeek.catch(() => {}); // The next iteration propagates a seek failure.
        const generation = this.generation;
        $('face-job-state').textContent =
          `Extracting video locally · ${Math.round((100 * i) / steps)}% · ${this.count} useful views saved`;
        // One upload may run while the worker analyzes the next frame. Keep
        // writes ordered and bounded, and let Stop await both in-flight tasks.
        this.inflight = Promise.all([this.process(bitmap), pendingSave]);
        try {
          const [data] = await this.inflight;
          data.timeSeconds = time;
          pendingSave = this.accept(data, generation);
          pendingSave.catch(() => {}); // Propagated at the next bounded join.
          processed++;
        } finally {
          this.inflight = pendingSave;
        }
      }
      await pendingSave;
      this.inflight = null;
      await nextSeek;
      let result = await api('face-timing', {
        id: this.id,
        kind: 'video',
        filename: file.name,
        durationSeconds: video.duration,
        extractionSeconds: (performance.now() - extractionStarted) / 1000,
        extractionComplete: processed === steps,
      });
      this.showTiming(result.source, result.timing);
      $('face-job-state').textContent =
        `Video processed. ${this.count} head views saved locally. Create 3D face will verify cameras and fit the head template.`;
    } finally {
      video.pause();
      video.removeAttribute('src');
      video.load();
      URL.revokeObjectURL(url);
      this.loading = false;
      this.importingVideo = false;
      await this.stop();
    }
  }

  async select(id) {
    if (this.running || this.saving) return;
    clearTimeout(this.pollTimer);
    this.id = id || null;
    $('face-scan-saved').value = id || '';
    this.ready = false;
    this.jobRunning = false;
    if (!id) {
      this.showTiming(null, null);
      this.controls();
      return;
    }
    const scan = this.saved.find((s) => s.id === id);
    this.count = scan?.frames || 0;
    $('face-coverage').textContent =
      `${this.count} saved views · ${Math.round(scan?.span || 0)}° estimated span`;
    await this.poll();
    this.controls();
  }

  async build() {
    if (this.engines.meshy) return this.engines.build();
    if (this.submitting) return;
    this.submitting = true;
    this.controls();
    try {
      await this.stop();
      const id = this.id;
      const result = await api('face-train', {
        id,
        cloudReview: $('face-cloud-review').checked,
      });
      trackScanJob({ id: result.id, engine: 'local' });
      if (id !== this.id || this.engines.meshy) return;
      this.jobRunning = true;
      this.ready = false;
      this.autoLoaded = null;
      await this.poll(result.id);
    } finally {
      this.submitting = false;
      this.controls();
    }
  }

  async poll(id = this.id) {
    clearTimeout(this.pollTimer);
    if (!id || this.closing || !$('face-scan-dialog').open) return;
    if (this.engines.meshy) return this.engines.poll(id);
    const current = () =>
      id === this.id &&
      !this.engines.meshy &&
      !this.closing &&
      $('face-scan-dialog').open;
    if (!current()) return;
    const state = await api('face-status?id=' + id);
    if (!current()) return;
    const wasRunning = this.jobRunning;
    this.jobRunning = state.status === 'running';
    if (this.jobRunning && !wasRunning)
      trackScanJob({ id, engine: 'local', resume: true });
    this.ready = state.photoModel === true;
    this.showTiming(state.source, state.timing);
    const orbit = state.evidence?.orbitCoverage;
    if (orbit)
      $('face-coverage').textContent =
        `${this.count} saved views · ${state.evidence.registeredViews} recovered cameras · ${orbit.recoveredSpanDegrees}° recovered span · ${orbit.registeredRearViews} rear views${orbit.completeOrbit ? ' · complete orbit' : ' · gaps remain estimated'}`;
    $('face-job-state').textContent =
      state.message +
      (state.evidence?.cloudReview?.nextCaptureInstruction
        ? ' OpenAI review: ' + state.evidence.cloudReview.nextCaptureInstruction
        : '') +
      (state.evidence?.cloudReview?.error
        ? ' OpenAI review: ' + state.evidence.cloudReview.error
        : '') +
      (state.evidence?.rearPrediction?.error
        ? ' Rear image API unavailable; using local hair-material continuation.'
        : '');
    this.controls();
    const pollAgain = async () => {
      if (!current()) return;
      try {
        await this.poll(id);
      } catch (error) {
        if (!current()) return;
        $('face-job-state').textContent = error.message;
        this.pollTimer = setTimeout(pollAgain, 4000);
      }
    };
    if (this.jobRunning) this.pollTimer = setTimeout(pollAgain, 2000);
  }

  async load() {
    if (this.engines.meshy) return this.engines.load();
    const id = this.id;
    this.loading = true;
    this.controls();
    const started = performance.now();
    try {
      await this.onReady(id);
      acknowledgeScanJob({ id, engine: 'local' });
      this.autoLoaded = id;
      const result = await api('face-timing', {
        id,
        kind: 'load',
        seconds: (performance.now() - started) / 1000,
      });
      this.showTiming(result.source, result.timing);
      $('face-job-state').textContent =
        'Face loaded into the interaction scene. Close this panel to inspect the surface and try a hook.';
    } finally {
      this.loading = false;
      this.controls();
    }
  }

  async remove() {
    if (
      this.deleting ||
      this.running ||
      this.saving ||
      this.loading ||
      this.submitting ||
      this.closing ||
      this.jobRunning
    )
      return;
    this.deleting = true;
    this.controls();
    try {
      await this.stop();
      const id = this.id;
      if (!id || this.jobRunning) return;
      await api('face-delete', { id });
      forgetScanJobs(id);
      clearTimeout(this.pollTimer);
      await this.onDelete(id);
      this.id = null;
      this.count = 0;
      this.samples = [];
      this.ready = false;
      this.jobRunning = false;
      this.autoLoaded = null;
      const c = $('face-scan-preview');
      c.getContext('2d').clearRect(0, 0, c.width, c.height);
      this.updateCoverage();
      $('face-job-state').textContent =
        'Scan and its local reconstruction outputs deleted.';
      await this.refresh();
    } finally {
      this.deleting = false;
      this.controls();
    }
  }
}
