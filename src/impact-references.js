import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
const $ = (id) => document.getElementById(id);

export class ImpactReferences {
  constructor() {
    this.selected = null;
    this.landmarks = null;
    document.body.insertAdjacentHTML(
      'beforeend',
      /* HTML */ `<dialog id="reference-dialog">
        <button class="close" id="reference-close" aria-label="Close impact references">
          ×</button
        ><span class="eyebrow">Evidence → landmarks → deformation controls</span>
        <h1>Study the surface response.</h1>
        <p class="muted">
          References guide contact compression, lip shear, and adjacent bulging. A still
          image cannot recover hidden geometry, impact force, or tissue stiffness.
        </p>
        <button id="reference-search" class="small full">
          Search the web for more references</button
        ><select
          id="reference-select"
          aria-label="Impact reference"
          class="full"
          style="margin:12px 0"
        ></select>
        <div id="reference-info" class="note"></div>
        <canvas
          id="reference-canvas"
          width="640"
          height="400"
          style="width:100%;background:#16281f;margin:12px 0"
        ></canvas
        ><label class="check"
          >Estimated landmark wireframe
          <input id="reference-wire" type="checkbox" checked /></label
        ><button id="reference-extract" class="full primary">
          Extract visible face landmarks</button
        ><button id="reference-save" class="small full" style="margin-top:8px">
          Save landmark evidence
        </button>
        <p id="reference-status" class="muted" role="status"></p>
        <p class="muted">
          Yellow topology is a detector estimate. Occluded landmarks behind a glove are
          predictions, not observations. Compare neutral and impact frames of the same
          person before fitting displacement targets.
        </p>
        <div id="reference-candidates"></div>
      </dialog>`,
    );
    $('reference-close').onclick = () => $('reference-dialog').close();
    $('reference-wire').onchange = () => this.draw();
    $('reference-select').onchange = () =>
      this.select(Number($('reference-select').value));
    $('reference-extract').onclick = () => this.extract();
    $('reference-save').onclick = () => this.save();
    $('reference-search').onclick = async () => {
      const button = $('reference-search');
      button.disabled = true;
      $('reference-status').textContent =
        'Searching public pages; no webcam data is sent.';
      try {
        const r = await fetch('/api/reference-search', { method: 'POST' });
        if (!r.ok) throw new Error('Reference search unavailable.');
        this.catalog = await r.json();
        this.render();
        $('reference-status').textContent =
          'Search candidates added below. They need relevance and source review before use.';
      } catch (e) {
        $('reference-status').textContent = e.message;
      } finally {
        button.disabled = false;
      }
    };
  }

  async open() {
    try {
      this.catalog = await fetch('/api/references').then((r) => r.json());
      this.render();
      $('reference-dialog').showModal();
      if (!this.selected) await this.select(0);
    } catch (e) {
      console.error(e);
    }
  }

  render() {
    const select = $('reference-select');
    select.replaceChildren();
    this.local = this.catalog.references.filter((r) => r.image);
    this.local.forEach((r, i) => {
      const o = document.createElement('option');
      o.value = i;
      o.textContent = r.title;
      select.append(o);
    });
    const list = $('reference-candidates');
    list.replaceChildren();
    for (const r of this.catalog.references.filter((r) => !r.image)) {
      const p = document.createElement('p'),
        a = document.createElement('a');
      a.href = r.url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = r.title;
      p.append(a, document.createTextNode(' · ' + r.status));
      list.append(p);
    }
  }

  async select(index) {
    this.selected = this.local[index];
    this.landmarks = null;
    this.contact = null;
    if (!this.selected) return;
    this.image = new Image();
    this.image.src = this.selected.image;
    await this.image.decode();
    $('reference-info').textContent = this.selected.features.join(' · ');
    $('reference-status').textContent =
      'Select the visible contact point on the image, then extract landmarks.';
    this.draw();
    $('reference-canvas').onclick = (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      this.contact = {
        x: (e.clientX - r.left) / r.width,
        y: (e.clientY - r.top) / r.height,
      };
      this.draw();
    };
  }

  draw() {
    if (!this.image) return;
    const c = $('reference-canvas');
    c.width = this.image.naturalWidth;
    c.height = this.image.naturalHeight;
    const ctx = c.getContext('2d');
    ctx.drawImage(this.image, 0, 0);
    if (this.landmarks && $('reference-wire').checked) {
      ctx.strokeStyle = '#e9e89c';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      for (const [a, b] of this.edges) {
        const p = this.landmarks[a],
          q = this.landmarks[b];
        ctx.moveTo(p.x * c.width, p.y * c.height);
        ctx.lineTo(q.x * c.width, q.y * c.height);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    if (this.contact) {
      ctx.strokeStyle = '#ffbd74';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(this.contact.x * c.width, this.contact.y * c.height, 12, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  async extract() {
    const button = $('reference-extract');
    button.disabled = true;
    $('reference-status').textContent = 'Estimating facial landmarks locally…';
    try {
      if (!this.detector) {
        const files = await FilesetResolver.forVisionTasks('/wasm');
        this.detector = await FaceLandmarker.createFromOptions(files, {
          baseOptions: {
            modelAssetPath: '/models/face_landmarker.task',
            delegate: 'CPU',
          },
          runningMode: 'IMAGE',
          numFaces: 1,
        });
        const obj = await fetch('/models/canonical_face_model.obj').then((r) =>
          r.text(),
        );
        this.edges = obj
          .split('\n')
          .filter((l) => l.startsWith('f '))
          .flatMap((l) => {
            const a = l
              .split(/\s+/)
              .slice(1)
              .map((x) => Number(x.split('/')[0]) - 1);
            return [
              [a[0], a[1]],
              [a[1], a[2]],
              [a[2], a[0]],
            ];
          });
      }
      const result = this.detector.detect(this.image);
      this.landmarks = result.faceLandmarks?.[0]?.slice(0, 468);
      if (!this.landmarks)
        throw new Error(
          'No reliable face detection. Occlusion or profile is too strong for this detector.',
        );
      this.draw();
      $('reference-status').textContent =
        '468 estimated landmarks. Review the glove occlusion and projected wireframe before using them.';
    } catch (e) {
      $('reference-status').textContent = e.message;
    } finally {
      button.disabled = false;
    }
  }

  async save() {
    if (!this.landmarks) {
      $('reference-status').textContent = 'Extract and review landmarks first.';
      return;
    }
    const data = {
      format: 'punching-face-impact-evidence',
      referenceId: this.selected.id,
      source: this.selected.url ?? 'user-provided',
      features: this.selected.features,
      contact: this.contact,
      landmarks: this.landmarks,
      limitations: [
        'Monocular detector estimate, not measured 3D deformation.',
        'Occluded landmarks are unverified.',
        'No neutral comparison frame or physical calibration.',
      ],
      imageSize: [this.image.naturalWidth, this.image.naturalHeight],
    };
    try {
      const response = await fetch('/api/reference-evidence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Evidence save failed.');
      $('reference-status').textContent =
        'Evidence saved locally with provenance and uncertainty.';
    } catch (e) {
      $('reference-status').textContent = e.message;
    }
  }
}
