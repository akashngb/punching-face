// Guest page: join the ring from any device. It watches the host's live 3D head, tracks your
// hands with MediaPipe on THIS device, and sends only tiny punch events over LiveKit's data
// channel. Your camera is published only if you tick the box. This page never talks to the
// host's local servers when deployed; it only knows LiveKit.
import './guest.css';
import { Room, RoomEvent, Track, TokenSource } from 'livekit-client';
import { PunchDetector } from './punch-detect.js';
import { encode, decode } from './sse.js';

const root = document.getElementById('guest'),
  hash = new URLSearchParams(location.hash.slice(1));
const local = ['127.0.0.1', 'localhost'].includes(location.hostname);
// Served next to the app in development; from public CDNs once the page is deployed on its own.
const WASM = local
  ? '/wasm'
  : 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm';
const MODEL = local
  ? '/models/hand_landmarker.task'
  : 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
// Remembered per tab and per room so a reload (or a flaky connection) drops you straight back in, under the
// same identity: LiveKit then replaces the stale connection instead of showing you twice.
const KEY = 'punching-face-guest-' + (hash.get('r') || '');
const saved = () => {
  try {
    return JSON.parse(sessionStorage.getItem(KEY) || 'null');
  } catch {
    return null;
  }
};
const remember = (value) => {
  try {
    value
      ? sessionStorage.setItem(KEY, JSON.stringify(value))
      : sessionStorage.removeItem(KEY);
  } catch {
    /* private mode */
  }
};
const identity =
  saved()?.identity ||
  'guest-' +
    [...crypto.getRandomValues(new Uint8Array(4))]
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
const escape = (text) =>
  String(text).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );

async function credentials(name) {
  const roomName = hash.get('r');
  if (!roomName)
    throw new Error('This link has no room in it. Ask the host for a fresh invite.');
  if (hash.get('d')) {
    // LiveKit Cloud's development token server: every guest gets a unique identity from one shared link.
    const details = await TokenSource.developmentTokenServer(hash.get('d')).fetch({
      roomName,
      participantName: name,
      participantIdentity: identity,
    });
    return { url: details.serverUrl, token: details.participantToken };
  }
  if (local) {
    try {
      const r = await fetch('http://127.0.0.1:5176/sponsors/livekit/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room: roomName, role: 'guest', name, identity }),
      });
      if (r.ok) return await r.json();
    } catch {
      /* fall through to the token in the link */
    }
  }
  if (hash.get('u') && hash.get('t'))
    return { url: hash.get('u'), token: hash.get('t') };
  throw new Error('This invite is incomplete. Ask the host for a fresh link.');
}

function joinScreen(message) {
  root.innerHTML = /* HTML */ `<div class="join">
    <h1>Join the ring</h1>
    <p>
      You’ll see the host’s live 3D head. Throw punches at your camera and they land on
      it.
    </p>
    <input
      type="text"
      id="name"
      placeholder="Your name"
      maxlength="24"
      autocomplete="nickname"
    />
    <label
      ><input type="checkbox" id="share" checked /><span
        >Share my camera with the room. Your hands are tracked on this device either
        way; without this, only punch events leave it.</span
      ></label
    >
    <button class="primary" id="join">Join</button>
    <div class="note ${message ? 'error' : ''}" id="note">
      ${message ? escape(message) : 'Room: ' + escape(hash.get('r') || '—')}
    </div>
  </div>`;
  const button = root.querySelector('#join');
  button.onclick = async () => {
    button.disabled = true;
    root.querySelector('#note').className = 'note';
    root.querySelector('#note').textContent = 'Connecting…';
    try {
      await enter(
        root.querySelector('#name').value.trim() || 'Guest',
        root.querySelector('#share').checked,
      );
    } catch (error) {
      joinScreen(error.message);
    }
  };
}

async function enter(name, share) {
  // Adaptive streaming pauses video while this page is hidden, which is right for phones on venue Wi-Fi.
  // Add &adaptive=0 to the invite to keep receiving in a background tab (also how the feed is tested headlessly).
  const { url, token } = await credentials(name);
  const room = new Room({
    adaptiveStream: hash.get('adaptive') !== '0',
    dynacast: true,
  });
  window.__guestRoom = room;
  root.innerHTML = /* HTML */ `<div class="ring">
    <div class="stage">
      <video class="model" autoplay playsinline muted></video>
      <div class="wait" id="wait">Waiting for the host’s head…</div>
      <div class="self">
        <video id="self" autoplay playsinline muted></video
        ><span id="track">camera off</span>
      </div>
      <div class="flash" id="flash"></div>
      <div class="toast" id="toast"></div>
    </div>
    <div class="hud">
      <span>Hits <b id="hits">0</b></span
      ><span>Best <b id="best">0.0</b> m/s</span><span class="grow"></span
      ><span>RTT <b id="rtt">—</b></span
      ><button id="leave">Leave</button>
    </div>
    <div class="pads">
      <button data-pad="left">LEFT HOOK</button><button data-pad="jab">JAB</button
      ><button data-pad="right">RIGHT HOOK</button>
    </div>
    <div class="board" id="board"></div>
  </div>`;
  const $ = (id) => root.querySelector('#' + id),
    model = root.querySelector('video.model'),
    self = $('self');
  let hits = 0,
    best = 0,
    running = true,
    toastTimer = 0;
  const toast = (text) => {
    $('toast').textContent = text;
    $('toast').classList.add('on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('on'), 1100);
  };
  const throwPunch = (punch) =>
    room.localParticipant
      .publishData(encode({ type: 'punch', ...punch }), {
        reliable: true,
        topic: 'punch',
      })
      .catch(() => toast('Not sent: reconnecting…'));

  room
    .on(RoomEvent.TrackSubscribed, (track, publication) => {
      if (publication.trackName === 'model') {
        track.attach(model);
        $('wait').hidden = true;
      } // also fires when the host republishes after a reload
      else if (track.kind === 'audio') {
        const audio = track.attach();
        audio.autoplay = true;
        root.append(audio);
      } // the coach's voice
    })
    .on(RoomEvent.TrackUnsubscribed, (track, publication) => {
      track.detach();
      if (publication.trackName !== 'model') return;
      // A reloading host briefly has two 'model' tracks; only show the wait card if none is left.
      const other = [...room.remoteParticipants.values()]
        .flatMap((p) => [...p.trackPublications.values()])
        .find((p) => p.trackName === 'model' && p.track && p !== publication);
      if (other) other.track.attach(model);
      else $('wait').hidden = false;
    })
    .on(RoomEvent.DataReceived, (payload) => {
      const message = decode(payload);
      if (!message) return;
      if (message.type === 'pong')
        $('rtt').textContent = Math.round(performance.now() - message.t) + ' ms';
      if (message.type === 'miss' && message.id === room.localParticipant.identity)
        toast('No contact: ' + String(message.reason || 'missed').slice(0, 40));
      if (message.board)
        $('board').innerHTML = message.board
          .map((row) => `<span>${escape(row.name)} <b>${row.count}</b></span>`)
          .join('');
      if (message.type === 'hit' && message.id === room.localParticipant.identity) {
        hits++;
        best = Math.max(best, message.speed);
        $('hits').textContent = hits;
        $('best').textContent = best.toFixed(1);
        toast(`${message.zone} · ${message.speed.toFixed(1)} m/s`);
        $('flash').classList.add('on');
        requestAnimationFrame(() =>
          requestAnimationFrame(() => $('flash').classList.remove('on')),
        );
        navigator.vibrate?.(30);
      }
    })
    .on(RoomEvent.Disconnected, () => {
      running = false;
      joinScreen('Disconnected. The host may have ended the session.');
    });
  await room.connect(url, token);
  remember({ name, share, identity });
  const ping = setInterval(
    () =>
      room.localParticipant
        .publishData(encode({ type: 'ping', t: performance.now() }), {
          reliable: false,
          topic: 'punch',
        })
        .catch(() => {}),
    3000,
  );
  $('leave').onclick = () => {
    remember(null);
    running = false;
    clearInterval(ping);
    room.disconnect();
  };

  // Pads always work, so a device that cannot run hand tracking can still play.
  const pads = {
    left: { u: -0.6, v: -0.05, lateral: 0.9, speed: 1.8, side: 'left', kind: 'hook' },
    jab: { u: 0, v: 0.05, lateral: 0, speed: 1.5, side: 'left', kind: 'straight' },
    right: { u: 0.6, v: -0.05, lateral: -0.9, speed: 1.8, side: 'right', kind: 'hook' },
  };
  for (const pad of root.querySelectorAll('[data-pad]'))
    pad.onclick = () => throwPunch(pads[pad.dataset.pad]);

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
      },
      audio: false,
    });
    self.srcObject = stream;
    await self.play();
    if (share)
      await room.localParticipant.publishTrack(stream.getVideoTracks()[0], {
        name: 'cam',
        source: Track.Source.Camera,
      });
    $('track').textContent = 'loading hand tracking…';
    const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
    const landmarker = await HandLandmarker.createFromOptions(
      await FilesetResolver.forVisionTasks(WASM),
      {
        baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
      },
    );
    const detector = new PunchDetector({
      aspect: (self.videoWidth || 4) / (self.videoHeight || 3),
    });
    let lastFrame = -1;
    $('track').textContent = share
      ? 'tracking · camera shared'
      : 'tracking · camera private';
    const loop = () => {
      if (!running) {
        landmarker.close();
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      if (self.readyState >= 2 && self.currentTime !== lastFrame) {
        lastFrame = self.currentTime;
        const now = performance.now(),
          result = landmarker.detectForVideo(self, now);
        result.landmarks.forEach((landmarks, i) => {
          const punch = detector.update(
            result.handednesses[i]?.[0]?.categoryName || 'hand' + i,
            landmarks,
            now,
          );
          if (punch) throwPunch(punch);
        });
      }
      requestAnimationFrame(loop);
    };
    loop();
  } catch (error) {
    $('track').textContent = 'no camera · use the pads';
    toast('No camera (' + error.name + '). The pads still work.');
  }
}

// Straight back in after a reload; one attempt, and a failure falls back to the join card.
const previous = saved();
if (previous)
  enter(previous.name, previous.share).catch((error) => {
    remember(null);
    joinScreen(error.message);
  });
else joinScreen();
