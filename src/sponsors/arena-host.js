// Arena host: streams the live 3D head into a LiveKit room and lets everyone in the call hit it.
// Guests track their own hands on their own device and send ~60-byte punch events over the data
// channel, so latency is a data-channel hop, not a video round trip, and no guest video is needed
// for hits to land. Guests are untrusted: every event is clamped and rate limited before physics.
import { sanitizePunch, encode, decode, RateLimit } from './sse.js';
import { obs } from './sentry.js';

const randomRoom = () => 'ring-' + Math.random().toString(36).slice(2, 6);
// The dev server reloads this page whenever any source file changes, which would otherwise end a live
// session mid-demo. The session is remembered per tab and rejoined with the same identity after a reload.
const SESSION = 'punching-face-arena-session';
const saved = () => {
  try {
    return JSON.parse(sessionStorage.getItem(SESSION) || 'null');
  } catch {
    return null;
  }
};
const remember = (value) => {
  try {
    value
      ? sessionStorage.setItem(SESSION, JSON.stringify(value))
      : sessionStorage.removeItem(SESSION);
  } catch {
    /* private mode */
  }
};

export function createArenaHost({
  api,
  panel,
  config,
  stats,
  refreshConfig,
  coach,
  record,
}) {
  panel.innerHTML = /* HTML */ ` <div class="sd-row" style="margin-top:0">
      <input
        type="text"
        data-k="name"
        placeholder="Your name"
        maxlength="24"
        value="Host"
      /><input type="text" data-k="room" placeholder="room" maxlength="40" />
    </div>
    <div class="sd-row">
      <button class="primary" data-k="toggle" style="flex:1">Go live</button
      ><span class="sd-badge" data-k="mode"></span>
    </div>
    <label class="sd-check"
      ><input type="checkbox" data-k="cam" /><span
        >Also share my webcam with the room (the 3D head is always shared)</span
      ></label
    >
    <div class="sd-status" data-k="status"></div>
    <div class="sd-invite" data-k="invite" hidden>
      <canvas data-k="qr" width="132" height="132"></canvas>
      <div style="min-width:0">
        <span class="sd-badge" data-k="kind"></span><code data-k="link"></code>
        <div class="sd-row">
          <button data-k="copy">Copy link</button
          ><button data-k="another" hidden>Next guest</button>
        </div>
        <div class="sd-status" data-k="reach"></div>
      </div>
    </div>
    <div class="sd-tiles" data-k="tiles"></div>
    <table data-k="board"></table>
    <details>
      <summary>LiveKit keys</summary>
      <div class="sd-status">
        Stored only on this computer (<code>.local/secrets/livekit.json</code>, mode
        0600). With nothing here, a local <code>livekit-server --dev</code> is used and
        guests must be on this computer. Other devices need LiveKit Cloud plus the guest
        page on an https host.
      </div>
      <div class="sd-row">
        <input
          type="text"
          data-k="url"
          placeholder="wss://your-project.livekit.cloud"
        />
      </div>
      <div class="sd-row">
        <input
          type="text"
          data-k="apiKey"
          placeholder="API key"
          autocomplete="off"
        /><input
          type="password"
          data-k="apiSecret"
          placeholder="API secret"
          autocomplete="off"
        />
      </div>
      <div class="sd-row">
        <input
          type="text"
          data-k="guestUrl"
          placeholder="https://…/guest.html (deployed guest page)"
        />
      </div>
      <div class="sd-row">
        <input
          type="text"
          data-k="tokenServerId"
          placeholder="Development token server id (one link for everyone)"
        /><button data-k="save">Save</button>
      </div>
    </details>`;
  const el = Object.fromEntries(
    [...panel.querySelectorAll('[data-k]')].map((n) => [n.dataset.k, n]),
  );
  el.room.value = randomRoom();
  let lk = null,
    room = null,
    capture = null,
    published = [];
  const limit = new RateLimit(6),
    tiles = new Map();
  const status = (text, error = false) => {
    el.status.textContent = text;
    el.status.classList.toggle('error', error);
  };
  const post = async (path, body) => {
    const r = await fetch(api + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Request failed.');
    return data;
  };
  for (const input of panel.querySelectorAll('input[type=text],input[type=password]')) {
    input.onkeydown = (e) => e.stopPropagation();
    input.onkeyup = (e) => e.stopPropagation();
  }

  function paint() {
    const cfg = config().livekit;
    el.mode.textContent = room ? 'LIVE' : cfg.configured ? cfg.mode : 'not configured';
    el.mode.classList.toggle('mock', !cfg.configured);
    el.toggle.textContent = room ? 'End session' : 'Go live';
    el.toggle.classList.toggle('danger', !!room);
    el.toggle.classList.toggle('primary', !room);
    el.name.disabled = el.room.disabled = !!room;
  }
  function board() {
    el.board.innerHTML = '';
    for (const row of stats.scoreboard().slice(0, 8)) {
      const tr = el.board.insertRow();
      tr.insertCell().textContent = row.name;
      tr.insertCell().textContent = row.count + ' hits';
      tr.insertCell().textContent = row.max + ' m/s';
    }
  }
  async function showInvite(data) {
    el.invite.hidden = false;
    el.link.textContent = data.invite;
    el.kind.textContent =
      data.inviteKind === 'reusable' ? 'ONE LINK FOR EVERYONE' : 'ONE GUEST PER LINK';
    el.another.hidden = data.inviteKind === 'reusable';
    el.reach.textContent = 'Works for: ' + data.inviteReach + '.';
    el.copy.onclick = () =>
      navigator.clipboard.writeText(data.invite).then(
        () => status('Invite copied.'),
        () => status('Copy failed; select the link instead.', true),
      );
    const { default: QRCode } = await import('qrcode');
    await QRCode.toCanvas(el.qr, data.invite, {
      width: 132,
      margin: 1,
      errorCorrectionLevel: 'L',
    });
  }
  function tile(participant) {
    let t = tiles.get(participant.identity);
    if (!t) {
      t = document.createElement('div');
      t.className = 'sd-tile';
      t.innerHTML = '<span></span>';
      el.tiles.append(t);
      tiles.set(participant.identity, t);
    }
    t.lastElementChild.textContent = participant.name || participant.identity;
    return t;
  }
  const send = (message, options = {}) =>
    room?.localParticipant
      .publishData(encode(message), { reliable: true, topic: 'state', ...options })
      .catch(() => {});

  function onData(payload, participant) {
    if (!participant) return;
    const message = decode(payload);
    if (message?.type === 'ping')
      return void send(
        { type: 'pong', t: message.t },
        { reliable: false, destinationIdentities: [participant.identity] },
      );
    const punch = sanitizePunch(message);
    if (!punch || !limit.allow(participant.identity, performance.now())) return;
    const name = participant.name || participant.identity;
    if (!window.__punchingFace?.remotePunch) {
      // The hook at the end of main.js was lost in an edit. Degrade to the scripted demo hook so the
      // room still sees a reaction, and say so plainly rather than failing silently.
      status(
        'main.js lost its remotePunch hook (see tests/sponsors-hook.test.mjs). Falling back to the scripted left/right hook.',
        true,
      );
      return void document
        .getElementById(punch.u < 0 ? 'left-hook' : 'right-hook')
        ?.click();
    }
    const landed = window.__punchingFace.remotePunch({
      ...punch,
      label: 'REMOTE · ' + name.toUpperCase(),
    });
    // Never drop a punch silently: right after a reload the physics session is still opening, and the
    // thrower deserves to know the head was not ready rather than wonder if their phone is broken.
    if (!landed)
      return void send(
        {
          type: 'miss',
          id: participant.identity,
          reason: window.__punchingFace.state?.physicsError
            ? 'physics error on the host'
            : 'the head is still loading',
        },
        { reliable: false, destinationIdentities: [participant.identity] },
      );
    const event = record(window.__lastContact, {
      id: participant.identity,
      name,
      side: punch.side,
    });
    const t = tile(participant);
    t.classList.add('hit');
    setTimeout(() => t.classList.remove('hit'), 220);
    obs.crumb('arena', 'remote punch', {
      speed: punch.speed,
      kind: punch.kind,
      zone: event.zone,
    });
  }

  async function publishCoach(stream) {
    const track = stream?.getAudioTracks()[0];
    if (!room || !track || published.some((p) => p.name === 'coach')) return;
    published.push({
      name: 'coach',
      pub: await room.localParticipant.publishTrack(track, {
        name: 'coach',
        source: lk.Track.Source.ScreenShareAudio,
      }),
    });
  }
  async function shareCamera(on) {
    const existing = published.find((p) => p.name === 'host-cam');
    if (!on && existing) {
      await room.localParticipant.unpublishTrack(existing.pub.track, true);
      published = published.filter((p) => p !== existing);
      return;
    }
    if (!on || existing || !room) return;
    const source = document.getElementById('webcam')?.srcObject?.getVideoTracks()[0];
    if (!source) {
      el.cam.checked = false;
      return status('Connect the tracking webcam first, then share it.', true);
    }
    // A clone, so LiveKit stopping its copy can never switch off the tracking camera.
    published.push({
      name: 'host-cam',
      pub: await room.localParticipant.publishTrack(source.clone(), {
        name: 'host-cam',
        source: lk.Track.Source.Camera,
      }),
    });
  }

  async function start() {
    const canvas =
      document.querySelector('#stage canvas') || document.querySelector('canvas');
    if (!canvas) return status('The 3D stage is not ready yet.', true);
    el.toggle.disabled = true;
    status('Connecting…');
    try {
      await obs.span(
        'arena.go_live',
        { 'arena.mode': config().livekit.mode || 'none' },
        async () => {
          lk ??= await import('livekit-client');
          const joined = await post('/sponsors/livekit/join', {
            room: el.room.value.trim(),
            role: 'host',
            name: el.name.value,
            identity:
              saved()?.room === el.room.value.trim() ? saved().identity : undefined,
          });
          const next = new lk.Room({ adaptiveStream: true, dynacast: true });
          next
            .on(lk.RoomEvent.DataReceived, onData)
            .on(lk.RoomEvent.TrackSubscribed, (track, pub, p) => {
              if (track.kind === 'video' && pub.trackName !== 'model') {
                const v = track.attach();
                v.muted = true;
                tile(p).prepend(v);
              }
            })
            .on(lk.RoomEvent.TrackUnsubscribed, (track) =>
              track.detach().forEach((n) => n.remove()),
            )
            .on(lk.RoomEvent.ParticipantConnected, (p) => {
              tile(p);
              status(
                `${p.name || p.identity} joined. ${next.remoteParticipants.size} in the ring.`,
              );
              send({ type: 'board', board: stats.scoreboard() });
            })
            .on(lk.RoomEvent.ParticipantDisconnected, (p) => {
              tiles.get(p.identity)?.remove();
              tiles.delete(p.identity);
              status(`${p.name || p.identity} left.`);
            })
            .on(lk.RoomEvent.Disconnected, () => {
              if (room === next) stop('Disconnected from the room.');
            });
          await next.connect(joined.url, joined.token);
          room = next;
          window.__arenaRoom = next; // QA handle, like window.__punchingFace
          // The head itself: the WebGL canvas as a 30 fps video track. main.js creates the renderer with
          // preserveDrawingBuffer, so captured frames are never blank.
          capture = canvas.captureStream(30);
          // One layer, capped at 2.5 Mbps: this exact configuration was verified end to end (a guest decoded the
          // full-resolution head). Simulcast for weak phone connections is a possible later step; it was tried and
          // could not be verified in a headless pane, so it is deliberately not shipped.
          published.push({
            name: 'model',
            pub: await room.localParticipant.publishTrack(capture.getVideoTracks()[0], {
              name: 'model',
              source: lk.Track.Source.ScreenShare,
              simulcast: false,
              videoEncoding: { maxBitrate: 2_500_000, maxFramerate: 30 },
            }),
          });
          await publishCoach(coach.outputStream);
          if (el.cam.checked) await shareCamera(true);
          remember({
            room: joined.room,
            name: joined.name,
            cam: el.cam.checked,
            identity: joined.identity,
          });
          await showInvite(joined);
          status(
            `Live in “${joined.room}”. Share the invite; hits from guests land on the head.`,
          );
          obs.tag('arena.room', joined.room);
        },
      );
    } catch (error) {
      status(error.message, true);
      obs.error(error, { feature: 'arena' });
      await stop();
    } finally {
      el.toggle.disabled = false;
      paint();
    }
  }
  async function stop(message) {
    const closing = room;
    room = null;
    window.__arenaRoom = null;
    capture?.getTracks().forEach((t) => t.stop());
    capture = null;
    published = [];
    try {
      await closing?.disconnect();
    } catch {
      /* already gone */
    }
    for (const t of tiles.values()) t.remove();
    tiles.clear();
    el.invite.hidden = true;
    paint();
    if (message) status(message);
  }

  el.toggle.onclick = () => {
    if (room) {
      remember(null);
      stop('Session ended. Nothing is being streamed.');
    } else start();
  };
  el.cam.onchange = () =>
    shareCamera(el.cam.checked).catch((e) => status(e.message, true));
  el.another.onclick = async () => {
    try {
      await showInvite(
        await post('/sponsors/livekit/invite', { room: el.room.value.trim() }),
      );
      status('New single-guest link ready.');
    } catch (e) {
      status(e.message, true);
    }
  };
  el.save.onclick = async () => {
    const body = { group: 'livekit' };
    for (const key of ['url', 'apiKey', 'apiSecret', 'guestUrl', 'tokenServerId']) {
      if (el[key].value.trim()) body[key] = el[key].value.trim();
      el[key].value = '';
    }
    try {
      await post('/sponsors/settings', body);
      await refreshConfig();
      paint();
      status('LiveKit settings saved on this computer.');
    } catch (e) {
      status(e.message, true);
    }
  };
  window.addEventListener('cornerman:audio', (e) =>
    publishCoach(e.detail).catch(() => {}),
  );
  paint();

  return {
    // Every landed punch, local or remote, is echoed to the room so guests get instant feedback
    // without waiting for the video of the dent to come back.
    onPunch(event) {
      board();
      if (room)
        send({
          type: 'hit',
          id: event.id,
          name: event.name,
          zone: event.zone,
          speed: +event.speed.toFixed(2),
          board: stats.scoreboard().slice(0, 8),
        });
    },
    get hostName() {
      return el.name.value.trim() || 'Host';
    },
    get live() {
      return !!room;
    },
    repaint: paint,
    // Called once the stage is ready. Rejoins the room this tab was hosting before a reload.
    resume() {
      const session = saved();
      if (!session || room) return false;
      el.room.value = session.room;
      el.name.value = session.name;
      el.cam.checked = !!session.cam;
      obs.log('arena.resume_after_reload', { room: session.room });
      start();
      return true;
    },
  };
}
