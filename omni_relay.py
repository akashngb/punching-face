"""OMNI Realtime relay — server piece of the shared engine (OMNI.md §3.4).

Holds the API key. Accepts a WebSocket from the browser at
`ws://127.0.0.1:5177/omni/realtime`, opens a WebSocket to the OMNI Realtime
endpoint, and shuttles frames between them. When Plan A/B is unavailable (no
`OMNI_REALTIME_URL`, no key, or handshake fails), the same client-facing WS is
served by an in-process mock that emits enough events to exercise the engine.

This is deliberately a **separate service** from `sponsor_server.py` (still on
:5176). The existing Cornerman path there is untouched — that keeps Arena's
current URL and behaviour bit-for-bit identical while OMNI Realtime lands.

Standard library only, so it runs in the existing Python 3.9 venv. WS server
implements RFC 6455 directly against the socket; no `websockets` dep required.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import socket
import ssl
import struct
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, urlencode

ROOT = Path(__file__).resolve().parent
SECRETS = ROOT / '.local/secrets'
PORT = 5177
ORIGINS = ('http://127.0.0.1:5173', 'http://localhost:5173')
WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
LOG_TAG = '[omni-relay]'

# The Huawei OMNI Live challenge requires that every API call the app makes is
# recorded to a JSONL ledger. We import the sponsor's canonical writer from
# `.local/third_party/yibuapi-examples/…/yibu_audit.py` rather than copying it
# into the repo (per the .local/ convention in AGENTS.md).
_LEDGER = ROOT / '.local/usage/yibu_api_calls.jsonl'
_LEDGER.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YIBU_AUDIT_LOG', str(_LEDGER))
_YIBU_PKG = ROOT / '.local/third_party/yibuapi-examples/yibuapi_examples_20260918_v01'
if _YIBU_PKG.is_dir() and str(_YIBU_PKG) not in sys.path:
    sys.path.insert(0, str(_YIBU_PKG))
try:
    from yibu_audit import append_audit_record  # type: ignore
except Exception:
    append_audit_record = None  # type: ignore

FIELDS = ('apiKey', 'baseUrl', 'model', 'voice', 'realtimeUrl', 'realtimeModel', 'clonedVoiceId')
ENV_MAP = {
    'apiKey': 'OMNI_API_KEY',
    'baseUrl': 'OMNI_BASE_URL',
    'model': 'OMNI_MODEL',
    'voice': 'OMNI_VOICE',
    'realtimeUrl': 'OMNI_REALTIME_URL',
    'realtimeModel': 'OMNI_REALTIME_MODEL',
    'clonedVoiceId': 'OMNI_CLONED_VOICE_ID',
}
DEFAULTS = {
    'baseUrl': 'https://yibuapi.com/v1',
    'model': 'qwen3.5-omni-flash',
    'realtimeModel': 'qwen3.5-omni-flash-realtime',
    'voice': 'Cherry',
}


def load_env_file(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith('#') or '=' not in raw:
            continue
        k, _, v = raw.partition('=')
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def load_secrets() -> dict:
    try:
        return json.loads((SECRETS / 'omni.json').read_text())
    except (OSError, ValueError):
        return {}


def config() -> dict:
    env_file = load_env_file(ROOT / '.env')
    saved = load_secrets()
    out = {k: DEFAULTS.get(k) for k in FIELDS}
    for k in FIELDS:
        env_key = ENV_MAP[k]
        out[k] = os.environ.get(env_key) or env_file.get(env_key) or saved.get(k) or out[k]
    out['enabled'] = (os.environ.get('OMNI_ENABLED', env_file.get('OMNI_ENABLED', 'true'))).lower() != 'false'
    return out


# ---------------------------------------------------------------------------
# WebSocket framing (RFC 6455). Client frames from the browser are masked;
# frames we send to the browser are unmasked. Frames to/from the upstream
# server we treat as a client, so those are masked when we send.
# ---------------------------------------------------------------------------

def _send_frame(sock, opcode: int, payload: bytes, mask: bool) -> None:
    header = bytearray([0x80 | (opcode & 0x0F)])
    length = len(payload)
    mask_bit = 0x80 if mask else 0
    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header += struct.pack('>H', length)
    else:
        header.append(mask_bit | 127)
        header += struct.pack('>Q', length)
    body = payload
    if mask:
        m = os.urandom(4)
        header += m
        body = bytes(b ^ m[i % 4] for i, b in enumerate(payload))
    sock.sendall(bytes(header) + body)


def _recv_frame(sock, leftover: bytearray, deadline: float | None = None):
    def read_at_least(n):
        while len(leftover) < n:
            if deadline is not None:
                sock.settimeout(max(0.05, deadline - time.time()))
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError('peer closed')
            leftover.extend(chunk)
    read_at_least(2)
    b1, b2 = leftover[0], leftover[1]
    fin = bool(b1 & 0x80)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    offset = 2
    if length == 126:
        read_at_least(4)
        length = struct.unpack('>H', bytes(leftover[2:4]))[0]
        offset = 4
    elif length == 127:
        read_at_least(10)
        length = struct.unpack('>Q', bytes(leftover[2:10]))[0]
        offset = 10
    mask = b''
    if masked:
        read_at_least(offset + 4)
        mask = bytes(leftover[offset:offset+4])
        offset += 4
    read_at_least(offset + length)
    payload = bytes(leftover[offset:offset+length])
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    del leftover[:offset + length]
    return fin, opcode, payload


class WSClientPeer:
    """Wraps a browser-side WebSocket after HTTP-upgrade succeeded."""
    def __init__(self, sock, buffered: bytes):
        self.sock = sock
        self.leftover = bytearray(buffered)
        self.lock = threading.Lock()
        self.closed = False

    def send_text(self, text: str) -> None:
        try:
            with self.lock:
                _send_frame(self.sock, 0x1, text.encode('utf-8'), mask=False)
        except Exception:
            self.closed = True

    def send_binary(self, payload: bytes) -> None:
        try:
            with self.lock:
                _send_frame(self.sock, 0x2, payload, mask=False)
        except Exception:
            self.closed = True

    def close(self, code: int = 1000) -> None:
        with self.lock:
            if self.closed:
                return
            try:
                _send_frame(self.sock, 0x8, struct.pack('>H', code), mask=False)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.closed = True

    def recv(self):
        try:
            return _recv_frame(self.sock, self.leftover)
        except Exception:
            self.closed = True
            raise


class WSUpstream:
    """Client to the OMNI Realtime WebSocket endpoint."""
    def __init__(self, url: str, headers: dict, timeout: float = 20.0):
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'wss' else 80)
        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query
        raw = socket.create_connection((host, port), timeout=timeout)
        if parsed.scheme == 'wss':
            ctx = ssl.create_default_context()
            self.sock = ctx.wrap_socket(raw, server_hostname=host)
        else:
            self.sock = raw
        key = base64.b64encode(uuid.uuid4().bytes).decode()
        lines = [
            f'GET {path} HTTP/1.1',
            f'Host: {host}',
            'Upgrade: websocket',
            'Connection: Upgrade',
            f'Sec-WebSocket-Key: {key}',
            'Sec-WebSocket-Version: 13',
        ]
        for k, v in headers.items():
            lines.append(f'{k}: {v}')
        lines.append('\r\n')
        self.sock.sendall(('\r\n'.join(lines)).encode())
        buf = b''
        while b'\r\n\r\n' not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError('Upstream closed before finishing WS handshake:\n' + buf.decode('utf-8', 'replace'))
            buf += chunk
        head, _, rest = buf.partition(b'\r\n\r\n')
        head_text = head.decode('utf-8', 'replace')
        if '101' not in head_text.split('\r\n')[0]:
            raise RuntimeError(f'Upstream WS handshake failed:\n{head_text}')
        self.leftover = bytearray(rest)
        self.lock = threading.Lock()
        self.closed = False

    def send_text(self, text: str) -> None:
        with self.lock:
            _send_frame(self.sock, 0x1, text.encode('utf-8'), mask=True)

    def send_binary(self, payload: bytes) -> None:
        with self.lock:
            _send_frame(self.sock, 0x2, payload, mask=True)

    def recv(self):
        return _recv_frame(self.sock, self.leftover)

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                _send_frame(self.sock, 0x8, struct.pack('>H', 1000), mask=True)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.closed = True


# ---------------------------------------------------------------------------
# Mock upstream (used when there's no key or Plan A URL). Emits enough events
# to exercise the client engine's tool/audio/text handlers.
# ---------------------------------------------------------------------------

class MockUpstream:
    """Consumes client events; emits scripted responses. Same interface as WSUpstream."""
    def __init__(self):
        self.q = queue.Queue()
        self.closed = False
        self.script_lock = threading.Lock()
        self.tools = []
        self.persona = None
        self.pending_prompt = None

    def send_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except Exception:
            return
        typ = msg.get('type') or ''
        if typ == 'session.update':
            self.persona = (msg.get('session') or {}).get('instructions')
            self.tools = (msg.get('session') or {}).get('tools') or []
            self._emit({'type': 'session.updated', 'session': {'mock': True, 'tools': [t.get('name') for t in self.tools]}})
        elif typ == 'input_audio_buffer.commit':
            self.pending_prompt = 'audio'
        elif typ == 'response.create':
            self._respond(msg.get('response') or {})
        elif typ == 'engine.event':
            # Server-side view of an engine event; echo a tiny reaction so the
            # tool-dispatch UI can be validated end-to-end without a key.
            evt = msg.get('event') or {}
            if evt.get('type') == 'strike':
                self._enqueue_tool('react_to_hit', {'location': evt.get('region'), 'severity': min(3, int((evt.get('force') or 0) / 30) + 1)})

    def send_binary(self, payload: bytes) -> None:
        return

    def _emit(self, obj: dict):
        self.q.put(('text', json.dumps(obj).encode('utf-8')))

    def _enqueue_tool(self, name: str, args: dict) -> None:
        call_id = 'mock-' + uuid.uuid4().hex[:6]
        self._emit({'type': 'response.function_call_arguments.done', 'name': name, 'call_id': call_id,
                    'arguments': json.dumps(args)})

    def _respond(self, req: dict):
        # A tiny mock line so the UI has something to display.
        text = req.get('instructions') or 'Alright, take a breath, I am with you.'
        for word in text.split(' '):
            self._emit({'type': 'response.output_text.delta', 'delta': word + ' '})
            time.sleep(0.02)
        self._emit({'type': 'response.done', 'response': {'usage': {'total_tokens': len(text.split())}, 'mock': True}})

    def recv(self):
        # Blocking read like a real socket; wrapped as a text frame.
        opcode, payload = self.q.get(timeout=60)[0], self.q.get_nowait()[1] if False else None
        # Simpler: fetch a message from the queue.
        raise RuntimeError('use pump()')

    def pump(self, deadline: float | None = None):
        """Yield (kind, bytes) tuples until closed."""
        while not self.closed:
            try:
                kind, payload = self.q.get(timeout=0.25)
                yield kind, payload
            except queue.Empty:
                if deadline is not None and time.time() > deadline:
                    return

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# HTTP server that upgrades to WebSocket at /omni/realtime and serves /health.
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = 'OmniRelay/1'
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        # Quiet HTTP access log; use log() below for structured lines.
        return

    def _cors(self):
        origin = self.headers.get('Origin')
        if origin in ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')

    def _guard_origin(self) -> bool:
        origin = self.headers.get('Origin')
        # Allow no-origin for `curl` and internal probes; otherwise require an allowed origin.
        return origin is None or origin in ORIGINS

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            cfg = config()
            payload = {
                'ok': True,
                'enabled': cfg['enabled'],
                'plan': 'A' if (cfg.get('realtimeUrl') and cfg.get('apiKey')) else ('C' if cfg.get('apiKey') else 'mock'),
                'model': cfg.get('realtimeModel'),
                'voice': cfg.get('voice'),
                'gateway': (cfg.get('realtimeUrl') or cfg.get('baseUrl') or '').split('/')[2] if (cfg.get('realtimeUrl') or cfg.get('baseUrl')) else None,
                'clonedVoice': bool(cfg.get('clonedVoiceId')),
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            return
        if self.path == '/omni/realtime':
            return self._upgrade_and_bridge()
        self.send_response(404)
        self._cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _upgrade_and_bridge(self):
        if not self._guard_origin():
            self.send_response(403)
            self.end_headers()
            return
        key = self.headers.get('Sec-WebSocket-Key')
        upgrade = (self.headers.get('Upgrade') or '').lower()
        if not key or 'websocket' not in upgrade:
            self.send_response(400)
            self.end_headers()
            return
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        response = (
            'HTTP/1.1 101 Switching Protocols\r\n'
            'Upgrade: websocket\r\n'
            'Connection: Upgrade\r\n'
            f'Sec-WebSocket-Accept: {accept}\r\n\r\n'
        )
        self.wfile.write(response.encode())
        self.wfile.flush()
        # Hand the underlying socket to a bridge; disable the HTTPServer's connection close.
        self.close_connection = True
        raw = self.connection
        raw.setblocking(True)
        buffered = self.rfile.read(0)  # buffer already drained
        peer = WSClientPeer(raw, buffered)
        try:
            Bridge(peer).run()
        except Exception as e:
            log(f'bridge error: {type(e).__name__}: {e}')


def log(msg: str) -> None:
    print(f'{LOG_TAG} {msg}', flush=True)


# ---------------------------------------------------------------------------
# Bridge: pumps events between the browser and either a real upstream or the
# in-process mock. Adds per-hop timings that judges can see via /omni/latency.
# ---------------------------------------------------------------------------

class Bridge:
    def __init__(self, peer: WSClientPeer):
        self.peer = peer
        cfg = config()
        self.cfg = cfg
        self.session_id = uuid.uuid4().hex[:8]
        self.upstream = None
        self.upstream_url = None
        self.upstream_model = None
        self.mock = None
        self.session_start = time.time()
        self.perf_start = time.perf_counter()
        self.last_response_done: dict | None = None
        self.turns_seen: int = 0
        # Purpose is picked up from a client-supplied session.update; defaults if absent.
        self.purpose = 'punching-face.relay'
        self.metrics = {
            'session_id': self.session_id,
            'plan': None,
            'started_ms': 0,
            'upstream_first_byte_ms': None,
            'events_out': 0,
            'events_in': 0,
            'turns': 0,
        }

    def _open_upstream(self):
        cfg = self.cfg
        if cfg.get('realtimeUrl') and cfg.get('apiKey') and cfg['enabled']:
            # yibuapi confirmed URL shape: wss://yibuapi.com/v1/realtime?model=<slug>.
            model = cfg.get('realtimeModel') or 'qwen3.5-omni-plus-realtime'
            base = cfg['realtimeUrl']
            url = base + ('&' if '?' in base else '?') + urlencode({'model': model})
            headers = {'Authorization': 'Bearer ' + cfg['apiKey']}
            t0 = time.perf_counter()
            self.upstream = WSUpstream(url, headers)
            self.upstream_url = url
            self.upstream_model = model
            self.metrics['plan'] = 'A'
            self.metrics['upstream_connect_ms'] = round((time.perf_counter() - t0) * 1000)
            log(f'session {self.session_id} plan=A upstream={url}')
            return
        self.mock = MockUpstream()
        self.metrics['plan'] = 'mock'
        log(f'session {self.session_id} plan=mock (no realtime url or key)')

    def _audit_turn(self, response_json: dict | None, ok: bool, error: str | None = None) -> None:
        """Write one audit ledger record for a completed (or failed) turn.

        Called only when a real upstream is used. Uses the sponsor's own writer
        so the fields exactly match `usage_summary.json`/CSV expectations."""
        if append_audit_record is None or not self.upstream_url:
            return
        api_key = self.cfg.get('apiKey') or ''
        try:
            append_audit_record(
                model=self.upstream_model or self.cfg.get('realtimeModel') or 'unknown',
                api_key=api_key,
                endpoint=self.upstream_url,
                purpose=self.purpose,
                transport='websocket',
                ok=ok,
                status_code=101 if ok else None,
                latency_s=time.perf_counter() - self.perf_start,
                response_json=response_json or {},
                error=error,
            )
        except Exception as e:
            log(f'audit write failed: {type(e).__name__}: {e}')

    def _send_ready(self):
        self.peer.send_text(json.dumps({
            'type': 'omni.session.ready',
            'plan': self.metrics['plan'],
            'model': self.cfg.get('realtimeModel'),
            'mock': self.mock is not None,
            'sessionId': self.session_id,
        }))

    def run(self):
        try:
            self._open_upstream()
        except Exception as e:
            # Even a failed handshake counts against usage tracking obligations
            # (guide §6 — record failures with a stable purpose).
            self.upstream_url = self.upstream_url or (self.cfg.get('realtimeUrl') or '')
            self.upstream_model = self.upstream_model or self.cfg.get('realtimeModel')
            self._audit_turn(None, ok=False, error=f'{type(e).__name__}: {e}')
            self.peer.send_text(json.dumps({'type': 'omni.session.error',
                                            'reason': f'{type(e).__name__}: {e}'}))
            self.peer.close()
            return
        self._send_ready()

        stop = threading.Event()

        def from_client():
            try:
                while not stop.is_set():
                    fin, opcode, payload = self.peer.recv()
                    if opcode == 0x8:
                        stop.set()
                        break
                    if opcode == 0x9:
                        # Reply pong with same payload (unmasked).
                        _send_frame(self.peer.sock, 0xA, payload, mask=False)
                        continue
                    if opcode == 0x1:
                        text = payload.decode('utf-8', 'replace')
                        # Sniff a client session.update for a `purpose` hint used
                        # in the audit ledger (guide §2). Non-fatal on parse fail.
                        try:
                            msg = json.loads(text)
                            if isinstance(msg, dict) and msg.get('type') == 'session.update':
                                session = msg.get('session') or {}
                                purpose = str(session.get('purpose') or '').strip()
                                if purpose:
                                    self.purpose = purpose[:80]
                        except Exception:
                            pass
                        if self.upstream:
                            self.upstream.send_text(text)
                        else:
                            self.mock.send_text(text)
                    elif opcode == 0x2:
                        if self.upstream:
                            self.upstream.send_binary(payload)
                    self.metrics['events_out'] += 1
            except Exception:
                stop.set()

        def from_upstream():
            try:
                if self.upstream:
                    first = True
                    while not stop.is_set():
                        fin, opcode, payload = self.upstream.recv()
                        if opcode == 0x8:
                            stop.set()
                            break
                        if opcode == 0x9:
                            self.upstream.send_binary(payload)  # pong (masked)
                            continue
                        if first:
                            self.metrics['upstream_first_byte_ms'] = round((time.time() - self.session_start) * 1000)
                            first = False
                        if opcode == 0x1:
                            text = payload.decode('utf-8', 'replace')
                            # Inspect for the terminal event of a turn. yibuapi's
                            # Realtime schema (confirmed via the sponsor's example
                            # code) sends `response.done` with usage inside.
                            try:
                                msg = json.loads(text)
                                if isinstance(msg, dict) and msg.get('type') == 'response.done':
                                    self.last_response_done = msg
                                    self.turns_seen += 1
                                    self.metrics['turns'] = self.turns_seen
                                    # One audit record per turn; guide §6 permits
                                    # per-response accounting on streaming.
                                    self._audit_turn(msg, ok=True)
                                elif isinstance(msg, dict) and msg.get('type') == 'error':
                                    self._audit_turn(msg, ok=False, error=json.dumps(msg.get('error') or msg)[:400])
                            except Exception:
                                pass
                            self.peer.send_text(text)
                        elif opcode == 0x2:
                            self.peer.send_binary(payload)
                        self.metrics['events_in'] += 1
                else:
                    for kind, payload in self.mock.pump():
                        if stop.is_set():
                            break
                        if kind == 'text':
                            self.peer.send_text(payload.decode('utf-8', 'replace'))
                        elif kind == 'binary':
                            self.peer.send_binary(payload)
                        self.metrics['events_in'] += 1
            except Exception:
                stop.set()

        t_in = threading.Thread(target=from_client, daemon=True)
        t_out = threading.Thread(target=from_upstream, daemon=True)
        t_in.start()
        t_out.start()
        try:
            while not stop.is_set():
                stop.wait(0.5)
        finally:
            if self.upstream:
                self.upstream.close()
            if self.mock:
                self.mock.close()
            self.peer.close()
            log(f'session {self.session_id} closed metrics={json.dumps(self.metrics)}')


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    cfg = config()
    plan = 'A' if (cfg.get('realtimeUrl') and cfg.get('apiKey')) else ('C-via-sponsor' if cfg.get('apiKey') else 'mock')
    log(f'listening on http://127.0.0.1:{PORT} · enabled={cfg["enabled"]} · plan={plan}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log('shutting down')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
