#!/usr/bin/env python3
"""OMNI smoke test — Step 1 of OMNI.md.

Detects which plan (A/B/C) the current key + gateway supports, saves any audio
reply to disk, tries a tool-call, and probes voice cloning. Runs against:

  - OMNI_REALTIME_URL if set (Plan A/B; tries WebSocket)
  - OMNI_BASE_URL + /chat/completions (Plan C; guaranteed by yibuapi)

Standard library only, so it runs in .venv (Python 3.9). WebSocket support uses
a tiny inline RFC 6455 client; no `websockets` dependency required.

Usage:
    OMNI_API_KEY=sk-... .venv/bin/python scripts/omni_smoke_test.py
    # optional overrides:
    OMNI_REALTIME_URL=wss://... OMNI_MODEL=qwen3.5-omni-plus \\
        .venv/bin/python scripts/omni_smoke_test.py --image path.jpg --audio path.wav

The Realtime protocol details are still being verified against Alibaba Cloud Model
Studio docs (https://www.alibabacloud.com/help/en/model-studio/realtime). If the
gateway's exact event schema diverges from that document, this script prints the
raw upstream reply so you can adjust the field names. It does not guess and
silently succeed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ROOT / '.local/secrets'
OUT_DIR = ROOT / '.local/omni-smoke'

# Yibuapi challenge audit ledger (required per §6). One canonical path so the
# report tool can find everything in one place.
LEDGER = ROOT / '.local/usage/yibu_api_calls.jsonl'
LEDGER.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YIBU_AUDIT_LOG', str(LEDGER))

_YIBU_PKG = ROOT / '.local/third_party/yibuapi-examples/yibuapi_examples_20260918_v01'
if _YIBU_PKG.is_dir() and str(_YIBU_PKG) not in sys.path:
    sys.path.insert(0, str(_YIBU_PKG))
try:
    from yibu_audit import append_audit_record as _audit  # type: ignore
except Exception:
    _audit = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration loading (env > .env > .local/secrets/omni.json)
# ---------------------------------------------------------------------------

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


def resolve_config() -> dict:
    env = load_env_file(ROOT / '.env')
    secrets = load_secrets()

    def pick(env_key, secret_key, default=None):
        return os.environ.get(env_key) or env.get(env_key) or secrets.get(secret_key) or default

    return {
        'api_key': pick('OMNI_API_KEY', 'apiKey'),
        'base_url': (pick('OMNI_BASE_URL', 'baseUrl', 'https://yibuapi.com/v1') or '').rstrip('/'),
        'realtime_url': pick('OMNI_REALTIME_URL', 'realtimeUrl'),
        'model': pick('OMNI_MODEL', 'model', 'qwen3.5-omni-flash'),
        'realtime_model': pick('OMNI_REALTIME_MODEL', 'realtimeModel', 'qwen3.5-omni-plus-realtime'),
        'voice': pick('OMNI_VOICE', 'voice', 'Ethan'),
        'cloned_voice': pick('OMNI_CLONED_VOICE_ID', 'clonedVoiceId'),
    }


# ---------------------------------------------------------------------------
# Test fixtures (a tiny gray JPEG + a 0.6 s tone WAV) — no assets required
# ---------------------------------------------------------------------------

def tiny_gray_jpeg() -> bytes:
    """A valid 16x16 gray JPEG, hand-encoded. About 350 bytes.

    Written this way so the smoke test has zero dependencies. Any real capture
    will be larger; this proves the image path accepts image/jpeg."""
    # Standard JFIF gray JPEG. Generated once and pinned; smaller than a Pillow
    # runtime just to avoid adding Pillow to the smoke test's dep footprint.
    return base64.b64decode(
        b'/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a'
        b'HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy'
        b'MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAQABADASIA'
        b'AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA'
        b'AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3'
        b'ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWm'
        b'p6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEB'
        b'AAA/APH6KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK/9k=')


def tiny_tone_wav(seconds: float = 0.6, freq: float = 440.0, rate: int = 16000) -> bytes:
    """A short mono 16 kHz PCM WAV, generated in-memory."""
    import math
    n = int(seconds * rate)
    buf = bytearray()
    for i in range(n):
        # Fade in/out so the tone doesn't click.
        env = min(1.0, i / (rate * 0.05), (n - i) / (rate * 0.05))
        s = int(env * 0.25 * 32767 * math.sin(2 * math.pi * freq * i / rate))
        buf += struct.pack('<h', s)
    import io
    out = io.BytesIO()
    with wave.open(out, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(buf))
    return out.getvalue()


# ---------------------------------------------------------------------------
# Plan C: HTTP chat/completions (matches sponsor_server.py::omni_request)
# ---------------------------------------------------------------------------

def _record(model: str, api_key: str, endpoint: str, transport: str, ok: bool,
            latency_s: float, response_json=None, status_code=None, error=None,
            purpose: str = 'punching-face.smoke') -> None:
    if _audit is None:
        return
    try:
        _audit(model=model, api_key=api_key, endpoint=endpoint, purpose=purpose,
               transport=transport, ok=ok, status_code=status_code,
               latency_s=latency_s, response_json=response_json or {}, error=error)
    except Exception:
        pass


def probe_plan_c(cfg: dict, image_b64: str, audio_b64: str, tool_test: bool) -> dict:
    url = cfg['base_url'].rstrip('/') + '/chat/completions'
    body = {
        'model': cfg['model'],
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant. Answer very briefly.'},
            {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image_b64}},
                {'type': 'text', 'text': 'What colour is this image? One word.'},
            ]},
            {'role': 'user', 'content': [
                {'type': 'input_audio', 'input_audio': {'data': 'data:;base64,' + audio_b64, 'format': 'wav'}},
            ]},
        ],
        'stream': False,
        'max_tokens': 60,
        # NOTE: yibuapi's Plan C rejects some voice IDs on some models
        # (e.g. Cherry on qwen3.5-omni-flash → 400). We keep the probe
        # text-only; voice discovery happens against the Realtime path.
    }
    if tool_test:
        body['tools'] = [{
            'type': 'function',
            'function': {
                'name': 'log_finding',
                'description': 'Report what you saw',
                'parameters': {
                    'type': 'object',
                    'properties': {'color': {'type': 'string'}, 'confidence': {'type': 'number'}},
                    'required': ['color'],
                },
            }
        }]
        body['tool_choice'] = 'auto'
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + cfg['api_key']},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        detail = e.read(1200).decode('utf-8', 'replace')
        _record(cfg['model'], cfg['api_key'], url, 'http', ok=False,
                latency_s=(time.perf_counter() - start), status_code=e.code,
                error=f'HTTP {e.code}: {detail[:200]}')
        return {'ok': False, 'plan': 'C', 'status': e.code, 'detail': detail, 'elapsed_ms': elapsed_ms}
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    payload = json.loads(raw)
    _record(cfg['model'], cfg['api_key'], url, 'http', ok=True,
            latency_s=(time.perf_counter() - start), status_code=status,
            response_json=payload)
    result = {'ok': True, 'plan': 'C', 'elapsed_ms': elapsed_ms, 'usage': payload.get('usage')}
    choices = payload.get('choices') or []
    if choices:
        msg = choices[0].get('message') or {}
        result['text'] = msg.get('content')
        result['tool_calls'] = msg.get('tool_calls')
        audio = msg.get('audio') or {}
        if audio.get('data'):
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUT_DIR / f'plan-c-reply-{int(time.time())}.wav'
            path.write_bytes(base64.b64decode(audio['data']))
            result['audio_wav'] = str(path)
    result['raw'] = payload
    return result


# ---------------------------------------------------------------------------
# Plan A: minimal WebSocket client (RFC 6455) — no `websockets` dep
# ---------------------------------------------------------------------------

def ws_connect(url: str, headers: dict, timeout: float = 20.0):
    """Return a (socket, remaining-response-bytes) pair after handshake."""
    import socket

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'wss' else 80)
    path = parsed.path or '/'
    if parsed.query:
        path += '?' + parsed.query
    key = base64.b64encode(uuid.uuid4().bytes).decode()

    raw = socket.create_connection((host, port), timeout=timeout)
    if parsed.scheme == 'wss':
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(raw, server_hostname=host)
    else:
        sock = raw

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
    sock.sendall(('\r\n'.join(lines)).encode())

    # Read response headers until \r\n\r\n
    buf = b''
    sock.settimeout(timeout)
    while b'\r\n\r\n' not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError('Upstream closed before finishing WS handshake.\n' + buf.decode('utf-8', 'replace'))
        buf += chunk
    head, _, rest = buf.partition(b'\r\n\r\n')
    head_text = head.decode('utf-8', 'replace')
    if '101' not in head_text.split('\r\n')[0]:
        raise RuntimeError(f'WS handshake failed:\n{head_text}')
    return sock, rest


def ws_send_text(sock, text: str) -> None:
    payload = text.encode('utf-8')
    _ws_send_frame(sock, 0x1, payload)


def ws_send_bytes(sock, opcode: int, payload: bytes) -> None:
    _ws_send_frame(sock, opcode, payload)


def _ws_send_frame(sock, opcode: int, payload: bytes) -> None:
    header = bytearray([0x80 | opcode])
    length = len(payload)
    mask_bit = 0x80
    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header += struct.pack('>H', length)
    else:
        header.append(mask_bit | 127)
        header += struct.pack('>Q', length)
    mask = os.urandom(4)
    header += mask
    masked = bytearray(len(payload))
    for i, b in enumerate(payload):
        masked[i] = b ^ mask[i % 4]
    sock.sendall(bytes(header) + bytes(masked))


def ws_recv_frame(sock, leftover: bytearray, deadline: float):
    """Read one complete frame. Blocks up to (deadline-now) seconds."""
    def _read_at_least(n):
        while len(leftover) < n:
            sock.settimeout(max(0.1, deadline - time.time()))
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError('Upstream closed while reading frame.')
            leftover.extend(chunk)
    _read_at_least(2)
    b1, b2 = leftover[0], leftover[1]
    fin = bool(b1 & 0x80)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    offset = 2
    if length == 126:
        _read_at_least(4)
        length = struct.unpack('>H', bytes(leftover[2:4]))[0]
        offset = 4
    elif length == 127:
        _read_at_least(10)
        length = struct.unpack('>Q', bytes(leftover[2:10]))[0]
        offset = 10
    mask = b''
    if masked:
        _read_at_least(offset + 4)
        mask = bytes(leftover[offset:offset+4])
        offset += 4
    _read_at_least(offset + length)
    payload = bytes(leftover[offset:offset+length])
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    del leftover[:offset+length]
    return fin, opcode, payload


def probe_plan_a(cfg: dict, image_b64: str, audio_b64: str) -> dict:
    """Try a Realtime WS session. If handshake fails, return the reason
    verbatim so the caller can see whether Plan A or B is the path forward."""
    url = cfg['realtime_url']
    if not url:
        return {'ok': False, 'plan': 'A', 'reason': 'OMNI_REALTIME_URL not set'}
    # yibuapi confirmed URL shape: wss://yibuapi.com/v1/realtime?model=<slug>
    from urllib.parse import urlencode as _q
    full_url = url + ('&' if '?' in url else '?') + _q({'model': cfg['realtime_model']})
    headers = {'Authorization': 'Bearer ' + cfg['api_key']}
    start = time.perf_counter()
    try:
        sock, rest = ws_connect(full_url, headers, timeout=20)
    except Exception as e:
        _record(cfg['realtime_model'], cfg['api_key'], full_url, 'websocket', ok=False,
                latency_s=(time.perf_counter() - start),
                error=f'WS handshake failed: {type(e).__name__}: {e}')
        return {'ok': False, 'plan': 'A', 'reason': f'WS handshake failed: {type(e).__name__}: {e}'}

    leftover = bytearray(rest)
    deadline = time.time() + 60
    done_event: dict = {}

    # Schema confirmed via yibuapi's own realtime example:
    #   1. server pushes session.created
    #   2. we send session.update (modalities, instructions, turn_detection)
    #   3. server responds session.updated
    #   4. we send conversation.item.create (input_text) + response.create
    #   5. server streams response.text.delta / response.audio_transcript.delta
    #   6. server sends response.done with usage
    try:
        # 1. wait for session.created (up to 20 s)
        while time.time() < deadline:
            fin, opcode, payload = ws_recv_frame(sock, leftover, min(time.time() + 20, deadline))
            if opcode == 0x1:
                created = json.loads(payload.decode('utf-8', 'replace'))
                if created.get('type') == 'session.created':
                    break
                if created.get('type') == 'error':
                    _record(cfg['realtime_model'], cfg['api_key'], full_url, 'websocket', ok=False,
                            latency_s=(time.perf_counter() - start),
                            error=json.dumps(created.get('error') or created)[:400])
                    return {'ok': False, 'plan': 'A', 'reason': 'Upstream error before session.created: ' + json.dumps(created)[:400]}

        # Voices verified against this key: Ethan, Serena, Dylan. Cherry / Chelsie
        # are rejected by yibuapi even though the public docs list them.
        session_update = {
            'modalities': ['text', 'audio'],
            'instructions': 'Answer the user briefly and directly.',
            'turn_detection': None,
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
        }
        if cfg.get('voice'):
            session_update['voice'] = cfg['voice']
        ws_send_text(sock, json.dumps({'type': 'session.update', 'session': session_update}))
        ws_send_text(sock, json.dumps({
            'type': 'conversation.item.create',
            'item': {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': 'What colour is this test image? Answer in a single word.'}],
            }
        }))
        # Yibuapi wants audio before image (verified experimentally: an
        # image-first ordering returns "Error append image before append audio").
        ws_send_text(sock, json.dumps({'type': 'input_audio_buffer.append', 'audio': audio_b64}))
        ws_send_text(sock, json.dumps({'type': 'input_audio_buffer.commit'}))
        ws_send_text(sock, json.dumps({'type': 'input_image_buffer.append', 'image': image_b64}))
        ws_send_text(sock, json.dumps({'type': 'response.create'}))

        collected_text: list = []
        collected_audio_b64: list = []
        events_seen: list = []
        while time.time() < deadline:
            fin, opcode, payload = ws_recv_frame(sock, leftover, deadline)
            if opcode == 0x8:
                break
            if opcode == 0x9:
                ws_send_bytes(sock, 0xA, payload)
                continue
            if opcode not in (0x1, 0x2):
                continue
            try:
                msg = json.loads(payload.decode('utf-8', 'replace'))
            except Exception:
                events_seen.append('non-json-frame')
                continue
            events_seen.append(msg.get('type') or 'unknown')
            typ = msg.get('type')
            if typ in ('response.text.delta', 'response.output_text.delta', 'response.audio_transcript.delta'):
                collected_text.append(msg.get('delta') or msg.get('text') or '')
            elif typ in ('response.audio.delta', 'response.output_audio.delta'):
                if msg.get('delta'):
                    collected_audio_b64.append(msg['delta'])
                elif msg.get('audio'):
                    collected_audio_b64.append(msg['audio'])
            elif typ == 'response.done':
                done_event = msg
                break
            elif typ == 'error':
                _record(cfg['realtime_model'], cfg['api_key'], full_url, 'websocket', ok=False,
                        latency_s=(time.perf_counter() - start),
                        error=json.dumps(msg.get('error') or msg)[:400])
                return {'ok': False, 'plan': 'A', 'reason': 'Upstream error: ' + json.dumps(msg)[:400],
                        'events': events_seen}
        result = {'ok': True, 'plan': 'A', 'events': events_seen, 'text': ''.join(collected_text)}
        if collected_audio_b64:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = OUT_DIR / f'plan-a-reply-{int(time.time())}.pcm16'
            audio_path.write_bytes(b''.join(base64.b64decode(chunk) for chunk in collected_audio_b64))
            result['audio_pcm16'] = str(audio_path)
        result['usage'] = (done_event.get('response') or {}).get('usage') or done_event.get('usage')
        _record(cfg['realtime_model'], cfg['api_key'], full_url, 'websocket', ok=True,
                latency_s=(time.perf_counter() - start), status_code=101,
                response_json=done_event)
        return result
    except Exception as e:
        _record(cfg['realtime_model'], cfg['api_key'], full_url, 'websocket', ok=False,
                latency_s=(time.perf_counter() - start),
                response_json=done_event, error=f'{type(e).__name__}: {e}')
        return {'ok': False, 'plan': 'A', 'reason': f'WS session error: {type(e).__name__}: {e}'}
    finally:
        try:
            sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Voice cloning probe (yibuapi may not expose this; document the failure mode)
# ---------------------------------------------------------------------------

def probe_voice_clone(cfg: dict, audio_b64: str) -> dict:
    if not cfg['api_key']:
        return {'ok': False, 'reason': 'no api key'}
    # Try DashScope-style voice registration endpoint (best-effort; the schema
    # differs between Alibaba's public docs and the yibuapi gateway).
    url = cfg['base_url'].rstrip('/') + '/audio/voices'
    body = {'name': f'omni-smoke-{int(time.time())}', 'sample_wav_b64': audio_b64}
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': 'Bearer ' + cfg['api_key']})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            _record('voice-clone', cfg['api_key'], url, 'http', ok=True,
                    latency_s=(time.perf_counter() - start), status_code=response.status,
                    purpose='punching-face.voice_clone')
            return {'ok': True, 'body': json.loads(response.read())}
    except urllib.error.HTTPError as e:
        detail = e.read(600).decode('utf-8', 'replace')
        _record('voice-clone', cfg['api_key'], url, 'http', ok=False,
                latency_s=(time.perf_counter() - start), status_code=e.code,
                error=f'HTTP {e.code}: {detail[:200]}',
                purpose='punching-face.voice_clone')
        return {'ok': False, 'status': e.code, 'detail': detail}
    except Exception as e:
        _record('voice-clone', cfg['api_key'], url, 'http', ok=False,
                latency_s=(time.perf_counter() - start), error=f'{type(e).__name__}: {e}',
                purpose='punching-face.voice_clone')
        return {'ok': False, 'reason': f'{type(e).__name__}: {e}'}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description='Detect OMNI plan (A/B/C).')
    p.add_argument('--image', help='Image file to send instead of the tiny gray fixture.')
    p.add_argument('--audio', help='WAV file (mono 16 kHz PCM16) instead of the fixture tone.')
    p.add_argument('--skip-realtime', action='store_true', help='Skip Plan A even if URL configured.')
    p.add_argument('--skip-voice-clone', action='store_true')
    p.add_argument('--skip-tool', action='store_true')
    args = p.parse_args()

    cfg = resolve_config()
    if not cfg['api_key']:
        print('OMNI_API_KEY not set — nothing to probe. Apply at https://luma.com/0fhypcu0.', file=sys.stderr)
        print('Config seen:', json.dumps({k: v for k, v in cfg.items() if k != 'api_key'}, indent=2))
        return 2

    if args.image:
        image_bytes = Path(args.image).read_bytes()
    else:
        image_bytes = tiny_gray_jpeg()
    if args.audio:
        audio_bytes = Path(args.audio).read_bytes()
    else:
        audio_bytes = tiny_tone_wav()
    image_b64 = base64.b64encode(image_bytes).decode()
    audio_b64 = base64.b64encode(audio_bytes).decode()

    report = {'config': {k: v for k, v in cfg.items() if k != 'api_key'}, 'api_key_present': True}

    if not args.skip_realtime:
        print('\n== Plan A (Realtime WebSocket) ==')
        report['plan_a'] = probe_plan_a(cfg, image_b64, audio_b64)
        print(json.dumps(report['plan_a'], indent=2)[:1200])
    else:
        report['plan_a'] = {'skipped': True}

    print('\n== Plan C (chat/completions) ==')
    try:
        report['plan_c'] = probe_plan_c(cfg, image_b64, audio_b64, tool_test=not args.skip_tool)
        print(json.dumps({k: v for k, v in report['plan_c'].items() if k != 'raw'}, indent=2)[:1200])
    except urllib.error.HTTPError as e:
        report['plan_c'] = {'ok': False, 'status': e.code, 'detail': e.read(600).decode('utf-8', 'replace')}
        print(json.dumps(report['plan_c'], indent=2))
    except Exception as e:
        report['plan_c'] = {'ok': False, 'reason': f'{type(e).__name__}: {e}'}
        print(json.dumps(report['plan_c'], indent=2))

    if not args.skip_voice_clone:
        print('\n== Voice cloning probe ==')
        report['voice_clone'] = probe_voice_clone(cfg, audio_b64)
        print(json.dumps(report['voice_clone'], indent=2)[:800])
    else:
        report['voice_clone'] = {'skipped': True}

    # Decide the plan.
    plan = 'C'
    if report.get('plan_a', {}).get('ok'):
        plan = 'A'
    elif not report.get('plan_c', {}).get('ok'):
        plan = 'blocked'
    report['recommended_plan'] = plan
    print(f'\n== Recommended plan: {plan} ==')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'Full report saved to {OUT_DIR/"report.json"}')

    return 0 if plan != 'blocked' else 3


if __name__ == '__main__':
    sys.exit(main())
