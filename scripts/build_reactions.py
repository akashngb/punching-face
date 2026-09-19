#!/usr/bin/env python3
"""Pre-generate cached reaction audio via ElevenLabs.

Reads `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` from env or `.env`, then
POSTs each short prompt in REACTIONS and writes a WAV under
`public/omni-reactions/<key>.wav`. Falls back to no-op if the key is absent
(the runtime synth in `src/omni/reactions.js` covers that case).

Standard library only; runs in .venv.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'public/omni-reactions'

# The face is the one being hit, so every clip is a reaction to taking a punch,
# never a cue given to the puncher. The ladder is force, not mood.
REACTIONS = {
    'arena.grunt.low':   {'text': 'hn.',      'style': 0.30},   # felt nothing worth admitting
    'arena.grunt.mid':   {'text': 'unh!',     'style': 0.60},   # felt it, will not say so
    'arena.grunt.high':  {'text': 'aagh—',    'style': 0.90},   # that one landed
    'arena.tap':         {'text': 'tsk',      'style': 0.20},   # a tap, not a punch
    'arena.scoff':       {'text': 'pff.',     'style': 0.45},   # mocking a weak shot
    'arena.laugh':       {'text': 'heh.',     'style': 0.50},   # they missed
    'arena.wheeze':      {'text': 'hhhh…',    'style': 0.75},   # getting the breath back
}


def load_env(path):
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


def wav(pcm: bytes, rate: int) -> bytes:
    """Wrap signed 16-bit mono PCM in a canonical 44-byte RIFF header."""
    import struct
    return (b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt '
            + struct.pack('<IHHIIHH', 16, 1, 1, rate, rate * 2, 2, 16)
            + b'data' + struct.pack('<I', len(pcm)) + pcm)


def main() -> int:
    env = load_env(ROOT / '.env')
    api_key = os.environ.get('ELEVENLABS_API_KEY') or env.get('ELEVENLABS_API_KEY')
    voice_id = os.environ.get('ELEVENLABS_VOICE_ID') or env.get('ELEVENLABS_VOICE_ID')
    if not api_key or not voice_id:
        print('ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID unset — nothing to generate.',
              'The runtime synth will cover reactions.', file=sys.stderr)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for key, spec in REACTIONS.items():
        # Raw PCM out, WAV in: `src/omni/reactions.js` fetches `<key>.wav`, so an
        # `.mp3` here would 404 at runtime and silently fall back to the synth.
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_24000'
        body = json.dumps({
            'text': spec['text'],
            'model_id': 'eleven_multilingual_v2',
            'voice_settings': {'stability': 0.6, 'similarity_boost': 0.8, 'style': spec['style']}
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            'xi-api-key': api_key,
            'Content-Type': 'application/json',
            'Accept': 'audio/pcm',
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                (OUT / f'{key}.wav').write_bytes(wav(response.read(), 24000))
                print(f'  wrote {key}.wav')
        except Exception as e:
            print(f'  {key}: FAILED — {type(e).__name__}: {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
