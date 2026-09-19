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

REACTIONS = {
    'arena.grunt.low':   {'text': 'ugh',      'style': 0.35},
    'arena.grunt.mid':   {'text': 'unh!',     'style': 0.55},
    'arena.grunt.high':  {'text': 'aagh!',    'style': 0.85},
    'arena.tap':         {'text': 'tsk',      'style': 0.20},
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
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128'
        body = json.dumps({
            'text': spec['text'],
            'model_id': 'eleven_multilingual_v2',
            'voice_settings': {'stability': 0.6, 'similarity_boost': 0.8, 'style': spec['style']}
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            'xi-api-key': api_key,
            'Content-Type': 'application/json',
            'Accept': 'audio/mpeg',
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                (OUT / f'{key}.mp3').write_bytes(response.read())
                print(f'  wrote {key}.mp3')
        except Exception as e:
            print(f'  {key}: FAILED — {type(e).__name__}: {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
