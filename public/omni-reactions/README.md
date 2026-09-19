# omni-reactions

Pre-generated short audio clips (100–400 ms) keyed by `<scenario>.<intent>.<intensity>`.
Named exactly to match `src/omni/reactions.js` — a missing file falls back to a
synthesized tone.

To pre-generate with ElevenLabs, run:

```bash
ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=... \
  .venv/bin/python scripts/build_reactions.py
```

Expected filenames (WAV, mono, 24 kHz preferred):

- `arena.grunt.low.wav`
- `arena.grunt.mid.wav`
- `arena.grunt.high.wav`
- `arena.tap.wav`

Files here are session-scoped for the *demo*; regenerate whenever the persona
voice changes. Commit only the synthetic fallback strategy, never real voice
clones.
