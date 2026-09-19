# omni-reactions

Pre-generated short audio clips (100–400 ms) keyed by `<scenario>.<intent>.<intensity>`.
Named exactly to match `src/omni/reactions.js` — a missing file falls back to a
synthesized tone.

These are the *face's* clips. It is the thing being punched, so every one of them
is a reaction to taking a hit — never a cue given to the person throwing.

To pre-generate with ElevenLabs, run:

```bash
ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=... \
  .venv/bin/python scripts/build_reactions.py
```

Expected filenames (WAV, mono, 24 kHz — what the script writes):

| File | The face is… |
| --- | --- |
| `arena.grunt.low.wav` | not admitting it felt that |
| `arena.grunt.mid.wav` | winded and covering it |
| `arena.grunt.high.wav` | genuinely rocked |
| `arena.tap.wav` | registering contact that wasn't a punch |
| `arena.scoff.wav` | mocking a weak shot |
| `arena.laugh.wav` | enjoying a miss |
| `arena.wheeze.wav` | getting its breath back after a big one |

Files here are session-scoped for the *demo*; regenerate whenever the persona
voice changes. Commit only the synthetic fallback strategy, never real voice
clones.
