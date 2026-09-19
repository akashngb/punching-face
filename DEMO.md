# DEMO.md — Arena script

The 3-minute rehearsal for the Arena punching demo. Setup checklist:

- Headset or lapel mic tested in the loudest spot on the floor.
- Phone hotspot standing by — never trust venue Wi-Fi.
- Pre-scanned head mesh ready (a teammate's) as the backup subject.
- Backup video ready on the laptop.
- `Ctrl-L` overlay hidden by default; toggle only when a judge asks about latency.

---

## Arena — 3 minutes

### 0:00 – 0:20 · Frame the demo

> "This is Punching Face — you scan your head, we reconstruct a personal 3D
> mesh, and then you can hit it. The physics is real Newton soft-tissue on the
> CPU; the AI sparring coach sees you and talks back."

### 0:20 – 1:00 · Live capture (optional; skip if the head is pre-loaded)

Open `http://127.0.0.1:5173/`. Record a short 360° pass (or upload a pre-shot
MOV) → **Create 3D face**. Show the trace/pipeline panel filling in while it
runs. Point out that COLMAP → MakeHuman → texture bake → Newton all happen
locally.

### 1:00 – 2:20 · The judge is the boxer

Open the URL with OMNI on: `http://127.0.0.1:5173/?arena_omni=1`. Camera and
mic on. Prompt the judge to try each beat if they hesitate:

1. **Throw a hook.** Cached grunt fires under 50 ms; the model reacts with a
   taunt on top around 1 s later.
2. **Drop the guard on purpose.** The model calls it out: "guard *up*, you're
   giving me the jab."
3. **Say "hold".** The model stops sparring immediately (barge-in).
4. **Land a combo.** The round summary reports zones hit, top speed and a
   coaching cue.

### 2:20 – 3:00 · Architecture

- "Perception, physics and cached audio stay on device — under 50 ms from
  contact to physical reaction. Model dialogue is around 1 s on top."
- "OMNI runs beside the existing pipeline, never in the critical path. Frame
  encoding and mic capture live in a Worker and an AudioWorklet; network I/O
  is async; the render loop is untouched."
- Optionally toggle `Ctrl-L` for the latency overlay when asked "how do you
  know it's under 1 s?"

---

## Failure modes and how to speak to them honestly

- **OMNI drops mid-round.** The "reconnecting" badge appears; the classifier
  keeps firing local reactions and cached audio. Say: "the physical reactions
  are local. We reconnect with backoff. Nothing dies."
- **Plan C only (no Realtime).** Say so once during the demo: "we're on the
  fallback path — turn-based, no streaming interruption today. The Realtime
  endpoint is a config flip." Overlay chip shows `plan C`.
- **Scan consent gate declines.** The scanning subject can retry; nothing else
  is affected.
- **The judge asks whether this is a deepfake tool.** "Every likeness needs
  live, verified consent from the person themselves, and it lives only inside
  the session."

---

## What we cut (say if directly asked)

- Arena's OMNI tools are behind `?arena_omni=1`; the original Arena works
  unchanged without them.
- Cloned voice degrades to a stock OMNI voice with emotion control if the
  gateway doesn't expose cloning.
- Auto-rigging isn't in the demo path yet; we present with a pre-rigged mesh.

The **never-cut** part: contact → cached reaction → model line. If that works,
the demo works.
