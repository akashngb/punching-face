import './demo-hud.css';

/** A contact-driven HUD: nothing here runs in the rendering or tracking loop. */
export function installDemoHUD({
  setMode,
  setView,
  resetHead,
  onAdvanced,
  onChooseModel,
  onCamera,
  onRoundEnd,
}) {
  const root = document.createElement('div');
  root.className = 'demo-hud';
  root.hidden = true;
  root.innerHTML = /* HTML */ `
    <section class="demo-score" aria-label="Punch score">
      <div class="demo-score-heading">Score</div>
      <div class="demo-score-total" aria-label="Total score">0</div>
      <div class="demo-score-meta">
        <span><strong class="demo-hit-count">0</strong> hits landed</span>
        <span class="demo-score-gain">+0</span>
      </div>
      <div class="demo-strength-heading">
        <span>LAST PUNCH</span><strong class="demo-strength-value">0%</strong>
      </div>
      <div
        class="demo-strength-meter"
        role="meter"
        aria-label="Last punch strength"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow="0"
      >
        <span class="demo-strength-fill"></span>
      </div>
      <div class="demo-round-clock">
        <span>Time left</span>
        <strong id="demo-round-time" role="timer" aria-live="off">1:00</strong>
      </div>
    </section>
    <section class="demo-controls" aria-label="Punching controls">
      <div class="demo-control-label" id="demo-response-label">Head response</div>
      <div class="demo-segment" role="group" aria-labelledby="demo-response-label">
        <button type="button" data-demo-mode="clay" aria-pressed="true">Clay</button>
        <button type="button" data-demo-mode="live" aria-pressed="false">
          Elastic
        </button>
      </div>
      <div class="demo-control-label" id="demo-view-label">View</div>
      <div class="demo-segment" role="group" aria-labelledby="demo-view-label">
        <button type="button" data-demo-view="mesh" aria-pressed="true">Surface</button>
        <button type="button" data-demo-view="clay" aria-pressed="false">
          Geometry
        </button>
        <button type="button" data-demo-view="wire" aria-pressed="false">
          Wireframe
        </button>
      </div>
      <div class="demo-session-actions">
        <button type="button" data-demo-action="reset">Reset head</button>
        <button type="button" data-demo-action="camera">Camera</button>
        <button type="button" data-demo-action="model">Change model</button>
        <button type="button" data-demo-action="advanced">
          Advanced mode <span aria-hidden="true">↗</span>
        </button>
      </div>
    </section>
    <div class="demo-punch-pop" role="status" aria-live="polite" aria-atomic="true">
      <span class="demo-punch-label"></span>
      <span class="demo-punch-points"></span>
    </div>
    <section class="demo-round-result" hidden aria-label="Round result">
      <div class="demo-round-result-heading">Time's up</div>
      <div class="demo-round-result-score">0</div>
      <div class="demo-round-result-caption">
        points · <span class="demo-round-result-hits">0</span> hits landed
      </div>
      <button type="button" class="demo-play-again" data-demo-action="replay">
        Play again
      </button>
      <button type="button" class="demo-result-advanced" data-demo-action="advanced">
        Advanced mode
      </button>
    </section>
    <div class="demo-key-hints">
      <span><kbd>Q</kbd> Left hook</span><span><kbd>E</kbd> Right hook</span>
      <span><kbd>Space</kbd> Uppercut</span><span><kbd>R</kbd> Reset head</span>
    </div>
  `;
  document.body.append(root);

  const totalText = root.querySelector('.demo-score-total');
  const hitText = root.querySelector('.demo-hit-count');
  const gainText = root.querySelector('.demo-score-gain');
  const strengthText = root.querySelector('.demo-strength-value');
  const strengthMeter = root.querySelector('.demo-strength-meter');
  const strengthFill = root.querySelector('.demo-strength-fill');
  const punchPop = root.querySelector('.demo-punch-pop');
  const punchLabel = root.querySelector('.demo-punch-label');
  const punchPoints = root.querySelector('.demo-punch-points');
  const roundTime = root.querySelector('#demo-round-time');
  const roundClock = root.querySelector('.demo-round-clock');
  const roundResult = root.querySelector('.demo-round-result');
  const resultScore = root.querySelector('.demo-round-result-score');
  const resultHits = root.querySelector('.demo-round-result-hits');
  const modeButtons = [...root.querySelectorAll('[data-demo-mode]')];
  const viewButtons = [...root.querySelectorAll('[data-demo-view]')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scoreFormat = new Intl.NumberFormat();
  let active = false;
  let total = 0;
  let hits = 0;
  let popAnimation;
  const roundDurationMs = 60_000;
  let roundStarted = false;
  let roundFinished = false;
  let remainingMs = roundDurationMs;
  let deadline = 0;
  let timer;
  let displayedSeconds = -1;

  function timeRemaining() {
    return active && deadline ? Math.max(0, deadline - performance.now()) : remainingMs;
  }

  function isRoundActive() {
    return active && roundStarted && !roundFinished && timeRemaining() > 0;
  }

  function stopTimer() {
    window.clearInterval(timer);
    timer = undefined;
  }

  function finishRound() {
    if (!roundStarted || roundFinished) return;
    roundFinished = true;
    remainingMs = 0;
    deadline = 0;
    stopTimer();
    popAnimation?.cancel();
    resultScore.textContent = scoreFormat.format(total);
    resultHits.textContent = scoreFormat.format(hits);
    roundResult.hidden = false;
    const result = { score: total, hits, duration: roundDurationMs / 1000 };
    window.dispatchEvent(
      new CustomEvent('punching-face-round-ended', { detail: result }),
    );
    onRoundEnd?.(result);
  }

  function updateTimer() {
    const seconds = Math.ceil(timeRemaining() / 1000);
    if (seconds !== displayedSeconds) {
      displayedSeconds = seconds;
      roundTime.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
      roundTime.setAttribute('aria-label', `${seconds} seconds left`);
      roundClock.classList.toggle('demo-round-urgent', seconds <= 10);
    }
    if (roundStarted && !roundFinished && seconds === 0) finishRound();
  }

  function startRound() {
    stopTimer();
    resetScore();
    remainingMs = roundDurationMs;
    roundStarted = true;
    roundFinished = false;
    deadline = active ? performance.now() + remainingMs : 0;
    roundResult.hidden = true;
    updateTimer();
    if (active) timer = window.setInterval(updateTimer, 250);
    window.dispatchEvent(
      new CustomEvent('punching-face-round-started', {
        detail: { score: 0, hits: 0, duration: roundDurationMs / 1000 },
      }),
    );
  }

  function sync({ mode, view } = {}) {
    if (mode) {
      for (const button of modeButtons) {
        button.setAttribute('aria-pressed', String(button.dataset.demoMode === mode));
      }
    }
    if (view) {
      for (const button of viewButtons) {
        button.setAttribute('aria-pressed', String(button.dataset.demoView === view));
      }
    }
  }

  function resetScore() {
    total = 0;
    hits = 0;
    totalText.textContent = '0';
    hitText.textContent = '0';
    gainText.textContent = '+0';
    strengthText.textContent = '0%';
    strengthMeter.setAttribute('aria-valuenow', '0');
    strengthFill.style.transform = 'scaleX(0)';
    popAnimation?.cancel();
    punchLabel.textContent = '';
    punchPoints.textContent = '';
    resultScore.textContent = '0';
    resultHits.textContent = '0';
  }

  root.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    if (button.dataset.demoMode) {
      setMode(button.dataset.demoMode);
      sync({ mode: button.dataset.demoMode });
    } else if (button.dataset.demoView) {
      setView(button.dataset.demoView);
      sync({ view: button.dataset.demoView });
    } else {
      switch (button.dataset.demoAction) {
        case 'reset':
          resetHead();
          break;
        case 'replay':
          resetHead();
          startRound();
          break;
        case 'camera':
          onCamera();
          break;
        case 'model':
          onChooseModel();
          break;
        case 'advanced':
          onAdvanced();
          break;
      }
    }
  });

  window.addEventListener('punching-face-contact', ({ detail }) => {
    if (!isRoundActive()) {
      if (active && roundStarted && !roundFinished) updateTimer();
      return;
    }
    if (!detail || !Number.isFinite(detail.magnitude)) return;
    const strength = Math.max(0, Math.min(1, detail.magnitude));
    const points = Math.round(strength * 100);
    total += points;
    hits += 1;
    totalText.textContent = scoreFormat.format(total);
    hitText.textContent = scoreFormat.format(hits);
    gainText.textContent = `+${points}`;
    strengthText.textContent = `${points}%`;
    strengthMeter.setAttribute('aria-valuenow', String(points));
    strengthFill.style.transform = `scaleX(${strength})`;

    const side = detail.side === 'left' ? 'LEFT' : 'RIGHT';
    const label =
      detail.mode === 'hook'
        ? `${side} HOOK`
        : detail.mode === 'uppercut'
          ? 'UPPERCUT'
          : detail.mode === 'jab'
            ? 'JAB'
            : 'DIRECT HIT';
    punchLabel.textContent = label;
    punchPoints.textContent = `+${points} POINTS`;
    popAnimation?.cancel();
    popAnimation = punchPop.animate(
      reducedMotion.matches
        ? [{ opacity: 1 }, { opacity: 1, offset: 0.75 }, { opacity: 0 }]
        : [
            { opacity: 0, transform: 'translateY(18px) scale(.72) rotate(-4deg)' },
            {
              opacity: 1,
              transform: 'translateY(0) scale(1.08) rotate(-2deg)',
              offset: 0.12,
            },
            {
              opacity: 1,
              transform: 'translateY(0) scale(1) rotate(-2deg)',
              offset: 0.24,
            },
            {
              opacity: 1,
              transform: 'translateY(-3px) scale(1) rotate(-2deg)',
              offset: 0.68,
            },
            { opacity: 0, transform: 'translateY(-18px) scale(.98) rotate(-2deg)' },
          ],
      { duration: 1050, easing: 'cubic-bezier(.2,.8,.2,1)' },
    );
  });

  return {
    setActive(value) {
      const nextActive = Boolean(value);
      if (nextActive === active) return;
      remainingMs = timeRemaining();
      stopTimer();
      active = nextActive;
      root.hidden = !active;
      deadline =
        active && roundStarted && !roundFinished ? performance.now() + remainingMs : 0;
      updateTimer();
      if (active && roundStarted && !roundFinished)
        timer = window.setInterval(updateTimer, 250);
      if (!active) popAnimation?.cancel();
    },
    startRound,
    get isRoundActive() {
      return isRoundActive();
    },
    sync,
    resetScore,
  };
}
