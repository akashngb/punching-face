import { createRoot } from 'react-dom/client';
import { useEffect, useRef } from 'react';
import { PlusCard } from '@/components/ui/ruixen-bento-cards';
import { GridCard } from '@/components/ui/grid-card';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import '@/design/index.css';

type PunchingFace = {
  __punchingFace?: {
    state?: { calibrated?: boolean; cameraActive?: boolean };
  };
};

function AppShell() {
  const headerSlot = useRef<HTMLDivElement>(null);
  const leftSlot = useRef<HTMLDivElement>(null);
  const stageSlot = useRef<HTMLDivElement>(null);
  const rightSlot = useRef<HTMLDivElement>(null);
  const appStaging = useRef<HTMLDivElement>(null);
  const done = useRef(false);

  useEffect(() => {
    if (done.current) return;
    done.current = true;

    (async () => {
      await import('@/main.js');

      const stage = appStaging.current;
      if (!stage) return;

      const header = stage.querySelector<HTMLElement>(':scope > header');
      const main = stage.querySelector<HTMLElement>(':scope > main');
      const dialogs = stage.querySelectorAll<HTMLDialogElement>(':scope > dialog');

      const left = main?.querySelector<HTMLElement>(':scope > aside.left');
      const stageShell = main?.querySelector<HTMLElement>(':scope > section.stage-shell');
      const right = main?.querySelector<HTMLElement>(':scope > aside.right');

      if (header && headerSlot.current) headerSlot.current.appendChild(header);
      if (left && leftSlot.current) leftSlot.current.appendChild(left);
      if (stageShell && stageSlot.current) stageSlot.current.appendChild(stageShell);
      if (right && rightSlot.current) rightSlot.current.appendChild(right);
      dialogs.forEach((d) => document.body.appendChild(d));

      // Punch detector: keep #slap-hud alive in the DOM (main.js still binds
      // the media stream to #slap-preview and toggles .active on the hud),
      // but hide it visually. Lift the hand-landmark SVG overlay out of the
      // hud and drop it on top of the #webcam preview.
      const slapHud = document.getElementById('slap-hud');
      const slapHand = document.getElementById('slap-view-hand');
      const webcam = document.getElementById('webcam');
      if (webcam && slapHand) {
        const wrap = document.createElement('div');
        wrap.className = 'camera-view';
        webcam.parentElement?.insertBefore(wrap, webcam);
        wrap.appendChild(webcam);
        wrap.appendChild(slapHand);
      }
      if (slapHud) slapHud.style.display = 'none';

      moveKeyHintToLeftPanel(stageShell, left);
      wrapAdvancedSection(right);
      customizeContactResponse(right);
      bakeBeatMeButton();
      addHeaderLogo();
      hideRoomSection(left);
      autoCalibrateOnCameraConnect(left);
      // The loader and tracking loop still update these status elements.
      const stageTop = stageShell?.querySelector<HTMLElement>('.stage-top');
      if (stageTop) stageTop.style.display = 'none';
      swapViewSwitch(stageShell);
      installImmersiveKeys();
    })();
  }, []);

  const plusSlot =
    'min-h-0 rounded-none p-0 bg-background border-foreground/60 dark:border-foreground/60 overflow-visible';
  const gridSlot = 'min-h-0 p-0';

  return (
    <>
      <div className="app-shell grid h-dvh grid-cols-12 grid-rows-[auto_1fr] gap-6 bg-background p-6 text-foreground">
        <div className="col-header col-span-12">
          <PlusCard className={`${plusSlot} h-16`}>
            <div ref={headerSlot} className="relative z-10 h-full w-full" />
          </PlusCard>
        </div>
        <div className="col-left col-span-12 min-h-0 sm:col-span-4 lg:col-span-3">
          <GridCard className={`${gridSlot} h-full`}>
            <div
              ref={leftSlot}
              className="relative z-10 min-h-0 w-full flex-1 overflow-auto"
            />
          </GridCard>
        </div>
        <div className="col-stage col-span-12 min-h-0 sm:col-span-8 lg:col-span-6">
          <PlusCard className={`${plusSlot} h-full`}>
            <div
              ref={stageSlot}
              className="relative z-10 h-full min-h-0 w-full flex-1"
            />
          </PlusCard>
        </div>
        <div className="col-right col-span-12 min-h-0 lg:col-span-3">
          <GridCard className={`${gridSlot} h-full`}>
            <div
              ref={rightSlot}
              className="relative z-10 min-h-0 w-full flex-1 overflow-auto"
            />
          </GridCard>
        </div>
      </div>

      <div
        id="app"
        ref={appStaging}
        style={{
          position: 'absolute',
          left: '-99999px',
          top: 0,
          width: '1400px',
          height: '900px',
          overflow: 'hidden',
          visibility: 'hidden',
        }}
      />
    </>
  );
}

function moveKeyHintToLeftPanel(
  stageShell: HTMLElement | null | undefined,
  leftPanel: HTMLElement | null | undefined,
) {
  if (!stageShell || !leftPanel) return;
  const hint = stageShell.querySelector<HTMLElement>('.stage-bottom > .hint');
  if (!hint) return;

  const clickEntry = document.createElement('span');
  clickEntry.innerHTML =
    '<kbd class="kbd-icon" aria-label="Click"><svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 9 5 12 1.774-5.226L21 14 9 9z"/><path d="M16.071 16.071 19.5 19.5"/><path d="M7.188 2.239 8.28 5.482"/><path d="M2.24 7.187l3.24 1.092"/><path d="M18.761 2.239 17.671 5.482"/><path d="M23.76 7.187l-3.239 1.092"/></svg>Click</kbd>Punch';
  hint.insertBefore(clickEntry, hint.firstChild);

  const section = document.createElement('section');
  section.className = 'panel-section';
  const heading = document.createElement('h2');
  heading.textContent = 'Keys';
  section.appendChild(heading);
  section.appendChild(hint);
  leftPanel.appendChild(section);
}

function wrapAdvancedSection(rightPanel: HTMLElement | null) {
  if (!rightPanel) return;
  const sections = rightPanel.querySelectorAll<HTMLElement>(':scope > section.panel-section');
  for (const section of sections) {
    const heading = section.querySelector('h2');
    if (!heading || heading.textContent?.trim() !== 'Surface & rig') continue;

    heading.textContent = 'Advanced';
    section.classList.add('advanced-section');
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'small full advanced-toggle';
    toggle.textContent = 'Advanced ▾';
    toggle.setAttribute('aria-expanded', 'false');
    toggle.addEventListener('click', () => {
      const expanded = section.classList.toggle('expanded');
      toggle.textContent = expanded ? 'Advanced ▴' : 'Advanced ▾';
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    });
    section.insertBefore(toggle, section.firstChild);
    return;
  }
}

function customizeContactResponse(rightPanel: HTMLElement | null) {
  if (!rightPanel) return;

  const advancedSection = rightPanel.querySelector<HTMLElement>('.advanced-section');
  const advancedHeading = advancedSection?.querySelector<HTMLElement>('h2');

  const relocate = (labelSelector: string, inputSelector: string) => {
    const label = rightPanel
      .querySelector<HTMLElement>(labelSelector)
      ?.closest<HTMLElement>('.controls-label');
    const input = rightPanel.querySelector<HTMLElement>(inputSelector);
    if (advancedSection && advancedHeading && label && input) {
      const anchor = advancedHeading.nextSibling;
      advancedSection.insertBefore(label, anchor);
      advancedSection.insertBefore(input, label.nextSibling);
    }
  };
  relocate('#distance-value', '#distance');
  relocate('#softness-value', '#softness');
  relocate('#impact-strength-value', '#impact-strength');

  const grid = rightPanel.querySelector<HTMLElement>('.metric-grid');
  const compressionEl = rightPanel.querySelector<HTMLElement>('#compression');
  if (grid && compressionEl) {
    const peakMetric = document.createElement('div');
    peakMetric.className = 'metric';
    peakMetric.innerHTML =
      '<strong><span class="peak-mirror">0.0</span><em>mm</em></strong><small>Peak</small>';
    grid.appendChild(peakMetric);
    const mirror = peakMetric.querySelector<HTMLElement>('.peak-mirror');
    const sync = () => {
      if (!mirror) return;
      const num = (compressionEl.textContent || '').replace(/[^\d.]/g, '');
      mirror.textContent = num || '0.0';
    };
    sync();
    new MutationObserver(sync).observe(compressionEl, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }

  const peakRow = compressionEl?.closest<HTMLElement>('.controls-label');
  if (peakRow) peakRow.style.display = 'none';

  // Hide the hook buttons, but retain them for meshControls() and Q/E input.
  const hookRow = rightPanel.querySelector('#left-hook')?.closest<HTMLElement>('.row');
  if (hookRow) hookRow.style.display = 'none';
}

function bakeBeatMeButton() {
  const meshyPanel = document.getElementById('meshy-panel');
  if (!meshyPanel) return;
  meshyPanel.querySelectorAll<HTMLElement>('.meshy-check').forEach((el) => {
    el.style.display = 'none';
  });
  const compress = document.getElementById('compress-toggle') as HTMLInputElement | null;
  if (compress) compress.checked = true;
  const btn = document.getElementById('beat-yourself') as HTMLButtonElement | null;
  if (btn) {
    btn.textContent = 'Beat me';
    btn.disabled = false;
    btn.classList.remove('small');
    btn.classList.add('primary');
  }
  const captureOpen = document.getElementById('capture-open');
  if (captureOpen) captureOpen.style.display = 'none';
}

// Replace the vanilla Surface/Geometry/Wireframe .view-switch with the
// AnimatedTabs React component. Preserves main.js's setView() wiring by
// capturing each original button's onclick and calling it when the matching
// tab activates.
function swapViewSwitch(stageShell: HTMLElement | null | undefined) {
  if (!stageShell) return;
  const box = stageShell.querySelector<HTMLElement>('.view-switch');
  if (!box) return;
  const buttons = Array.from(box.querySelectorAll<HTMLButtonElement>('button'));
  if (!buttons.length) return;

  const tabs = buttons.map((b) => ({ id: b.id, label: b.textContent?.trim() ?? b.id }));
  const handlers = new Map<string, (() => void) | null>(
    buttons.map((b) => [b.id, b.onclick ? (b.onclick.bind(b) as () => void) : null]),
  );
  const defaultTab = buttons.find((b) => b.classList.contains('active'))?.id ?? tabs[0].id;

  // Keep the original IDs and handlers alive for installMesh() and setView().
  const originalControls = document.createElement('div');
  originalControls.style.display = 'none';
  originalControls.append(...buttons);
  const tabsMount = document.createElement('div');
  box.replaceChildren(originalControls, tabsMount);
  box.classList.add('view-switch-react');
  createRoot(tabsMount).render(
    <AnimatedTabs
      tabs={tabs}
      defaultTab={defaultTab}
      onChange={(id) => handlers.get(id)?.()}
    />,
  );
}

// After the user connects the webcam, wait for cameraActive to flip true,
// then trigger calibration once and hide the now-redundant Calibrate +
// Scan my arms buttons so the left sidebar stays scroll-free.
function autoCalibrateOnCameraConnect(leftPanel: HTMLElement | null | undefined) {
  const calibrate = document.getElementById('calibrate') as HTMLButtonElement | null;
  const scanArms = document.getElementById('scan-arms') as HTMLElement | null;
  const armAppearance = document.getElementById('arm-appearance');
  if (armAppearance) armAppearance.style.display = 'none';
  if (scanArms) scanArms.style.display = 'none';

  if (!calibrate) return;
  let done = false;
  const id = window.setInterval(() => {
    const win = window as unknown as PunchingFace;
    const state = win.__punchingFace?.state;
    if (!done && state?.cameraActive) {
      done = true;
      window.setTimeout(() => calibrate.click(), 250);
      window.setTimeout(() => {
        // cameraToggle() still updates this control when disconnecting.
        calibrate.style.display = 'none';
        clearInterval(id);
      }, 750);
    }
  }, 250);
  // Failsafe: stop polling after a minute even if no camera connects.
  window.setTimeout(() => clearInterval(id), 60_000);
  // If the sidebar has a solitary muted line under the buttons, drop it.
  leftPanel?.querySelectorAll('.muted').forEach((el) => {
    const text = el.textContent?.trim() ?? '';
    if (/awaiting capture|calibrat/i.test(text)) (el as HTMLElement).style.display = 'none';
  });
}

// Immersive fullscreen keyboard support:
//   ESC     — exit immersive
//   Enter   — while immersive, toggle a fixed bottom-left camera panel
function installImmersiveKeys() {
  window.addEventListener('keydown', (e) => {
    const inField =
      e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
    if (inField) return;
    const body = document.body;
    if (e.key === 'Escape' && body.classList.contains('immersive')) {
      body.classList.remove('immersive', 'camera-open');
      e.preventDefault();
    } else if (e.key === 'Enter' && body.classList.contains('immersive')) {
      body.classList.toggle('camera-open');
      e.preventDefault();
    }
  });
}

function hideRoomSection(leftPanel: HTMLElement | null | undefined) {
  if (!leftPanel) return;
  leftPanel
    .querySelectorAll<HTMLElement>(':scope > section.panel-section')
    .forEach((section) => {
      const h2 = section.querySelector('h2');
      if (h2 && h2.textContent?.trim() === 'Room') section.style.display = 'none';
    });
}

function addHeaderLogo() {
  const brand = document.querySelector<HTMLElement>('.app-shell header .brand');
  if (!brand) return;
  brand.querySelector(':scope > svg')?.remove();
  const img = document.createElement('img');
  img.src = '/punching-face-logo.png';
  img.alt = 'Punching Face';
  img.className = 'brand-logo';
  brand.insertBefore(img, brand.firstChild);
}

const root = document.getElementById('react-root');
if (!root) throw new Error('#react-root not found');
createRoot(root).render(<AppShell />);
