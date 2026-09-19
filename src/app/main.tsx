import { createRoot } from 'react-dom/client';
import { useEffect, useRef } from 'react';
import { PlusCard } from '@/components/ui/ruixen-bento-cards';
import { GridCard } from '@/components/ui/grid-card';
import '@/design/index.css';

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

      const slapHud = document.getElementById('slap-hud');
      if (slapHud && right) right.insertBefore(slapHud, right.firstChild);

      moveKeyHintToLeftPanel(stageShell, left);
      wrapAdvancedSection(right);
      customizeContactResponse(right);
      bakeBeatMeButton();
      collapseSlapBody();
      addHeaderLogo();

      stage.removeAttribute('style');
      stage.style.display = 'none';
    })();
  }, []);

  const plusSlot =
    'min-h-0 rounded-none p-0 bg-background border-foreground/60 dark:border-foreground/60 overflow-visible';
  const gridSlot = 'min-h-0 p-0';

  return (
    <>
      <div className="app-shell grid h-dvh grid-cols-12 grid-rows-[auto_1fr] gap-6 bg-background p-6 text-foreground">
        <div className="col-span-12">
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
        <div className="col-span-12 min-h-0 sm:col-span-8 lg:col-span-6">
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

// Lift the Q/E/Space/Drag key hint out of the stage overlay and drop it into
// the left sidebar as its own panel-section (so it matches Face/Camera/Room).
function moveKeyHintToLeftPanel(
  stageShell: HTMLElement | null | undefined,
  leftPanel: HTMLElement | null | undefined,
) {
  if (!stageShell || !leftPanel) return;
  const hint = stageShell.querySelector<HTMLElement>('.stage-bottom > .hint');
  if (!hint) return;

  // Menu-only "Click → Punch" entry. No handler wired — placeholder for a
  // future click-to-punch command.
  const clickEntry = document.createElement('span');
  clickEntry.innerHTML =
    '<kbd class="kbd-icon" aria-label="Click"><svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 9 5 12 1.774-5.226L21 14 9 9z"/><path d="M16.071 16.071 19.5 19.5"/><path d="M7.188 2.239 8.28 5.482"/><path d="M2.24 7.187l3.24 1.092"/><path d="M18.761 2.239 17.671 5.482"/><path d="M23.76 7.187l-3.239 1.092"/></svg>Click</kbd>Punch';
  hint.appendChild(clickEntry);

  const section = document.createElement('section');
  section.className = 'panel-section';
  const heading = document.createElement('h2');
  heading.textContent = 'Keys';
  section.appendChild(heading);
  section.appendChild(hint);
  leftPanel.appendChild(section);
}

// Wrap the technical rig controls behind an Advanced toggle. This same
// section becomes the destination for Softness + Distance sliders too, so
// its heading is renamed to just "Advanced".
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

// Contact-response tweaks: promote Peak deformation into the top metric grid
// (mirroring #compression via observer) and move Softness + Distance sliders
// into the Advanced section.
function customizeContactResponse(rightPanel: HTMLElement | null) {
  if (!rightPanel) return;

  const advancedSection = rightPanel.querySelector<HTMLElement>('.advanced-section');
  const advancedHeading = advancedSection?.querySelector<HTMLElement>('h2');

  // Move Softness + Distance sliders (label + input pairs) to the top of the
  // Advanced section, just below the heading.
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
  // Insert Distance first so, after Softness is inserted after the heading,
  // the visible order becomes Softness → Distance.
  relocate('#distance-value', '#distance');
  relocate('#softness-value', '#softness');

  // Add Peak metric to the grid; mirror #compression via MutationObserver.
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

  // Hide the standalone Peak deformation row. #compression stays in the DOM
  // so main.js's textContent updates keep firing.
  const peakRow = compressionEl?.closest<HTMLElement>('.controls-label');
  if (peakRow) peakRow.style.display = 'none';
}

// Header: hide the two meshy checkboxes (Photo → 3D self, Compress GLB),
// force compress-toggle on so GLBs are always compressed, and promote the
// beat-yourself button to a prominent, always-clickable "Beat me". The
// button already knows how to build a Meshy head on first click if none is
// loaded yet (main.js line 781).
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
}

// Collapse the punch-detector metric list (.slap-body) behind an in-HUD
// Advanced toggle. Metric rows: Approach / Rise / Palm size / Last hit /
// Latency. The webcam view and title stay visible.
function collapseSlapBody() {
  const hud = document.getElementById('slap-hud');
  if (!hud) return;
  const body = hud.querySelector<HTMLElement>('.slap-body');
  if (!body) return;
  body.classList.add('slap-body-collapsed');
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'slap-advanced-toggle';
  toggle.textContent = 'Advanced ▾';
  toggle.setAttribute('aria-expanded', 'false');
  toggle.addEventListener('click', () => {
    const collapsed = body.classList.toggle('slap-body-collapsed');
    toggle.textContent = collapsed ? 'Advanced ▾' : 'Advanced ▴';
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  });
  body.parentElement?.insertBefore(toggle, body);
}

// Prepend the PNG logo to .brand (leftmost item in the top menu bar).
function addHeaderLogo() {
  const brand = document.querySelector<HTMLElement>('.app-shell header .brand');
  if (!brand) return;
  const img = document.createElement('img');
  img.src = '/punching-face-logo.png';
  img.alt = 'Punching Face';
  img.className = 'brand-logo';
  brand.insertBefore(img, brand.firstChild);
}

const root = document.getElementById('react-root');
if (!root) throw new Error('#react-root not found');
createRoot(root).render(<AppShell />);
