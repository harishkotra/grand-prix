/*
 * Grand Prix renderer.
 *
 * One PixiJS Application and one WebGL context for the whole board. Each racer
 * gets a masked viewport inside that single context rather than its own canvas,
 * because five WebGL contexts on an 8 GB machine is how you lose the frame rate
 * you were trying to show off.
 *
 * The thing worth looking at is the belief cone: a translucent fan drawn from
 * the car, one wedge per steering option, its angle fixed by what that option
 * means and its length and opacity set by the probability the model returned.
 * That fan is the model's whole answer, not a summary of it, and no chat model
 * can draw one.
 */

import * as PIXI from './vendor/pixi.min.mjs';

const stageEl = document.getElementById('stage');

const app = new PIXI.Application();
await app.init({
  background: 0x05070d,
  antialias: true,
  resizeTo: stageEl,
  resolution: Math.min(2, window.devicePixelRatio || 1),
  autoDensity: true,
});

// PixiJS v8 builds the canvas but does not put it in the document, so without this
// line the whole board renders into a detached canvas and the page looks empty
// while every DOM-driven part (standings, catalogue, clock) works perfectly.
stageEl.appendChild(app.canvas);

// resizeTo only fires on window resize, and the stage is taller before the
// catalogue chips and the footer have rendered. Without this the canvas keeps that
// first measurement and the bottom row of panels is clipped.
// The panels must be re-laid-out too: their rects are in canvas pixels, so a
// resize without a re-layout leaves every panel drawn for the old canvas size.
new ResizeObserver(() => {
  app.resize();
  layoutPanels();
}).observe(stageEl);

const FONT = 'ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const STEER_ORDER = ['hard_left', 'left', 'straight', 'right', 'hard_right'];
const STEER_ANGLE = { hard_left: -34, left: -17, straight: 0, right: 17, hard_right: 34 };
const STEER_COLOR = { hard_left: 0x4cc9f0, left: 0x3a86ff, straight: 0xe9ecef, right: 0xff9f1c, hard_right: 0xff6b6b };
const STEER_SHORT = { hard_left: 'HL', left: 'L', straight: 'S', right: 'R', hard_right: 'HR' };

let state = null;
let catalogue = [];
let selected = new Set();
let tracks = [];
let trackName = '';

function text(content, size, color, weight = '400') {
  return new PIXI.Text({
    text: content,
    style: { fontFamily: FONT, fontSize: size, fill: color, fontWeight: weight },
  });
}

function lerp(from, to, factor) {
  return from + (to - from) * factor;
}

function lerpAngle(from, to, factor) {
  let delta = to - from;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  return from + delta * factor;
}

class Panel {
  constructor(meta) {
    this.meta = meta;
    this.root = new PIXI.Container();
    this.maskG = new PIXI.Graphics();
    this.root.addChild(this.maskG);
    this.root.mask = this.maskG;
    this.bg = new PIXI.Graphics();
    this.trackG = new PIXI.Graphics();
    this.coneG = new PIXI.Graphics();
    this.carG = new PIXI.Graphics();
    this.gaugeG = new PIXI.Graphics();
    this.title = text(meta.label, 15, 0xffffff, '600');
    this.badge = text('', 12, 0x8b949e, '600');
    this.stats = text('', 11.5, 0x9aa4b2);
    this.message = text('', 13, 0xffb4a2, '500');
    this.root.addChild(this.bg, this.trackG, this.coneG, this.carG, this.gaugeG, this.title, this.badge, this.stats, this.message);
    app.stage.addChild(this.root);
    this.rect = { x: 0, y: 0, w: 0, h: 0 };
    this.transform = null;
    this.pos = null;
    this.heading = 0;
    this.points = null;
    this.halfWidth = 9;
  }

  layout(rect) {
    this.rect = rect;
    const { x, y, w, h } = rect;
    this.maskG.clear().rect(x, y, w, h).fill(0xffffff);
    this.bg
      .clear()
      .roundRect(x, y, w, h, 12)
      .fill({ color: 0x0b1020, alpha: 0.92 })
      .stroke({ width: 1, color: 0x1e2a44, alpha: 1 });
    this.title.position.set(x + 14, y + 10);
    this.badge.anchor.set(1, 0);
    this.badge.position.set(x + w - 14, y + 11);
    this.stats.position.set(x + 14, y + h - 46);
    this.message.position.set(x + 14, y + 46);
    if (this.points) this.drawTrack();
  }

  // Map track metres into this panel's pixels, fitting the whole circuit.
  setTrack(points, halfWidth) {
    this.points = points;
    this.halfWidth = halfWidth;
    this.drawTrack();
  }

  drawTrack() {
    if (!this.points || !this.rect.w) return;
    const pad = 16;
    const top = this.rect.y + 34;
    const areaW = this.rect.w - pad * 2;
    const areaH = this.rect.h - 34 - 62;
    const xs = this.points.map((point) => point[0]);
    const ys = this.points.map((point) => point[1]);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const spanX = Math.max(1, Math.max(...xs) - minX);
    const spanY = Math.max(1, Math.max(...ys) - minY);
    const scale = Math.min(areaW / spanX, areaH / spanY);
    const offsetX = this.rect.x + pad + (areaW - spanX * scale) / 2;
    const offsetY = top + (areaH - spanY * scale) / 2;
    this.transform = { scale, offsetX, offsetY, minX, minY };

    const g = this.trackG;
    g.clear();
    const path = () => {
      g.moveTo(...this.map(...this.points[0]));
      for (const point of this.points.slice(1)) g.lineTo(...this.map(...point));
      g.lineTo(...this.map(...this.points[0]));
    };
    path();
    g.stroke({ width: this.halfWidth * 2 * scale + 3, color: 0x10162a, alpha: 1, cap: 'round', join: 'round' });
    path();
    g.stroke({ width: this.halfWidth * 2 * scale, color: 0x1b2437, alpha: 1, cap: 'round', join: 'round' });
    // Centre line, dashed, so the racing line is legible.
    for (let i = 0; i < this.points.length; i += 4) {
      const [x1, y1] = this.map(...this.points[i]);
      const [x2, y2] = this.map(...this.points[(i + 2) % this.points.length]);
      g.moveTo(x1, y1).lineTo(x2, y2);
    }
    g.stroke({ width: 1, color: 0x33405e, alpha: 0.55 });
  }

  map(x, y) {
    const t = this.transform;
    return [t.offsetX + (x - t.minX) * t.scale, t.offsetY + (y - t.minY) * t.scale];
  }

  update(racer, car, delta) {
    const { x, y, w, h } = this.rect;
    if (!car || !this.transform) {
      this.message.text = racer ? 'waiting for the first decision' : '';
      return;
    }
    const target = this.map(car.x, car.y);
    if (!this.pos) {
      this.pos = target;
      this.heading = car.heading;
    } else {
      const factor = Math.min(1, delta * 0.02);
      this.pos = [lerp(this.pos[0], target[0], factor), lerp(this.pos[1], target[1], factor)];
      this.heading = lerpAngle(this.heading, car.heading, factor);
    }

    const unavailable = racer.status !== 'ready';
    const accent = PIXI.Color.shared.setValue(racer.color || '#ffffff').toNumber();
    const probabilities = racer.last_probabilities;

    // Belief cone: one wedge per option, sized by the probability returned.
    const cone = this.coneG;
    cone.clear();
    if (probabilities) {
      const reach = Math.max(38, Math.min(w, h) * 0.30);
      for (const label of STEER_ORDER) {
        const p = probabilities[label] ?? 0;
        const angle = this.heading + (STEER_ANGLE[label] * Math.PI) / 180;
        const length = reach * (0.32 + 1.5 * p);
        const spread = 0.16;
        const tipX = this.pos[0] + Math.cos(angle) * length;
        const tipY = this.pos[1] + Math.sin(angle) * length;
        const leftX = this.pos[0] + Math.cos(angle - spread) * length * 0.55;
        const leftY = this.pos[1] + Math.sin(angle - spread) * length * 0.55;
        const rightX = this.pos[0] + Math.cos(angle + spread) * length * 0.55;
        const rightY = this.pos[1] + Math.sin(angle + spread) * length * 0.55;
        cone
          .poly([this.pos[0], this.pos[1], leftX, leftY, tipX, tipY, rightX, rightY])
          .fill({ color: STEER_COLOR[label], alpha: 0.07 + 0.30 * p });
      }
    }

    // The car itself, with a heading tick so its orientation is readable.
    const body = this.carG;
    body.clear();
    if (unavailable) {
      body.circle(this.pos[0], this.pos[1], 6).fill({ color: 0x3d4757, alpha: 1 });
      body.circle(this.pos[0], this.pos[1], 11).stroke({ width: 1, color: 0x3d4757, alpha: 0.6 });
    } else {
      const off = car.off_track;
      body.circle(this.pos[0], this.pos[1], 13).fill({ color: accent, alpha: 0.12 });
      body.circle(this.pos[0], this.pos[1], 7.5).fill({ color: off ? 0xff6b6b : accent, alpha: 1 });
      body
        .moveTo(this.pos[0], this.pos[1])
        .lineTo(this.pos[0] + Math.cos(this.heading) * 14, this.pos[1] + Math.sin(this.heading) * 14)
        .stroke({ width: 2, color: 0xffffff, alpha: 0.9 });
    }

    // Steering gauge: the five options as bars, the chosen one marked.
    const gauge = this.gaugeG;
    gauge.clear();
    const barW = (w - 28 - 4 * 6) / 5;
    const baseY = y + h - 34;
    const maxH = 20;
    for (let i = 0; i < STEER_ORDER.length; i += 1) {
      const label = STEER_ORDER[i];
      const p = probabilities ? probabilities[label] ?? 0 : 0;
      const barX = x + 14 + i * (barW + 6);
      const barH = Math.max(2, maxH * p);
      const chosen = racer.last_label === label;
      // A dim full-height slot first, so all five options stay readable even when
      // the distribution is peaked and four probabilities are close to zero. Without
      // it those bars collapse to a 2 px sliver and the gauge reads as one bar.
      gauge.roundRect(barX, baseY, barW, maxH, 2).fill({ color: STEER_COLOR[label], alpha: 0.1 });
      gauge.roundRect(barX, baseY + (maxH - barH), barW, barH, 2).fill({
        color: STEER_COLOR[label],
        alpha: probabilities ? 0.55 + 0.45 * p : 0.12,
      });
      if (chosen) {
        gauge.roundRect(barX - 1, baseY - 3, barW + 2, 3, 1.5).fill({ color: 0xffffff, alpha: 0.95 });
      }
    }

    this.title.style.fill = unavailable ? 0x6e7681 : 0xffffff;
    const stateWord = racer.status === 'ready' ? (car.frozen ? 'frozen' : 'racing') : racer.status;
    this.badge.text = stateWord.toUpperCase();
    this.badge.style.fill = racer.status === 'ready' ? 0x7ee787 : 0xff7b72;
    const p50 = racer.latency_p50_ms ? `${racer.latency_p50_ms.toFixed(0)} ms` : '-';
    const entropy = racer.mean_entropy === null || racer.mean_entropy === undefined ? '-' : racer.mean_entropy.toFixed(2);
    const chosenLabel = racer.last_label ? STEER_SHORT[racer.last_label] : '-';
    this.stats.text =
      `decisions ${racer.decisions}   ${racer.decisions_per_sec.toFixed(1)}/s   p50 ${p50}   ` +
      `certainty ${entropy}\n` +
      `chose ${chosenLabel}   lap ${car.lap}/${state?.race?.laps ?? '?'}   ` +
      `lateral ${car.lateral.toFixed(1)} m   spins ${car.spins}`;
    this.message.text = racer.status === 'ready' ? '' : `${racer.detail}`.slice(0, 92);
  }
}

const panels = new Map();

function syncPanels() {
  const racers = state?.racers ?? [];
  for (const racer of racers) {
    if (!panels.has(racer.key)) panels.set(racer.key, new Panel(racer));
  }
  for (const [key, panel] of panels) {
    if (!racers.some((racer) => racer.key === key)) {
      panel.root.destroy();
      panels.delete(key);
    }
  }
  layoutPanels();
}

function layoutPanels() {
  const racers = state?.racers ?? [];
  const count = racers.length;
  const width = app.screen.width;
  const height = app.screen.height;
  if (!count) return;
  const cols = count <= 3 ? count : count === 4 ? 2 : 3;
  const rows = Math.ceil(count / cols);
  const gap = 12;
  const cellW = (width - gap * (cols + 1)) / cols;
  const cellH = (height - gap * (rows + 1)) / rows;
  racers.forEach((racer, index) => {
    const panel = panels.get(racer.key);
    if (!panel) return;
    const col = index % cols;
    const row = Math.floor(index / cols);
    panel.layout({
      x: gap + col * (cellW + gap),
      y: gap + row * (cellH + gap),
      w: cellW,
      h: cellH,
    });
    if (state?.race?.track) panel.setTrack(state.race.track.points, state.race.track.half_width_m);
  });
}

function updateChrome() {
  const race = state?.race;
  document.getElementById('race-id').textContent = state?.race_id ?? '-';
  document.getElementById('clock').textContent = race ? `${race.time.toFixed(1)}s` : '0.0s';
  document.getElementById('laps').textContent = race ? `${race.laps}` : '-';
  const current = state?.track_name ?? trackName;
  const label = tracks.find((track) => track.name === current);
  document.getElementById('track-name').textContent = label ? label.label : (current || '-');
  // The board is empty until a race runs, so say so instead of showing a blank box.
  const hint = document.getElementById('stage-hint');
  if (hint) hint.style.display = race ? 'none' : 'block';

  const standings = state?.standings ?? [];
  const racers = state?.racers ?? [];
  const byKey = Object.fromEntries(racers.map((racer) => [racer.key, racer]));
  document.getElementById('standings').innerHTML = standings
    .map((row) => {
      const racer = byKey[row.key];
      if (!racer) return '';
      const finish = row.finish_time === null ? '' : ` · finished ${row.finish_time.toFixed(1)}s`;
      return `<div class="row"><span class="dot" style="background:${racer.color}"></span>
        <span class="pos">${row.position}</span>
        <span class="name">${racer.label}</span>
        <span class="meta">lap ${row.lap} · ${row.distance_m.toFixed(0)} m · ${row.spins} spins${finish}</span></div>`;
    })
    .join('');

  const notes = state?.notes ?? [];
  document.getElementById('notes').innerHTML = notes
    .map((note) => `<div class="note">${note}</div>`)
    .join('');
}

async function poll() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    const next = await response.json();
    const firstRace = !state?.race && next?.race;
    state = next;
    if (firstRace || panels.size !== (next?.racers?.length ?? 0)) syncPanels();
    updateChrome();
  } catch (error) {
    document.getElementById('notes').innerHTML = `<div class="note">dashboard lost the server: ${error}</div>`;
  }
}

async function loadEngines() {
  const response = await fetch('/api/engines', { cache: 'no-store' });
  const previous = new Set(selected);
  catalogue = (await response.json()).engines ?? [];
  // First load selects everything that answers; a refresh keeps the user's choice.
  selected = new Set(catalogue.filter((engine) => previous.size === 0 || previous.has(engine.key)).map((engine) => engine.key));
  renderCatalogue();
}

function renderCatalogue() {
  const el = document.getElementById('catalogue');
  el.innerHTML = catalogue
    .map((engine) => {
      const on = selected.has(engine.key);
      return `<button class="chip ${engine.probe_ok ? 'ok' : 'bad'} ${on ? 'on' : 'off'}" data-key="${engine.key}">
        <span class="dot" style="background:${engine.color}"></span>
        <b>${engine.label}</b>
        <span class="meta">${engine.kind} · ${engine.probe_detail}</span>
        <span class="tick">${on ? '✓' : ''}</span>
      </button>`;
    })
    .join('');
  for (const chip of el.querySelectorAll('.chip')) {
    chip.addEventListener('click', () => {
      const key = chip.dataset.key;
      if (selected.has(key)) selected.delete(key);
      else selected.add(key);
      renderCatalogue();
    });
  }
  document.getElementById('selection-count').textContent = `${selected.size} of ${catalogue.length} selected`;
}

async function loadTracks() {
  const response = await fetch('/api/tracks', { cache: 'no-store' });
  tracks = (await response.json()).tracks ?? [];
  const select = document.getElementById('track');
  select.innerHTML = tracks
    .map((track) => `<option value="${track.name}">${track.label} · ${Math.round(track.length_m)} m</option>`)
    .join('');
  if (!trackName && tracks.length) trackName = tracks[0].name;
  select.value = trackName;
  select.addEventListener('change', () => {
    trackName = select.value;
  });
}

async function start() {
  const racers = catalogue.filter((engine) => selected.has(engine.key)).map((engine) => engine.key);
  if (!racers.length) {
    document.getElementById('notes').innerHTML = '<div class="note">select at least one racer first</div>';
    return;
  }
  await fetch('/api/race/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ racers, track: trackName }),
  });
}

document.getElementById('start').addEventListener('click', () => start());
document.getElementById('stop').addEventListener('click', () => fetch('/api/race/stop', { method: 'POST' }));
document.getElementById('refresh').addEventListener('click', async () => {
  await loadEngines();
  await loadTracks();
});
document.getElementById('select-all').addEventListener('click', () => {
  selected = new Set(catalogue.map((engine) => engine.key));
  renderCatalogue();
});
document.getElementById('select-none').addEventListener('click', () => {
  selected = new Set();
  renderCatalogue();
});

// Debug handle. Open the console and inspect window.__gp to see exactly what the
// renderer has drawn and where. A blank board is otherwise hard to tell apart from
// a board that is drawing off-screen.
window.__gp = { app, panels, get state() { return state; } };

await loadTracks();
await loadEngines();
syncPanels();
setInterval(poll, 33);
poll();

app.ticker.add(() => {
  if (!state?.racers) return;
  const cars = Object.fromEntries((state.race?.cars ?? []).map((car) => [car.key, car]));
  const delta = app.ticker.deltaMS;
  for (const racer of state.racers) {
    const panel = panels.get(racer.key);
    if (panel) panel.update(racer, cars[racer.key], delta);
  }
});
