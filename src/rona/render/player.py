"""A self-contained interactive player for a kinetic ensemble.

Writes a single HTML file with the whole time course embedded - no server, no
network, no build step.  Opening it gives a scrubbable, playable view of the
folding ensemble: the structure morphs between sampled time points, base pairs
are shaded by their ensemble probability, and the population and energy panels
carry a synchronised cursor.

This is the counterpart to :mod:`rona.render.movie`.  A movie is what you put in
a talk; the player is what you use to actually look at a result, because you can
stop on a frame and read off which structures are competing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..struct import helices_from_pairtable, iter_pairs, parse_dotbracket
from . import colors
from .layout import LayoutOptions, bounding_box, camera_path, layout_series
from .overview import OPEN_CHAIN, kymograph_data_uri, top_bands


@dataclass(frozen=True, slots=True)
class PlayerOptions:
    title: str = "Cotranscriptional folding ensemble"
    subtitle: str = ""
    pair_threshold: float = 0.05
    max_bands: int = 10
    #: Playback speed in sampled frames per second.
    fps: int = 20


def _frame_payload(
    ensemble, opt: PlayerOptions, layout_options: LayoutOptions | None
) -> dict:
    structures = [s for s, _p in ensemble.dominant()]
    coords = layout_series(structures, options=layout_options)
    n_total = len(ensemble.seq)
    box = bounding_box(coords)
    # per-frame view boxes, so the player pans and zooms like the movie
    camera = camera_path(coords, margin=0.05, min_fraction=0.35)
    probabilities = ensemble.pair_probabilities()

    frames = []
    for index, (structure, frame) in enumerate(zip(structures, coords)):
        from ..energy.pseudoknot import split_crossing

        pt = parse_dotbracket(structure)
        _core, pk_helices = split_crossing(helices_from_pairtable(pt))
        pk = {p for helix in pk_helices for p in helix.pairs}

        matrix = probabilities[index]
        visible = ensemble.lengths[index]
        pairs = []
        for i in range(visible):
            for j in range(i + 1, visible):
                weight = float(matrix[i, j])
                if weight >= opt.pair_threshold:
                    pairs.append([i, j, round(weight, 3), 1 if (i, j) in pk else 0])
        padded = np.zeros((n_total, 2))
        count = len(frame)
        padded[:count] = frame
        if count and count < n_total:
            padded[count:] = frame[count - 1]
        frames.append(
            {
                "xy": [[round(float(x), 2), round(float(y), 2)] for x, y in padded],
                "pairs": pairs,
                "db": structure,
                "len": visible,
                "view": [round(float(v), 3) for v in camera[index]],
            }
        )

    labels, occupancy = ensemble.occupancy(min_population=0.02)
    labels, occupancy = top_bands(labels, occupancy, ensemble.times, opt.max_bands)

    return {
        "sequence": ensemble.seq,
        "times": [round(float(t), 4) for t in ensemble.times],
        "lengths": list(ensemble.lengths),
        "frames": frames,
        "box": [round(v, 3) for v in box],
        "occupancy": [[round(float(v), 4) for v in row] for row in occupancy],
        "structures": labels,
        "energy": [round(float(v), 3) for v in ensemble.mean_energy()],
        "energySd": [round(float(v), 3) for v in ensemble.energy_spread()],
        "pkFraction": [round(float(v), 4) for v in ensemble.pseudoknot_fraction()],
        "nTrajectories": ensemble.n_trajectories,
        "palette": list(colors.OCCUPANCY_PALETTE),
        "baseColors": colors.BASE_COLORS,
        "elementColors": colors.ELEMENT_COLORS,
        "fps": opt.fps,
        "kymograph": kymograph_data_uri(ensemble),
        "openChain": OPEN_CHAIN,
    }


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #ffffff; --panel: #f5f6f8; --line: #dfe2e7; --text: #1c2028;
  --muted: #6d7580; --accent: #2e86ab; --pk: #d1495b;
  --pair: #404752; --backbone: #7a828e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14171c; --panel: #1c2027; --line: #2c323b; --text: #e8eaed;
    --muted: #9aa2ae; --accent: #5aa9d6; --pk: #f07182;
    --pair: #c2c9d4; --backbone: #8b939f;
  }
}
:root[data-theme="dark"] {
  --bg: #14171c; --panel: #1c2027; --line: #2c323b; --text: #e8eaed;
  --muted: #9aa2ae; --accent: #5aa9d6; --pk: #f07182;
  --pair: #c2c9d4; --backbone: #8b939f;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
header { padding: 18px 16px 6px; }
h1 { margin: 0; font-size: 19px; font-weight: 650; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 13px; margin-top: 3px; }
main {
  display: grid; grid-template-columns: minmax(0,1.5fr) minmax(0,1fr);
  gap: 14px; padding: 12px 16px 20px; align-items: start;
}
.span2 { grid-column: 1 / -1; }
@media (max-width: 860px) {
  main { grid-template-columns: 1fr; }
  .span2 { grid-column: auto; }
}
.card {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 12px; min-width: 0;
}
.card h2 {
  margin: 0 0 8px; font-size: 12px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
}
canvas { width: 100%; display: block; border-radius: 6px; }
.controls {
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  padding: 10px 16px; border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line); background: var(--panel);
  position: sticky; top: 0; z-index: 5;
}
button, select {
  font: inherit; color: var(--text); background: var(--bg);
  border: 1px solid var(--line); border-radius: 7px; padding: 5px 12px;
  cursor: pointer;
}
button:hover, select:hover { border-color: var(--accent); }
input[type=range] { flex: 1 1 220px; min-width: 160px; accent-color: var(--accent); }
.readout {
  font-variant-numeric: tabular-nums; color: var(--muted);
  font-size: 13px; white-space: nowrap;
}
.db {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px; word-break: break-all; color: var(--muted);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: 8px; margin-top: 8px; line-height: 1.45;
}
.stats { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 10px; }
.stat { font-variant-numeric: tabular-nums; }
.stat b { display: block; font-size: 17px; font-weight: 650; }
.stat span { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
</header>
<div class="controls">
  <button id="play">&#9654;&nbsp; Play</button>
  <input type="range" id="scrub" min="0" max="100" value="0" step="0.01">
  <span class="readout" id="readout"></span>
  <select id="speed">
    <option value="0.25">0.25&times;</option>
    <option value="0.5">0.5&times;</option>
    <option value="1" selected>1&times;</option>
    <option value="2">2&times;</option>
    <option value="4">4&times;</option>
  </select>
  <select id="mode">
    <option value="ensemble" selected>Ensemble (pairs by probability)</option>
    <option value="dominant">Dominant structure only</option>
  </select>
  <button id="letters">Letters: auto</button>
  <button id="theme">Theme</button>
</div>
<main>
  <section class="card">
    <h2>Structure</h2>
    <canvas id="structure" height="520"></canvas>
    <div class="legend">
      <span><i style="background:#e4572e"></i>A</span>
      <span><i style="background:#2e86ab"></i>C</span>
      <span><i style="background:#3fa34d"></i>G</span>
      <span><i style="background:#8f5fd6"></i>U</span>
      <span><i style="background:var(--pk)"></i>pseudoknot pair</span>
      <span><i style="border:2px solid var(--muted);background:transparent"></i>polymerase</span>
    </div>
    <div class="db" id="dbtext"></div>
  </section>
  <section>
    <div class="card">
      <h2>Time overview &middot; colour = helix identity</h2>
      <canvas id="kymograph" height="190"></canvas>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Structure populations</h2>
      <canvas id="occupancy" height="190"></canvas>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Ensemble free energy &amp; chain growth</h2>
      <canvas id="energy" height="200"></canvas>
      <div class="stats">
        <div class="stat"><b id="statLen">0</b><span>length (nt)</span></div>
        <div class="stat"><b id="statG">0</b><span>&lt;G&gt; kcal/mol</span></div>
        <div class="stat"><b id="statPk">0%</b><span>pseudoknotted</span></div>
        <div class="stat"><b id="statN">0</b><span>trajectories</span></div>
      </div>
    </div>
  </section>
  <section class="card span2">
    <h2>Structures in the ensemble now &middot; tile width = population</h2>
    <canvas id="gallery" height="150"></canvas>
  </section>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("payload").textContent);
  const N = D.sequence.length;
  const F = D.frames.length;

  const css = (name) => getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();

  // ---- playback state -----------------------------------------------------
  let position = 0;      // continuous frame index, so motion is interpolated
  let playing = false;
  let speed = 1;
  let lettersMode = "auto";
  let mode = "ensemble";
  let lastTick = 0;

  const el = (id) => document.getElementById(id);
  const scrub = el("scrub");
  scrub.max = String(F - 1);
  scrub.step = "0.01";

  // ---- canvas helpers -----------------------------------------------------
  function fit(canvas) {
    const ratio = window.devicePixelRatio || 1;
    // The CSS height must be remembered separately: setting canvas.height
    // rewrites the attribute that, absent an explicit CSS height, drives
    // layout - so reading it back on the next frame would compound the device
    // pixel ratio and the canvas would grow without bound on a HiDPI display.
    if (!canvas.dataset.cssHeight) {
      canvas.dataset.cssHeight = canvas.getAttribute("height");
    }
    const height = Number(canvas.dataset.cssHeight);
    const width = canvas.clientWidth || 600;
    canvas.style.height = height + "px";
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx: ctx, w: width, h: height };
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  const OPEN_CHARS = "([{<ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const CLOSE_CHARS = ")]}>abcdefghijklmnopqrstuvwxyz";

  // Dot-bracket with the extended alphabet, so pseudoknots parse too.
  const dbCache = new Map();
  function parseDb(db) {
    let cached = dbCache.get(db);
    if (cached) return cached;
    const stacks = {}, pairs = [];
    for (let k = 0; k < db.length; k++) {
      const oi = OPEN_CHARS.indexOf(db[k]);
      if (oi >= 0) { (stacks[oi] = stacks[oi] || []).push(k); continue; }
      const ci = CLOSE_CHARS.indexOf(db[k]);
      if (ci >= 0) {
        const st = stacks[ci];
        if (st && st.length) pairs.push([st.pop(), k]);
      }
    }
    dbCache.set(db, pairs);
    return pairs;
  }

  // A base pair's colour is set by its "imaginary centre" (i+j)/2, so every
  // pair of one helix shares a colour and keeps it for the helix's lifetime.
  const HUE_CYCLES = 3.5;
  function pairCentreColor(i, j, n, light) {
    const h = (((i + j) / 2 / n) * HUE_CYCLES) % 1;
    const s = 0.62, l = light === undefined ? 0.46 : light;
    const a = s * Math.min(l, 1 - l);
    const f = (shift) => {
      const k = (shift + h * 12) % 12;
      const v = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
      return Math.round(Math.max(0, Math.min(1, v)) * 255);
    };
    return "rgb(" + f(0) + "," + f(8) + "," + f(4) + ")";
  }

  // occupancy column at a continuous frame position, so tiles glide
  function occupancyAt(index) {
    const lo = Math.min(Math.floor(index), F - 1);
    const hi = Math.min(lo + 1, F - 1);
    const t = index - lo;
    return D.occupancy.map((band) => lerp(band[lo], band[hi], t));
  }

  function interpolatedPoints(index) {
    const lo = Math.min(Math.floor(index), F - 1);
    const hi = Math.min(lo + 1, F - 1);
    const raw = index - lo;
    const t = raw * raw * (3 - 2 * raw);   // smoothstep easing
    const A = D.frames[lo].xy, B = D.frames[hi].xy;
    const out = new Array(N);
    for (let k = 0; k < N; k++) {
      out[k] = [lerp(A[k][0], B[k][0], t), lerp(A[k][1], B[k][1], t)];
    }
    return out;
  }

  // ---- structure panel ----------------------------------------------------
  function drawStructure(index) {
    const canvas = el("structure");
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);

    const key = Math.min(Math.round(index), F - 1);
    const frame = D.frames[key];
    const pts = interpolatedPoints(index);
    // interpolate the camera between key frames too, so it glides
    const loK = Math.min(Math.floor(index), F - 1);
    const hiK = Math.min(loK + 1, F - 1);
    const tK = index - loK;
    const va = D.frames[loK].view, vb = D.frames[hiK].view;
    const box = [0, 1, 2, 3].map((k) => lerp(va[k], vb[k], tK));
    const pad = 22;
    const spanX = Math.max(box[2] - box[0], 1e-6);
    const spanY = Math.max(box[3] - box[1], 1e-6);
    const scale = Math.min((w - 2 * pad) / spanX, (h - 2 * pad) / spanY);
    const ox = pad + ((w - 2 * pad) - spanX * scale) / 2;
    const oy = pad + ((h - 2 * pad) - spanY * scale) / 2;
    const X = (p) => (p[0] - box[0]) * scale + ox;
    const Y = (p) => (box[3] - p[1]) * scale + oy;   // flip y for screen coords

    const visible = frame.len;
    const radius = Math.max(1.6, 0.30 * scale);
    const showLetters = lettersMode === "on" ||
      (lettersMode === "auto" && radius >= 5.5);

    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.strokeStyle = css("--backbone");
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = Math.max(1, scale * 0.09);
    ctx.beginPath();
    for (let k = 0; k < visible; k++) {
      const x = X(pts[k]), y = Y(pts[k]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    const pkColor = css("--pk");
    const pairColor = css("--pair");
    for (const [i, j, weight, isPk] of frame.pairs) {
      // opacity AND width carry the ensemble probability, so a pair the
      // population has committed to reads as solid while a minority pairing
      // stays a faint hairline instead of clutter
      const alpha = mode === "dominant" ? (weight > 0.5 ? 0.95 : 0) : 0.08 + 0.87 * weight;
      if (alpha <= 0.02) continue;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = isPk ? pkColor : pairCentreColor(i, j, N);
      ctx.lineWidth = Math.max(
        0.6, scale * (isPk ? 0.11 : 0.085) * (0.3 + 0.7 * weight)
      );
      if (isPk) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(X(pts[i]), Y(pts[i]));
      ctx.lineTo(X(pts[j]), Y(pts[j]));
      ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    for (let k = 0; k < visible; k++) {
      const x = X(pts[k]), y = Y(pts[k]);
      ctx.fillStyle = D.baseColors[D.sequence[k]] || D.baseColors.N;
      ctx.beginPath(); ctx.arc(x, y, radius, 0, 6.2832); ctx.fill();
      ctx.lineWidth = Math.max(0.6, radius * 0.16);
      ctx.strokeStyle = css("--panel"); ctx.stroke();
      if (showLetters) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "600 " + (radius * 1.15).toFixed(1) + "px Inter, sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(D.sequence[k], x, y + 0.5);
      }
    }
    if (visible > 0) {
      const last = pts[visible - 1];
      ctx.strokeStyle = css("--muted");
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(X(last), Y(last), radius + 7, 0, 6.2832);
      ctx.stroke();
      ctx.fillStyle = css("--text");
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(visible >= N ? "3\\u2032" : "RNAP", X(last), Y(last) + radius + 22);
      ctx.fillText("5\\u2032", X(pts[0]), Y(pts[0]) - radius - 10);
    }
    el("dbtext").textContent = frame.db;
  }

  // ---- gallery: one tile per structure, width = its population ------------
  function drawGallery(index) {
    const canvas = el("gallery");
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const column = occupancyAt(index);
    const pad = 6, labelH = 16;
    let cursor = 0;
    const tiles = [];
    for (let k = 0; k < column.length; k++) {
      const v = Math.max(0, column[k]);
      if (v > 0.02) tiles.push({ label: D.structures[k], start: cursor, width: v });
      cursor += v;
    }
    const rest = Math.max(0, 1 - cursor);
    if (rest > 0.02) tiles.push({ label: null, start: cursor, width: rest });

    for (const tile of tiles) {
      const x0 = pad + tile.start * (w - 2 * pad);
      const tw = tile.width * (w - 2 * pad);
      if (tw < 2) continue;
      ctx.fillStyle = css("--bg");
      ctx.strokeStyle = css("--line");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(x0 + 1, 2, Math.max(1, tw - 2), h - labelH - 4);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = css("--muted");
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(Math.round(tile.width * 100) + "%", x0 + tw / 2, h - 4);

      if (!tile.label || tile.label === D.openChain) continue;
      drawArcs(ctx, tile.label, x0 + 4, x0 + tw - 4, h - labelH - 6, 6);
    }
  }

  // A miniature arc diagram: readable at tile size in a way a 2D drawing is not.
  function drawArcs(ctx, db, x0, x1, yBase, yTop) {
    const pairs = parseDb(db);
    const span = x1 - x0;
    if (span <= 2) return;
    ctx.strokeStyle = css("--muted");
    ctx.globalAlpha = 0.6;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, yBase);
    ctx.lineTo(x0 + span * (db.length / N), yBase);
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (!pairs.length) return;
    let widest = 1;
    for (const [i, j] of pairs) widest = Math.max(widest, j - i);
    for (const [i, j] of pairs) {
      const cx = x0 + span * ((i + j) / 2 / N);
      const rx = span * ((j - i) / 2 / N);
      const ry = (yBase - yTop) * Math.pow((j - i) / widest, 0.55);
      ctx.strokeStyle = pairCentreColor(i, j, N);
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.ellipse(cx, yBase, Math.max(rx, 0.5), Math.max(ry, 0.5), 0, Math.PI, 2 * Math.PI);
      ctx.stroke();
    }
  }

  // ---- kymograph ----------------------------------------------------------
  const kymoImage = new Image();
  let kymoReady = false;
  kymoImage.onload = function () { kymoReady = true; render(); };
  kymoImage.src = D.kymograph;

  function drawKymograph(index) {
    const canvas = el("kymograph");
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const padL = 30, padB = 18, padT = 4, padR = 6;
    const pw = w - padL - padR, ph = h - padT - padB;
    if (kymoReady) {
      ctx.imageSmoothingEnabled = false;
      // flip vertically so nucleotide 1 sits at the bottom, as on an axis
      ctx.save();
      ctx.translate(padL, padT + ph);
      ctx.scale(1, -1);
      ctx.drawImage(kymoImage, 0, 0, pw, ph);
      ctx.restore();
    }
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, pw, ph);
    ctx.strokeStyle = css("--text"); ctx.globalAlpha = 0.85; ctx.lineWidth = 1.3;
    const x = padL + (index / Math.max(F - 1, 1)) * pw;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = css("--muted"); ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(String(N), padL - 4, padT + 8);
    ctx.fillText("1", padL - 4, padT + ph);
    ctx.textAlign = "center";
    ctx.fillText(D.times[0].toFixed(1) + " s", padL, h - 5);
    ctx.fillText(D.times[F - 1].toFixed(1) + " s", padL + pw, h - 5);
  }

  // ---- occupancy panel ----------------------------------------------------
  function drawOccupancy(index) {
    const canvas = el("occupancy");
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const padL = 34, padB = 22, padT = 8, padR = 8;
    const pw = w - padL - padR, ph = h - padT - padB;
    const X = (k) => padL + (k / Math.max(F - 1, 1)) * pw;
    const Y = (v) => padT + ph - v * ph;

    const bands = D.occupancy.slice();
    const residual = new Array(F).fill(1);
    for (const band of bands) for (let k = 0; k < F; k++) residual[k] -= band[k];
    for (let k = 0; k < F; k++) residual[k] = Math.max(0, residual[k]);
    const palette = D.structures.map((name, k) =>
      name === D.openChain ? css("--line") : D.palette[k % D.palette.length]);
    if (residual.some((v) => v > 1e-6)) { bands.push(residual); palette.push(css("--muted")); }

    const base = new Array(F).fill(0);
    bands.forEach((band, b) => {
      ctx.beginPath();
      for (let k = 0; k < F; k++) ctx.lineTo(X(k), Y(base[k] + band[k]));
      for (let k = F - 1; k >= 0; k--) ctx.lineTo(X(k), Y(base[k]));
      ctx.closePath();
      ctx.fillStyle = palette[b % palette.length];
      ctx.globalAlpha = 0.88; ctx.fill(); ctx.globalAlpha = 1;
      for (let k = 0; k < F; k++) base[k] += band[k];
    });

    ctx.strokeStyle = css("--line");
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, pw, ph);
    ctx.strokeStyle = css("--text");
    ctx.globalAlpha = 0.8; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(index), padT); ctx.lineTo(X(index), padT + ph); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = css("--muted");
    ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "right";
    ctx.fillText("1.0", padL - 5, padT + 8);
    ctx.fillText("0", padL - 5, padT + ph);
    ctx.textAlign = "center";
    ctx.fillText(D.times[0].toFixed(1) + " s", padL, h - 6);
    ctx.fillText(D.times[F - 1].toFixed(1) + " s", padL + pw, h - 6);
  }

  // ---- energy panel -------------------------------------------------------
  function drawEnergy(index) {
    const canvas = el("energy");
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const padL = 40, padB = 22, padT = 8, padR = 34;
    const pw = w - padL - padR, ph = h - padT - padB;
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < F; k++) {
      lo = Math.min(lo, D.energy[k] - D.energySd[k]);
      hi = Math.max(hi, D.energy[k] + D.energySd[k]);
    }
    if (!isFinite(lo)) { lo = -1; hi = 1; }
    const span = (hi - lo) || 1;
    const X = (k) => padL + (k / Math.max(F - 1, 1)) * pw;
    const Y = (v) => padT + ph - ((v - lo) / span) * ph;

    ctx.beginPath();
    for (let k = 0; k < F; k++) ctx.lineTo(X(k), Y(D.energy[k] + D.energySd[k]));
    for (let k = F - 1; k >= 0; k--) ctx.lineTo(X(k), Y(D.energy[k] - D.energySd[k]));
    ctx.closePath();
    ctx.fillStyle = css("--accent"); ctx.globalAlpha = 0.2; ctx.fill(); ctx.globalAlpha = 1;

    ctx.beginPath();
    for (let k = 0; k < F; k++) ctx.lineTo(X(k), Y(D.energy[k]));
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.stroke();

    const maxLen = Math.max.apply(null, D.lengths) || 1;
    ctx.beginPath();
    for (let k = 0; k < F; k++) ctx.lineTo(X(k), padT + ph - (D.lengths[k] / maxLen) * ph);
    ctx.strokeStyle = "#f2c14e"; ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);

    ctx.strokeStyle = css("--text"); ctx.globalAlpha = 0.8; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(X(index), padT); ctx.lineTo(X(index), padT + ph); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = css("--muted"); ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(hi.toFixed(0), padL - 5, padT + 8);
    ctx.fillText(lo.toFixed(0), padL - 5, padT + ph);
    ctx.textAlign = "left";
    ctx.fillText(maxLen + " nt", padL + pw + 5, padT + 8);
  }

  // ---- frame update -------------------------------------------------------
  function render() {
    const key = Math.min(Math.round(position), F - 1);
    drawStructure(position);
    drawGallery(position);
    drawKymograph(position);
    drawOccupancy(position);
    drawEnergy(position);
    el("readout").textContent =
      "t = " + D.times[key].toFixed(2) + " s   \\u00b7   frame " + (key + 1) + " / " + F;
    el("statLen").textContent = String(D.lengths[key]);
    el("statG").textContent = D.energy[key].toFixed(1);
    el("statPk").textContent = (D.pkFraction[key] * 100).toFixed(0) + "%";
    el("statN").textContent = String(D.nTrajectories);
    scrub.value = String(position);
  }

  function tick(now) {
    if (playing) {
      if (!lastTick) lastTick = now;
      const dt = (now - lastTick) / 1000;
      lastTick = now;
      position += dt * D.fps * speed;
      if (position >= F - 1) { position = F - 1; setPlaying(false); }
      render();
    }
    requestAnimationFrame(tick);
  }

  function setPlaying(value) {
    playing = value;
    lastTick = 0;
    el("play").innerHTML = value ? "&#10074;&#10074;&nbsp; Pause" : "&#9654;&nbsp; Play";
  }

  el("play").addEventListener("click", function () {
    if (!playing && position >= F - 1) position = 0;
    setPlaying(!playing);
  });
  scrub.addEventListener("input", function () {
    position = Number(scrub.value); setPlaying(false); render();
  });
  el("speed").addEventListener("change", function () { speed = Number(this.value); });
  el("mode").addEventListener("change", function () { mode = this.value; render(); });
  el("letters").addEventListener("click", function () {
    lettersMode = lettersMode === "auto" ? "on" : (lettersMode === "on" ? "off" : "auto");
    this.textContent = "Letters: " + lettersMode;
    render();
  });
  el("theme").addEventListener("click", function () {
    const root = document.documentElement;
    const now = root.getAttribute("data-theme");
    root.setAttribute("data-theme", now === "dark" ? "light" : "dark");
    render();
  });
  window.addEventListener("resize", render);
  document.addEventListener("keydown", function (event) {
    if (event.key === " ") { event.preventDefault(); setPlaying(!playing); }
    if (event.key === "ArrowRight") { position = Math.min(F - 1, position + 1); setPlaying(false); render(); }
    if (event.key === "ArrowLeft") { position = Math.max(0, position - 1); setPlaying(false); render(); }
  });

  render();
  requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""


def player_html(
    ensemble,
    *,
    options: PlayerOptions | None = None,
    layout_options: LayoutOptions | None = None,
) -> str:
    """Build the complete HTML document for an ensemble."""
    opt = options or PlayerOptions()
    payload = _frame_payload(ensemble, opt, layout_options)
    subtitle = opt.subtitle or (
        f"{len(ensemble.seq)} nt · {ensemble.n_trajectories} trajectories · "
        f"{len(ensemble.times)} sampled time points"
    )
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__TITLE__", opt.title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__DATA__", data)
    )


def write_player(
    ensemble,
    path: str,
    *,
    options: PlayerOptions | None = None,
    layout_options: LayoutOptions | None = None,
) -> str:
    """Write the interactive player to ``path`` and return it."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            player_html(ensemble, options=options, layout_options=layout_options)
        )
    return path
