"""Dependency-free SVG plots of a kinetic ensemble.

The three views that matter for cotranscriptional folding:

``occupancy_plot``
    A stacked-area plot of structure populations against time - the folding
    pathway.  Bands are ordered by first appearance, so the plot reads
    left-to-right as "this structure forms, then gives way to that one".

``energy_plot``
    Mean free energy with its ensemble spread, plus the transcript length, which
    makes it obvious which features are transcription-driven.

``pair_probability_plot``
    ``P(i, j)`` at one instant, drawn as the usual upper-triangular dot plot,
    with crossing pairs highlighted.

All of them emit plain SVG, so they work anywhere and stay sharp at any size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import colors
from .svg import _esc


@dataclass(frozen=True, slots=True)
class PlotOptions:
    width: int = 900
    height: int = 340
    left: float = 62.0
    right: float = 168.0
    top: float = 40.0
    bottom: float = 46.0
    title: str = ""
    background: str = colors.ELEMENT_COLORS["background"]


def _axes(
    opt: PlotOptions,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    x_label: str,
    y_label: str,
    x_ticks: int = 6,
    y_ticks: int = 5,
    y_format: str = "{:.2f}",
) -> tuple[list[str], callable, callable]:
    """Emit axis furniture and return the two data->pixel mappings."""
    plot_w = opt.width - opt.left - opt.right
    plot_h = opt.height - opt.top - opt.bottom
    x0, x1 = x_range
    y0, y1 = y_range
    span_x = (x1 - x0) or 1.0
    span_y = (y1 - y0) or 1.0

    def sx(value: float) -> float:
        return opt.left + (value - x0) / span_x * plot_w

    def sy(value: float) -> float:
        return opt.top + plot_h - (value - y0) / span_y * plot_h

    parts = [
        f'<rect width="{opt.width}" height="{opt.height}" fill="{opt.background}"/>'
    ]
    if opt.title:
        parts.append(
            f'<text x="{opt.left}" y="24" font-size="15" font-weight="600" '
            f'fill="{colors.ELEMENT_COLORS["text"]}">{_esc(opt.title)}</text>'
        )
    grid = colors.ELEMENT_COLORS["grid"]
    muted = colors.ELEMENT_COLORS["muted"]
    for k in range(y_ticks + 1):
        value = y0 + span_y * k / y_ticks
        y = sy(value)
        parts.append(
            f'<line x1="{opt.left}" y1="{y:.1f}" x2="{opt.left + plot_w}" '
            f'y2="{y:.1f}" stroke="{grid}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{opt.left - 8}" y="{y + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="{muted}">{y_format.format(value)}</text>'
        )
    for k in range(x_ticks + 1):
        value = x0 + span_x * k / x_ticks
        x = sx(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{opt.top + plot_h}" x2="{x:.1f}" '
            f'y2="{opt.top + plot_h + 5}" stroke="{muted}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{opt.top + plot_h + 19}" font-size="11" '
            f'text-anchor="middle" fill="{muted}">{value:.3g}</text>'
        )
    parts.append(
        f'<text x="{opt.left + plot_w / 2}" y="{opt.height - 8}" font-size="12" '
        f'text-anchor="middle" fill="{colors.ELEMENT_COLORS["text"]}">'
        f"{_esc(x_label)}</text>"
    )
    parts.append(
        f'<text x="14" y="{opt.top + plot_h / 2}" font-size="12" '
        f'text-anchor="middle" fill="{colors.ELEMENT_COLORS["text"]}" '
        f'transform="rotate(-90 14 {opt.top + plot_h / 2})">{_esc(y_label)}</text>'
    )
    return parts, sx, sy


def occupancy_plot(
    ensemble,
    *,
    options: PlotOptions | None = None,
    min_population: float = 0.05,
    max_bands: int = 12,
) -> str:
    """Stacked-area plot of structure populations against time."""
    opt = options or PlotOptions(
        title=opt_title(ensemble, "Structure populations during transcription")
    )
    labels, matrix = ensemble.occupancy(min_population=min_population)
    if len(labels) > max_bands:
        order = np.argsort(-matrix.max(axis=1))[:max_bands]
        order = sorted(order)
        labels = [labels[k] for k in order]
        matrix = matrix[order]
    residual = 1.0 - matrix.sum(axis=0)
    residual = np.clip(residual, 0.0, 1.0)

    times = np.asarray(ensemble.times)
    parts, sx, sy = _axes(
        opt,
        (float(times[0]), float(times[-1])),
        (0.0, 1.0),
        x_label="time (s)",
        y_label="population",
        y_format="{:.1f}",
    )

    bands = list(matrix) + ([residual] if residual.max() > 1e-9 else [])
    names = list(labels) + (["other"] if residual.max() > 1e-9 else [])
    baseline = np.zeros(len(times))
    for index, band in enumerate(bands):
        top = baseline + band
        color = (
            colors.ELEMENT_COLORS["muted"]
            if names[index] == "other"
            else colors.occupancy_color(index)
        )
        upper = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times, top))
        lower = " ".join(
            f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times[::-1], baseline[::-1])
        )
        parts.append(
            f'<polygon points="{upper} {lower}" fill="{color}" opacity="0.85"/>'
        )
        baseline = top

    # legend
    legend_x = opt.width - opt.right + 12
    parts.append(
        f'<text x="{legend_x}" y="{opt.top + 2}" font-size="11" font-weight="600" '
        f'fill="{colors.ELEMENT_COLORS["text"]}">structures</text>'
    )
    for index, name in enumerate(names):
        y = opt.top + 18 + index * 16
        if y > opt.height - opt.bottom:
            break
        color = (
            colors.ELEMENT_COLORS["muted"]
            if name == "other"
            else colors.occupancy_color(index)
        )
        parts.append(
            f'<rect x="{legend_x}" y="{y - 8}" width="10" height="10" fill="{color}"/>'
        )
        short = name if name == "other" else f"S{index + 1}"
        parts.append(
            f'<text x="{legend_x + 15}" y="{y + 1}" font-size="11" '
            f'fill="{colors.ELEMENT_COLORS["muted"]}">{_esc(short)}</text>'
        )
    return _wrap(parts, opt)


def opt_title(ensemble, default: str) -> str:
    return f"{default} ({ensemble.n_trajectories} trajectories)"


def energy_plot(ensemble, *, options: PlotOptions | None = None) -> str:
    """Mean free energy with ensemble spread, plus transcript length."""
    opt = options or PlotOptions(title="Ensemble free energy and chain growth")
    times = np.asarray(ensemble.times)
    mean = ensemble.mean_energy()
    spread = ensemble.energy_spread()
    lengths = np.asarray(ensemble.lengths, dtype=float)

    low = float(np.min(mean - spread))
    high = float(np.max(mean + spread))
    pad = max(1.0, 0.08 * (high - low))
    parts, sx, sy = _axes(
        opt,
        (float(times[0]), float(times[-1])),
        (low - pad, high + pad),
        x_label="time (s)",
        y_label="free energy (kcal/mol)",
        y_format="{:.0f}",
    )

    upper = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times, mean + spread))
    lower = " ".join(
        f"{sx(t):.1f},{sy(v):.1f}"
        for t, v in zip(times[::-1], (mean - spread)[::-1])
    )
    parts.append(
        f'<polygon points="{upper} {lower}" fill="{colors.OCCUPANCY_PALETTE[0]}" '
        f'opacity="0.22"/>'
    )
    line = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times, mean))
    parts.append(
        f'<polyline points="{line}" fill="none" '
        f'stroke="{colors.OCCUPANCY_PALETTE[0]}" stroke-width="2.2"/>'
    )

    # transcript length on a secondary scale
    plot_h = opt.height - opt.top - opt.bottom
    max_len = float(lengths.max()) or 1.0
    growth = " ".join(
        f"{sx(t):.1f},{opt.top + plot_h - (v / max_len) * plot_h:.1f}"
        for t, v in zip(times, lengths)
    )
    parts.append(
        f'<polyline points="{growth}" fill="none" '
        f'stroke="{colors.ELEMENT_COLORS["nascent"]}" stroke-width="1.8" '
        f'stroke-dasharray="5 4"/>'
    )
    legend_x = opt.width - opt.right + 12
    parts.append(
        f'<rect x="{legend_x}" y="{opt.top + 4}" width="10" height="10" '
        f'fill="{colors.OCCUPANCY_PALETTE[0]}"/>'
        f'<text x="{legend_x + 15}" y="{opt.top + 13}" font-size="11" '
        f'fill="{colors.ELEMENT_COLORS["muted"]}">&lt;G&gt; &#177; sd</text>'
    )
    parts.append(
        f'<rect x="{legend_x}" y="{opt.top + 24}" width="10" height="10" '
        f'fill="{colors.ELEMENT_COLORS["nascent"]}"/>'
        f'<text x="{legend_x + 15}" y="{opt.top + 33}" font-size="11" '
        f'fill="{colors.ELEMENT_COLORS["muted"]}">length ({int(max_len)} nt)</text>'
    )
    return _wrap(parts, opt)


def pair_probability_plot(
    ensemble,
    t_index: int = -1,
    *,
    options: PlotOptions | None = None,
) -> str:
    """Dot plot of ``P(i, j)`` at one time point."""
    opt = options or PlotOptions(width=620, height=620, right=90, title="")
    probabilities = ensemble.pair_probabilities()[t_index]
    n = probabilities.shape[0]
    title = (
        opt.title
        or f"Pairing probability at t = {ensemble.times[t_index]:.2f} s"
    )
    opt = PlotOptions(
        width=opt.width,
        height=opt.height,
        left=opt.left,
        right=opt.right,
        top=opt.top,
        bottom=opt.bottom,
        title=title,
        background=opt.background,
    )
    parts, sx, sy = _axes(
        opt,
        (1.0, float(n)),
        (1.0, float(n)),
        x_label="nucleotide j",
        y_label="nucleotide i",
        y_format="{:.0f}",
    )
    plot_w = opt.width - opt.left - opt.right
    cell = plot_w / max(n, 1)
    for i in range(n):
        for j in range(i + 1, n):
            p = float(probabilities[i, j])
            if p < 0.01:
                continue
            size = cell * (0.35 + 0.65 * p)
            parts.append(
                f'<rect x="{sx(j + 1) - size/2:.2f}" y="{sy(i + 1) - size/2:.2f}" '
                f'width="{size:.2f}" height="{size:.2f}" rx="{size*0.25:.2f}" '
                f'fill="{colors.ramp_color(p)}"/>'
            )
    # colour key
    legend_x = opt.width - opt.right + 14
    for k, color in enumerate(colors.PROBABILITY_RAMP):
        parts.append(
            f'<rect x="{legend_x}" y="{opt.top + 10 + k * 14}" width="12" '
            f'height="14" fill="{color}"/>'
        )
    parts.append(
        f'<text x="{legend_x + 17}" y="{opt.top + 22}" font-size="10" '
        f'fill="{colors.ELEMENT_COLORS["muted"]}">0</text>'
    )
    parts.append(
        f'<text x="{legend_x + 17}" y="{opt.top + 10 + len(colors.PROBABILITY_RAMP)*14}" '
        f'font-size="10" fill="{colors.ELEMENT_COLORS["muted"]}">1</text>'
    )
    return _wrap(parts, opt)


def pseudoknot_plot(ensemble, *, options: PlotOptions | None = None) -> str:
    """Fraction of the ensemble carrying a pseudoknot, over time."""
    opt = options or PlotOptions(height=240, title="Pseudoknotted fraction")
    times = np.asarray(ensemble.times)
    fraction = ensemble.pseudoknot_fraction()
    parts, sx, sy = _axes(
        opt,
        (float(times[0]), float(times[-1])),
        (0.0, max(0.05, float(fraction.max()) * 1.15)),
        x_label="time (s)",
        y_label="fraction",
        y_format="{:.2f}",
    )
    line = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times, fraction))
    parts.append(
        f'<polyline points="{line}" fill="none" '
        f'stroke="{colors.ELEMENT_COLORS["pseudoknot"]}" stroke-width="2.2"/>'
    )
    return _wrap(parts, opt)


def _wrap(parts: Sequence[str], opt: PlotOptions) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{opt.width}" '
        f'height="{opt.height}" viewBox="0 0 {opt.width} {opt.height}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif">'
    )
    return head + "\n" + "\n".join(parts) + "\n</svg>"
