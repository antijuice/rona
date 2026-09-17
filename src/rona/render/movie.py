"""Movies of a cotranscriptional kinetic ensemble.

Three modes, all built from the same multi-panel frame:

``ensemble`` (default)
    The structure panel shows the dominant structure's layout, but every base
    pair is drawn with opacity equal to its *ensemble* probability at that
    instant.  The picture is therefore the whole distribution, not one
    realisation: pairs fade in as the population commits to them and fade out
    when the kinetics abandons them.

``dominant``
    Only the most populated structure, drawn solid.  Cleaner, and useful when
    the ensemble is sharply peaked.

``trajectory``
    A single stochastic pathway, which is what you want when showing that
    individual molecules take different routes.

Frames are laid out with :func:`rona.render.layout.layout_series` and tweened,
so the structure morphs continuously instead of cutting between drawings.

Requires ``matplotlib``; MP4 output additionally needs ``imageio`` with an
ffmpeg backend.  Both are optional extras (``pip install rona[movie]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..struct import helices_from_pairtable, iter_pairs, parse_dotbracket
from . import colors
from .layout import LayoutOptions, bounding_box, interpolate, layout_series


@dataclass(frozen=True, slots=True)
class MovieOptions:
    """Movie appearance and encoding settings."""

    mode: str = "ensemble"  # ensemble | dominant | trajectory
    fps: int = 24
    #: Tweened frames inserted between consecutive sampled time points.
    interpolation: int = 4
    width: int = 1280
    height: int = 720
    dpi: int = 100
    #: Base pairs below this ensemble probability are not drawn.
    pair_threshold: float = 0.04
    show_letters: bool | None = None
    title: str = ""
    #: Trajectory index to follow in ``trajectory`` mode.
    trajectory: int = 0


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "movie rendering needs matplotlib; install with "
            "`pip install 'rona[movie]'`"
        ) from exc


def _series_structures(ensemble, opt: MovieOptions) -> list[str]:
    if opt.mode == "trajectory":
        index = opt.trajectory
        return [row[index] for row in ensemble.structures]
    return [s for s, _p in ensemble.dominant()]


def _pk_pairs(structure: str) -> set[tuple[int, int]]:
    from ..energy.pseudoknot import split_crossing

    pt = parse_dotbracket(structure)
    _core, pk = split_crossing(helices_from_pairtable(pt))
    return {p for helix in pk for p in helix.pairs}


def render_frames(
    ensemble,
    *,
    options: MovieOptions | None = None,
    layout_options: LayoutOptions | None = None,
):
    """Yield RGB frames (``uint8`` arrays) for the whole time course."""
    plt = _require_matplotlib()
    from matplotlib.collections import LineCollection

    opt = options or MovieOptions()
    sequence = ensemble.seq
    n_total = len(sequence)

    structures = _series_structures(ensemble, opt)
    key_coords = layout_series(structures, options=layout_options)
    steps = max(1, opt.interpolation)
    coords = interpolate(key_coords, steps, n_total)

    probabilities = (
        ensemble.pair_probabilities() if opt.mode == "ensemble" else None
    )
    occupancy_labels, occupancy = ensemble.occupancy(min_population=0.05)
    mean_energy = ensemble.mean_energy()
    times = np.asarray(ensemble.times)
    lengths = ensemble.lengths
    dominant = ensemble.dominant()
    box = bounding_box(coords)
    pad = 0.06 * max(box[2] - box[0], box[3] - box[1], 1.0)

    figure = plt.figure(
        figsize=(opt.width / opt.dpi, opt.height / opt.dpi), dpi=opt.dpi
    )
    figure.patch.set_facecolor(colors.ELEMENT_COLORS["background"])
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0],
        left=0.035, right=0.975, top=0.90, bottom=0.085, wspace=0.16, hspace=0.32,
    )
    ax_structure = figure.add_subplot(grid[:, 0])
    ax_occupancy = figure.add_subplot(grid[0, 1])
    ax_energy = figure.add_subplot(grid[1, 1])

    base_colors = [colors.base_color(c) for c in sequence]

    for frame_index, frame in enumerate(coords):
        key = min(frame_index // steps, len(structures) - 1)
        structure = structures[key]
        length = lengths[key]
        time = float(times[key])

        ax_structure.clear()
        ax_structure.set_facecolor(colors.ELEMENT_COLORS["background"])
        ax_structure.set_xlim(box[0] - pad, box[2] + pad)
        ax_structure.set_ylim(box[1] - pad, box[3] + pad)
        ax_structure.set_aspect("equal")
        ax_structure.axis("off")

        visible = min(length, n_total)
        # backbone
        ax_structure.plot(
            frame[:visible, 0],
            frame[:visible, 1],
            color=colors.ELEMENT_COLORS["backbone"],
            linewidth=1.6,
            alpha=0.8,
            zorder=1,
            solid_capstyle="round",
        )

        pk = _pk_pairs(structure)
        segments: list[np.ndarray] = []
        stroke: list[tuple] = []
        widths: list[float] = []
        if probabilities is not None:
            matrix = probabilities[key]
            candidates = [
                (i, j, float(matrix[i, j]))
                for i in range(visible)
                for j in range(i + 1, visible)
                if matrix[i, j] >= opt.pair_threshold
            ]
        else:
            candidates = [(i, j, 1.0) for i, j in iter_pairs(parse_dotbracket(structure))]
        for i, j, weight in candidates:
            if i >= len(frame) or j >= len(frame):
                continue
            segments.append(np.array([frame[i], frame[j]]))
            color = (
                colors.ELEMENT_COLORS["pseudoknot"]
                if (i, j) in pk
                else colors.ELEMENT_COLORS["pair"]
            )
            stroke.append(_rgba(color, 0.18 + 0.72 * weight))
            widths.append(1.0 + 1.8 * weight)
        if segments:
            ax_structure.add_collection(
                LineCollection(segments, colors=stroke, linewidths=widths, zorder=2)
            )

        ax_structure.scatter(
            frame[:visible, 0],
            frame[:visible, 1],
            c=base_colors[:visible],
            s=46,
            zorder=3,
            edgecolors=colors.ELEMENT_COLORS["background"],
            linewidths=0.7,
        )
        if visible:
            ax_structure.scatter(
                [frame[visible - 1, 0]],
                [frame[visible - 1, 1]],
                s=170,
                facecolors="none",
                edgecolors=colors.ELEMENT_COLORS["polymerase"],
                linewidths=1.6,
                zorder=4,
            )
            ax_structure.annotate(
                "5′",
                (frame[0, 0], frame[0, 1]),
                textcoords="offset points",
                xytext=(-12, 8),
                fontsize=11,
                weight="bold",
                color=colors.ELEMENT_COLORS["text"],
            )

        pop = dominant[key][1]
        label = "RNAP" if length < n_total else "3′ released"
        ax_structure.set_title(
            f"t = {time:7.2f} s     {length} nt     {label}"
            f"     dominant population {pop:.0%}",
            fontsize=12,
            color=colors.ELEMENT_COLORS["text"],
            loc="left",
        )

        _draw_occupancy(ax_occupancy, times, occupancy, occupancy_labels, time)
        _draw_energy(ax_energy, times, mean_energy, ensemble.energy_spread(), lengths, time)

        figure.suptitle(
            opt.title or f"Cotranscriptional folding kinetics — {n_total} nt",
            fontsize=14,
            weight="600",
            color=colors.ELEMENT_COLORS["text"],
            x=0.035,
            ha="left",
        )
        figure.canvas.draw()
        buffer = np.asarray(figure.canvas.buffer_rgba())
        yield buffer[:, :, :3].copy()

    plt.close(figure)


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    return (r, g, b, min(max(alpha, 0.0), 1.0))


def _draw_occupancy(ax, times, occupancy, labels, cursor: float) -> None:
    ax.clear()
    ax.set_facecolor(colors.ELEMENT_COLORS["background"])
    if len(occupancy):
        bands = list(occupancy)
        residual = np.clip(1.0 - occupancy.sum(axis=0), 0.0, 1.0)
        palette = [colors.occupancy_color(k) for k in range(len(bands))]
        if residual.max() > 1e-9:
            bands.append(residual)
            palette.append(colors.ELEMENT_COLORS["muted"])
        ax.stackplot(times, *bands, colors=palette, alpha=0.9)
    ax.axvline(cursor, color=colors.ELEMENT_COLORS["text"], linewidth=1.2, alpha=0.75)
    ax.set_ylim(0, 1)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("population", fontsize=10)
    ax.set_title("structure populations", fontsize=11, loc="left")
    ax.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_energy(ax, times, mean, spread, lengths, cursor: float) -> None:
    ax.clear()
    ax.set_facecolor(colors.ELEMENT_COLORS["background"])
    ax.fill_between(
        times, mean - spread, mean + spread,
        color=colors.OCCUPANCY_PALETTE[0], alpha=0.2, linewidth=0,
    )
    ax.plot(times, mean, color=colors.OCCUPANCY_PALETTE[0], linewidth=2.0)
    ax.axvline(cursor, color=colors.ELEMENT_COLORS["text"], linewidth=1.2, alpha=0.75)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("<G>  kcal/mol", fontsize=10)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_title("ensemble free energy", fontsize=11, loc="left")
    ax.tick_params(labelsize=9)
    twin = ax.twinx()
    twin.plot(
        times, lengths, color=colors.ELEMENT_COLORS["nascent"],
        linewidth=1.5, linestyle="--",
    )
    twin.set_ylabel("length (nt)", fontsize=9, color=colors.ELEMENT_COLORS["nascent"])
    twin.tick_params(labelsize=8, colors=colors.ELEMENT_COLORS["nascent"])
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        twin.spines[spine].set_visible(False)


def render_movie(
    ensemble,
    path: str,
    *,
    options: MovieOptions | None = None,
    layout_options: LayoutOptions | None = None,
) -> str:
    """Write an MP4 or GIF of the ensemble's time course.

    The container is chosen from the file extension.  Returns ``path``.
    """
    opt = options or MovieOptions()
    frames = render_frames(
        ensemble, options=opt, layout_options=layout_options
    )
    suffix = str(path).lower().rsplit(".", 1)[-1]
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "writing movies needs imageio; install with `pip install 'rona[movie]'`"
        ) from exc

    if suffix == "gif":
        imageio.mimsave(path, list(frames), duration=1.0 / opt.fps, loop=0)
        return path

    writer = imageio.get_writer(
        path, fps=opt.fps, codec="libx264", quality=8, macro_block_size=8
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return path
