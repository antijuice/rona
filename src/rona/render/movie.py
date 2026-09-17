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
from .layout import (
    LayoutOptions,
    bounding_box,
    camera_path,
    fit_aspect,
    interpolate,
    layout_series,
)
from .overview import OPEN_CHAIN, kymograph_array, tile_spans, top_bands


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
    pair_threshold: float = 0.05
    show_letters: bool | None = None
    title: str = ""
    #: Trajectory index to follow in ``trajectory`` mode.
    trajectory: int = 0
    #: Draw the occupancy-proportional gallery of competing structures.
    gallery: bool = True
    #: Draw the time-overview kymograph.
    kymograph: bool = True
    #: Colour base pairs by their centre (a helix keeps one colour) rather
    #: than by nested/pseudoknot status.
    color_pairs_by_centre: bool = True
    #: Structures below this population get no tile.
    gallery_threshold: float = 0.02


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
    occupancy_labels, occupancy = ensemble.occupancy(min_population=0.02)
    occupancy_labels, occupancy = top_bands(
        occupancy_labels, occupancy, ensemble.times, 11
    )
    mean_energy = ensemble.mean_energy()
    times = np.asarray(ensemble.times)
    # panels are drawn against sample index and labelled with real times, so a
    # log-spaced grid reads correctly
    sample_index = np.arange(len(times), dtype=float)
    lengths = ensemble.lengths
    dominant = ensemble.dominant()
    camera = camera_path(coords, margin=0.05, min_fraction=0.35)

    figure = plt.figure(
        figsize=(opt.width / opt.dpi, opt.height / opt.dpi), dpi=opt.dpi
    )
    figure.patch.set_facecolor(colors.ELEMENT_COLORS["background"])
    grid = figure.add_gridspec(
        3, 2, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.0, 0.62],
        left=0.025, right=0.945, top=0.895, bottom=0.075, wspace=0.20, hspace=0.45,
    )
    ax_structure = figure.add_subplot(grid[0:2, 0])
    ax_gallery = figure.add_subplot(grid[2, 0])
    ax_kymograph = figure.add_subplot(grid[0, 1])
    ax_occupancy = figure.add_subplot(grid[1, 1])
    ax_energy = figure.add_subplot(grid[2, 1])

    base_colors = [colors.base_color(c) for c in sequence]

    # The population and energy panels never change - only the time cursor
    # moves - so they are drawn once and the cursor is repositioned per frame.
    # Redrawing them every frame dominated the render time.
    spread = ensemble.energy_spread()
    _draw_occupancy(ax_occupancy, sample_index, occupancy, occupancy_labels, times)
    _draw_energy(ax_energy, sample_index, mean_energy, spread, lengths, times)
    cursors = []
    cursors.append(ax_occupancy.axvline(
        0.0, color=colors.ELEMENT_COLORS["text"], linewidth=1.3, alpha=0.8
    ))
    cursors.append(ax_energy.axvline(
        0.0, color=colors.ELEMENT_COLORS["text"], linewidth=1.3, alpha=0.8
    ))
    if opt.kymograph:
        _draw_kymograph(ax_kymograph, ensemble, sample_index, times)
        cursors.append(ax_kymograph.axvline(
            0.0, color=colors.ELEMENT_COLORS["text"], linewidth=1.2, alpha=0.9
        ))
    else:
        ax_kymograph.axis("off")
    figure.suptitle(
        opt.title or f"Cotranscriptional folding kinetics \u2014 {n_total} nt",
        fontsize=14,
        weight="bold",
        color=colors.ELEMENT_COLORS["text"],
        x=0.025,
        ha="left",
    )
    # the per-frame caption belongs to the figure: an axes title would move
    # with the structure panel's box
    caption = figure.text(
        0.025, 0.925, "", fontsize=12,
        color=colors.ELEMENT_COLORS["text"],
        family="monospace", va="bottom",
    )

    # Expand the view box to the panel's own aspect ratio, so an equal-aspect
    # drawing fills the panel instead of being shrunk to fit inside it.  One
    # draw is needed first to learn the panel's final size.
    figure.canvas.draw()
    position = ax_structure.get_position()
    panel_aspect = (position.width * opt.width) / max(
        position.height * opt.height, 1e-6
    )
    views = [fit_aspect(row, panel_aspect) for row in camera]

    for frame_index, frame in enumerate(coords):
        key = min(frame_index // steps, len(structures) - 1)
        structure = structures[key]
        length = lengths[key]
        time = float(times[key])

        ax_structure.clear()
        ax_structure.set_facecolor(colors.ELEMENT_COLORS["background"])
        view = views[frame_index]
        ax_structure.set_xlim(view[0], view[2])
        ax_structure.set_ylim(view[1], view[3])
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
            if (i, j) in pk:
                color = colors.ELEMENT_COLORS["pseudoknot"]
            elif opt.color_pairs_by_centre:
                color = colors.pair_center_color(i, j, n_total)
            else:
                color = colors.ELEMENT_COLORS["pair"]
            # opacity and width both carry the ensemble probability
            stroke.append(_rgba(color, 0.10 + 0.85 * weight))
            widths.append(0.6 + 2.2 * weight)
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
        caption.set_text(
            f"t = {time:7.2f} s      {length} nt      {label}"
            f"      dominant population {pop:.0%}"
        )

        for cursor in cursors:
            cursor.set_xdata([key, key])
        if opt.gallery:
            _draw_gallery(
                ax_gallery, occupancy_labels, occupancy[:, key], n_total, opt
            )
        else:
            ax_gallery.axis("off")

        figure.canvas.draw()
        buffer = np.asarray(figure.canvas.buffer_rgba())
        yield buffer[:, :, :3].copy()

    plt.close(figure)


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    return (r, g, b, min(max(alpha, 0.0), 1.0))


def _draw_kymograph(ax, ensemble, index, times) -> None:
    """Time overview: nucleotide vs time, coloured by which helix it is in."""
    rgb, alpha = kymograph_array(ensemble)
    image = np.dstack(
        [rgb.astype(float) / 255.0, np.clip(alpha, 0.0, 1.0)[:, :, None]]
    )
    ax.clear()
    ax.set_facecolor(colors.ELEMENT_COLORS["background"])
    ax.imshow(
        image, aspect="auto", origin="lower", interpolation="nearest",
        extent=(float(index[0]) - 0.5, float(index[-1]) + 0.5, 0.5, rgb.shape[0] + 0.5),
    )
    ax.set_ylabel("nucleotide", fontsize=10)
    ax.set_title(
        "time overview \u00b7 colour = helix identity", fontsize=11, loc="left"
    )
    ax.tick_params(labelsize=9)
    _tick_times(ax, index, times)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_gallery(ax, labels, column, n_total, opt) -> None:
    """Competing structures as tiles whose width is their population.

    Each tile carries a miniature arc diagram rather than a 2D drawing: at tile
    size an arc diagram stays readable, and with pairs coloured by centre the
    differences between competing structures are visible directly.
    """
    ax.clear()
    ax.set_facecolor(colors.ELEMENT_COLORS["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.22, 1.0)
    ax.axis("off")

    tiles = tile_spans(labels, column, min_width=opt.gallery_threshold)
    for tile in tiles:
        pad = min(0.006, tile.width * 0.12)
        x0, x1 = tile.start + pad, tile.start + tile.width - pad
        if x1 <= x0:
            continue
        ax.add_patch(
            plt_rectangle(
                (x0, -0.16), x1 - x0, 1.10,
                facecolor=colors.ELEMENT_COLORS["panel"],
                edgecolor=colors.ELEMENT_COLORS["grid"], linewidth=0.7,
            )
        )
        ax.text(
            (x0 + x1) / 2.0, -0.13, f"{tile.population:.0%}",
            ha="center", va="bottom", fontsize=8.5,
            color=colors.ELEMENT_COLORS["muted"],
        )
        if tile.index < 0 or tile.label == OPEN_CHAIN:
            continue
        _mini_arcs(ax, tile.label, x0, x1, 0.02, 0.92, n_total)

    ax.set_title(
        "structures in the ensemble now \u00b7 width = population",
        fontsize=11, loc="left", color=colors.ELEMENT_COLORS["text"],
    )


def plt_rectangle(xy, width, height, **kwargs):
    from matplotlib.patches import Rectangle

    return Rectangle(xy, width, height, **kwargs)


def _mini_arcs(ax, structure, x0, x1, y0, y1, n_total) -> None:
    """A compact arc diagram of one structure inside a tile."""
    pt = parse_dotbracket(structure)
    length = len(pt)
    if length < 2:
        return
    span = x1 - x0
    ax.plot(
        [x0, x0 + span * (length / max(n_total, 1))], [y0, y0],
        color=colors.ELEMENT_COLORS["muted"], linewidth=0.9, alpha=0.7,
    )
    pairs = list(iter_pairs(pt))
    if not pairs:
        return
    widest = max(j - i for i, j in pairs) or 1
    theta = np.linspace(0.0, np.pi, 18)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for i, j in pairs:
        centre = x0 + span * ((i + j) / 2.0 / max(n_total, 1))
        radius = span * ((j - i) / 2.0 / max(n_total, 1))
        height = (y1 - y0) * ((j - i) / widest) ** 0.55
        ax.plot(
            centre - radius * cos_t, y0 + height * sin_t,
            color=colors.pair_center_color(i, j, n_total),
            linewidth=1.0, alpha=0.9, solid_capstyle="round",
        )


def _tick_times(ax, index, times) -> None:
    """Label an index axis with the real sample times."""
    count = len(times)
    positions = np.linspace(0, count - 1, min(5, count))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{times[int(round(p))]:.3g}" for p in positions]
    )


def _draw_occupancy(ax, index, occupancy, labels, times) -> None:
    ax.clear()
    ax.set_facecolor(colors.ELEMENT_COLORS["background"])
    if len(occupancy):
        bands = list(occupancy)
        residual = np.clip(1.0 - occupancy.sum(axis=0), 0.0, 1.0)
        palette = [
            colors.ELEMENT_COLORS["grid"] if labels[k] == OPEN_CHAIN
            else colors.occupancy_color(k)
            for k in range(len(bands))
        ]
        if residual.max() > 1e-9:
            bands.append(residual)
            palette.append(colors.ELEMENT_COLORS["muted"])
        ax.stackplot(index, *bands, colors=palette, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlim(index[0], index[-1])
    _tick_times(ax, index, times)
    ax.set_ylabel("population", fontsize=10)
    ax.set_title("structure populations", fontsize=11, loc="left")
    ax.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_energy(ax, index, mean, spread, lengths, times) -> None:
    ax.clear()
    ax.set_facecolor(colors.ELEMENT_COLORS["background"])
    ax.fill_between(
        index, mean - spread, mean + spread,
        color=colors.OCCUPANCY_PALETTE[0], alpha=0.2, linewidth=0,
    )
    ax.plot(index, mean, color=colors.OCCUPANCY_PALETTE[0], linewidth=2.0)
    ax.set_xlim(index[0], index[-1])
    _tick_times(ax, index, times)
    ax.set_ylabel("<G>  kcal/mol", fontsize=10)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_title("ensemble free energy", fontsize=11, loc="left")
    ax.tick_params(labelsize=9)
    twin = ax.twinx()
    twin.plot(
        index, lengths, color=colors.ELEMENT_COLORS["nascent"],
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
