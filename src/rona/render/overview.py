"""Ensemble-level views: the time overview, and occupancy-sized structure tiles.

Both ideas are taken from DrForna (Tang, Badelt, Lorenz, Hofacker et al., 2023),
which solved the hard part of showing a folding ensemble: how to make a
*distribution over structures* legible while it changes.

``kymograph_array``
    One column per sampled time, one row per nucleotide, coloured by the pair
    that nucleotide is in - using :func:`rona.render.colors.pair_center_color`,
    so a helix keeps one colour for its whole lifetime.  A rearrangement is
    then a visible colour change at a glance, and the eye can follow a motif
    across the whole time course.

``tile_spans``
    Structures laid out as rectangles whose area is proportional to their
    occupancy.  Tiles keep the band order of the occupancy plot rather than
    being re-sorted every frame, so in a movie they grow and shrink in place
    instead of jumping past each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..struct import iter_pairs, parse_dotbracket
from . import colors


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


def kymograph_array(
    ensemble, *, mode: str = "dominant", unpaired_shade: int = 232
) -> tuple[np.ndarray, np.ndarray]:
    """``(rgb, alpha)`` arrays of shape ``(n, n_times[, 3])``.

    Row ``i`` is nucleotide ``i + 1`` and column ``t`` is ``ensemble.times[t]``.

    ``mode="dominant"`` colours by the most populated structure and fades the
    column by that structure's population, so a split ensemble reads as washed
    out.  ``mode="ensemble"`` instead colours each nucleotide by its most
    probable partner and fades by that pair's probability, which shows where
    the ensemble genuinely disagrees.
    """
    size = len(ensemble.seq)
    n_times = ensemble.n_times
    rgb = np.full((size, n_times, 3), unpaired_shade, dtype=np.uint8)
    alpha = np.zeros((size, n_times), dtype=float)

    if mode == "ensemble":
        probabilities = ensemble.pair_probabilities()
        for t in range(n_times):
            matrix = probabilities[t]
            symmetric = matrix + matrix.T
            partner = symmetric.argmax(axis=1)
            weight = symmetric.max(axis=1)
            for i in range(min(ensemble.lengths[t], size)):
                alpha[i, t] = 1.0
                if weight[i] > 0.02:
                    j = int(partner[i])
                    rgb[i, t] = _hex_to_rgb(
                        colors.pair_center_color(min(i, j), max(i, j), size)
                    )
                    alpha[i, t] = 0.25 + 0.75 * float(weight[i])
                else:
                    alpha[i, t] = 0.35
        return rgb, alpha

    dominant = ensemble.dominant()
    for t, (structure, population) in enumerate(dominant):
        pt = parse_dotbracket(structure)
        length = min(len(pt), size)
        for i in range(length):
            alpha[i, t] = 0.30 + 0.70 * population
        for i, j in iter_pairs(pt):
            if j >= size:
                continue
            colour = _hex_to_rgb(colors.pair_center_color(i, j, size))
            rgb[i, t] = colour
            rgb[j, t] = colour
    return rgb, alpha


OPEN_CHAIN = "open chain"


def merge_open_chain(
    labels: Sequence[str], matrix: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Collapse every pair-free structure into one band.

    During elongation each prefix length gives a *different* pair-free
    dot-bracket string, so an unfolded chain would otherwise appear as a row of
    unrelated one-frame bands.  They are all the same state as far as folding
    is concerned.
    """
    open_rows = [k for k, label in enumerate(labels) if "(" not in label]
    if len(open_rows) < 2:
        return list(labels), matrix
    keep = [k for k in range(len(labels)) if k not in set(open_rows)]
    merged = matrix[open_rows].sum(axis=0)
    return [OPEN_CHAIN] + [labels[k] for k in keep], np.vstack(
        [merged[None, :], matrix[keep]]
    )


def occupied_time(matrix: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Time-integrated population of each band (trapezoid rule).

    Written out rather than using ``np.trapezoid``/``np.trapz``, whose name
    changed between numpy 1.x and 2.x.
    """
    if matrix.shape[1] < 2:
        return matrix.sum(axis=1)
    widths = np.diff(times)
    return (0.5 * (matrix[:, :-1] + matrix[:, 1:]) * widths).sum(axis=1)


def top_bands(
    labels: Sequence[str],
    matrix: np.ndarray,
    times: Sequence[float],
    max_bands: int,
) -> tuple[list[str], np.ndarray]:
    """Merge the open chain, then keep the bands that are occupied longest.

    Ranking by peak population would let a structure that flickers through a
    single frame at 100% displace one that carries the pathway.
    """
    labels, matrix = merge_open_chain(labels, matrix)
    if len(labels) <= max_bands:
        return list(labels), matrix
    weight = occupied_time(matrix, np.asarray(times, dtype=float))
    order = sorted(np.argsort(-weight)[:max_bands])
    return [labels[k] for k in order], matrix[order]


def kymograph_rgba(ensemble, *, mode: str = "dominant") -> np.ndarray:
    """The kymograph as an ``(n, n_times, 4)`` uint8 RGBA image.

    Row 0 is nucleotide 1.  Untranscribed positions are transparent, so the
    growing chain reads as the image filling in from the top-left.
    """
    rgb, alpha = kymograph_array(ensemble, mode=mode)
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    out[:, :, :3] = rgb
    out[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return out


def kymograph_data_uri(ensemble, *, mode: str = "dominant") -> str:
    """The kymograph encoded as a PNG data URI, for SVG and HTML embedding."""
    from .png import data_uri

    return data_uri(kymograph_rgba(ensemble, mode=mode))


@dataclass(frozen=True, slots=True)
class Tile:
    """One structure's rectangle in the gallery."""

    label: str
    index: int
    start: float
    width: float
    population: float

    @property
    def centre(self) -> float:
        return self.start + self.width / 2.0


def tile_spans(
    labels: Sequence[str],
    column: Sequence[float],
    *,
    min_width: float = 0.0,
) -> list[Tile]:
    """Lay out one time point's structures as a strip of proportional tiles.

    Tiles are placed in the order given - which is the occupancy plot's band
    order, i.e. first appearance - so they never swap places from frame to
    frame.  The remaining probability, if any, becomes a final "other" tile.
    """
    tiles: list[Tile] = []
    cursor = 0.0
    for index, (label, value) in enumerate(zip(labels, column)):
        value = float(max(value, 0.0))
        if value > min_width:
            tiles.append(Tile(label, index, cursor, value, value))
        cursor += value
    remainder = max(0.0, 1.0 - cursor)
    if remainder > max(min_width, 1e-6):
        tiles.append(Tile("other", -1, cursor, remainder, remainder))
    return tiles


def structure_palette(structure: str, n: int) -> dict[tuple[int, int], str]:
    """Pair-centre colour for every pair of a structure."""
    pt = parse_dotbracket(structure)
    return {(i, j): colors.pair_center_color(i, j, n) for i, j in iter_pairs(pt)}


def transition_index(ensemble) -> int:
    """Sample index at which transcription finishes.

    DrForna splits its time axis here - linear while the chain grows, then
    logarithmic - because the post-transcriptional relaxation usually spans far
    more decades than transcription itself.
    """
    full = len(ensemble.seq)
    for index, length in enumerate(ensemble.lengths):
        if length >= full:
            return index
    return ensemble.n_times - 1


def hybrid_time_positions(
    ensemble, *, split: float = 0.75, decades: float = 3.0
) -> np.ndarray:
    """Horizontal position in ``[0, 1]`` for each sample, on a split axis.

    Linear up to the end of transcription, which is placed at ``split`` of the
    width, then logarithmic for the post-transcriptional phase.
    """
    times = np.asarray(ensemble.times, dtype=float)
    n_times = len(times)
    cut = transition_index(ensemble)
    out = np.zeros(n_times)
    if cut <= 0:
        t_end = times[-1] or 1.0
        return times / t_end
    end_time = times[cut] or 1.0
    out[: cut + 1] = times[: cut + 1] / end_time * split
    if cut + 1 < n_times:
        tail = times[cut + 1 :] - times[cut]
        floor = max(tail.max() * 10.0 ** (-decades), 1e-9)
        scaled = np.log10(np.maximum(tail, floor) / floor)
        span = scaled.max() or 1.0
        out[cut + 1 :] = split + (1.0 - split) * scaled / span
    return out
