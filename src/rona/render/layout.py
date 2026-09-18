"""Pseudoknot-aware 2D layout of RNA secondary structures.

The nested part of a structure is drawn with the classical loop-circle
algorithm: every loop becomes a circle whose radius is chosen so that its
members - unpaired bases at backbone spacing, base pairs at pair spacing - fit
exactly around the circumference, and every helix becomes a straight ladder
leaving its parent loop at a right angle.  The exterior loop is laid out along a
horizontal line.

Crossing (pseudoknot) pairs cannot be honoured by that construction, so they are
added afterwards as springs and the whole drawing is relaxed: pseudoknot
partners attract, the backbone keeps its spacing, and non-adjacent nucleotides
repel so loops do not collapse into each other.

Temporal coherence
------------------
For a movie, laying out each frame independently produces a drawing that jumps
and flips between frames even when little has changed.  :func:`layout_series`
instead

1. seeds each frame's relaxation from the previous frame's coordinates,
2. rotates and translates each frame onto the previous one by a Kabsch fit over
   the nucleotides they share, and
3. optionally interpolates intermediate frames,

so the structure visibly *morphs* from one state to the next instead of being
redrawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..struct import helices_from_pairtable, iter_pairs, parse_dotbracket


@dataclass(frozen=True, slots=True)
class LayoutOptions:
    """Geometry and relaxation settings."""

    #: Distance between consecutive backbone nucleotides.
    bond: float = 1.0
    #: Distance between the two nucleotides of a base pair.
    pair: float = 2.0
    #: Relaxation sweeps used to accommodate pseudoknot pairs.
    relax_iterations: int = 400
    #: Strength of the pseudoknot spring, per sweep.
    pk_strength: float = 0.35
    #: Strength of the backbone and base-pair constraints, per sweep.
    bond_strength: float = 0.5
    #: Short-range repulsion radius (in units of ``bond``).
    repulsion_radius: float = 1.6
    repulsion_strength: float = 0.12


def _solve_radius(chords: Sequence[float]) -> float:
    """Radius of the circle on which the given chords close a full turn.

    Solves ``sum_k 2 asin(c_k / 2r) = 2 pi`` by bisection; the left-hand side is
    monotonically decreasing in ``r``, so the root is unique.
    """
    longest = max(chords)
    lo = longest / 2.0 + 1e-9
    total = sum(chords)
    hi = max(total / (2.0 * math.pi), lo) * 4.0 + longest
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        angle = sum(2.0 * math.asin(min(1.0, c / (2.0 * mid))) for c in chords)
        if angle > 2.0 * math.pi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _loop_members(
    pt: Sequence[int], closing: tuple[int, int] | None, n: int
) -> list[int]:
    """Nucleotides directly on a loop, in 5'->3' order."""
    if closing is None:
        lo, hi = 0, n
        members: list[int] = []
    else:
        lo, hi = closing[0] + 1, closing[1]
        members = [closing[0]]
    k = lo
    while k < hi:
        partner = pt[k]
        if partner < 0 or not (lo <= partner < hi) or partner < k:
            members.append(k)
            k += 1
        else:
            members.append(k)
            members.append(partner)
            k = partner + 1
    if closing is not None:
        members.append(closing[1])
    return members


def _nested_layout(
    pt: Sequence[int], n: int, opt: LayoutOptions
) -> np.ndarray:
    """Loop-circle layout of a nested pair table."""
    coords = np.zeros((n, 2))
    placed = np.zeros(n, dtype=bool)

    def chord(a: int, b: int) -> float:
        return opt.pair if pt[a] == b else opt.bond

    # --- exterior loop: straight line, 5' to 3' ---
    members = _loop_members(pt, None, n)
    x = 0.0
    for index, base in enumerate(members):
        coords[base] = (x, 0.0)
        placed[base] = True
        if index + 1 < len(members):
            x += chord(base, members[index + 1])

    stack: list[tuple[int, int, np.ndarray]] = []
    for base in members:
        partner = pt[base]
        if partner > base:
            stack.append((base, partner, np.array([0.0, 1.0])))

    while stack:
        i, j, direction = stack.pop()
        # walk down the helix, one rung at a time
        helix_len = 0
        a, b = i, j
        while a < b and pt[a] == b:
            helix_len += 1
            a, b = a + 1, b - 1
        for k in range(1, helix_len):
            coords[i + k] = coords[i] + direction * (opt.bond * k)
            coords[j - k] = coords[j] + direction * (opt.bond * k)
            placed[i + k] = placed[j - k] = True
        inner = (i + helix_len - 1, j - helix_len + 1)
        if inner[1] - inner[0] <= 1:
            continue

        loop = _loop_members(pt, inner, n)
        chords = [
            chord(loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop))
        ]
        radius = _solve_radius(chords)

        start, end = coords[inner[0]], coords[inner[1]]
        midpoint = 0.5 * (start + end)
        half = 0.5 * float(np.linalg.norm(end - start))
        height = math.sqrt(max(radius * radius - half * half, 0.0))
        centre = midpoint + direction * height

        angle = math.atan2(start[1] - centre[1], start[0] - centre[0])
        # traverse so that the loop opens away from the parent helix
        # 2D scalar cross product; np.cross on 2-vectors is deprecated
        u, v = start - centre, end - centre
        cross = float(u[0] * v[1] - u[1] * v[0])
        sign = -1.0 if cross > 0 else 1.0
        for k in range(len(loop) - 1):
            step = 2.0 * math.asin(min(1.0, chords[k] / (2.0 * radius)))
            angle += sign * step
            base = loop[k + 1]
            if not placed[base]:
                coords[base] = centre + radius * np.array(
                    [math.cos(angle), math.sin(angle)]
                )
                placed[base] = True

        for k in range(1, len(loop) - 1):
            base = loop[k]
            partner = pt[base]
            if partner > base and partner != inner[1]:
                outward = 0.5 * (coords[base] + coords[partner]) - centre
                norm = float(np.linalg.norm(outward))
                if norm < 1e-9:
                    outward = direction
                    norm = 1.0
                stack.append((base, partner, outward / norm))

    return coords


def _ladder_constraints(
    helices: Iterable, opt: LayoutOptions
) -> list[tuple[int, int, float]]:
    """Diagonal braces that keep a helix a rigid ladder rather than a shear.

    Backbone and base-pair springs alone leave a helix free to shear into a
    parallelogram, which is what makes naive spring layouts of pseudoknots look
    wrong.  Bracing both diagonals of every rung fixes the shape exactly.
    """
    diagonal = math.hypot(opt.pair, opt.bond)
    out: list[tuple[int, int, float]] = []
    for helix in helices:
        for k in range(helix.length - 1):
            a, b = helix.i + k, helix.j - k
            out.append((a, b - 1, diagonal))
            out.append((a + 1, b, diagonal))
    return out


def _relax(
    coords: np.ndarray,
    pt: Sequence[int],
    crossing: Sequence[tuple[int, int]],
    n: int,
    opt: LayoutOptions,
    braces: Sequence[tuple[int, int, float]] = (),
) -> np.ndarray:
    """Spring relaxation that pulls pseudoknot partners together."""
    if n == 0:
        return coords
    pos = coords.copy()
    backbone = [(k, k + 1, opt.bond) for k in range(n - 1)]
    pairs = [(i, j, opt.pair) for i, j in iter_pairs(pt)]
    pk = [(i, j, opt.pair) for i, j in crossing]
    if not pk:
        return pos

    warmup = opt.relax_iterations // 3
    for sweep in range(opt.relax_iterations):
        cooling = 1.0 - sweep / (opt.relax_iterations + 1)
        delta = np.zeros_like(pos)
        for a, b, target in backbone:
            delta = _spring(pos, delta, a, b, target, opt.bond_strength * cooling)
        for a, b, target in pairs:
            delta = _spring(pos, delta, a, b, target, opt.bond_strength * cooling)
        for a, b, target in braces:
            delta = _spring(pos, delta, a, b, target, opt.bond_strength * cooling)
        for a, b, target in pk:
            delta = _spring(pos, delta, a, b, target, opt.pk_strength * cooling)
        if sweep >= warmup:
            # repulsion only once the springs have pulled the partners together,
            # so it cannot build a barrier around a tangled configuration
            delta = _repel(pos, delta, n, opt, cooling)
        pos = pos + delta
    return pos


def _spring(
    pos: np.ndarray,
    delta: np.ndarray,
    a: int,
    b: int,
    target: float,
    strength: float,
) -> np.ndarray:
    vec = pos[b] - pos[a]
    dist = float(np.hypot(vec[0], vec[1]))
    if dist < 1e-9:
        vec = np.array([1e-3, 0.0])
        dist = 1e-3
    shift = 0.5 * strength * (dist - target) * vec / dist
    delta[a] += shift
    delta[b] -= shift
    return delta


def _repel(
    pos: np.ndarray, delta: np.ndarray, n: int, opt: LayoutOptions, cooling: float
) -> np.ndarray:
    """Short-range repulsion, evaluated on a grid so it stays near-linear."""
    radius = opt.repulsion_radius * opt.bond
    if radius <= 0 or opt.repulsion_strength <= 0:
        return delta
    cells: dict[tuple[int, int], list[int]] = {}
    for k in range(n):
        key = (int(pos[k, 0] // radius), int(pos[k, 1] // radius))
        cells.setdefault(key, []).append(k)
    for (cx, cy), members in cells.items():
        neighbours: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbours.extend(cells.get((cx + dx, cy + dy), ()))
        for a in members:
            for b in neighbours:
                if b <= a or abs(a - b) <= 1:
                    continue
                vec = pos[b] - pos[a]
                dist = float(np.hypot(vec[0], vec[1]))
                if dist >= radius or dist < 1e-9:
                    continue
                push = (
                    0.5
                    * opt.repulsion_strength
                    * cooling
                    * (radius - dist)
                    * vec
                    / dist
                )
                delta[a] -= push
                delta[b] += push
    return delta


def _ideal_ladder(length: int, opt: LayoutOptions) -> np.ndarray:
    """Coordinates of a textbook helix: two antiparallel strands, ``2L`` points.

    Row ``k`` is nucleotide ``i + k``; row ``L + k`` is its partner ``j - k``.
    """
    out = np.zeros((2 * length, 2))
    for k in range(length):
        out[k] = (k * opt.bond, 0.0)
        out[length + k] = (k * opt.bond, opt.pair)
    return out


def _place_pseudoknot_helices(
    coords: np.ndarray, pk_helices: Sequence, opt: LayoutOptions
) -> np.ndarray:
    """Snap each pseudoknot helix onto a rigid ladder before relaxing.

    Springs alone cannot reliably untangle a pseudoknot: the nested layout puts
    the crossing partners on opposite sides of the drawing, and the relaxation
    then settles into a knotted local minimum.  Fitting an ideal ladder onto the
    helix's current positions by a Kabsch superposition - and writing those
    positions back - starts the relaxation from an untangled configuration, so
    the loops simply stretch around a helix that already looks like a helix.
    """
    out = coords.copy()
    for helix in pk_helices:
        indices = [helix.i + k for k in range(helix.length)] + [
            helix.j - k for k in range(helix.length)
        ]
        ideal = _ideal_ladder(helix.length, opt)
        out[indices] = kabsch_align(ideal, out[indices], len(indices))
    return out


def layout_structure(
    structure: str,
    *,
    options: LayoutOptions | None = None,
    seed_coords: np.ndarray | None = None,
) -> np.ndarray:
    """2D coordinates for one dot-bracket structure.

    ``seed_coords`` supplies starting positions (from the previous movie frame)
    for the pseudoknot relaxation, which keeps successive frames consistent.
    """
    opt = options or LayoutOptions()
    pt = parse_dotbracket(structure)
    n = len(pt)
    if n == 0:
        return np.zeros((0, 2))

    nested_pt = list(pt)
    crossing: list[tuple[int, int]] = []
    helices = helices_from_pairtable(pt)
    from ..energy.pseudoknot import split_crossing

    core, pk = split_crossing(helices)
    for helix in pk:
        for a, b in helix.pairs:
            nested_pt[a] = nested_pt[b] = -1
            crossing.append((a, b))

    coords = _nested_layout(nested_pt, n, opt)
    if seed_coords is not None and len(seed_coords) >= n:
        blend = 0.35
        coords = (1.0 - blend) * coords + blend * seed_coords[:n]
    if pk:
        coords = _place_pseudoknot_helices(coords, pk, opt)
    braces = _ladder_constraints(helices, opt)
    return _relax(coords, nested_pt, crossing, n, opt, braces)


def orient_horizontally(coords: np.ndarray) -> np.ndarray:
    """Rotate a layout so its longest extent runs left to right.

    A structure drawn in a wide, short box - a gallery tile - is legible in
    proportion to how much of the box it fills, and a fold's natural orientation
    has nothing to do with the box it lands in.  The principal axis is put on the
    horizontal and the 5' end on the left, which also makes the orientation a
    function of the structure rather than of how the layout happened to converge.
    """
    if len(coords) < 2:
        return coords
    centred = coords - coords.mean(axis=0)
    # principal axis of the point cloud
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    rotated = centred @ vt.T
    if rotated[0, 0] > rotated[-1, 0]:
        rotated = -rotated
    return rotated


def kabsch_align(
    moving: np.ndarray, reference: np.ndarray, count: int | None = None
) -> np.ndarray:
    """Rigidly superpose ``moving`` onto ``reference`` (rotation + translation).

    Only the first ``count`` points are used to fit, which lets a growing chain
    be aligned on the nucleotides both frames actually have.
    """
    if count is None:
        count = min(len(moving), len(reference))
    if count < 2:
        return moving
    a = moving[:count]
    b = reference[:count]
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    covariance = (a - ca).T @ (b - cb)
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:  # forbid reflections
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return (moving - ca) @ rotation.T + cb


def layout_series(
    structures: Sequence[str],
    *,
    options: LayoutOptions | None = None,
    align: bool = True,
) -> list[np.ndarray]:
    """Lay out a time series of structures with frame-to-frame coherence."""
    opt = options or LayoutOptions()
    out: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for structure in structures:
        coords = layout_structure(structure, options=opt, seed_coords=previous)
        if align and previous is not None and len(coords) and len(previous):
            coords = kabsch_align(coords, previous, min(len(coords), len(previous)))
        out.append(coords)
        previous = coords
    return out


def interpolate(
    frames: Sequence[np.ndarray], steps: int, n_total: int | None = None
) -> list[np.ndarray]:
    """Insert ``steps - 1`` tweened frames between each pair of key frames.

    Nucleotides that do not yet exist in the earlier frame are held at the
    position of their 5' neighbour, so a newly transcribed base grows out of the
    chain rather than appearing from nowhere.
    """
    if steps <= 1 or len(frames) < 2:
        return [f.copy() for f in frames]
    size = n_total or max(len(f) for f in frames)

    def padded(frame: np.ndarray) -> np.ndarray:
        out = np.zeros((size, 2))
        count = len(frame)
        if count:
            out[:count] = frame
            if count < size:
                out[count:] = frame[count - 1]
        return out

    out: list[np.ndarray] = []
    for index in range(len(frames) - 1):
        a, b = padded(frames[index]), padded(frames[index + 1])
        for step in range(steps):
            weight = step / steps
            # smoothstep easing removes the visible velocity jump at key frames
            eased = weight * weight * (3.0 - 2.0 * weight)
            out.append((1.0 - eased) * a + eased * b)
    out.append(padded(frames[-1]))
    return out


def camera_path(
    frames: Sequence[np.ndarray],
    *,
    smoothing: float = 0.18,
    margin: float = 0.08,
    min_fraction: float = 0.45,
) -> np.ndarray:
    """A smoothly moving view box that follows the molecule, one row per frame.

    A single fixed box over the whole time course is dominated by the early,
    fully-extended chain, which leaves the folded structure small for the rest
    of the film.  Framing each frame independently instead makes the view jump.

    This smooths the per-frame boxes with a centred moving average, then takes
    the element-wise maximum against the raw box so the molecule can never be
    clipped, and finally applies a floor on the span so a ten-nucleotide chain
    is not magnified out of proportion.  The result reads as a camera gently
    zooming and panning.

    Returns an ``(n_frames, 4)`` array of ``(x0, y0, x1, y1)``.
    """
    if not frames:
        return np.zeros((0, 4))
    raw = np.array([bounding_box([f]) for f in frames], dtype=float)
    centre = np.stack(
        [(raw[:, 0] + raw[:, 2]) / 2.0, (raw[:, 1] + raw[:, 3]) / 2.0], axis=1
    )
    half = np.stack(
        [(raw[:, 2] - raw[:, 0]) / 2.0, (raw[:, 3] - raw[:, 1]) / 2.0], axis=1
    )

    count = len(frames)
    window = max(3, int(count * smoothing) | 1)
    kernel = np.ones(window) / window
    pad = window // 2

    def smooth(values: np.ndarray) -> np.ndarray:
        padded = np.concatenate(
            [np.full(pad, values[0]), values, np.full(pad, values[-1])]
        )
        return np.convolve(padded, kernel, mode="valid")[:count]

    centre = np.stack([smooth(centre[:, 0]), smooth(centre[:, 1])], axis=1)
    half = np.stack([smooth(half[:, 0]), smooth(half[:, 1])], axis=1)
    # The smoothed centre can sit off the frame's own centre, so the half-span
    # is measured *from the smoothed centre* to the furthest edge of the raw
    # box.  That keeps the pan smooth while guaranteeing nothing is ever
    # clipped - which a movie makes immediately obvious.
    need = np.stack(
        [
            np.maximum(centre[:, 0] - raw[:, 0], raw[:, 2] - centre[:, 0]),
            np.maximum(centre[:, 1] - raw[:, 1], raw[:, 3] - centre[:, 1]),
        ],
        axis=1,
    )
    half = np.maximum(half, need)
    floor = min_fraction * half.max(axis=0)
    half = np.maximum(half, floor) * (1.0 + margin)
    half = np.maximum(half, 1e-6)

    out = np.empty((count, 4))
    out[:, 0] = centre[:, 0] - half[:, 0]
    out[:, 1] = centre[:, 1] - half[:, 1]
    out[:, 2] = centre[:, 0] + half[:, 0]
    out[:, 3] = centre[:, 1] + half[:, 1]
    return out


def fit_aspect(
    box: Sequence[float], aspect: float
) -> tuple[float, float, float, float]:
    """Grow a view box to a target width/height ratio, keeping it centred."""
    x0, y0, x1, y1 = (float(v) for v in box)
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    if width / height < aspect:
        target = height * aspect
        centre = 0.5 * (x0 + x1)
        x0, x1 = centre - target / 2.0, centre + target / 2.0
    else:
        target = width / aspect
        centre = 0.5 * (y0 + y1)
        y0, y1 = centre - target / 2.0, centre + target / 2.0
    return (x0, y0, x1, y1)


def bounding_box(frames: Iterable[np.ndarray]) -> tuple[float, float, float, float]:
    """``(xmin, ymin, xmax, ymax)`` over a set of frames."""
    xs: list[float] = []
    ys: list[float] = []
    for frame in frames:
        if len(frame) == 0:
            continue
        xs.extend([float(frame[:, 0].min()), float(frame[:, 0].max())])
        ys.extend([float(frame[:, 1].min()), float(frame[:, 1].max())])
    if not xs:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs), min(ys), max(xs), max(ys))
