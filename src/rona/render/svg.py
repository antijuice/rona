"""Dependency-free SVG drawing of RNA secondary structures.

Everything here emits SVG text directly, so a structure picture can always be
produced even in an environment with nothing but numpy installed.  The movie
renderer uses matplotlib instead, because it needs rasterisation.

The drawing distinguishes the things that matter for cotranscriptional folding:
nucleotides that have not been transcribed yet are omitted, those still inside
the polymerase footprint are drawn muted, and crossing (pseudoknot) pairs are
drawn in a contrasting colour with a dashed stroke so they are never confused
with nested pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..struct import helices_from_pairtable, iter_pairs, parse_dotbracket
from . import colors
from .layout import LayoutOptions, bounding_box, layout_structure


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass(frozen=True, slots=True)
class DrawOptions:
    """Appearance of a structure drawing."""

    width: int = 760
    height: int = 620
    margin: float = 34.0
    #: Nucleotide marker radius, in SVG units at unit layout scale.
    base_radius: float = 0.30
    show_letters: bool | None = None  # None = automatic, based on scale
    number_every: int = 10
    color_by: str = "base"  # base | pairing | none
    background: str = colors.ELEMENT_COLORS["background"]
    title: str = ""
    subtitle: str = ""
    footprint: int = 0
    #: Optional per-nucleotide weights in [0, 1] for ``color_by="pairing"``.
    highlight: tuple[float, ...] | None = None


def _transform(
    coords: np.ndarray, opt: DrawOptions, box: tuple[float, float, float, float] | None
) -> tuple[np.ndarray, float]:
    """Fit layout coordinates into the canvas, returning points and the scale."""
    x0, y0, x1, y1 = box if box is not None else bounding_box([coords])
    span_x = max(x1 - x0, 1e-6)
    span_y = max(y1 - y0, 1e-6)
    usable_w = opt.width - 2 * opt.margin
    usable_h = opt.height - 2 * opt.margin - (34 if opt.title else 0)
    scale = min(usable_w / span_x, usable_h / span_y)
    offset_x = opt.margin + (usable_w - span_x * scale) / 2.0
    offset_y = opt.margin + (34 if opt.title else 0) + (usable_h - span_y * scale) / 2.0
    points = np.empty_like(coords)
    points[:, 0] = (coords[:, 0] - x0) * scale + offset_x
    # SVG y grows downwards; flip so 5'->3' reads the way a chemist draws it
    points[:, 1] = (y1 - coords[:, 1]) * scale + offset_y
    return points, scale


def structure_svg(
    sequence: str,
    structure: str,
    *,
    coords: np.ndarray | None = None,
    options: DrawOptions | None = None,
    layout_options: LayoutOptions | None = None,
    box: tuple[float, float, float, float] | None = None,
) -> str:
    """Render one structure as a standalone SVG document."""
    opt = options or DrawOptions()
    if coords is None:
        coords = layout_structure(structure, options=layout_options)
    n = len(structure)
    points, scale = _transform(coords[:n], opt, box)
    radius = max(1.6, opt.base_radius * scale)
    show_letters = (
        opt.show_letters if opt.show_letters is not None else radius >= 5.5
    )
    pt = parse_dotbracket(structure)

    from ..energy.pseudoknot import split_crossing

    _core, pk_helices = split_crossing(helices_from_pairtable(pt))
    pk_pairs = {p for helix in pk_helices for p in helix.pairs}

    available = n - opt.footprint if opt.footprint else n
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{opt.width}" '
        f'height="{opt.height}" viewBox="0 0 {opt.width} {opt.height}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif">',
        f'<rect width="{opt.width}" height="{opt.height}" fill="{opt.background}"/>',
    ]

    if opt.title:
        out.append(
            f'<text x="{opt.margin}" y="26" font-size="16" font-weight="600" '
            f'fill="{colors.ELEMENT_COLORS["text"]}">{_esc(opt.title)}</text>'
        )
    if opt.subtitle:
        out.append(
            f'<text x="{opt.width - opt.margin}" y="26" font-size="13" '
            f'text-anchor="end" fill="{colors.ELEMENT_COLORS["muted"]}">'
            f"{_esc(opt.subtitle)}</text>"
        )

    # --- backbone ---
    if n > 1:
        path = " ".join(
            ("M" if k == 0 else "L") + f"{points[k,0]:.2f},{points[k,1]:.2f}"
            for k in range(n)
        )
        out.append(
            f'<path d="{path}" fill="none" '
            f'stroke="{colors.ELEMENT_COLORS["backbone"]}" '
            f'stroke-width="{max(1.0, scale * 0.09):.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>'
        )

    # --- base pairs ---
    for i, j in iter_pairs(pt):
        is_pk = (i, j) in pk_pairs
        color = (
            colors.ELEMENT_COLORS["pseudoknot"]
            if is_pk
            else colors.ELEMENT_COLORS["pair"]
        )
        dash = ' stroke-dasharray="4 3"' if is_pk else ""
        width = max(1.0, scale * (0.11 if is_pk else 0.085))
        out.append(
            f'<line x1="{points[i,0]:.2f}" y1="{points[i,1]:.2f}" '
            f'x2="{points[j,0]:.2f}" y2="{points[j,1]:.2f}" stroke="{color}" '
            f'stroke-width="{width:.2f}" opacity="{0.95 if is_pk else 0.6}"{dash}/>'
        )

    # --- nucleotides ---
    for k in range(n):
        base = sequence[k] if k < len(sequence) else "N"
        if opt.color_by == "base":
            fill = colors.base_color(base)
        elif opt.color_by == "pairing" and opt.highlight is not None:
            fill = colors.ramp_color(
                opt.highlight[k] if k < len(opt.highlight) else 0.0
            )
        else:
            fill = colors.ELEMENT_COLORS["muted"]
        # nucleotides still inside the polymerase are drawn faded
        opacity = 0.35 if k >= available else 1.0
        out.append(
            f'<circle cx="{points[k,0]:.2f}" cy="{points[k,1]:.2f}" '
            f'r="{radius:.2f}" fill="{fill}" opacity="{opacity}" '
            f'stroke="{opt.background}" stroke-width="{max(0.6, radius*0.16):.2f}"/>'
        )
        if show_letters:
            out.append(
                f'<text x="{points[k,0]:.2f}" y="{points[k,1] + radius*0.36:.2f}" '
                f'font-size="{radius*1.15:.1f}" text-anchor="middle" fill="#ffffff" '
                f'font-weight="600">{base}</text>'
            )

    # --- numbering ---
    if opt.number_every and n > 1:
        step = opt.number_every
        centroid = points.mean(axis=0)
        for k in range(step - 1, n, step):
            # push the label radially outwards, which keeps it clear of the
            # backbone whichever way the chain happens to run
            away = points[k] - centroid
            norm = float(np.hypot(*away))
            if norm < 1e-6:
                previous = points[k - 1]
                away = points[k] - previous
                norm = float(np.hypot(*away)) or 1.0
            pos = points[k] + away / norm * (radius + 10.0)
            out.append(
                f'<text x="{pos[0]:.2f}" y="{pos[1] + 3:.2f}" font-size="10" '
                f'text-anchor="middle" fill="{colors.ELEMENT_COLORS["muted"]}">'
                f"{k + 1}</text>"
            )

    # --- 5' / 3' labels ---
    if n:
        out.append(
            f'<text x="{points[0,0]:.2f}" y="{points[0,1] - radius - 9:.2f}" '
            f'font-size="12" text-anchor="middle" font-weight="600" '
            f'fill="{colors.ELEMENT_COLORS["text"]}">5′</text>'
        )
        label = "3′" if available >= n else "RNAP"
        color = (
            colors.ELEMENT_COLORS["text"]
            if available >= n
            else colors.ELEMENT_COLORS["polymerase"]
        )
        out.append(
            f'<text x="{points[n-1,0]:.2f}" y="{points[n-1,1] + radius + 15:.2f}" '
            f'font-size="12" text-anchor="middle" font-weight="600" '
            f'fill="{color}">{label}</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


def legend_svg(width: int = 760, height: int = 40) -> str:
    """A small standalone legend for base colours and pair types."""
    entries = [(b, colors.base_color(b)) for b in "ACGU"] + [
        ("pseudoknot", colors.ELEMENT_COLORS["pseudoknot"])
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif">'
    ]
    x = 16.0
    for label, color in entries:
        out.append(
            f'<circle cx="{x + 6}" cy="{height/2}" r="6" fill="{color}"/>'
        )
        out.append(
            f'<text x="{x + 18}" y="{height/2 + 4}" font-size="12" '
            f'fill="{colors.ELEMENT_COLORS["text"]}">{_esc(label)}</text>'
        )
        x += 30 + 7 * len(label)
    out.append("</svg>")
    return "\n".join(out)


def write_svg(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
