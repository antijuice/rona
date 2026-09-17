"""Colour schemes for structure drawings.

Palettes are chosen to stay distinguishable in both light and dark contexts and
to survive the most common forms of colour-vision deficiency: the categorical
base colours are separated in lightness as well as hue, so they remain readable
even when hue information is lost.
"""

from __future__ import annotations

#: Nucleotide colours.
BASE_COLORS = {
    "A": "#e4572e",
    "C": "#2e86ab",
    "G": "#3fa34d",
    "U": "#8f5fd6",
    "N": "#9aa0a6",
}

#: Structural-element colours.
ELEMENT_COLORS = {
    "backbone": "#5a6270",
    "pair": "#404752",
    "pseudoknot": "#d1495b",
    "nascent": "#f2c14e",
    "polymerase": "#7f8896",
    "text": "#1c2028",
    "muted": "#7a828e",
    "background": "#ffffff",
    "panel": "#f4f5f7",
    "grid": "#dfe2e7",
}

#: Qualitative palette for stacked occupancy plots (structure identities).
OCCUPANCY_PALETTE = (
    "#2e86ab",
    "#e4572e",
    "#3fa34d",
    "#8f5fd6",
    "#f2c14e",
    "#00898a",
    "#d1495b",
    "#7a5195",
    "#6a8d3f",
    "#bc5090",
    "#356b87",
    "#ef8354",
    "#4f7942",
    "#a05195",
    "#c7b446",
    "#5f7d95",
)

#: Sequential ramp for pairing-probability heatmaps (low -> high).
PROBABILITY_RAMP = (
    "#f7f9fb",
    "#d9e6ef",
    "#b4d0e2",
    "#84b5d3",
    "#5596c0",
    "#2e76a6",
    "#1a5685",
    "#0d3a60",
)


#: Hue cycles across the sequence for pair-centre colouring.
PAIR_HUE_CYCLES = 3.5


def _hsl(hue: float, saturation: float, lightness: float) -> str:
    """HSL -> ``#rrggbb``."""
    hue = hue % 1.0

    def channel(shift: float) -> int:
        k = (shift + hue * 12.0) % 12.0
        a = saturation * min(lightness, 1.0 - lightness)
        value = lightness - a * max(-1.0, min(k - 3.0, 9.0 - k, 1.0))
        return max(0, min(255, round(value * 255)))

    return "#%02x%02x%02x" % (channel(0.0), channel(8.0), channel(4.0))


def pair_center_color(i: int, j: int, n: int, *, lightness: float = 0.46) -> str:
    """Colour a base pair by its "imaginary centre" ``(i + j) / 2``.

    Every pair of one helix shares the same centre - ``(i+k, j-k)`` has centre
    ``(i+j)/2`` for all ``k`` - so a helix is drawn in a single colour, and it
    keeps that colour for as long as it exists.  Following a colour through a
    time course therefore means following one structural motif, which is the
    whole point when the ensemble is rearranging.

    The hue wraps several times across the sequence so that helices only a few
    nucleotides apart are still easy to tell apart.

    The scheme is the one introduced by DrForna (Tang et al., 2023).
    """
    if n <= 0:
        return ELEMENT_COLORS["pair"]
    centre = (i + j) / 2.0 / n
    return _hsl(centre * PAIR_HUE_CYCLES, 0.62, lightness)


def base_color(base: str) -> str:
    return BASE_COLORS.get(base.upper(), BASE_COLORS["N"])


def occupancy_color(index: int) -> str:
    return OCCUPANCY_PALETTE[index % len(OCCUPANCY_PALETTE)]


def ramp_color(value: float) -> str:
    """Map ``value`` in ``[0, 1]`` onto the sequential ramp."""
    value = min(max(value, 0.0), 1.0)
    index = int(value * (len(PROBABILITY_RAMP) - 1) + 0.5)
    return PROBABILITY_RAMP[index]


def mix(color_a: str, color_b: str, weight: float) -> str:
    """Linear blend of two ``#rrggbb`` colours."""
    weight = min(max(weight, 0.0), 1.0)
    a = [int(color_a[k : k + 2], 16) for k in (1, 3, 5)]
    b = [int(color_b[k : k + 2], 16) for k in (1, 3, 5)]
    out = [round(x + (y - x) * weight) for x, y in zip(a, b)]
    return "#" + "".join(f"{v:02x}" for v in out)
