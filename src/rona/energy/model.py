"""Turner nearest-neighbour energy model with a loop-local evaluation API.

Two things distinguish this implementation from a plain "score a structure"
routine, and both exist to serve the kinetic engine:

1. **Loop-local evaluation.**  :meth:`NearestNeighbourModel.loop_energy` scores
   a single loop given its closing pair.  Because the total free energy is a
   sum over loops, a move that adds or removes one helix changes only two or
   three loops, so its ``dG`` costs O(loop) rather than O(n).

2. **Exactness.**  The published Turner 2004 tables are used verbatim,
   including the 1x1/2x1/2x2 interior-loop tables and the special tri-, tetra-
   and hexaloop bonuses, so energies agree with ViennaRNA's ``eval_structure``
   to within integer rounding (see ``tests/test_energy_vs_vienna.py``).

All public functions return **kcal/mol**; the tables themselves are kept in
dekacal/mol to match the parameter file and to keep the arithmetic in integers
for as long as possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..seq import NS, PAIR_TYPE
from . import paramfile
from .paramfile import INF, TurnerParams

#: Gas constant in kcal/(mol K).
GAS_CONSTANT = 0.0019872
#: 0 degrees Celsius in Kelvin, as used by the parameter tables.
K0 = 273.15
#: Reference temperature of the tabulated free energies.
T37 = 37.0

MAXLOOP = 30
#: Energies at or above this are treated as forbidden.
FORBIDDEN = INF / 100.0


def _rescale(dg: np.ndarray, dh: np.ndarray, factor: float) -> np.ndarray:
    """Linear ``dH``/``dG`` interpolation to another temperature.

    ``dG(T) = dH - (dH - dG37) * T / T37`` with both temperatures in Kelvin,
    the standard extrapolation used with the Turner tables.  Results are
    truncated to whole dekacalories, as the reference implementations do, so
    that energies stay bit-comparable across tools.  Forbidden entries stay
    forbidden.
    """
    out = np.trunc(dh - (dh - dg) * factor)
    return np.where(dg >= INF, float(INF), out)


@dataclass(slots=True)
class LoopContext:
    """Description of one loop, used for reporting and pseudoknot scoring."""

    kind: str  # exterior | hairpin | stack | bulge | interior | multi
    closing: tuple[int, int] | None
    branches: tuple[tuple[int, int], ...]
    unpaired: int
    energy: float


class NearestNeighbourModel:
    """Turner nearest-neighbour model at a given temperature.

    Parameters
    ----------
    params:
        Parsed parameter tables; defaults to the bundled Turner 2004 set.
    temperature:
        Folding temperature in degrees Celsius.
    dangles:
        ``2`` (default) applies terminal-mismatch stabilisation to every helix
        end in exterior and multi loops, matching ViennaRNA's default model.
        ``0`` disables dangling-end contributions entirely.
    special_hairpins:
        Apply the tabulated tri-/tetra-/hexaloop bonuses.
    min_loop:
        Minimum hairpin loop size (3 for sterically allowed RNA).
    """

    def __init__(
        self,
        params: TurnerParams | None = None,
        *,
        temperature: float = T37,
        dangles: int = 2,
        special_hairpins: bool = True,
        min_loop: int = 3,
    ) -> None:
        if dangles not in (0, 2):
            raise ValueError("dangles must be 0 or 2")
        self.params = params or paramfile.default_params()
        self.temperature = float(temperature)
        self.dangles = dangles
        self.special_hairpins = special_hairpins
        self.min_loop = min_loop
        self.kT = GAS_CONSTANT * (K0 + self.temperature)
        self._build_tables()

    # ------------------------------------------------------------------
    def _build_tables(self) -> None:
        p = self.params
        factor = (K0 + self.temperature) / (K0 + T37)

        def tab(name: str) -> np.ndarray:
            return _rescale(p[name], p[name + "_enthalpies"], factor)

        self.stack = tab("stack")
        self.mismatch_hairpin = tab("mismatch_hairpin")
        self.mismatch_internal = tab("mismatch_internal")
        self.mismatch_internal_1n = tab("mismatch_internal_1n")
        self.mismatch_internal_23 = tab("mismatch_internal_23")
        self.mismatch_multi = np.minimum(0.0, tab("mismatch_multi"))
        self.mismatch_exterior = np.minimum(0.0, tab("mismatch_exterior"))
        # Dangling ends and the exterior/multiloop terminal mismatches are
        # stabilising by construction; after temperature rescaling they are
        # clamped at zero so an extrapolation can never turn them repulsive.
        self.dangle5 = np.minimum(0.0, tab("dangle5"))
        self.dangle3 = np.minimum(0.0, tab("dangle3"))
        self.int11 = tab("int11")
        self.int21 = tab("int21")
        self.int22 = tab("int22")
        self.hairpin_init = tab("hairpin")
        self.bulge_init = tab("bulge")
        self.internal_init = tab("internal")

        def scalar(dg: float, dh: float) -> float:
            return float(math.trunc(dh - (dh - dg) * factor))

        self.ml_base = scalar(p.ml_base, p.ml_base_dh)
        self.ml_closing = scalar(p.ml_closing, p.ml_closing_dh)
        self.ml_intern = scalar(p.ml_intern, p.ml_intern_dh)
        self.ninio = scalar(p.ninio, p.ninio_dh)
        self.ninio_max = p.ninio_max
        self.terminal_au = scalar(p.terminal_au, p.terminal_au_dh)
        self.lxc = p.lxc * factor

        self.tetraloops = {
            k: scalar(dg, dh) for k, (dg, dh) in p.tetraloops.items()
        }
        self.triloops = {k: scalar(dg, dh) for k, (dg, dh) in p.triloops.items()}
        self.hexaloops = {k: scalar(dg, dh) for k, (dg, dh) in p.hexaloops.items()}

        if self.dangles == 0:
            zero = np.zeros_like(self.dangle5)
            self.dangle5 = zero
            self.dangle3 = np.zeros_like(self.dangle3)
            self.mismatch_multi = np.zeros_like(self.mismatch_multi)
            self.mismatch_exterior = np.zeros_like(self.mismatch_exterior)

    # ------------------------------------------------------------------
    # elementary contributions (dekacal/mol)
    # ------------------------------------------------------------------
    def _pt(self, enc: Sequence[int], i: int, j: int) -> int:
        t = PAIR_TYPE[enc[i]][enc[j]]
        return t if t else NS

    def _terminal_au(self, ptype: int) -> float:
        return self.terminal_au if ptype > 2 else 0.0

    def _size_penalty(self, table: np.ndarray, size: int) -> float:
        if size <= MAXLOOP:
            return float(table[size])
        return float(table[MAXLOOP]) + self.lxc * math.log(size / MAXLOOP)

    def ext_stem(self, ptype: int, s5: int, s3: int) -> float:
        """Contribution of one helix end sitting in the exterior loop."""
        e = 0.0
        if s5 >= 0 and s3 >= 0:
            e += float(self.mismatch_exterior[ptype][s5][s3])
        elif s5 >= 0:
            e += float(self.dangle5[ptype][s5])
        elif s3 >= 0:
            e += float(self.dangle3[ptype][s3])
        return e + self._terminal_au(ptype)

    def ml_stem(self, ptype: int, s5: int, s3: int) -> float:
        """Contribution of one helix end sitting in a multiloop."""
        e = 0.0
        if s5 >= 0 and s3 >= 0:
            e += float(self.mismatch_multi[ptype][s5][s3])
        elif s5 >= 0:
            e += float(self.dangle5[ptype][s5])
        elif s3 >= 0:
            e += float(self.dangle3[ptype][s3])
        return e + self._terminal_au(ptype) + self.ml_intern

    def hairpin(self, enc: Sequence[int], seq: str, i: int, j: int) -> float:
        """Free energy of the hairpin loop closed by ``(i, j)``, dekacal/mol."""
        size = j - i - 1
        if size < self.min_loop:
            return float(INF)
        ptype = self._pt(enc, i, j)
        e = self._size_penalty(self.hairpin_init, size)
        if self.special_hairpins:
            if size == 4:
                motif = seq[i : i + 6]
                if motif in self.tetraloops:
                    return self.tetraloops[motif]
            elif size == 6:
                motif = seq[i : i + 8]
                if motif in self.hexaloops:
                    return self.hexaloops[motif]
            elif size == 3:
                motif = seq[i : i + 5]
                if motif in self.triloops:
                    return self.triloops[motif]
                return e + self._terminal_au(ptype)
        elif size == 3:
            return e + self._terminal_au(ptype)
        return e + float(self.mismatch_hairpin[ptype][enc[i + 1]][enc[j - 1]])

    def interior(
        self, enc: Sequence[int], i: int, j: int, p: int, q: int
    ) -> float:
        """Free energy of the loop closed by ``(i, j)`` enclosing ``(p, q)``.

        Covers stacks (``n1 = n2 = 0``), bulges and all interior loops, using
        the tabulated 1x1, 2x1 and 2x2 values where they apply.
        """
        n1 = p - i - 1
        n2 = j - q - 1
        if n1 < 0 or n2 < 0:
            raise ValueError(f"({p},{q}) is not enclosed by ({i},{j})")
        t1 = self._pt(enc, i, j)
        t2 = self._pt(enc, q, p)  # inner pair, reversed orientation
        nl, ns = (n1, n2) if n1 > n2 else (n2, n1)

        if nl == 0:
            return float(self.stack[t1][t2])

        if ns == 0:  # bulge
            e = self._size_penalty(self.bulge_init, nl)
            if nl == 1:
                e += float(self.stack[t1][t2])
            else:
                e += self._terminal_au(t1) + self._terminal_au(t2)
            return e

        si1, sj1 = enc[i + 1], enc[j - 1]
        sp1, sq1 = enc[p - 1], enc[q + 1]

        if ns == 1:
            if nl == 1:
                return float(self.int11[t1][t2][si1][sj1])
            if nl == 2:
                if n1 == 1:
                    return float(self.int21[t1][t2][si1][sq1][sj1])
                return float(self.int21[t2][t1][sq1][si1][sp1])
            e = self._size_penalty(self.internal_init, nl + 1)
            e += min(self.ninio_max, (nl - ns) * self.ninio)
            e += float(self.mismatch_internal_1n[t1][si1][sj1])
            e += float(self.mismatch_internal_1n[t2][sq1][sp1])
            return e
        if ns == 2:
            if nl == 2:
                return float(self.int22[t1][t2][si1][sp1][sq1][sj1])
            if nl == 3:
                e = float(self.internal_init[5]) + self.ninio
                e += float(self.mismatch_internal_23[t1][si1][sj1])
                e += float(self.mismatch_internal_23[t2][sq1][sp1])
                return e

        e = self._size_penalty(self.internal_init, nl + ns)
        e += min(self.ninio_max, (nl - ns) * self.ninio)
        e += float(self.mismatch_internal[t1][si1][sj1])
        e += float(self.mismatch_internal[t2][sq1][sp1])
        return e

    # ------------------------------------------------------------------
    # loop-local evaluation
    # ------------------------------------------------------------------
    def loop_branches(
        self, pt: Sequence[int], closing: tuple[int, int] | None, n: int | None = None
    ) -> tuple[list[tuple[int, int]], int]:
        """Branches directly inside a loop, and the count of unpaired bases.

        ``closing=None`` walks the exterior loop.  Positions that are paired to
        a partner outside the window (which happens for pseudoknot helices that
        have been split off) are reported as unpaired here; the pseudoknot
        module accounts for them separately.
        """
        if closing is None:
            lo, hi = 0, (len(pt) if n is None else n)
        else:
            lo, hi = closing[0] + 1, closing[1]
        branches: list[tuple[int, int]] = []
        unpaired = 0
        k = lo
        while k < hi:
            partner = pt[k]
            if partner < 0 or not (lo <= partner < hi) or partner < k:
                unpaired += 1
                k += 1
                continue
            branches.append((k, partner))
            k = partner + 1
        return branches, unpaired

    def loop_energy(
        self,
        enc: Sequence[int],
        seq: str,
        pt: Sequence[int],
        closing: tuple[int, int] | None,
        n: int | None = None,
    ) -> float:
        """Free energy (dekacal/mol) of the single loop closed by ``closing``."""
        length = len(pt) if n is None else n
        branches, unpaired = self.loop_branches(pt, closing, length)

        if closing is None:
            e = 0.0
            for a, b in branches:
                t = self._pt(enc, a, b)
                s5 = enc[a - 1] if a > 0 else -1
                s3 = enc[b + 1] if b + 1 < length else -1
                e += self.ext_stem(t, s5, s3)
            return e

        i, j = closing
        if not branches:
            return self.hairpin(enc, seq, i, j)
        if len(branches) == 1:
            p, q = branches[0]
            return self.interior(enc, i, j, p, q)

        # multiloop: closing stem is scored with the reversed pair type
        t = self._pt(enc, j, i)
        e = self.ml_closing + self.ml_base * unpaired
        e += self.ml_stem(t, enc[j - 1], enc[i + 1])
        for a, b in branches:
            tb = self._pt(enc, a, b)
            e += self.ml_stem(tb, enc[a - 1], enc[b + 1])
        return e

    def loop_kind(
        self, pt: Sequence[int], closing: tuple[int, int] | None, n: int | None = None
    ) -> str:
        if closing is None:
            return "exterior"
        branches, _ = self.loop_branches(pt, closing, n)
        if not branches:
            return "hairpin"
        if len(branches) > 1:
            return "multi"
        p, q = branches[0]
        n1, n2 = p - closing[0] - 1, closing[1] - q - 1
        if n1 == 0 and n2 == 0:
            return "stack"
        if n1 == 0 or n2 == 0:
            return "bulge"
        return "interior"

    # ------------------------------------------------------------------
    def closing_pairs(
        self, pt: Sequence[int], n: int | None = None
    ) -> list[tuple[int, int] | None]:
        """Every loop in a nested pair table, identified by its closing pair."""
        length = len(pt) if n is None else n
        out: list[tuple[int, int] | None] = [None]
        for i in range(length):
            j = pt[i]
            if j > i:
                out.append((i, j))
        return out

    def eval_nested(
        self, enc: Sequence[int], seq: str, pt: Sequence[int], n: int | None = None
    ) -> float:
        """Total free energy (kcal/mol) of a **nested** structure."""
        total = 0.0
        for closing in self.closing_pairs(pt, n):
            e = self.loop_energy(enc, seq, pt, closing, n)
            if e >= FORBIDDEN * 100:
                return float("inf")
            total += e
        return total / 100.0

    def decompose(
        self, enc: Sequence[int], seq: str, pt: Sequence[int], n: int | None = None
    ) -> list[LoopContext]:
        """Per-loop breakdown, useful for diagnostics and for rendering."""
        out: list[LoopContext] = []
        for closing in self.closing_pairs(pt, n):
            branches, unpaired = self.loop_branches(pt, closing, n)
            out.append(
                LoopContext(
                    kind=self.loop_kind(pt, closing, n),
                    closing=closing,
                    branches=tuple(branches),
                    unpaired=unpaired,
                    energy=self.loop_energy(enc, seq, pt, closing, n) / 100.0,
                )
            )
        return out

    def boltzmann(self, dg: float) -> float:
        """``exp(-dG / kT)`` with ``dG`` in kcal/mol."""
        return math.exp(-dg / self.kT)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NearestNeighbourModel({self.params.name}, T={self.temperature}C, "
            f"dangles={self.dangles})"
        )
