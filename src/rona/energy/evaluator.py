"""Sequence-bound energy evaluation, including fast single-helix deltas.

:class:`FoldingEnergy` is the interface the kinetic engine talks to.  It offers

* :meth:`FoldingEnergy.energy` - total free energy of any structure, nested or
  pseudoknotted, over the first ``n`` transcribed nucleotides;
* :meth:`FoldingEnergy.delta_add` / :meth:`FoldingEnergy.delta_remove` - the
  free-energy change of forming or melting one helix.

The delta methods take an O(loop) path whenever the structure is nested and
stays nested, which is the overwhelming majority of moves.  Anything that
touches a crossing helix falls back to two full evaluations; those are rare, so
the amortised cost stays low while the answer stays exact.  ``tests`` fuzz the
fast path against the slow one.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..seq import encode
from ..struct import Helix, pairtable_from_helices
from .model import FORBIDDEN, NearestNeighbourModel
from .pseudoknot import (
    PseudoknotModel,
    helix_stack_energy,
    split_crossing,
    topology_penalty,
)


def enclosing_pair(
    pt: Sequence[int], pos: int, n: int
) -> tuple[int, int] | None:
    """Innermost pair enclosing ``pos``, or ``None`` if it is in the exterior loop."""
    k = pos - 1
    while k >= 0:
        partner = pt[k]
        if partner < 0:
            k -= 1
        elif partner > k:
            if partner > pos:
                return (k, partner)
            k -= 1  # a sub-helix entirely to our left
        else:
            k = partner - 1  # skip back over a closed sub-helix
    return None


class FoldingEnergy:
    """Energy evaluation bound to one sequence."""

    def __init__(
        self,
        seq: str,
        model: NearestNeighbourModel | None = None,
        pk_model: PseudoknotModel | None = None,
    ) -> None:
        self.seq = seq
        self.enc = encode(seq)
        self.n = len(seq)
        self.model = model or NearestNeighbourModel()
        self.pk_model = pk_model or PseudoknotModel()

    # ------------------------------------------------------------------
    @property
    def kT(self) -> float:
        return self.model.kT

    def helix_stability(self, helix: Helix) -> float:
        """Context-free stacking energy of one helix, kcal/mol."""
        return helix_stack_energy(self.model, self.enc, helix)

    # ------------------------------------------------------------------
    def energy(self, helices: Iterable[Helix], n: int | None = None) -> float:
        """Total free energy in kcal/mol of a (possibly pseudoknotted) state."""
        helices = list(helices)
        length = self.n if n is None else n
        if not helices:
            return 0.0
        core, pk = split_crossing(
            helices, stability=[self.helix_stability(h) for h in helices]
        )
        pt_core = pairtable_from_helices(core, self.n)
        total = self.model.eval_nested(self.enc, self.seq, pt_core, length)
        if pk:
            pt_full = pairtable_from_helices(helices, self.n)
            for h in pk:
                total += self.helix_stability(h)
                total += topology_penalty(self.pk_model, h, core, pt_full)
        return total

    def energy_pt(self, pt: Sequence[int], n: int | None = None) -> float:
        """Total free energy from a pair table."""
        from ..struct import helices_from_pairtable

        return self.energy(helices_from_pairtable(pt), n)

    # ------------------------------------------------------------------
    # fast single-helix deltas
    # ------------------------------------------------------------------
    def _stack_run(self, helix: Helix) -> float:
        """Sum of the stacked-pair terms interior to one helix (dekacal)."""
        total = 0.0
        for k in range(helix.length - 1):
            i, j = helix.i + k, helix.j - k
            total += self.model.interior(self.enc, i, j, i + 1, j - 1)
        return total

    def delta_add(
        self,
        pt: list[int],
        helix: Helix,
        n: int,
        *,
        nested: bool = True,
    ) -> float:
        """Free-energy change of forming ``helix`` in the structure ``pt``.

        ``pt`` is left unmodified.  ``nested`` asserts that the current
        structure has no crossing helices *and* that ``helix`` does not cross
        anything in it; the caller knows this cheaply, and it selects the
        O(loop) path.
        """
        if not nested:
            return self._delta_full(pt, helix, n, add=True)

        closing = enclosing_pair(pt, helix.i, n)
        before = self.model.loop_energy(self.enc, self.seq, pt, closing, n)

        for a, b in helix.pairs:
            pt[a], pt[b] = b, a
        try:
            after = self.model.loop_energy(self.enc, self.seq, pt, closing, n)
            inner = self.model.loop_energy(self.enc, self.seq, pt, helix.inner, n)
            stacks = self._stack_run(helix)
        finally:
            for a, b in helix.pairs:
                pt[a], pt[b] = -1, -1

        if max(after, inner) >= FORBIDDEN * 100:
            return float("inf")
        return (after - before + inner + stacks) / 100.0

    def delta_remove(
        self,
        pt: list[int],
        helix: Helix,
        n: int,
        *,
        nested: bool = True,
    ) -> float:
        """Free-energy change of melting ``helix`` out of the structure ``pt``."""
        if not nested:
            return self._delta_full(pt, helix, n, add=False)

        closing = enclosing_pair(pt, helix.i, n)
        before = self.model.loop_energy(self.enc, self.seq, pt, closing, n)
        inner = self.model.loop_energy(self.enc, self.seq, pt, helix.inner, n)
        stacks = self._stack_run(helix)

        for a, b in helix.pairs:
            pt[a], pt[b] = -1, -1
        try:
            after = self.model.loop_energy(self.enc, self.seq, pt, closing, n)
        finally:
            for a, b in helix.pairs:
                pt[a], pt[b] = b, a

        if before >= FORBIDDEN * 100:
            return float("-inf")
        return (after - before - inner - stacks) / 100.0

    def _delta_full(
        self, pt: Sequence[int], helix: Helix, n: int, *, add: bool
    ) -> float:
        from ..struct import helices_from_pairtable

        current = list(helices_from_pairtable(pt))
        before = self.energy(current, n)
        if add:
            after = self.energy(current + [helix], n)
        else:
            remaining = [h for h in current if h != helix]
            if len(remaining) == len(current):
                # the helix is embedded in a longer ladder; rebuild from pairs
                scratch = list(pt)
                for a, b in helix.pairs:
                    scratch[a], scratch[b] = -1, -1
                remaining = list(helices_from_pairtable(scratch))
            after = self.energy(remaining, n)
        return after - before

    def __repr__(self) -> str:  # pragma: no cover
        return f"FoldingEnergy(len={self.n}, {self.model!r})"
