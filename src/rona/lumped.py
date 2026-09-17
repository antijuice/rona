"""Lumped folding kinetics: the helix-window degree of freedom removed.

The microscopic simulator spends 99.7% of its events zipping base pairs onto and
off helix ends — a fast mode at internal equilibrium that changes nothing about
which stems are formed.  This module removes that variable.

A state here is a **set of stem indices**.  The window each stem occupies is a
deterministic, pure function of that set, :func:`canonical_windows`, so

* the free energy is a genuine state function, and the Metropolis rule therefore
  imposes detailed balance on the lumped chain exactly;
* ``FORM(s)`` and ``MELT(s)`` are exact inverses, with no history dependence;
* a helix can never be stranded at a stale window, because windows are
  recomputed from the stem set after every move.  Stranding was the defect that
  forced zipping into the microscopic move set to begin with.

The approximation is confined to the free energy: the window *entropy* is
discarded, at most ``RT ln|W|`` per helix, much of which cancels between states
because it appears on both sides of every ``dG``.  Its size is measured, not
assumed - see ``docs/lumping.md`` and ``docs/validation.md``.

Because the event count collapses by three orders of magnitude, each candidate's
``dG`` is obtained from full O(n) energy evaluations instead of an incremental
update.  That removes the whole invalidation apparatus from this mode, along
with the class of bug it kept producing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .energy.evaluator import FoldingEnergy
from .kinetics import MAX_EXPONENT, FORM, MELT, Move, RateModel
from .moves import MoveSet
from .struct import Helix, to_dotbracket

LUMPED_MODE = "lumped"


def canonical_windows(
    moveset: MoveSet,
    stems: Iterable[int],
    available: int,
    min_helix: int,
) -> dict[int, Helix]:
    """Window for each stem in the set - a pure function of the set.

    Stems are placed in increasing index order, each taking the longest run of
    its ladder whose nucleotides are still free and transcribed.  Fixing the
    order is what makes this a function of the set rather than of the history,
    which is what the detailed-balance argument rests on.

    A stem that cannot reach ``min_helix`` under this placement is omitted; the
    caller treats such a set as unreachable.
    """
    taken: set[int] = set()
    out: dict[int, Helix] = {}
    for index in sorted(stems):
        stem = moveset.stems[index]
        best_len = best_off = run = start = 0
        for k in range(stem.length):
            a, b = stem.i + k, stem.j - k
            if b < available and a not in taken and b not in taken:
                if run == 0:
                    start = k
                run += 1
                if run > best_len:
                    best_len, best_off = run, start
            else:
                run = 0
        if best_len < min_helix:
            continue
        helix = Helix(stem.i + best_off, stem.j - best_off, best_len)
        out[index] = helix
        taken.update(helix.positions())
    return out


def _has_crossing(windows: dict[int, Helix]) -> bool:
    helices = list(windows.values())
    return any(
        a.crosses(b)
        for index, a in enumerate(helices)
        for b in helices[index + 1 :]
    )


@dataclass(slots=True)
class LumpedState:
    """Folding state as a set of formed stems."""

    n_total: int
    length: int = 0
    available: int = 0
    stems: set[int] = field(default_factory=set)
    windows: dict[int, Helix] = field(default_factory=dict)
    energy: float = 0.0

    def helices(self) -> list[Helix]:
        return list(self.windows.values())

    def pair_table(self) -> list[int]:
        pt = [-1] * self.n_total
        for helix in self.windows.values():
            for a, b in helix.pairs:
                pt[a], pt[b] = b, a
        return pt

    def dotbracket(self) -> str:
        return to_dotbracket(self.pair_table()[: self.length])

    def copy(self) -> "LumpedState":
        return LumpedState(
            n_total=self.n_total,
            length=self.length,
            available=self.available,
            stems=set(self.stems),
            windows=dict(self.windows),
            energy=self.energy,
        )


class LumpedEngine:
    """Gillespie SSA over sets of formed stems."""

    def __init__(
        self,
        energy: FoldingEnergy,
        moveset: MoveSet,
        rates: RateModel,
        *,
        min_helix: int = 3,
    ) -> None:
        self.energy = energy
        self.moveset = moveset
        self.rates = rates
        self.min_helix = max(min_helix, moveset.nucleation_size)
        self.n = energy.n
        self.state = LumpedState(n_total=self.n)
        self._active = 0
        self._order = sorted(
            range(len(moveset.stems)), key=lambda k: moveset.stems[k].j
        )
        self._moves: list[Move] = []
        self._total = 0.0
        self._dirty = True

    # ------------------------------------------------------------------
    def _windows_for(self, stems: Iterable[int]) -> dict[int, Helix]:
        return canonical_windows(
            self.moveset, stems, self.state.available, self.min_helix
        )

    def _others_fixed(self, windows: dict[int, Helix], index: int) -> bool:
        """Whether ``windows`` agrees with the current state away from ``index``.

        Toggling one stem may only be a transition when it leaves the rest of
        the structure alone; the condition is the same whether ``index`` is
        being added or removed, which is what makes FORM and MELT exact
        inverses.  Nothing becomes unreachable: any valid set is still built up
        in increasing index order, because a stem's canonical window depends
        only on stems of lower index.
        """
        for key, helix in self.state.windows.items():
            if key != index and windows.get(key) != helix:
                return False
        return True

    def _energy_of(self, windows: dict[int, Helix]) -> float:
        return self.energy.energy(list(windows.values()), self.state.available)

    def grow(self, new_length: int, footprint: int) -> None:
        """Extend the transcript and re-derive the windows."""
        st = self.state
        st.length = min(new_length, self.n)
        st.available = max(0, st.length - footprint)
        while (
            self._active < len(self.moveset.stems)
            and self.moveset.stems[self._order[self._active]].j < st.available
        ):
            self._active += 1
        # more sequence may let existing helices extend
        st.windows = self._windows_for(st.stems)
        st.stems = set(st.windows)
        st.energy = self._energy_of(st.windows)
        self._dirty = True

    # ------------------------------------------------------------------
    def _candidates(self) -> list[Move]:
        st = self.state
        kT = self.energy.kT
        out: list[Move] = []

        for position in range(self._active):
            index = self._order[position]
            if index in st.stems:
                continue
            trial = st.stems | {index}
            windows = self._windows_for(trial)
            if index not in windows or not self._others_fixed(windows, index):
                # (the checks below also reject sets the energy model forbids)
                # either the new stem cannot be placed, or placing it would
                # move an existing helix; both make the move something other
                # than the inverse of melting this stem again
                continue
            if not self.energy.pk_model.enabled and _has_crossing(windows):
                # pseudoknots switched off: the state does not exist, exactly
                # as the incremental engine's infinite dG says
                continue
            dg = self._energy_of(windows) - st.energy
            rate = self.rates.rate(FORM, dg, kT)
            if rate > 0.0:
                out.append(Move(FORM, windows[index], index, dg, rate))

        for index in sorted(st.stems):
            trial = st.stems - {index}
            windows = self._windows_for(trial)
            if set(windows) != trial or not self._others_fixed(windows, index):
                continue
            dg = self._energy_of(windows) - st.energy
            rate = self.rates.rate(MELT, dg, kT)
            if rate > 0.0:
                out.append(Move(MELT, st.windows[index], index, dg, rate))
        return out

    def propensity(self) -> float:
        if self._dirty:
            self._moves = self._candidates()
            self._total = math.fsum(m.rate for m in self._moves)
            self._dirty = False
        return self._total

    def select(self, u: float) -> Move | None:
        if self._total <= 0.0 or not self._moves:
            return None
        target = u * self._total
        acc = 0.0
        for move in self._moves:
            acc += move.rate
            if acc >= target:
                return move
        return self._moves[-1]

    def apply(self, move: Move) -> None:
        st = self.state
        if move.kind == FORM:
            st.stems.add(move.stem)
        else:
            st.stems.discard(move.stem)
        st.windows = self._windows_for(st.stems)
        st.stems = set(st.windows)
        st.energy += move.dg
        self._dirty = True

    # ------------------------------------------------------------------
    def has_pseudoknot(self) -> bool:
        helices = self.state.helices()
        return any(
            a.crosses(b)
            for index, a in enumerate(helices)
            for b in helices[index + 1 :]
        )

    def resync_energy(self) -> None:
        self.state.energy = self._energy_of(self.state.windows)

    def window_entropy_bound(self) -> float:
        """Upper bound on the free energy discarded by ignoring window entropy.

        ``RT ln|W|`` summed over formed helices, where ``|W|`` counts the
        windows of at least ``min_helix`` pairs available to that stem.  This is
        a worst case: it is attained only if every window were degenerate.
        """
        total = 0.0
        for index in self.state.stems:
            stem = self.moveset.stems[index]
            count = sum(
                1
                for length in range(self.min_helix, stem.length + 1)
                for _offset in range(stem.length - length + 1)
            )
            if count > 1:
                total += self.energy.kT * math.log(count)
        return total
