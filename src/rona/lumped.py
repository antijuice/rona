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

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable

from .energy.evaluator import FoldingEnergy
from .kinetics import EXCHANGE, MAX_EXPONENT, FORM, MELT, Move, RateModel
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


def nucleation_window(
    energy: FoldingEnergy, helix: Helix, min_helix: int
) -> Helix:
    """The nucleus of ``helix``: its most stable ``min_helix``-pair run.

    Lumping the window away also lumps away the *transition state* between two
    stem sets, which is the nucleus - a helix does not appear whole, it
    nucleates and then zips.  Keeping the barrier is what keeps the mode
    kinetic rather than a fast equilibrium sampler, so the nucleus has to be
    named explicitly.  It is a pure function of the window (and the sequence),
    which is what lets both directions of a transition agree on it.
    """
    if helix.length <= min_helix:
        return helix
    best = None
    best_cost = float("inf")
    for offset in range(helix.length - min_helix + 1):
        trial = Helix(helix.i + offset, helix.j - offset, min_helix)
        cost = energy.helix_stability(trial)
        if cost < best_cost:
            best, best_cost = trial, cost
    return best


def barrier_rates(
    prefactor: float, dg: float, dg_nucleus: float, kT: float
) -> tuple[float, float]:
    """``(form, melt)`` rates for a transition whose top is the nucleus.

    The transition state is the highest of the three free energies involved -
    the starting set, the set with only the nucleus formed, and the set with
    the whole window formed - and each direction is
    ``A exp(-(G_top - G_start)/RT)``.  The ratio is ``exp(-dg/RT)`` whatever
    the barrier, so detailed balance is exact by construction, and with no
    barrier this reduces to the Metropolis rule.

    This also recovers the microscopic kinetics rather than merely its
    equilibrium: a microscopic nucleation followed by fast zipping commits at
    ``k_nucleate min(1, e^(-dG_nuc/RT))``, which is what this returns.
    """
    if dg != dg or dg == float("inf") or dg_nucleus != dg_nucleus:
        return 0.0, 0.0
    top = max(0.0, dg, dg_nucleus)
    if top == float("inf"):
        return 0.0, 0.0
    forward = -min(top / kT, MAX_EXPONENT)
    reverse = -min((top - dg) / kT, MAX_EXPONENT)
    return prefactor * math.exp(forward), prefactor * math.exp(reverse)


def _intervals(
    moveset: MoveSet, windows: dict[int, Helix]
) -> dict[int, tuple[int, int]]:
    """Each window as ``(offset, length)`` in its stem's ladder."""
    return {
        index: (helix.i - moveset.stems[index].i, helix.length)
        for index, helix in windows.items()
    }


def _helix_of(moveset: MoveSet, index: int, span: tuple[int, int]) -> Helix:
    stem = moveset.stems[index]
    offset, length = span
    return Helix(stem.i + offset, stem.j - offset, length)


def _steps(
    moveset: MoveSet,
    current: dict[int, tuple[int, int]],
    target: dict[int, tuple[int, int]],
    min_helix: int,
    available: int,
):
    """Single microscopic moves that take ``current`` closer to ``target``.

    Only moves that remove a pair ``target`` does not want, or add one it does,
    so the walk is monotone and terminates.  These are the microscopic engine's
    own moves: nucleate ``min_helix`` pairs, zip or unzip one, melt at
    nucleation length.
    """
    occupied: set[int] = set()
    for index, span in current.items():
        occupied.update(_helix_of(moveset, index, span).positions())
    for index, want in target.items():
        if index in current:
            continue
        # nucleate, at the nucleus of the window it is heading for
        stem = moveset.stems[index]
        length = min(min_helix, want[1])
        for start in range(want[0], want[0] + want[1] - length + 1):
            span = (start, length)
            helix = _helix_of(moveset, index, span)
            if helix.j >= available:
                continue
            if occupied.isdisjoint(helix.positions()):
                yield {**current, index: span}
    for index, span in current.items():
        offset, length = span
        want = target.get(index)
        if want == span:
            continue
        if want is None:
            if length == min_helix:
                rest = dict(current)
                del rest[index]
                yield rest
            else:
                yield {**current, index: (offset + 1, length - 1)}
                yield {**current, index: (offset, length - 1)}
            continue
        want_off, want_len = want
        # retract the ends the target does not want
        if offset < want_off and length > min_helix:
            yield {**current, index: (offset + 1, length - 1)}
        if offset + length > want_off + want_len and length > min_helix:
            yield {**current, index: (offset, length - 1)}
        if length == min_helix and (
            offset + length <= want_off or want_off + want_len <= offset
        ):
            # nowhere to shrink to and no overlap: it has to go
            rest = dict(current)
            del rest[index]
            yield rest
        # extend towards the ends it does want
        stem = moveset.stems[index]
        if offset > want_off:
            grown = _helix_of(moveset, index, (offset - 1, length + 1))
            if grown.j < available and occupied.isdisjoint(
                ((grown.i, grown.j))
            ):
                yield {**current, index: (offset - 1, length + 1)}
        if offset + length < want_off + want_len:
            grown = _helix_of(moveset, index, (offset, length + 1))
            inner = (grown.i + length, grown.j - length)
            if grown.j < available and occupied.isdisjoint(inner):
                yield {**current, index: (offset, length + 1)}


def slide_barrier(
    energy: FoldingEnergy,
    moveset: MoveSet,
    min_helix: int,
    windows_a: dict[int, Helix],
    windows_b: dict[int, Helix],
    available: int,
    cache: dict,
    budget: int = 20000,
    ceiling: float = 30.0,
) -> float:
    """Free energy of the lowest saddle between two window assignments.

    Two lumped states that differ in *where* a helix sits are not separated by
    a barrier as high as melting it: the incumbent retracts pair by pair while
    the challenger zips into the nucleotides that frees, and the top of that
    trade is far below either helix's nucleus.  Getting this wrong is the
    difference between a chain that resolves a kinetic trap and one that
    freezes in it - a greedy walk gets it wrong, because retracting a helix
    from its cheaper end is exactly the direction that does not unblock the
    challenger.

    So the path is optimised, not guessed: a minimum-bottleneck search (Dijkstra
    with ``max`` in place of ``+``) over the *microscopic* move set - nucleate
    ``min_helix``, zip or unzip one pair, melt at nucleation length - restricted
    to moves the two endpoints disagree about, which bounds the search.  The
    result is the exact lowest saddle over monotone paths, and it is computed in
    a canonical direction so both directions of the transition read the same
    number, which is what detailed balance needs.

    Memoised on the pair of assignments: the lumped chain revisits the same
    transitions constantly, which is the whole point of lumping.  The search is
    also stopped once the saddle is ``ceiling`` kcal/mol above both endpoints,
    and that value returned: such a transition has a rate below 10^-16 s^-1 in
    either direction, so its exact height cannot matter, and the cut-off is the
    same number from both sides, which is all detailed balance asks.  Without it
    the hopeless transitions - which are most of them - cost as much to score as
    the ones that carry the kinetics.
    """
    span_a = _intervals(moveset, windows_a)
    span_b = _intervals(moveset, windows_b)
    key_a = tuple(sorted(span_a.items()))
    key_b = tuple(sorted(span_b.items()))
    if key_b < key_a:
        key_a, key_b = key_b, key_a
        span_a, span_b = span_b, span_a
    cache_key = (key_a, key_b, available)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    seen = cache.setdefault("energies", {})

    def energy_of(span, key):
        value = seen.get((key, available))
        if value is None:
            value = energy.energy(
                [_helix_of(moveset, i, v) for i, v in span.items()], available
            )
            seen[(key, available)] = value
        return value

    counter = itertools.count()
    start = tuple(sorted(span_a.items()))
    goal = tuple(sorted(span_b.items()))
    g_start = energy_of(span_a, start)
    g_goal = energy_of(span_b, goal)
    limit = max(g_start, g_goal) + ceiling
    best = {start: g_start}
    heap = [(g_start, next(counter), start)]
    answer = float("inf")
    spent = 0
    while heap:
        top, _tie, key = heapq.heappop(heap)
        if key == goal:
            answer = top
            break
        if top > best.get(key, float("inf")):
            continue
        if top > limit or spent > budget:
            answer = limit
            break
        span = dict(key)
        for trial in _steps(moveset, span, span_b, min_helix, available):
            trial_key = tuple(sorted(trial.items()))
            value = energy_of(trial, trial_key)
            spent += 1
            reached = top if top > value else value
            if reached > limit:
                continue
            if reached < best.get(trial_key, float("inf")):
                best[trial_key] = reached
                heapq.heappush(heap, (reached, next(counter), trial_key))
    if answer > limit:
        answer = limit
    cache[cache_key] = answer
    return answer


def _crossing_list(helices) -> bool:
    return any(
        a.crosses(b)
        for index, a in enumerate(helices)
        for b in helices[index + 1 :]
    )


def _has_crossing(windows: dict[int, Helix]) -> bool:
    return _crossing_list(list(windows.values()))


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
        from .kinetics import _competitor_lists

        self._competitors = _competitor_lists(moveset.stems)
        self._barrier_cache: dict = {}

    # ------------------------------------------------------------------
    def _windows_for(self, stems: Iterable[int]) -> dict[int, Helix]:
        return canonical_windows(
            self.moveset, stems, self.state.available, self.min_helix
        )

    def _ladder(self, index: int) -> set[int]:
        stem = self.moveset.stems[index]
        return {p for k in range(stem.length) for p in (stem.i + k, stem.j - k)}

    def _run_length(self, index: int, taken: set[int]) -> int:
        stem = self.moveset.stems[index]
        avail = self.state.available
        best = run = 0
        for k in range(stem.length):
            a, b = stem.i + k, stem.j - k
            if b < avail and a not in taken and b not in taken:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    def _blocked_by(
        self, windows: dict[int, Helix], holder: int, blocked: int
    ) -> bool:
        """Whether ``blocked`` cannot be placed, and ``holder`` alone is why."""
        taken: set[int] = set()
        held: set[int] = set()
        for key, helix in windows.items():
            (held if key == holder else taken).update(helix.positions())
        if not (held & self._ladder(blocked)):
            return False
        return self._run_length(blocked, taken | held) < self.min_helix <= (
            self._run_length(blocked, taken)
        )

    def _exchanges(self, holder: int) -> list[Move]:
        """One helix replacing a competitor, over a two-nucleus barrier.

        See :meth:`rona.kinetics.KineticEngine._lumped_exchanges`.
        """
        st = self.state
        kT = self.energy.kT
        common = st.stems - {holder}
        out: list[Move] = []
        for challenger in sorted(self._competitors[holder]):
            if challenger in st.stems:
                continue
            target = common | {challenger}
            windows_b = self._windows_for(target)
            if set(windows_b) != target:
                continue
            if not self._blocked_by(st.windows, holder, challenger):
                continue
            if not self._blocked_by(windows_b, challenger, holder):
                continue
            after = list(windows_b.values())
            if not self.energy.pk_model.enabled and _has_crossing(windows_b):
                continue
            dg = self.energy.energy(after, st.available) - st.energy
            dg_top = self._slide(windows_b) - st.energy
            rate, _reverse = barrier_rates(
                self.rates.prefactor(EXCHANGE), dg, dg_top, kT
            )
            if rate > 0.0:
                out.append(
                    Move(EXCHANGE, windows_b[challenger], challenger, dg, rate, holder)
                )
        return out

    def _others_moved(self, windows: dict[int, Helix], index: int) -> bool:
        for key, helix in self.state.windows.items():
            if key != index and windows.get(key) != helix:
                return True
        for key, helix in windows.items():
            if key != index and self.state.windows.get(key) != helix:
                return True
        return False

    def _slide(self, windows: dict[int, Helix]) -> float:
        return slide_barrier(
            self.energy,
            self.moveset,
            self.min_helix,
            self.state.windows,
            windows,
            self.state.available,
            self._barrier_cache,
        )

    def _transition_energy(self, windows: dict[int, Helix], index: int) -> float:
        """Free energy of the transition state for toggling ``index``.

        The two states differ by one stem, and possibly by where the stems it
        competes with sit.  The path between them retracts those competitors to
        their crowded windows and nucleates this stem: so the top of the path
        is ``windows`` with every other stem where the crowded state puts it and
        ``index`` present only as its nucleus.  It is a function of the
        unordered pair of states, which is all detailed balance needs, and it is
        the microscopic path - a helix retracts pair by pair while its
        competitor nucleates in the nucleotides that frees.
        """
        nucleus = nucleation_window(self.energy, windows[index], self.min_helix)
        structure = [h for key, h in windows.items() if key != index]
        structure.append(nucleus)
        return self.energy.energy(structure, self.state.available)

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
            if index not in windows or set(windows) != trial:
                # either the new stem cannot be placed, or placing it would
                # squeeze an existing helix below the minimum; both make the
                # target set one the chain does not contain
                continue
            if not self.energy.pk_model.enabled and _has_crossing(windows):
                # pseudoknots switched off: the state does not exist, exactly
                # as the incremental engine's infinite dG says
                continue
            dg = self._energy_of(windows) - st.energy
            top = (
                self._slide(windows)
                if self._others_moved(windows, index)
                else self._transition_energy(windows, index)
            )
            dg_nuc = top - st.energy
            rate, _ = barrier_rates(
                self.rates.prefactor(FORM), dg, dg_nuc, kT
            )
            if rate > 0.0:
                out.append(Move(FORM, windows[index], index, dg, rate))

        for index in sorted(st.stems):
            trial = st.stems - {index}
            windows = self._windows_for(trial)
            if set(windows) != trial:
                continue
            dg = self._energy_of(windows) - st.energy
            # the transition state is named from the *crowded* side, which is
            # this state, so both directions agree on it
            top = (
                self._slide(windows)
                if self._others_moved(windows, index)
                else self._transition_energy(st.windows, index)
            )
            dg_nuc = top - (st.energy + dg)
            _, rate = barrier_rates(
                self.rates.prefactor(MELT), -dg, dg_nuc, kT
            )
            if rate > 0.0:
                out.append(Move(MELT, st.windows[index], index, dg, rate))
            out.extend(self._exchanges(index))
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
        if move.kind == EXCHANGE:
            st.stems.discard(move.other)
            st.stems.add(move.stem)
        elif move.kind == FORM:
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
