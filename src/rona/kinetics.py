"""Stochastic helix-level folding kinetics (Gillespie SSA).

This is a genuine continuous-time Markov simulation over structure space, not a
sequence of equilibrium calculations.  Each trajectory is an exact realisation
of the master equation ``dP/dt = K P``, with ``K`` defined implicitly by the
move set below.

Move sets
---------
``helix`` (default)
    A move forms the longest currently-unobstructed ladder of a maximal stem,
    or melts a formed helix in its entirety.  Single-base-pair zipping is
    treated as instantaneous, which it effectively is: adding a pair to an
    existing helix end takes ~100 ns, against ~10 us for a nucleation event.
    Coarse-graining it away removes a stiff fast mode that would otherwise
    consume ~10^8 events per second of simulated time while leaving the
    coarse folding pathway untouched.

``breathe``
    Adds base-pair resolution: helices nucleate at a fixed window size and then
    zip or unzip one pair at a time with the much larger ``k_zip`` prefactor.
    Physically finer, dramatically more expensive - intended for short
    sequences or short time windows.

Both move sets are closed under reversal with exactly one forward and one
reverse transition per connected pair of states, which is what lets the rate
rule below impose detailed balance.

Rates
-----
``metropolis`` (default)
    ``k = k0 * exp(-max(0, dG) / RT)``
``kawasaki``
    ``k = k0 * exp(-dG / (2 RT))``

Both satisfy ``k(A->B) / k(B->A) = exp(-(G_B - G_A) / RT)``, so a fixed-length
chain relaxes to the correct Boltzmann ensemble.  A *growing* chain does not,
and that is the entire point.

Performance
-----------
Propensities are cached in a Fenwick tree and invalidated loop-locally: after a
move, only candidates with a nucleotide directly inside the one loop that
changed are rescored.  Cost per event therefore scales with the size of the
affected loop, not with the length of the transcript.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from .energy.evaluator import FoldingEnergy, enclosing_pair
from .energy.pseudoknot import PseudoknotModel, pseudoknot_region
from .moves import FORM, MELT, UNZIP_IN, UNZIP_OUT, ZIP_IN, ZIP_OUT, MoveSet
from .struct import Helix, to_dotbracket

#: Exponent clamp, keeping ``exp`` away from overflow/underflow.
MAX_EXPONENT = 400.0

HELIX_MODE = "helix"
BREATHE_MODE = "breathe"
#: Window degree of freedom removed; see :mod:`rona.lumped` and docs/lumping.md.
LUMPED_MODE = "lumped"
#: One helix replacing a competitor in a single transition (lumped mode only).
EXCHANGE = "exchange"
MOVE_SETS = (HELIX_MODE, BREATHE_MODE, LUMPED_MODE)


class Fenwick:
    """Binary indexed tree supporting O(1) total and O(log n) update/sampling.

    The running total is tracked separately rather than read off the tree, since
    ``tree[size]`` only equals the full prefix sum when ``size`` is a power of
    two.
    """

    __slots__ = ("size", "_tree", "_values", "_total", "_top")

    def __init__(self, size: int) -> None:
        self.size = size
        self._tree = [0.0] * (size + 1)
        self._values = [0.0] * size
        self._total = 0.0
        self._top = 1 << max(0, size.bit_length() - 1)

    def value(self, index: int) -> float:
        return self._values[index]

    def set(self, index: int, value: float) -> None:
        delta = value - self._values[index]
        if delta == 0.0:
            return
        self._values[index] = value
        self._total += delta
        k = index + 1
        while k <= self.size:
            self._tree[k] += delta
            k += k & -k

    def total(self) -> float:
        return self._total

    def resum(self) -> None:
        """Recompute the cached total, clearing accumulated round-off."""
        self._total = math.fsum(self._values)

    def find(self, target: float) -> int:
        """Largest index whose exclusive prefix sum is <= ``target``."""
        pos = 0
        bit = self._top
        remaining = target
        while bit:
            nxt = pos + bit
            if nxt <= self.size and self._tree[nxt] <= remaining:
                pos = nxt
                remaining -= self._tree[nxt]
            bit >>= 1
        return min(pos, self.size - 1)


@dataclass(frozen=True, slots=True)
class RateModel:
    """Kinetic prefactors and the detailed-balance-preserving rate rule."""

    #: Attempt frequency for nucleating a helix (s^-1).
    k_nucleate: float = 1.0e5
    #: Attempt frequency for adding/removing one pair at a helix end (s^-1).
    #
    # Physically this is ~10^7 s^-1.  The default is deliberately lower,
    # because zipping is a futile fast mode: measured on a 58 nt transcript,
    # 99.7% of all events are zip/unzip, with forward and reverse counts equal
    # to three significant figures - the helix length is already at internal
    # equilibrium and merely jittering.  What the coarse kinetics needs is only
    # that zipping be fast compared with nucleation (10^5 s^-1) and with the
    # observation timescale (seconds), which 10^6 s^-1 amply satisfies, and the
    # event count scales linearly with it.  ``examples/05_timescale_separation.py``
    # checks that the coarse result does not move when this is varied.
    k_zip: float = 1.0e6
    #: ``metropolis`` or ``kawasaki``.
    scheme: str = "metropolis"

    def prefactor(self, kind: str) -> float:
        return self.k_zip if kind in (ZIP_IN, ZIP_OUT, UNZIP_IN, UNZIP_OUT) else self.k_nucleate

    def rate(self, kind: str, dg: float, kT: float) -> float:
        if dg == float("inf") or dg != dg:
            return 0.0
        if self.scheme == "kawasaki":
            x = -dg / (2.0 * kT)
        elif self.scheme == "metropolis":
            x = -max(0.0, dg) / kT
        else:  # pragma: no cover - validated by config
            raise ValueError(f"unknown rate scheme {self.scheme!r}")
        if x < -MAX_EXPONENT:
            return 0.0
        return self.prefactor(kind) * math.exp(min(x, MAX_EXPONENT))


def loop_positions(
    pt: Sequence[int], closing: tuple[int, int] | None, n: int
) -> Iterator[int]:
    """Every nucleotide that sits *directly* in one loop.

    That is the loop's unpaired bases plus both ends of each branch helix, and
    the closing pair itself.  Nucleotides buried in nested sub-loops are
    excluded - their energetic context is unchanged by anything happening in
    this loop, which is what makes loop-local invalidation sound.
    """
    if closing is None:
        lo, hi = 0, n
    else:
        yield closing[0]
        yield closing[1]
        lo, hi = closing[0] + 1, closing[1]
    k = lo
    while k < hi:
        partner = pt[k]
        if partner < 0 or not (lo <= partner < hi):
            yield k
            k += 1
        elif partner > k:
            yield k
            yield partner
            k = partner + 1
        else:
            yield k
            k += 1


def crosses_structure(pt: Sequence[int], i: int, j: int, n: int) -> bool:
    """True if the pair ``(i, j)`` would cross an existing pair."""
    k = i + 1
    while k < j:
        partner = pt[k]
        if partner < 0:
            k += 1
        elif partner < i or partner > j:
            return True
        elif partner > k:
            k = partner + 1
        else:
            k += 1
    return False


@dataclass(slots=True)
class Move:
    """One elementary transition out of the current state."""

    kind: str
    helix: Helix
    stem: int
    dg: float
    rate: float
    #: For an exchange, the stem that gives way; unused otherwise.
    other: int = -1


@dataclass(slots=True)
class FoldingState:
    """Folding state of a transcript prefix."""

    n_total: int
    length: int = 0
    available: int = 0
    pt: list[int] = field(default_factory=list)
    formed: dict[int, Helix] = field(default_factory=dict)
    energy: float = 0.0

    def __post_init__(self) -> None:
        if not self.pt:
            self.pt = [-1] * self.n_total

    def dotbracket(self) -> str:
        return to_dotbracket(self.pt[: self.length])

    def helices(self) -> list[Helix]:
        return list(self.formed.values())

    def copy(self) -> "FoldingState":
        return FoldingState(
            n_total=self.n_total,
            length=self.length,
            available=self.available,
            pt=list(self.pt),
            formed=dict(self.formed),
            energy=self.energy,
        )


def _any_crossing(helices) -> bool:
    return any(
        a.crosses(b)
        for index, a in enumerate(helices)
        for b in helices[index + 1 :]
    )


def _ladder_positions(stem) -> set[int]:
    return {p for k in range(stem.length) for p in (stem.i + k, stem.j - k)}


def _competitor_lists(stems) -> tuple[tuple[int, ...], ...]:
    """For each stem, the stems whose ladders share a nucleotide with it.

    Only these can ever be in competition, so only these can be the two ends of
    an exchange.  Computed once: it depends on the sequence, not the structure.
    """
    ladders = [_ladder_positions(stem) for stem in stems]
    out: list[tuple[int, ...]] = []
    for index, own in enumerate(ladders):
        out.append(
            tuple(
                other
                for other, theirs in enumerate(ladders)
                if other != index and not own.isdisjoint(theirs)
            )
        )
    return tuple(out)


class KineticEngine:
    """Propensity bookkeeping and move application for one trajectory."""

    def __init__(
        self,
        energy: FoldingEnergy,
        moveset: MoveSet,
        rates: RateModel,
        *,
        pk_model: PseudoknotModel | None = None,
        mode: str = HELIX_MODE,
        min_helix: int = 3,
    ) -> None:
        if mode not in MOVE_SETS:
            raise ValueError(f"mode must be one of {MOVE_SETS}")
        self.energy = energy
        self.moveset = moveset
        self.rates = rates
        self.pk_model = pk_model or energy.pk_model
        self.mode = mode
        self.min_helix = max(min_helix, moveset.nucleation_size)
        self.n = energy.n
        self.state = FoldingState(n_total=self.n)

        stems = moveset.stems
        if mode in (HELIX_MODE, LUMPED_MODE):
            self.n_slots = len(stems)
            self._slot_stem = list(range(len(stems)))
            self._slot_max_pos = [s.j for s in stems]
        else:
            self.n_slots = len(moveset.sites)
            self._slot_stem = [moveset.sites[k][0] for k in range(self.n_slots)]
            self._slot_max_pos = list(moveset.site_max_pos)

        # activation order: a candidate becomes usable once its 3'-most
        # nucleotide has been transcribed and left the polymerase footprint
        self._activation = sorted(
            range(self.n_slots), key=lambda slot: self._slot_max_pos[slot]
        )
        self._fen = Fenwick(self.n_slots)
        self._slot_helix: list[Helix | None] = [None] * self.n_slots
        self._slot_dg: list[float] = [float("inf")] * self.n_slots
        self._dirty: set[int] = set()
        # Candidates that cross the current structure are scored by full
        # evaluation, which depends on the whole structure rather than on one
        # loop.  Loop-local invalidation cannot see that, so they are
        # re-scored after every move.  There are few of them.
        self._crossing_slots: set[int] = set()
        self._active = 0
        self._is_active = bytearray(self.n_slots)

        # position -> slots that could be affected by a change there
        self._slots_by_pos: list[list[int]] = [[] for _ in range(self.n)]
        for slot in range(self.n_slots):
            stem = stems[self._slot_stem[slot]]
            if mode in (HELIX_MODE, LUMPED_MODE):
                positions = range(stem.length)
                touch = [
                    p
                    for k in positions
                    for p in (stem.i + k, stem.j - k)
                ]
            else:
                touch = list(moveset.helix(slot).positions())
            for pos in touch:
                self._slots_by_pos[pos].append(slot)

        # first slot belonging to each stem, for the melt-window check
        self._stem_slot: dict[int, int] = {}
        for slot in range(self.n_slots):
            self._stem_slot.setdefault(self._slot_stem[slot], slot)

        # Dynamic (melt / zip / unzip) moves are cached per stem.  Almost
        # every event is a zip on one helix, which leaves every other helix's
        # moves untouched; rebuilding them all was costing ~h energy
        # evaluations per event for no reason.
        self._dyn_cache: dict[int, list[Move]] = {}
        self._dyn_dirty_stems: set[int] = set()
        self._dyn: list[Move] = []
        self._dyn_total = 0.0
        self._dyn_dirty = True
        self._crossing: set[int] = set()
        # Pair table of the nested core only.  While the structure has no
        # crossings this *is* the full pair table, so the fast path costs
        # nothing; once a pseudoknot forms the two diverge and the core keeps
        # its loop decomposition (and therefore its O(loop) deltas).
        self._core_pt: list[int] = self.state.pt
        self._owner: list[int] = [-1] * self.n
        self._nucleus_cache: dict[tuple[int, int, int], Helix] = {}
        self._barrier_cache: dict = {}
        self._competitors: tuple[tuple[int, ...], ...] = ()
        if mode == LUMPED_MODE:
            # bound here rather than imported at module scope: rona.lumped
            # imports this module for the move kinds and the rate model
            from .lumped import barrier_rates, nucleation_window

            self._barrier_rates = barrier_rates
            self._nucleation_window = nucleation_window
            self._competitors = _competitor_lists(stems)
        self._pk_regions: tuple[tuple[int, int], ...] = ()
        self._pk_keys: set[tuple[int, int, int]] = set()
        self._pk_helices: tuple[Helix, ...] = ()

    # ------------------------------------------------------------------
    # activation and invalidation
    # ------------------------------------------------------------------
    def grow(self, new_length: int, footprint: int) -> None:
        """Extend the transcript, activating candidates that are now complete."""
        st = self.state
        st.length = min(new_length, self.n)
        st.available = max(0, st.length - footprint)
        order = self._activation
        while (
            self._active < self.n_slots
            and self._slot_max_pos[order[self._active]] < st.available
        ):
            self._dirty.add(order[self._active])
            self._is_active[order[self._active]] = 1
            self._active += 1
        # the exterior loop just gained a nucleotide: every helix end sitting in
        # it has a new dangling-end context
        self._touch(loop_positions(st.pt, None, st.available))
        if self.mode == LUMPED_MODE:
            self._recanonicalise()
        self._rebuild_core()
        self._dyn_dirty = True

    def _touch(self, positions) -> None:
        by_pos = self._slots_by_pos
        dirty = self._dirty
        for pos in positions:
            if pos < self.n:
                dirty.update(by_pos[pos])

    def _rebuild_core(self) -> None:
        """Refresh the nested-core pair table and the pseudoknot regions.

        Also invalidates every candidate inside a pseudoknot region, because
        the topology penalty depends on unpaired counts across the whole
        region rather than on any single loop.
        """
        st = self.state
        if not self._crossing:
            self._core_pt = st.pt
            self._pk_regions = ()
            self._pk_keys = set()
            self._pk_helices = ()
            return
        helices = st.helices()
        core, pk = self.energy.split(helices)
        core_pt = [-1] * self.n
        for helix in core:
            for a, b in helix.pairs:
                core_pt[a], core_pt[b] = b, a
        self._core_pt = core_pt
        regions = []
        for helix in pk:
            lo, hi = pseudoknot_region(
                helix, [c for c in core if helix.crosses(c)]
            )
            regions.append((lo, hi))
            self._touch(range(lo, min(hi + 1, self.n)))
        self._pk_regions = tuple(regions)
        self._pk_keys = {(h.i, h.j, h.length) for h in pk}
        self._pk_helices = tuple(pk)

    def _pk_unpaired_delta(self, helix: Helix, *, adding: bool) -> float:
        """Change in the pseudoknot topology penalty from one helix.

        A helix that crosses nothing cannot change the core/pseudoknot split,
        nor which core helices each pseudoknot threads through, nor therefore
        the extent of any pseudoknot region.  The only term that moves is the
        per-unpaired-nucleotide cost, and it moves by exactly the number of the
        helix's nucleotides that lie inside each region.  That is O(1) per
        region, where re-deriving it from two full evaluations costs a conflict
        graph and two O(n) energy sums.
        """
        regions = self._pk_regions
        if not regions:
            return 0.0
        per = self.pk_model.per_unpaired
        if per == 0.0:
            return 0.0
        count = 0
        for lo, hi in regions:
            for a, b in helix.pairs:
                if lo <= a <= hi:
                    count += 1
                if lo <= b <= hi:
                    count += 1
        return -per * count if adding else per * count

    def _crossed_by_pseudoknot(self, helix: Helix) -> bool:
        return any(helix.crosses(other) for other in self._pk_helices)

    def _in_pk_region(self, helix: Helix) -> bool:
        for lo, hi in self._pk_regions:
            if helix.i <= hi and lo <= helix.j:
                return True
        return False

    # ------------------------------------------------------------------
    # candidate construction
    # ------------------------------------------------------------------
    def _candidate(self, slot: int, *, ignore_own: bool = False) -> Helix | None:
        """The helix that slot ``slot`` would form, given current occupancy.

        ``ignore_own`` treats this stem's already-formed helix as absent, which
        is what the melt rule needs: a helix may only melt when it *is* the
        window that forming it again would produce.
        """
        st = self.state
        pt = st.pt
        avail = st.available
        stem_index = self._slot_stem[slot]
        own = st.formed.get(stem_index)
        if own is not None and not ignore_own:
            return None
        # the helix's own nucleotides, as two contiguous ranges: an O(1) test
        # per position, where building a set would allocate on every event
        if own is not None and ignore_own:
            own_lo1, own_hi1, own_lo2, own_hi2 = own.arms()
        else:
            own_lo1 = own_hi1 = own_lo2 = own_hi2 = -1
        stem = self.moveset.stems[stem_index]

        if self.mode == BREATHE_MODE:
            helix = self.moveset.helix(slot)
            if helix.j >= avail:
                return None
            for a, b in helix.pairs:
                if pt[a] >= 0 or pt[b] >= 0:
                    return None
            return helix

        # helix mode: longest unobstructed ladder inside the stem
        best_len = best_off = run = start = 0
        for k in range(stem.length):
            a, b = stem.i + k, stem.j - k
            free_a = pt[a] < 0 or own_lo1 <= a <= own_hi1 or own_lo2 <= a <= own_hi2
            free_b = pt[b] < 0 or own_lo1 <= b <= own_hi1 or own_lo2 <= b <= own_hi2
            if b < avail and free_a and free_b:
                if run == 0:
                    start = k
                run += 1
                if run > best_len:
                    best_len, best_off = run, start
            else:
                run = 0
        if best_len < self.min_helix:
            return None
        return Helix(stem.i + best_off, stem.j - best_off, best_len)

    # ------------------------------------------------------------------
    # lumped mode: windows are a function of the stem set
    # ------------------------------------------------------------------
    def _canonical(self, stems) -> dict[int, Helix]:
        from .lumped import canonical_windows

        return canonical_windows(
            self.moveset, stems, self.state.available, self.min_helix
        )

    def _rebuild_owner(self) -> None:
        """``position -> the stem occupying it``, or -1.

        The canonical placement lays stems down in increasing index, so which
        stem owns a nucleotide is exactly what decides whether a candidate is
        blocked there.  Keeping it as an array turns each candidate's placement
        into a scan of its own ladder, instead of a placement of the whole
        structure - which was half the cost of this mode.
        """
        owner = [-1] * self.n
        for index, helix in self.state.formed.items():
            for position in helix.positions():
                owner[position] = index
        self._owner = owner

    def _nucleus(self, helix: Helix) -> Helix:
        key = (helix.i, helix.j, helix.length)
        found = self._nucleus_cache.get(key)
        if found is None:
            found = self._nucleation_window(self.energy, helix, self.min_helix)
            self._nucleus_cache[key] = found
        return found

    def _lumped_melt(
        self, stem_index: int, helix: Helix, *, local: bool, helices
    ) -> tuple[float, float] | None:
        """``(dG, rate)`` for melting one stem, or ``None`` if it cannot.

        The barrier is the same transition state the form move crosses - both
        directions have to agree on it, or the ratio of the two rates is no
        longer ``exp(-dG/RT)``.
        """
        st = self.state
        kT = self.energy.kT
        if not self._lumped_melt_local(stem_index):
            # competitors grow into the nucleotides this frees, so neither the
            # energy change nor the barrier is loop-local
            rest = self._canonical(set(st.formed) - {stem_index})
            if len(rest) != len(st.formed) - 1:
                return None
            after = self.energy.energy(list(rest.values()), st.available)
            dg = after - st.energy
            top = self._slide(rest)
            _form, melt = self._barrier_rates(
                self.rates.prefactor(MELT), -dg, top - after, kT
            )
            return dg, melt
        if local:
            dg = self.energy.delta_remove(
                self._core_pt, helix, st.available, nested=True
            ) + self._pk_unpaired_delta(helix, adding=False)
        else:
            dg = self.energy.delta_full(
                helices, helix, st.available, add=False, before=st.energy
            )
        nucleus = self._nucleus(helix)
        if nucleus == helix:
            dg_nuc = -dg
        elif local:
            # the pair table of the set without this helix; adding the nucleus
            # to it is then an ordinary loop-local change.  ``local`` already
            # guarantees the pseudoknot split and every region are untouched,
            # so only the per-unpaired term moves
            table = list(self._core_pt)
            for a, b in helix.pairs:
                table[a] = table[b] = -1
            dg_nuc = self.energy.delta_add(
                table, nucleus, st.available, nested=True
            ) + self._pk_unpaired_delta(nucleus, adding=True)
        else:
            rest = [h for h in st.helices() if h != helix]
            dg_nuc = self.energy.energy(
                rest + [nucleus], st.available
            ) - self.energy.energy(rest, st.available)
        _form, melt = self._barrier_rates(
            self.rates.prefactor(MELT), -dg, dg_nuc, kT
        )
        return dg, melt

    def _lumped_form_window(
        self, stem_index: int
    ) -> tuple[Helix, dict[int, Helix] | None] | None:
        """``(window, crowded windows)`` for forming ``stem_index``.

        The second element is ``None`` in the common case where nothing else
        moves, which is what lets the energy change stay loop-local.  When the
        new helix does compete for nucleotides, it is the canonical placement of
        the whole target set, and the caller scores that state in full.

        Placement is read off the owner array: stems of lower index are already
        down and block, stems of higher index are not yet placed and do not.
        That is exactly what :func:`rona.lumped.canonical_windows` computes for
        this stem, at the cost of one scan of its own ladder.
        """
        st = self.state
        if stem_index in st.formed:
            return None
        owner = self._owner
        stem = self.moveset.stems[stem_index]
        avail = st.available
        best_len = best_off = run = start = 0
        for k in range(stem.length):
            a, b = stem.i + k, stem.j - k
            own_a, own_b = owner[a], owner[b]
            if (
                b < avail
                and (own_a < 0 or own_a > stem_index)
                and (own_b < 0 or own_b > stem_index)
            ):
                if run == 0:
                    start = k
                run += 1
                if run > best_len:
                    best_len, best_off = run, start
            else:
                run = 0
        if best_len < self.min_helix:
            return None
        i, j = stem.i + best_off, stem.j - best_off
        helix = Helix(i, j, best_len)
        crowded = False
        for k in range(best_len):
            if owner[i + k] >= 0 or owner[j - k] >= 0:
                crowded = True
                break
        if not crowded:
            return helix, None
        windows = self._canonical(set(st.formed) | {stem_index})
        if len(windows) != len(st.formed) + 1 or windows.get(stem_index) != helix:
            # a competitor would be squeezed out of existence, so the target
            # set is not one the chain contains
            return None
        return helix, windows

    def _lumped_melt_local(self, stem_index: int) -> bool:
        """Whether melting ``stem_index`` leaves every other window in place.

        Freeing a helix's nucleotides can only let stems of *higher* index grow
        - lower ones are placed before it and never saw it - so those are the
        only windows that have to be re-derived.  When none of them moves the
        melt is a loop-local change; otherwise the caller scores both states in
        full.
        """
        st = self.state
        owner = self._owner
        for other, helix in st.formed.items():
            if other <= stem_index:
                continue
            stem = self.moveset.stems[other]
            avail = st.available
            best_len = best_off = run = start = 0
            for k in range(stem.length):
                a, b = stem.i + k, stem.j - k
                own_a, own_b = owner[a], owner[b]
                if (
                    b < avail
                    and (own_a < 0 or own_a >= other or own_a == stem_index)
                    and (own_b < 0 or own_b >= other or own_b == stem_index)
                ):
                    if run == 0:
                        start = k
                    run += 1
                    if run > best_len:
                        best_len, best_off = run, start
                else:
                    run = 0
            if best_len != helix.length or stem.i + best_off != helix.i:
                return False
        return True

    def _blocked_by(self, windows, holder: int, blocked: int) -> bool:
        """Whether ``blocked`` cannot be placed, and ``holder`` alone is why.

        A predicate on the *pair of states*, computed from each side's canonical
        windows, so both directions of an exchange agree on whether the move
        exists.  Without that the exchange would be a one-way door.
        """
        stem = self.moveset.stems[blocked]
        avail = self.state.available
        taken: set[int] = set()
        held: set[int] = set()
        for key, helix in windows.items():
            (held if key == holder else taken).update(helix.positions())
        if not (held & _ladder_positions(stem)):
            return False
        return self._run_length(stem, avail, taken | held) < self.min_helix <= (
            self._run_length(stem, avail, taken)
        )

    def _run_length(self, stem, avail: int, taken) -> int:
        best = run = 0
        for k in range(stem.length):
            a, b = stem.i + k, stem.j - k
            if b < avail and a not in taken and b not in taken:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        return best

    def _lumped_exchanges(self, holder: int) -> list[Move]:
        """Moves that replace helix ``holder`` with a competitor of its own.

        Two helices that want the same nucleotides cannot both be in a state,
        so in the lumped chain one has to melt before the other can form - and
        a full melt is a barrier so high that a trap never resolves, which is
        not what the microscopic simulator does.  Microscopically the incumbent
        retracts pair by pair while the challenger nucleates in the nucleotides
        that frees, and both are only ever a few pairs from their nucleus at
        the top.  This move is that path: one transition between the two states,
        over a transition state where both helices are present as nuclei.
        """
        st = self.state
        kT = self.energy.kT
        common = set(st.formed) - {holder}
        out: list[Move] = []
        for challenger in self._competitors[holder]:
            if challenger in st.formed:
                continue
            target = common | {challenger}
            windows_b = self._canonical(target)
            if len(windows_b) != len(target) or challenger not in windows_b:
                continue
            if not self._blocked_by(st.formed, holder, challenger):
                continue
            if not self._blocked_by(windows_b, challenger, holder):
                continue
            after = list(windows_b.values())
            if not self.pk_model.enabled and _any_crossing(after):
                continue
            g_b = self.energy.energy(after, st.available)
            g_top = self._slide(windows_b)
            dg = g_b - st.energy
            rate, _reverse = self._barrier_rates(
                self.rates.prefactor(EXCHANGE), dg, g_top - st.energy, kT
            )
            if rate > 0.0:
                out.append(
                    Move(
                        EXCHANGE,
                        windows_b[challenger],
                        challenger,
                        dg,
                        rate,
                        holder,
                    )
                )
        return out

    def _slide(self, windows) -> float:
        """Top of a greedy microscopic path from the current windows to these."""
        from .lumped import slide_barrier

        return slide_barrier(
            self.energy,
            self.moveset,
            self.min_helix,
            self.state.formed,
            windows,
            self.state.available,
            self._barrier_cache,
        )

    def _lumped_candidate(
        self, slot: int
    ) -> tuple[Helix, float, float, bool] | None:
        """``(window, dG, rate, globally dependent)`` for one form candidate."""
        stem_index = self._slot_stem[slot]
        found = self._lumped_form_window(stem_index)
        if found is None:
            return None
        helix, windows = found
        st = self.state
        kT = self.energy.kT
        if windows is None:
            dg, crossed = self._form_dg(helix)
            nucleus = self._nucleus(helix)
            dg_nuc = dg if nucleus == helix else self._form_dg(nucleus)[0]
        else:
            # a competitor retracts, so neither end of the move is a loop-local
            # change and both states are evaluated in full.  Rare enough not to
            # matter, and it is the case that carries the interesting kinetics:
            # one helix giving way to another.
            crossed = True
            if not self.pk_model.enabled and any(
                a.crosses(b)
                for pos, a in enumerate(windows.values())
                for b in list(windows.values())[pos + 1 :]
            ):
                return None
            after = self.energy.energy(list(windows.values()), st.available)
            dg = after - st.energy
            dg_nuc = self._slide(windows) - st.energy
        rate, _melt = self._barrier_rates(
            self.rates.prefactor(FORM), dg, dg_nuc, kT
        )
        return helix, dg, rate, crossed

    def _form_dg(self, helix: Helix) -> tuple[float, bool]:
        """``(dG, crossed)`` for forming ``helix``.

        ``crossed`` tells the caller the value came from a full evaluation and
        must not be cached across a structural change.
        """
        st = self.state
        crossing = crosses_structure(st.pt, helix.i, helix.j, st.available)
        if crossing:
            if not self.pk_model.enabled or helix.length < self.pk_model.min_helix:
                return float("inf"), True
            if self.pk_model.max_helices is not None:
                _core, pk = self.energy.split(st.helices())
                if len(pk) >= self.pk_model.max_helices:
                    return float("inf"), True
            return (
                self.energy.delta_full(
                    st.helices(), helix, st.available, add=True, before=st.energy
                ),
                True,
            )
        # the helix crosses nothing, so the split and every region are fixed;
        # only the per-unpaired term of the topology penalty moves
        return (
            self.energy.delta_add(self._core_pt, helix, st.available, nested=True)
            + self._pk_unpaired_delta(helix, adding=True),
            False,
        )

    def _refresh_slots(self) -> None:
        if not self._dirty:
            return
        kT = self.energy.kT
        fen = self._fen
        active = self._is_active
        formed = self.state.formed
        slot_stem = self._slot_stem
        for slot in self._dirty:
            # a dirty candidate whose stem is formed means that helix's own
            # loop context moved, so its melt/zip rates are stale too
            stem_index = slot_stem[slot]
            if stem_index in formed:
                self._dyn_dirty_stems.add(stem_index)
            if not active[slot]:
                continue
            if self.mode == LUMPED_MODE:
                scored = self._lumped_candidate(slot)
                if scored is None:
                    self._slot_helix[slot] = None
                    self._slot_dg[slot] = float("inf")
                    self._crossing_slots.discard(slot)
                    fen.set(slot, 0.0)
                    continue
                helix, dg, rate, crossed = scored
                if crossed:
                    self._crossing_slots.add(slot)
                else:
                    self._crossing_slots.discard(slot)
                self._slot_helix[slot] = helix
                self._slot_dg[slot] = dg
                fen.set(slot, rate)
                continue
            helix = self._candidate(slot)
            if helix is None:
                self._slot_helix[slot] = None
                self._slot_dg[slot] = float("inf")
                self._crossing_slots.discard(slot)
                fen.set(slot, 0.0)
                continue
            dg, crossed = self._form_dg(helix)
            if crossed:
                self._crossing_slots.add(slot)
            else:
                self._crossing_slots.discard(slot)
            self._slot_helix[slot] = helix
            self._slot_dg[slot] = dg
            fen.set(slot, self.rates.rate(FORM, dg, kT))
        self._dirty.clear()

    # ------------------------------------------------------------------
    def _refresh_dynamic(self) -> None:
        """Melt and zip/unzip moves for formed helices, rebuilt per stem."""
        st = self.state
        formed = st.formed
        # drop helices that no longer exist
        if len(self._dyn_cache) != len(formed):
            for stem_index in list(self._dyn_cache):
                if stem_index not in formed:
                    del self._dyn_cache[stem_index]
        stale = [s for s in formed if s not in self._dyn_cache]
        self._dyn_dirty_stems.update(stale)
        if not self._dyn_dirty_stems and not self._dyn_dirty:
            return

        kT = self.energy.kT
        helices = st.helices() if self._crossing else ()
        targets = (
            list(formed.items())
            if self._dyn_dirty
            else [(s, formed[s]) for s in self._dyn_dirty_stems if s in formed]
        )
        for stem_index, helix in targets:
            out: list[Move] = []
            stem = self.moveset.stems[stem_index]
            local = not self._crossing or not (
                (helix.i, helix.j, helix.length) in self._pk_keys
                or self._crossed_by_pseudoknot(helix)
            )
            if self.mode == LUMPED_MODE:
                # the window is a function of the stem set, so there is nothing
                # to zip: melting is the only dynamic move
                found = self._lumped_melt(
                    stem_index, helix, local=local, helices=helices
                )
                if found is not None:
                    dg, rate = found
                    out.append(Move(MELT, helix, stem_index, dg, rate))
                out.extend(self._lumped_exchanges(stem_index))
                self._dyn_cache[stem_index] = out
                continue
            if self.mode == HELIX_MODE:
                # A helix may only melt when it is the window that forming it
                # again would produce, so melt and form are exact inverses.
                # Otherwise the state would be a one-way door and detailed
                # balance would fail.
                window = self._candidate(
                    self._stem_slot[stem_index], ignore_own=True
                )
                can_melt = window is not None and window == helix
            else:
                can_melt = helix.length == self.moveset.nucleation_size
            if can_melt:
                if local:
                    dg = self.energy.delta_remove(
                        self._core_pt, helix, st.available, nested=True
                    ) + self._pk_unpaired_delta(helix, adding=False)
                else:
                    dg = self.energy.delta_full(
                        helices, helix, st.available, add=False, before=st.energy
                    )
                out.append(
                    Move(MELT, helix, stem_index, dg, self.rates.rate(MELT, dg, kT))
                )
            offset = helix.i - stem.i
            if offset > 0:
                pair = Helix(helix.i - 1, helix.j + 1, 1)
                if (
                    pair.j < st.available
                    and st.pt[pair.i] < 0
                    and st.pt[pair.j] < 0
                ):
                    dg = self._pair_delta(
                        pair, add=True, local=local, kind=ZIP_OUT, stem=stem_index
                    )
                    out.append(
                        Move(
                            ZIP_OUT,
                            pair,
                            stem_index,
                            dg,
                            self.rates.rate(ZIP_OUT, dg, kT),
                        )
                    )
            if offset + helix.length < stem.length:
                a, b = helix.inner
                pair = Helix(a + 1, b - 1, 1)
                if st.pt[pair.i] < 0 and st.pt[pair.j] < 0:
                    dg = self._pair_delta(
                        pair, add=True, local=local, kind=ZIP_IN, stem=stem_index
                    )
                    out.append(
                        Move(
                            ZIP_IN, pair, stem_index, dg, self.rates.rate(ZIP_IN, dg, kT)
                        )
                    )
            if helix.length > self.moveset.nucleation_size:
                for kind, pair in (
                    (UNZIP_OUT, Helix(helix.i, helix.j, 1)),
                    (UNZIP_IN, Helix(*helix.inner, 1)),
                ):
                    dg = self._pair_delta(
                        pair, add=False, local=local, kind=kind, stem=stem_index
                    )
                    out.append(
                        Move(kind, pair, stem_index, dg, self.rates.rate(kind, dg, kT))
                    )
            self._dyn_cache[stem_index] = out

        self._dyn_dirty_stems.clear()
        self._dyn_dirty = False
        self._dyn = [m for moves in self._dyn_cache.values() for m in moves]
        self._dyn_total = math.fsum(m.rate for m in self._dyn)

    # ------------------------------------------------------------------
    @staticmethod
    def _resulting_helix(kind: str, parent: Helix, pair: Helix) -> Helix | None:
        """The helix that ``kind`` leaves behind, or ``None`` if it is gone."""
        if kind == FORM:
            return pair
        if kind == MELT:
            return None
        if kind == ZIP_OUT:
            return Helix(pair.i, pair.j, parent.length + 1)
        if kind == ZIP_IN:
            return Helix(parent.i, parent.j, parent.length + 1)
        if kind == UNZIP_OUT:
            return Helix(parent.i + 1, parent.j - 1, parent.length - 1)
        if kind == UNZIP_IN:
            return Helix(parent.i, parent.j, parent.length - 1)
        raise ValueError(f"unknown move kind {kind!r}")

    def _full_pair_delta(self, kind: str, stem_index: int, pair: Helix) -> float:
        """Exact delta for a zip/unzip when the fast path does not apply.

        ``delta_full`` works on a *set of helices*, and a zipped pair is not one
        of them - it is part of a larger helix.  Passing the pair to it removed
        nothing and silently returned zero.  The resulting helix set is built
        explicitly here instead.
        """
        st = self.state
        parent = st.formed[stem_index]
        replacement = self._resulting_helix(kind, parent, pair)
        helices = [
            (replacement if key == stem_index else helix)
            for key, helix in st.formed.items()
            if key != stem_index or replacement is not None
        ]
        return self.energy.energy(helices, st.available) - st.energy

    def _pair_delta(
        self, pair: Helix, *, add: bool, local: bool, kind: str = "", stem: int = -1
    ) -> float:
        """Delta for adding or removing one base pair at a helix end.

        ``local`` describes the parent helix, but zipping adds a *new* pair
        that can cross a pseudoknot the parent does not, which would change the
        core/pseudoknot split and invalidate the O(1) correction.  Adding a
        pair is therefore re-checked on its own.  Removing one cannot create a
        crossing, so the parent's status is enough there.
        """
        st = self.state
        if local and add and crosses_structure(
            st.pt, pair.i, pair.j, st.available
        ):
            local = False
        if local:
            table = self._core_pt
            correction = self._pk_unpaired_delta(pair, adding=add)
            if add:
                return (
                    self.energy.delta_add(table, pair, st.available, nested=True)
                    + correction
                )
            return (
                self.energy.delta_remove(table, pair, st.available, nested=True)
                + correction
            )
        return self._full_pair_delta(kind, stem, pair)

    def propensity(self) -> float:
        """Total escape rate from the current state."""
        if not self._crossing and self._core_pt is not self.state.pt:
            # the caller replaced the pair table rather than mutating it
            self._core_pt = self.state.pt
        if self.mode == LUMPED_MODE:
            self._rebuild_owner()
        self._refresh_slots()
        self._refresh_dynamic()
        return self._fen.total() + self._dyn_total

    def select(self, u: float) -> Move | None:
        """Pick a move; ``u`` is a uniform draw on ``[0, 1)``."""
        total = self._fen.total() + self._dyn_total
        if total <= 0.0:
            return None
        target = u * total
        form_total = self._fen.total()
        if target < form_total:
            slot = self._fen.find(target)
            helix = self._slot_helix[slot]
            if helix is not None:
                return Move(
                    FORM,
                    helix,
                    self._slot_stem[slot],
                    self._slot_dg[slot],
                    self._fen.value(slot),
                )
            target = form_total  # numerical edge case: fall through to dynamics
        acc = form_total
        for move in self._dyn:
            acc += move.rate
            if acc >= target:
                return move
        return self._dyn[-1] if self._dyn else None

    def apply(self, move: Move) -> None:
        """Commit a move, updating the pair table, energy and dirty sets."""
        st = self.state
        helix = move.helix
        if self.mode == LUMPED_MODE and self._is_concerted(move):
            self._apply_concerted(move)
            return
        closing = enclosing_pair(st.pt, helix.i, st.available)

        # everything sharing the affected loop loses its cached rate
        self._touch(loop_positions(st.pt, closing, st.available))
        if move.kind in (MELT, UNZIP_OUT, UNZIP_IN):
            self._touch(loop_positions(st.pt, helix.inner, st.available))
        self._touch(helix.positions())

        if move.kind in (FORM, ZIP_OUT, ZIP_IN):
            for a, b in helix.pairs:
                st.pt[a], st.pt[b] = b, a
            current = st.formed.get(move.stem)
            if move.kind == FORM:
                st.formed[move.stem] = helix
            elif move.kind == ZIP_OUT:
                st.formed[move.stem] = Helix(helix.i, helix.j, current.length + 1)
            else:
                st.formed[move.stem] = Helix(
                    current.i, current.j, current.length + 1
                )
            self._touch(loop_positions(st.pt, helix.inner, st.available))
        else:
            for a, b in helix.pairs:
                st.pt[a], st.pt[b] = -1, -1
            current = st.formed[move.stem]
            if move.kind == MELT:
                del st.formed[move.stem]
            elif move.kind == UNZIP_OUT:
                st.formed[move.stem] = Helix(
                    current.i + 1, current.j - 1, current.length - 1
                )
            else:
                st.formed[move.stem] = Helix(
                    current.i, current.j, current.length - 1
                )

        self._touch(loop_positions(st.pt, closing, st.available))
        st.energy += move.dg
        # globally-dependent candidate scores cannot survive a structural change
        self._dirty.update(self._crossing_slots)
        self._dyn_dirty_stems.add(move.stem)
        if self.mode == LUMPED_MODE:
            self._recanonicalise()
        had_crossing = bool(self._crossing)
        self._update_crossings()
        if bool(self._crossing) != had_crossing or self._crossing:
            # pseudoknot penalties are not loop-local, so every helix is stale
            self._dyn_dirty = True
        self._rebuild_core()

    def _is_concerted(self, move: Move) -> bool:
        """Whether this move also moves helices other than its own.

        Such a move cannot be committed by the incremental bookkeeping: the new
        window may lie on nucleotides another helix still holds, so the pair
        table is rebuilt from the stem set instead.
        """
        if move.kind == EXCHANGE:
            return True
        if move.kind == FORM:
            owner = self._owner
            return any(owner[p] >= 0 for p in move.helix.positions())
        return not self._lumped_melt_local(move.stem)

    def _apply_concerted(self, move: Move) -> None:
        """Commit a move that rearranges more than one helix.

        Membership is the state and the windows follow from it, so the whole
        structure is re-derived rather than patched.  Every cached rate goes
        with it; these moves are rare, and they are the ones that carry a helix
        giving way to a competitor.
        """
        st = self.state
        if move.kind == EXCHANGE:
            del st.formed[move.other]
            st.formed[move.stem] = move.helix
        elif move.kind == FORM:
            st.formed[move.stem] = move.helix
        else:
            del st.formed[move.stem]
        st.energy += move.dg
        self._recanonicalise(force=True)
        self._update_crossings()
        self._dyn_dirty = True
        self._rebuild_core()

    def _recanonicalise(self, *, force: bool = False) -> None:
        """Re-derive every window from the stem set, as lumped mode requires.

        Without zipping, a helix would otherwise stay at whatever length it had
        when it formed - the stranding defect.  Here the window is a pure
        function of the set of formed stems, so it is simply recomputed.  In
        practice nothing but the stem just touched changes, because placement
        only interacts between stems that share nucleotides; when something else
        does move, every cached rate is dropped.
        """
        from .lumped import canonical_windows

        st = self.state
        fresh = canonical_windows(
            self.moveset, st.formed.keys(), st.available, self.min_helix
        )
        if fresh == st.formed and not force:
            return
        # rebuilt in place: _core_pt aliases this list while the structure is
        # nested, and replacing it would orphan the alias
        for position in range(len(st.pt)):
            st.pt[position] = -1
        st.formed = fresh
        for helix in fresh.values():
            for a, b in helix.pairs:
                st.pt[a], st.pt[b] = b, a
        st.energy = self.energy.energy(st.helices(), st.available)
        self._dirty = set(range(self.n_slots))
        self._dyn_dirty = True
        self._dyn_cache.clear()

    def _update_crossings(self) -> None:
        items = list(self.state.formed.items())
        crossing: set[int] = set()
        for idx, (key_a, hel_a) in enumerate(items):
            for key_b, hel_b in items[idx + 1 :]:
                if hel_a.crosses(hel_b):
                    crossing.add(key_a)
                    crossing.add(key_b)
        self._crossing = crossing

    def has_pseudoknot(self) -> bool:
        return bool(self._crossing)

    def resync_energy(self) -> None:
        """Recompute the total energy from scratch (guards against drift)."""
        self.state.energy = self.energy.energy(
            self.state.helices(), self.state.available
        )
