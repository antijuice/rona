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
MOVE_SETS = (HELIX_MODE, BREATHE_MODE)


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
    k_zip: float = 1.0e7
    #: ``metropolis`` or ``kawasaki``.
    scheme: str = "metropolis"

    def prefactor(self, kind: str) -> float:
        return self.k_nucleate if kind in (FORM, MELT) else self.k_zip

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
        if mode == HELIX_MODE:
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
        self._active = 0
        self._is_active = bytearray(self.n_slots)

        # position -> slots that could be affected by a change there
        self._slots_by_pos: list[list[int]] = [[] for _ in range(self.n)]
        for slot in range(self.n_slots):
            stem = stems[self._slot_stem[slot]]
            if mode == HELIX_MODE:
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

        self._dyn: list[Move] = []
        self._dyn_total = 0.0
        self._dyn_dirty = True
        self._crossing: set[int] = set()
        # Pair table of the nested core only.  While the structure has no
        # crossings this *is* the full pair table, so the fast path costs
        # nothing; once a pseudoknot forms the two diverge and the core keeps
        # its loop decomposition (and therefore its O(loop) deltas).
        self._core_pt: list[int] = self.state.pt
        self._pk_regions: tuple[tuple[int, int], ...] = ()
        self._pk_keys: set[tuple[int, int, int]] = set()

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

    def _in_pk_region(self, helix: Helix) -> bool:
        for lo, hi in self._pk_regions:
            if helix.i <= hi and lo <= helix.j:
                return True
        return False

    # ------------------------------------------------------------------
    # candidate construction
    # ------------------------------------------------------------------
    def _candidate(self, slot: int) -> Helix | None:
        """The helix that slot ``slot`` would form, given current occupancy."""
        st = self.state
        pt = st.pt
        avail = st.available
        stem_index = self._slot_stem[slot]
        if stem_index in st.formed:
            return None
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
            if b < avail and pt[a] < 0 and pt[b] < 0:
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

    def _form_dg(self, helix: Helix) -> float:
        st = self.state
        crossing = crosses_structure(st.pt, helix.i, helix.j, st.available)
        if crossing:
            if not self.pk_model.enabled or helix.length < self.pk_model.min_helix:
                return float("inf")
            if self.pk_model.max_helices is not None:
                _core, pk = self.energy.split(st.helices())
                if len(pk) >= self.pk_model.max_helices:
                    return float("inf")
            return self.energy.delta_full(
                st.helices(), helix, st.available, add=True, before=st.energy
            )
        if self._pk_regions and self._in_pk_region(helix):
            return self.energy.delta_full(
                st.helices(), helix, st.available, add=True, before=st.energy
            )
        # no crossing and clear of every pseudoknot region: the topology
        # penalties are unchanged, so the core delta is the whole answer
        return self.energy.delta_add(
            self._core_pt, helix, st.available, nested=True
        )

    def _refresh_slots(self) -> None:
        if not self._dirty:
            return
        kT = self.energy.kT
        fen = self._fen
        active = self._is_active
        for slot in self._dirty:
            if not active[slot]:
                continue
            helix = self._candidate(slot)
            if helix is None:
                self._slot_helix[slot] = None
                self._slot_dg[slot] = float("inf")
                fen.set(slot, 0.0)
                continue
            dg = self._form_dg(helix)
            self._slot_helix[slot] = helix
            self._slot_dg[slot] = dg
            fen.set(slot, self.rates.rate(FORM, dg, kT))
        self._dirty.clear()

    # ------------------------------------------------------------------
    def _refresh_dynamic(self) -> None:
        """Melt (and, in breathe mode, zip/unzip) moves for formed helices."""
        if not self._dyn_dirty:
            return
        st = self.state
        kT = self.energy.kT
        out: list[Move] = []
        helices = st.helices() if self._crossing else ()
        for stem_index, helix in st.formed.items():
            stem = self.moveset.stems[stem_index]
            local = not self._crossing or not (
                (helix.i, helix.j, helix.length) in self._pk_keys
                or self._in_pk_region(helix)
            )
            can_melt = (
                self.mode == HELIX_MODE
                or helix.length == self.moveset.nucleation_size
            )
            if can_melt:
                if local:
                    dg = self.energy.delta_remove(
                        self._core_pt, helix, st.available, nested=True
                    )
                else:
                    dg = self.energy.delta_full(
                        helices, helix, st.available, add=False, before=st.energy
                    )
                out.append(
                    Move(MELT, helix, stem_index, dg, self.rates.rate(MELT, dg, kT))
                )
            if self.mode != BREATHE_MODE:
                continue

            offset = helix.i - stem.i
            if offset > 0:
                pair = Helix(helix.i - 1, helix.j + 1, 1)
                if (
                    pair.j < st.available
                    and st.pt[pair.i] < 0
                    and st.pt[pair.j] < 0
                ):
                    dg = self._pair_delta(pair, add=True, local=local)
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
                    dg = self._pair_delta(pair, add=True, local=local)
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
                    dg = self._pair_delta(pair, add=False, local=local)
                    out.append(
                        Move(kind, pair, stem_index, dg, self.rates.rate(kind, dg, kT))
                    )
        self._dyn = out
        self._dyn_total = math.fsum(m.rate for m in out)
        self._dyn_dirty = False

    # ------------------------------------------------------------------
    def _pair_delta(self, pair: Helix, *, add: bool, local: bool) -> float:
        """Delta for a single-base-pair zip/unzip in breathe mode."""
        st = self.state
        if local:
            table = self._core_pt
            if add:
                return self.energy.delta_add(table, pair, st.available, nested=True)
            return self.energy.delta_remove(table, pair, st.available, nested=True)
        helices = st.helices()
        return self.energy.delta_full(
            helices, pair, st.available, add=add, before=st.energy
        )

    def propensity(self) -> float:
        """Total escape rate from the current state."""
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
        self._update_crossings()
        self._rebuild_core()
        self._dyn_dirty = True

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
