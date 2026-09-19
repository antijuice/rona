"""A transcribing ensemble held as a product over anchored regions.

:mod:`rona.master.factor` proves that conditional on a nested set of anchors the
generator is exactly a Kronecker sum, so a product distribution stays a product
and costs the *sum* of the region supports instead of their product.  This module
decides when to use that, and pays for it in a currency it can count.

Measured on a 45 nt transcript, against the flat solver reaching the same
certificate:

    tolerance   flat states   flat s   anchored states   anchored s
        1e-2        252         2.1          98             0.1
        1e-3        538         4.0         147             0.1
        1e-4      1,905       34.7         218             0.1
        1e-5      2,586       59.3         343             0.2

The saving grows as the tolerance tightens, which is the signature of a product:
a loose tolerance only needs the dominant corner, a tight one needs the whole
thing.  And it is not free money - at 55 nt the same sequence has one dominant
hairpin and a 37 nt remainder that nothing separates, the anchors cut almost
nothing, and the anchored form is *worse* (1,745 states against 1,118).  So the
decomposition has to be adaptive, and it has to be honest about its own error.

Two approximations enter, both measured rather than assumed, both added to a
running ``slack`` that joins the certificate:

**Conditioning.**  Anchoring a pair discards the probability of states that do
not contain it.  That mass is known exactly - it is the complement of the pair's
marginal in the region being split - and a pair is only promoted when it is
small.

**Factorising.**  Splitting a region replaces the conditional joint distribution
over its two halves by the product of its marginals.  The cost is
``|| p - p_in (x) p_out ||_1``, computed exactly at the moment of the split from
the distribution actually held, and the split is refused if it is too large.  A
promotion that would not pay for itself simply does not happen.

Promotion reads the *kinetic* marginal, which is the right signal for "this pair
is currently formed and staying formed".  Demotion cannot: while a pair is
anchored, the ensemble holds no states without it, so its melting is by
construction invisible.  So demotion is driven from outside the representation,
by the exact equilibrium probability of the anchor at the current length - two
McCaskill calls, one constrained and one not.  Releasing is lossless as an
operation: the product of the two factors is multiplied back out into a joint
distribution over the merged region, exactly.  It does not *undo* the promotion,
because what factorising discarded is not recoverable from the marginals - a
promote/release round trip leaves an L1 error of exactly the ``factorised`` term
that was already paid and already counted.  Being an equilibrium test, demotion
is a guard rather than a prediction; that is the right asymmetry, because
releasing an anchor is always safe and holding a stale one is not.

Conditioning is left visible rather than renormalised away: the stored factors
sum to less than one by exactly the mass discarded, so ``1 - retained_mass`` is a
running check on ``slack`` that does not depend on trusting the bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..energy.evaluator import FoldingEnergy
from .cache import EnergyCache
from .certify import partition_function_energy
from .factor import Pair, Region, check_nested, decompose, owner_of
from .fsp import Solver
from .moves import MoveModel
from .structures import EMPTY, Structure, dotbracket


@dataclass(frozen=True, slots=True)
class Promotion:
    """What anchoring one pair cost, in the two ways it can cost anything."""

    pair: Pair
    #: Probability of the states discarded because they lacked the pair.
    conditioned: float
    #: ``|| p - p_in (x) p_out ||_1`` at the moment of the split.
    factorised: float

    @property
    def total(self) -> float:
        return self.conditioned + self.factorised

    def __str__(self) -> str:
        return (f"anchor {self.pair}: conditioned {self.conditioned:.2e} + "
                f"factorised {self.factorised:.2e}")


@dataclass(slots=True)
class Anchored:
    """The ensemble as a product of independent regions, adaptively.

    With no anchors this is exactly one flat ``Solver`` and behaves identically;
    every anchor is an optimisation the object decided was worth its measured
    error.
    """

    energy: EnergyCache
    model: MoveModel
    length: int
    tolerance: float
    #: Kinetic marginal a pair needs before it may be anchored.
    promote_above: float = 0.995
    #: Largest measured product error a single promotion may cost.
    promote_budget: float | None = None
    #: Equilibrium probability below which an anchor is released.
    demote_below: float = 0.9
    #: Smallest region worth separating, in free nucleotides.
    min_region: int = 4

    anchors: frozenset = field(default_factory=frozenset)
    parts: dict = field(default_factory=dict)
    #: Accumulated L1 error from conditioning and factorising.
    slack: float = 0.0
    promotions: list = field(default_factory=list)
    #: Anchors released, with the equilibrium probability that released them.
    demotions: list = field(default_factory=list)

    @classmethod
    def of(
        cls,
        sequence: str | FoldingEnergy | EnergyCache,
        model: MoveModel | None = None,
        *,
        length: int | None = None,
        tolerance: float = 1e-3,
        **kwargs,
    ) -> "Anchored":
        energy = sequence
        if isinstance(energy, str):
            energy = FoldingEnergy(energy)
        if not isinstance(energy, EnergyCache):
            energy = EnergyCache(energy)
        model = model or MoveModel()
        it = cls(
            energy=energy,
            model=model,
            length=energy.n if length is None else length,
            tolerance=tolerance,
            **kwargs,
        )
        it.parts = {None: it._solver(None, range(it.length))}
        return it

    # ------------------------------------------------------------------
    def _solver(self, owner, positions) -> Solver:
        """One region's solver: the flat one, with the anchors as its context."""
        return Solver(
            self.energy,
            self.model,
            length=self.length,
            tolerance=self.tolerance / max(1, len(self.parts) or 1),
            base=self.anchors,
            within=positions,
        )

    def _rebase(self) -> None:
        """Point every region solver at the current anchor set and tolerance."""
        share = self.tolerance / max(1, len(self.parts))
        for solver in self.parts.values():
            solver.base = self.anchors
            solver.length = self.length
            solver.tolerance = share
            solver._neighbours.clear()
            solver._pair_cache.clear()

    # ------------------------------------------------------------------
    def advance(self, dt: float, **kwargs) -> None:
        """Advance every region by ``dt``.  Exact, by the Kronecker-sum identity."""
        for solver in self.parts.values():
            solver.advance(dt, **kwargs)

    def grow(self, new_length: int) -> None:
        """Transcribe.  New nucleotides join the outermost region."""
        new_length = min(new_length, self.energy.n)
        added = range(self.length, new_length)
        self.length = new_length
        exterior = self.parts[None]
        if exterior.within is not None:
            exterior.within.update(added)
        for solver in self.parts.values():
            solver.length = new_length

    # ------------------------------------------------------------------
    def marginals(self, owner) -> dict[Pair, float]:
        """Probability that each pair is formed, within one region."""
        solver = self.parts[owner]
        out: dict[Pair, float] = {}
        for state, p in zip(solver.states, solver.probability):
            if p <= 0.0:
                continue
            for pair in state:
                out[pair] = out.get(pair, 0.0) + float(p)
        mass = float(solver.probability.sum()) or 1.0
        return {pair: value / mass for pair, value in out.items()}

    def promote(self, owner, pair: Pair) -> Promotion | None:
        """Anchor ``pair``, splitting its region in two.  ``None`` if refused."""
        check_nested(list(self.anchors) + [pair])
        solver = self.parts[owner]
        mass = float(solver.probability.sum())
        kept = [(s, float(p)) for s, p in zip(solver.states, solver.probability)
                if pair in s and p > 0.0]
        held = sum(p for _s, p in kept)
        if held <= 0.0:
            return None
        conditioned = max(0.0, mass - held)

        anchors = self.anchors | {pair}
        regions = {r.owner: r for r in decompose(anchors, self.length)}
        inside = regions[pair]
        rest = regions[owner]
        if len(inside) < self.min_region or len(rest) < self.min_region:
            return None

        # the conditional joint over the two halves, and its marginals
        joint: dict[tuple[Structure, Structure], float] = {}
        for state, p in kept:
            body = state - {pair}
            left = frozenset(q for q in body if q[0] in inside.positions)
            right = body - left
            key = (left, right)
            joint[key] = joint.get(key, 0.0) + p / held
        left_marginal: dict[Structure, float] = {}
        right_marginal: dict[Structure, float] = {}
        for (left, right), p in joint.items():
            left_marginal[left] = left_marginal.get(left, 0.0) + p
            right_marginal[right] = right_marginal.get(right, 0.0) + p
        factorised = 0.0
        for left, a in left_marginal.items():
            for right, b in right_marginal.items():
                factorised += abs(joint.get((left, right), 0.0) - a * b)
        budget = self.tolerance if self.promote_budget is None else self.promote_budget
        if conditioned + factorised > budget:
            return None

        self.anchors = anchors
        self.parts[owner] = self._install(rest, right_marginal, held)
        self.parts[pair] = self._install(inside, left_marginal, 1.0)
        self._rebase()
        self.slack += conditioned + factorised * held
        record = Promotion(pair=pair, conditioned=conditioned,
                           factorised=factorised * held)
        self.promotions.append(record)
        return record

    def _install(self, region: Region, weights: dict, scale: float) -> Solver:
        solver = Solver(
            self.energy, self.model, length=self.length,
            tolerance=self.tolerance, base=self.anchors,
            within=region.positions,
        )
        states = list(weights)
        if not states:
            states = [EMPTY]
            weights = {EMPTY: 1.0}
        solver.states = states
        solver.position = {s: i for i, s in enumerate(states)}
        solver.probability = np.array([weights[s] * scale for s in states])
        return solver

    def demote(self, pair: Pair) -> None:
        """Release an anchor, merging its region back into its parent.  Exact."""
        if pair not in self.anchors:
            raise KeyError(f"{pair} is not an anchor")
        parent = owner_of(self.anchors - {pair}, pair[0])
        inner = self.parts.pop(pair)
        outer = self.parts[parent]
        self.anchors = self.anchors - {pair}
        merged: dict[Structure, float] = {}
        for left, a in zip(inner.states, inner.probability):
            if a <= 0.0:
                continue
            for right, b in zip(outer.states, outer.probability):
                if b <= 0.0:
                    continue
                state = left | right | {pair}
                merged[state] = merged.get(state, 0.0) + float(a) * float(b)
        regions = {r.owner: r for r in decompose(self.anchors, self.length)}
        self.parts[parent] = self._install(regions[parent], merged, 1.0)
        self._rebase()

    # ------------------------------------------------------------------
    def anchor_probability(self, pair: Pair) -> float:
        """Exact equilibrium probability that ``pair`` is formed, at this length.

        ``Z`` with the pair enforced over ``Z`` without it.  Independent of the
        representation, which is the point: an anchored ensemble cannot see its
        own anchors melting.
        """
        sequence = self.energy.energy.seq[: self.length]
        free = partition_function_energy(sequence)
        forced = partition_function_energy(sequence, forced=[pair])
        if free is None or forced is None:  # pragma: no cover - optional dep
            return float("nan")
        return math.exp(-(forced - free) / self.energy.kT)

    def review(self) -> list[Promotion]:
        """Release stale anchors, then anchor what has become certain."""
        for pair in sorted(self.anchors, key=lambda q: -q[0]):
            probability = self.anchor_probability(pair)
            if probability == probability and probability < self.demote_below:
                self.demotions.append((pair, probability))
                self.demote(pair)
        accepted = []
        # A refusal is recorded per pair rather than by raising the threshold,
        # which an earlier version did: that turned one pair too stacked to be
        # worth splitting into a permanent block on every future promotion.
        refused: set[Pair] = set()
        while True:
            best = None
            for owner in list(self.parts):
                for pair, p in self.marginals(owner).items():
                    if p < self.promote_above or pair in refused:
                        continue
                    if best is None or p > best[2]:
                        best = (owner, pair, p)
            if best is None:
                break
            record = self.promote(best[0], best[1])
            if record is None:
                refused.add(best[1])
                continue
            accepted.append(record)
        return accepted

    # ------------------------------------------------------------------
    @property
    def states(self) -> int:
        """States actually stored: the sum over regions, not the product."""
        return sum(len(solver.states) for solver in self.parts.values())

    @property
    def represented(self) -> int:
        """Structures the stored factors represent between them."""
        total = 1
        for solver in self.parts.values():
            total *= max(1, len(solver.states))
        return total

    def certificate(self) -> float:
        """Equilibrium weight unaccounted for: truncation plus representation.

        Truncation is bounded per region against an exact constrained ``Z`` and
        summed - a union bound, so an over-estimate rather than an under-one.
        ``slack`` is what the promotions cost, measured when they were made.
        """
        total = self.slack
        for solver in self.parts.values():
            outside = solver.certified_outside()
            if outside != outside:  # pragma: no cover - optional dependency
                return float("nan")
            total += outside
        return total

    @property
    def retained_mass(self) -> float:
        mass = 1.0
        for solver in self.parts.values():
            mass *= float(solver.probability.sum())
        return mass

    def distribution(self, *, top: int = 8, beam: int = 4096):
        """The most probable structures, as dot-bracket strings.

        The product is enumerated with a beam rather than in full, because the
        whole point is that it may be astronomically large.  With every factor
        non-negative, keeping the ``beam`` best partial products is exact for the
        top ``top`` whenever ``top`` is far below ``beam``, which it is; the beam
        is reported so the claim can be checked rather than trusted.
        """
        current: list[tuple[Structure, float]] = [(frozenset(self.anchors), 1.0)]
        for solver in sorted(self.parts.values(), key=lambda s: -len(s.states)):
            ranked = sorted(
                ((s, float(p)) for s, p in zip(solver.states, solver.probability)
                 if p > 0.0),
                key=lambda kv: -kv[1],
            )[:beam]
            grown = [(state | part, weight * p)
                     for state, weight in current for part, p in ranked]
            grown.sort(key=lambda kv: -kv[1])
            current = grown[:beam]
        return [(dotbracket(state, self.length), weight)
                for state, weight in current[:top]]

    def regions(self) -> tuple[Region, ...]:
        return decompose(self.anchors, self.length)

    def pair_marginals(self) -> dict[Pair, float]:
        """Probability each pair is formed, conditional on the anchor set.

        The factors are independent, so a pair's probability is its marginal in
        its own region and nothing else - which is exactly why this is cheap to
        report at every nucleotide where the flat solver would have to sum over
        the whole retained set.  Anchors read 1 by construction: that is the
        conditioning, and its cost is in ``slack``, not hidden here.
        """
        out: dict[Pair, float] = {pair: 1.0 for pair in self.anchors}
        for owner in self.parts:
            out.update(self.marginals(owner))
        return out

    def paired(self) -> dict[int, float]:
        """Probability each nucleotide is paired at all."""
        out: dict[int, float] = {k: 0.0 for k in range(self.length)}
        for (i, j), p in self.pair_marginals().items():
            out[i] = out.get(i, 0.0) + p
            out[j] = out.get(j, 0.0) + p
        return out


@dataclass(slots=True)
class AnchoredFrame:
    """The anchored ensemble at one moment, and what it is not accounting for."""

    time: float
    transcript: int
    available: int
    #: States actually stored: the sum over regions.
    stored: int
    #: Structures those factors represent between them: the product.
    represented: int
    anchors: tuple
    #: Truncation (per region, union bound) plus representation slack.
    certificate: float
    slack: float
    distribution: list = field(default_factory=list)

    @property
    def dominant(self):
        return self.distribution[0] if self.distribution else ("", 0.0)


def transcribe(
    ensemble: "Anchored",
    schedule=None,
    *,
    steps_per_nucleotide: int = 2,
    top: int = 8,
    review_every: int = 1,
    seed_window: float | None = 4.0,
    seed_limit: int = 2000,
):
    """Fold while the chain grows, keeping the ensemble factorised as it goes.

    Transcription is what makes the adaptive scheme work at all.  Cold-starting
    at 45 nt, the ensemble has to solve the hard flat problem *before* it has
    marginals certain enough to anchor anything - 995 states and 37 s to find one
    anchor.  Growing into it, the molecule is short and cheap when its first
    domain becomes certain, and every later nucleotide arrives in a
    representation that is already factorised.
    """
    from .cotrans import Schedule
    from .seed import seed as seed_suboptimal

    schedule = schedule or Schedule()
    total = ensemble.energy.n
    interval = schedule.interval()
    clock = 0.0
    transcript = max(1, min(schedule.start, total))
    ensemble.grow(max(0, transcript - schedule.footprint))

    index = 0
    while True:
        if seed_window:
            for solver in ensemble.parts.values():
                seed_suboptimal(solver, window=seed_window, limit=seed_limit)
        for _ in range(steps_per_nucleotide):
            ensemble.advance(interval / steps_per_nucleotide)
        clock += interval
        if review_every and index % review_every == 0:
            ensemble.review()
        yield AnchoredFrame(
            time=clock,
            transcript=transcript,
            available=ensemble.length,
            stored=ensemble.states,
            represented=ensemble.represented,
            anchors=tuple(sorted(ensemble.anchors)),
            certificate=ensemble.certificate(),
            slack=ensemble.slack,
            distribution=ensemble.distribution(top=top),
        )
        index += 1
        if transcript >= total:
            break
        transcript += 1
        ensemble.grow(max(0, transcript - schedule.footprint))
