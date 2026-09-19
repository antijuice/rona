"""Which structures are one move apart, and at what rate.

**One elementary move: add or remove a single base pair.**  Not "form a helix",
which is what v1 and Kinefold do and what an earlier draft of this module did.
The reason is reversibility, and it is worth being precise about because the
helix-level version fails in a way that is easy to miss.

Define moves on *maximal helices* and they stop being self-inverse: nucleating a
three-pair block next to an existing helix produces one longer helix, whose
whole-melt is not an offered move, so the inverse of that nucleation does not
exist.  Measured on a 25 nt sequence, 119 of 1155 edges were one-way.  The
chain is then not reversible, its stationary distribution is not Boltzmann, and
it drifts further from Boltzmann the longer it runs - which is a silent,
thermodynamically wrong answer rather than a crash.

With single base pairs, ``add(i,j)`` and ``remove(i,j)`` are inverses by
construction.  Reversibility is structural and cannot be broken by an
unconsidered case.  This is Kinfold's move set, and it is the right foundation
even though it makes the state space larger, because the solver prunes states
that carry no probability and *counts what it pruned*.

Energies are evaluated exactly, every time.  v1 needed incremental loop-local
deltas and an invalidation apparatus because it touched one *event* per step and
there were 10^6 of them per simulated second; here each state is visited once
per time step regardless, so a full O(n) evaluation per neighbour is affordable -
and the entire class of stale-cache bug that produced most of v1's defects does
not exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..energy.evaluator import FoldingEnergy
from ..seq import PAIR_TYPE
from .cache import EnergyCache
from .structures import Structure, occupied

#: Exponent clamp, keeping ``exp`` away from overflow.
MAX_EXPONENT = 400.0


@dataclass(frozen=True, slots=True)
class MoveModel:
    """The move set and its rate rule."""

    #: Attempt frequency for adding or removing one base pair (s^-1).
    k_pair: float = 1.0e7
    #: Minimum hairpin loop.
    min_loop: int = 3
    #: Longest span a pair may bridge; ``None`` for no limit.
    max_span: int | None = None

    def rate(self, dg: float, kT: float) -> float:
        """Metropolis on the exact free-energy difference.

        Detailed balance is then exact for a reason worth stating: both
        directions read the same energy function on the same two states, and
        there are no incremental deltas that could disagree with each other.
        """
        if dg != dg or dg == float("inf"):
            return 0.0
        x = -max(0.0, dg) / kT
        if x < -MAX_EXPONENT:
            return 0.0
        return self.k_pair * math.exp(x)


def candidate_pairs(energy: FoldingEnergy, length: int, model: MoveModel):
    """Every pair the sequence could form within ``length``.

    A property of the sequence, not of any structure, so the caller can compute
    it once and reuse it for every state.
    """
    enc = energy.enc
    span = model.max_span
    out = []
    for i in range(length):
        top = length if span is None else min(length, i + span + 1)
        for j in range(i + model.min_loop + 1, top):
            if PAIR_TYPE[enc[i]][enc[j]]:
                out.append((i, j))
    return out


def neighbours(
    energy: FoldingEnergy,
    structure: Structure,
    length: int,
    model: MoveModel,
    pairs=None,
) -> list[tuple[Structure, float]]:
    """``(neighbour, rate)`` for every move out of ``structure``.

    ``length`` is how much of the sequence has been transcribed; nothing beyond
    it may pair.
    """
    if not isinstance(energy, EnergyCache):
        energy = EnergyCache(energy)
    kT = energy.kT
    here = energy.of(structure, length)
    used = occupied(structure)
    if pairs is None:
        pairs = candidate_pairs(energy, length, model)
    out: list[tuple[Structure, float]] = []

    def offer(target: Structure) -> None:
        there = energy.of(target, length)
        rate = model.rate(there - here, kT)
        if rate > 0.0:
            out.append((target, rate))

    for pair in pairs:
        if pair in structure:
            offer(structure - {pair})
            continue
        i, j = pair
        if i in used or j in used or j >= length:
            continue
        offer(structure | {pair})
    return out
