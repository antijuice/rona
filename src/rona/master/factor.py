"""Anchored decomposition: where the state space is a product, exactly.

Milestone 3 hit a wall at about 50 nt, and the wall was not arithmetic.  The
retained set has to cover a *product*: once the molecule holds two structural
domains that fold independently, keeping both to a tolerance costs the product
of their supports.  238 states at 35 nt became 1,224 at 45 nt for exactly that
reason - a third domain appeared.  No amount of faster linear algebra fixes a
representation whose size is multiplicative in the number of domains.

The fix has to come from the physics, and it does.  Two facts, both measured
rather than assumed:

**A formed pair separates inside from outside.**  Fix a nested set of pairs
``A`` - call them anchors.  For a signature ``S`` (a subset of ``A``), consider
the structures that contain every pair of ``S``.  The pairs of ``S`` cut the
free nucleotides into *regions*: each anchor owns the nucleotides inside it but
not inside any deeper anchor of ``S``, and the exterior owns the rest.  Then

1.  every pair that does not cross an anchor lies wholly within one region, so
    the state space is the Cartesian product of per-region state spaces; and
2.  the free energy is additive, ``G(S u parts) = sum_r G(S u part_r) - (k-1)
    G(S)``, to the last bit of the evaluator's output.

Both are checked in ``tests/test_factor.py`` - the second as an exact equality,
because the nearest-neighbour model really is a sum over loops and a region
boundary is a loop boundary.

**So the generator is a Kronecker sum.**  A single-pair move changes one region.
Its rate is Metropolis on ``dG``, and by additivity that ``dG`` is the same
whether the other regions are empty or full.  Admissibility likewise: a pair
inside a region can only clash with that region's pairs or with an anchor.  So
on the block of states containing ``S``,

    Q = Q_1 (+) Q_2 (+) ... (+) Q_k

and ``exp(tQ) = exp(t Q_1) (x) ... (x) exp(t Q_k)``.  A product distribution
stays a product forever, at a cost that is the *sum* of the region supports
rather than their product.  Nothing is approximated: this is an identity.

What it does not do is survive an anchor event.  When an anchor breaks, two
regions merge and their joint distribution is generally not a product; when one
forms, a region splits and the split is not a product either.  That is where
approximation has to enter, and it enters where it can be measured - see
:mod:`rona.master.lowrank`.

This module is only the algebra: regions, restriction, and the per-region move
set.  It deliberately reuses ``neighbours`` unchanged, passing the region's own
candidate pairs, so a region's rates come from the same code path as the flat
solver's and cannot drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..energy.evaluator import FoldingEnergy
from .moves import MoveModel, candidate_pairs
from .structures import Structure

Pair = tuple[int, int]


class NotNested(ValueError):
    """The anchor set has a crossing pair, so it cuts nothing."""


@dataclass(frozen=True, slots=True)
class Region:
    """One independent domain of an anchored decomposition."""

    #: The innermost anchor containing this region, or ``None`` for the exterior.
    owner: Pair | None
    #: The nucleotides still free to pair, ascending.
    positions: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.positions)


def check_nested(anchors) -> frozenset[Pair]:
    """Reject a crossing or overlapping anchor set."""
    items = sorted(anchors)
    seen: set[int] = set()
    for i, j in items:
        if i >= j:
            raise NotNested(f"anchor {(i, j)} is not a pair")
        if i in seen or j in seen:
            raise NotNested(f"anchor {(i, j)} shares a nucleotide with another")
        seen.update((i, j))
    for index, (a, b) in enumerate(items):
        for c, d in items[index + 1 :]:
            if a < c < b < d:
                raise NotNested(f"anchors {(a, b)} and {(c, d)} cross")
    return frozenset(items)


def owner_of(anchors, position: int) -> Pair | None:
    """The innermost anchor strictly containing ``position``."""
    best: Pair | None = None
    for i, j in anchors:
        if i < position < j and (best is None or i > best[0]):
            best = (i, j)
    return best


def decompose(anchors, length: int) -> tuple[Region, ...]:
    """The regions an anchor signature cuts the first ``length`` nt into.

    Ordered outermost first, which is the order a transcribing molecule creates
    them in and the order the tensor factors are stored in.
    """
    anchors = check_nested(anchors)
    taken = {k for pair in anchors for k in pair}
    groups: dict[Pair | None, list[int]] = {None: []}
    for pair in anchors:
        groups[pair] = []
    for position in range(length):
        if position in taken:
            continue
        groups[owner_of(anchors, position)].append(position)
    keys = sorted(groups, key=lambda key: (0, 0) if key is None else (1, key[0]))
    return tuple(Region(owner=key, positions=tuple(groups[key])) for key in keys)


def region_of(anchors, pair: Pair) -> Pair | None:
    """Which region ``pair`` belongs to.

    Raises if it crosses an anchor or straddles two regions - by the theorem
    above neither can happen for an admissible pair, so either is a bug.
    """
    i, j = pair
    for a, b in anchors:
        if (a < i < b < j) or (i < a < j < b):
            raise NotNested(f"pair {pair} crosses anchor {(a, b)}")
    left, right = owner_of(anchors, i), owner_of(anchors, j)
    if left != right:
        raise NotNested(f"pair {pair} straddles {left} and {right}")
    return left


def restrict(structure: Structure, anchors, regions) -> tuple[Structure, ...]:
    """Split ``structure`` into its per-region parts, aligned to ``regions``.

    ``structure`` must contain every anchor; the anchors themselves are not part
    of any region.
    """
    anchors = frozenset(anchors)
    missing = anchors - structure
    if missing:
        raise ValueError(f"structure is missing anchors {sorted(missing)}")
    parts: dict[Pair | None, set[Pair]] = {region.owner: set() for region in regions}
    for pair in structure:
        if pair in anchors:
            continue
        parts[region_of(anchors, pair)].add(pair)
    return tuple(frozenset(parts[region.owner]) for region in regions)


def rejoin(anchors, parts) -> Structure:
    """Put per-region parts back together with their anchors."""
    out: set[Pair] = set(anchors)
    for part in parts:
        out.update(part)
    return frozenset(out)


def region_candidates(
    energy: FoldingEnergy,
    anchors,
    region: Region,
    length: int,
    model: MoveModel,
) -> list[Pair]:
    """The pairs a region may form, for :func:`rona.master.moves.neighbours`.

    Passing these as that function's ``pairs`` argument is what makes a region's
    move set a restriction of the flat one rather than a reimplementation of it:
    same rates, same admissibility, same energy evaluator, no second code path
    to keep in step.
    """
    inside = set(region.positions)
    out = []
    for pair in candidate_pairs(energy, length, model):
        if pair[0] in inside and pair[1] in inside:
            out.append(pair)
    return out
