"""Pseudoknot energetics.

The nearest-neighbour model is defined on a loop decomposition, which only
exists for nested structures.  rona handles crossing helices by splitting the
structure into

* a **core** - a maximal nested subset of the helices, scored exactly with the
  Turner model, and
* a **pseudoknot set** - the helices that had to be removed to make the core
  nested.

Each pseudoknot helix then contributes its own stacking energy (it really is a
helix, so its stacks are worth their full nearest-neighbour value), its
terminal AU/GU penalties, and a topology penalty for threading through the
core.  Bases belonging to pseudoknot helices are counted as *unpaired* inside
the core loops, so the entropic cost of holding them in place is not
double-counted.

The topology penalty follows the shape used by Dirks & Pierce (2003):

    dG_pk = init + per_unpaired * n_unpaired + per_branch * n_crossed

evaluated over the *pseudoknot region* - the window spanned jointly by the
pseudoknot helix and the core helices it crosses - which keeps the linear term
bounded no matter how long the transcript is.

The defaults are a deliberately simple, transparent parameterisation rather
than a fitted parameter set; they are chosen so that characterised H-type
pseudoknots form during simulation while incidental crossings do not.  Tune
them from the command line (``--pk-init``, ``--pk-unpaired``, ``--pk-branch``)
or disable pseudoknots entirely with ``--no-pseudoknots``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..seq import PAIR_TYPE
from ..struct import Helix


@dataclass(frozen=True, slots=True)
class PseudoknotModel:
    """Tunable parameters of the H-type pseudoknot penalty (kcal/mol)."""

    #: Constant cost of threading one helix through the nested core.
    init: float = 7.0
    #: Cost per unpaired nucleotide inside the pseudoknot region.
    per_unpaired: float = 0.1
    #: Cost per core helix that the pseudoknot helix crosses.
    per_branch: float = 0.2
    #: Shortest helix allowed to form a pseudoknot.
    min_helix: int = 2
    #: Upper bound on simultaneously formed pseudoknot helices (None = no cap).
    max_helices: int | None = None
    #: Set False to forbid crossing helices altogether.
    enabled: bool = True

    def disabled(self) -> "PseudoknotModel":
        return PseudoknotModel(
            init=self.init,
            per_unpaired=self.per_unpaired,
            per_branch=self.per_branch,
            min_helix=self.min_helix,
            max_helices=self.max_helices,
            enabled=False,
        )


def helix_stack_energy(model, enc: Sequence[int], helix: Helix) -> float:
    """Stacking + terminal penalties of an isolated helix, in kcal/mol.

    This is the helix's own nearest-neighbour content with no loop context -
    exactly what a pseudoknot helix contributes once the core has accounted for
    the surrounding loops.
    """
    total = 0.0
    pairs = helix.pairs
    for k in range(len(pairs) - 1):
        i, j = pairs[k]
        p, q = pairs[k + 1]
        t1 = PAIR_TYPE[enc[i]][enc[j]] or 7
        t2 = PAIR_TYPE[enc[q]][enc[p]] or 7
        total += float(model.stack[t1][t2])
    for i, j in (helix.outer, helix.inner):
        t = PAIR_TYPE[enc[i]][enc[j]] or 7
        if t > 2:
            total += model.terminal_au
    return total / 100.0


def conflict_graph(helices: Sequence[Helix]) -> list[set[int]]:
    """Adjacency sets of the "these two helices cross" graph."""
    adj: list[set[int]] = [set() for _ in helices]
    for a in range(len(helices)):
        helix_a = helices[a]
        for b in range(a + 1, len(helices)):
            if helix_a.crosses(helices[b]):
                adj[a].add(b)
                adj[b].add(a)
    return adj


def any_crossing(helices: Sequence[Helix]) -> bool:
    """Cheap test for "is this structure pseudoknotted at all"."""
    for a in range(len(helices)):
        helix_a = helices[a]
        for b in range(a + 1, len(helices)):
            if helix_a.crosses(helices[b]):
                return True
    return False


def split_crossing(
    helices: Sequence[Helix],
    *,
    stability: Sequence[float] | None = None,
) -> tuple[list[Helix], list[Helix]]:
    """Partition helices into a nested core and a pseudoknot set.

    Finding the *maximum* nested subset is NP-hard in general, but real folding
    states carry only a handful of crossings, so a greedy rule is both fast and
    effectively optimal here: repeatedly move out the helix with the most
    crossings, breaking ties in favour of keeping the more stable helix in the
    core.

    ``stability`` optionally supplies a per-helix free energy (more negative =
    more stable); when omitted, helix length is used as a proxy.

    Ties are broken by the helices' own coordinates, never by the order they
    arrive in.  That matters more than it looks: the partition decides which
    helices pay the topology penalty, so an order-dependent tie would make the
    free energy of a state depend on the order its helices happened to be
    stored in - and a free energy that is not a function of the state breaks
    detailed balance.
    """
    adj = conflict_graph(helices)
    if not any(adj):
        return list(helices), []

    weight = (
        list(stability)
        if stability is not None
        else [-float(h.length) for h in helices]
    )
    removed: set[int] = set()
    while True:
        live = [
            (len(adj[k] - removed), weight[k], k)
            for k in range(len(helices))
            if k not in removed and (adj[k] - removed)
        ]
        if not live:
            break
        # most crossings first; among those, drop the least stable helix;
        # among those, the one earliest in sequence
        live.sort(
            key=lambda t: (
                -t[0],
                -t[1],
                helices[t[2]].i,
                helices[t[2]].j,
                helices[t[2]].length,
            )
        )
        removed.add(live[0][2])

    core = [h for k, h in enumerate(helices) if k not in removed]
    pk = [h for k, h in enumerate(helices) if k in removed]
    return core, pk


def pseudoknot_region(
    helix: Helix, crossed: Iterable[Helix]
) -> tuple[int, int]:
    """Window jointly spanned by a pseudoknot helix and what it crosses."""
    lo, hi = helix.i, helix.j
    for other in crossed:
        lo = min(lo, other.i)
        hi = max(hi, other.j)
    return lo, hi


def topology_penalty(
    pk_model: PseudoknotModel,
    helix: Helix,
    core: Sequence[Helix],
    pt: Sequence[int],
) -> float:
    """Cost of threading ``helix`` through the ``core``, in kcal/mol."""
    crossed = [h for h in core if helix.crosses(h)]
    lo, hi = pseudoknot_region(helix, crossed)
    unpaired = sum(1 for k in range(lo, hi + 1) if k < len(pt) and pt[k] < 0)
    return (
        pk_model.init
        + pk_model.per_unpaired * unpaired
        + pk_model.per_branch * len(crossed)
    )
