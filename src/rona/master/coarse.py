"""Grouping structures into basins, for reporting and for comparison.

The solver's answer is a distribution over individual structures.  That is the
right thing to compute and the wrong thing to read: two structures differing by
one marginal base pair at the end of a helix are the same conformation to anyone
looking at it, and every other tool in this area reports coarse-grained
macrostates instead - DrTransformer's occupancies, BarMap's basins, Kinefold's
representative structures.

So comparing against them requires coarse-graining *this* ensemble by the same
rule, not comparing microstate probabilities against macrostate ones.  Doing that
matters: at 60 nt DrTransformer puts 0.9999 on one structure where this solver
spreads 0.55/0.21/0.15 over three that differ only in a marginal hairpin.  Read
as microstates those disagree badly; read as basins they are the same answer.

The rule here is steepest descent in the solver's own move set, which is the same
notion of basin the barrier-tree literature uses (Flamm, Hofacker et al.): a
structure belongs to the local minimum a greedy downhill walk reaches from it.
Ties are broken by sorted pair order so the map is deterministic - an arbitrary
tie-break would make basin membership depend on set iteration order, which is
exactly the class of bug that made v1's pseudoknot energies order-dependent.

This is a *reporting* transform.  Nothing in the dynamics goes through it, and
the certificate is unaffected: grouping states does not change how much weight
lies outside the retained set.
"""

from __future__ import annotations

from .cache import EnergyCache
from .moves import MoveModel, neighbours
from .structures import Structure


def local_minimum(
    energy,
    structure: Structure,
    length: int,
    model: MoveModel | None = None,
    *,
    pairs=None,
    memo: dict | None = None,
) -> Structure:
    """The local minimum a steepest-descent walk reaches from ``structure``."""
    model = model or MoveModel()
    if not isinstance(energy, EnergyCache):
        energy = EnergyCache(energy)
    if memo is not None:
        found = memo.get((structure, length))
        if found is not None:
            return found
    path = []
    current = structure
    seen = {current}
    while True:
        path.append(current)
        here = energy.of(current, length)
        best = None
        for target, _rate in neighbours(energy, current, length, model, pairs):
            if target in seen:
                continue
            there = energy.of(target, length)
            # strict descent, with a deterministic tie-break so the basin map is
            # a function of the structures and not of iteration order
            key = (there, sorted(target))
            if there < here - 1e-9 and (best is None or key < best[0]):
                best = (key, target)
        if best is None:
            break
        current = best[1]
        seen.add(current)
    if memo is not None:
        for state in path:
            memo[(state, length)] = current
    return current


def basins(
    energy,
    states,
    probability,
    length: int,
    model: MoveModel | None = None,
    *,
    memo: dict | None = None,
) -> dict[Structure, float]:
    """Probability grouped by basin: ``{local minimum: summed probability}``."""
    if memo is None:
        memo = {}
    out: dict[Structure, float] = {}
    for state, p in zip(states, probability):
        if p <= 0.0:
            continue
        minimum = local_minimum(energy, state, length, model, memo=memo)
        out[minimum] = out.get(minimum, 0.0) + float(p)
    return out
