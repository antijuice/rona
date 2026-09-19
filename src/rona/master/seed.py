"""Seeding the retained set with structures the walk would struggle to reach.

Expansion admits states adjacent to what is already held, which finds a basin
only by walking to it.  A basin separated from the current set by a ridge of
individually-insignificant structures is therefore invisible: on a 40 nt
transcript the solver reported 7e-2 of the equilibrium weight missing and could
not close the gap, because the weight sat behind states none of which was worth
admitting on its own.

Low-energy structures are exactly the ones carrying that weight, and this domain
can enumerate them directly - ``RNAsubopt`` lists every structure within a free
energy window of the minimum, in time proportional to how many there are.  So
the basins are handed to the solver and the walk is left to do what it is good
at: filling in the paths between them.

This is not the model.  It seeds a state space whose error is then certified; a
structure that turns out to carry nothing is pruned like any other.  That is the
difference between using suboptimal structures as a starting point and using
them *as* the ensemble, which is what a sampling-and-selecting method does.
"""

from __future__ import annotations

from .structures import Structure, from_dotbracket


def suboptimal(
    sequence: str,
    *,
    window: float = 4.0,
    limit: int = 2000,
    temperature: float = 37.0,
) -> list[Structure]:
    """Structures within ``window`` kcal/mol of the minimum, or ``[]``."""
    try:
        import RNA
    except ImportError:  # pragma: no cover - optional dependency
        return []
    model = RNA.md()
    model.temperature = temperature
    fold = RNA.fold_compound(sequence, model)
    found = fold.subopt(int(round(window * 100.0)))
    out: list[Structure] = []
    for entry in found[:limit]:
        structure = getattr(entry, "structure", None)
        if not structure:
            continue
        out.append(from_dotbracket(structure))
    return out


def seed(solver, *, window: float = 4.0, limit: int = 2000) -> int:
    """Admit the low-energy structures of the current prefix; return how many.

    A region solver is seeded with the *projection* of each structure onto its
    own nucleotides.  That needs no constrained enumeration: a nested structure's
    pairs that lie wholly inside one region are mutually nested and cannot cross
    an anchor, so the projection is always a legal region state, and a pair
    running from inside the region to outside it is dropped because no single
    region could hold it anyway.  Seeding is not the model - a projection that
    carries no weight is pruned like anything else - so it is enough that every
    seed be legal and cheap to obtain.
    """
    sequence = solver.energy.energy.seq[: solver.length]
    if not sequence:
        return 0
    added = 0
    for structure in suboptimal(sequence, window=window, limit=limit):
        if solver.within is not None:
            structure = frozenset(
                pair for pair in structure
                if pair[0] in solver.within and pair[1] in solver.within
            )
        if structure not in solver.position:
            solver._add(structure)
            added += 1
    if added:
        solver.probability = solver._padded()
    return added
