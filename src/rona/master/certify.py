"""A rigorous bound on what the retained set is missing.

The reflecting chain relaxes to the Boltzmann distribution *conditioned on the
retained set*, so its equilibrium error is governed by the weight lying outside:

    || pi|_S - pi ||_1  =  2 (1 - Z_S / Z)

``Z_S`` is a sum over the states actually held.  ``Z`` is the partition function
over *every* secondary structure - which in this domain can be computed exactly
without enumerating anything, by McCaskill's O(n^3) algorithm.  That is a
resource the generic chemical-master-equation setting does not have, and it
turns the boundary indicator into a real certificate: not "the states one move
out look light", but "the retained set holds all but this fraction of the
equilibrium weight, and here is the number".

Two honest limits.  It certifies the *equilibrium* component; a transient
distribution can be wrong in ways this does not see, which is why the dynamic
indicator is still reported alongside.  And it inherits whatever disagreement
exists between this package's energy model and the one computing ``Z`` - here
0.009 kcal/mol against ViennaRNA, which is small but not nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Certificate:
    """What fraction of the equilibrium weight the retained set does not hold."""

    outside: float
    #: L1 distance between the retained equilibrium and the true one.
    l1: float
    retained_states: int
    #: ``None`` when the exact partition function was unavailable.
    available: bool = True

    def __str__(self) -> str:
        if not self.available:
            return "no certificate (ViennaRNA not installed)"
        return (
            f"{self.retained_states} states hold all but {self.outside:.3e} "
            f"of the equilibrium weight (L1 <= {self.l1:.3e})"
        )


def partition_function_energy(sequence: str, temperature: float = 37.0) -> float | None:
    """Ensemble free energy over *all* structures, or ``None`` without ViennaRNA."""
    try:
        import RNA
    except ImportError:  # pragma: no cover - optional dependency
        return None
    model = RNA.md()
    model.temperature = temperature
    fold = RNA.fold_compound(sequence, model)
    _structure, ensemble = fold.pf()
    return float(ensemble)


def certify(solver, *, temperature: float = 37.0) -> Certificate:
    """How much equilibrium weight lies outside ``solver``'s retained set."""
    sequence = solver.energy.energy.seq[: solver.length]
    ensemble = partition_function_energy(sequence, temperature)
    if ensemble is None:  # pragma: no cover - optional dependency
        return Certificate(float("nan"), float("nan"), len(solver.states), False)

    kT = solver.energy.kT
    # sum in the shifted scale the ensemble free energy already sets, so the
    # exponentials stay in range for long sequences
    retained = 0.0
    for state in solver.states:
        retained += math.exp(-(solver.energy.of(state, solver.length) - ensemble) / kT)
    outside = max(0.0, 1.0 - retained)
    return Certificate(outside, 2.0 * outside, len(solver.states))
