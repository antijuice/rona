"""Free energy of a structure, computed once.

The move set proposes O(n^2) neighbours per state, and each neighbour is itself
a state reached again from its own neighbours.  Without memoisation a
structure's energy is recomputed once per edge incident on it, which on a 25 nt
sequence is about a hundred times more evaluations than there are states.

Keyed by ``(structure, length)``: transcription changes the energy of a fixed
structure, because what dangles off its ends changes.
"""

from __future__ import annotations

from ..energy.evaluator import FoldingEnergy
from .structures import Structure, helices_of


class EnergyCache:
    """``FoldingEnergy`` with the answers kept."""

    __slots__ = ("energy", "_values", "hits", "misses")

    def __init__(self, energy: FoldingEnergy) -> None:
        self.energy = energy
        self._values: dict[tuple[Structure, int], float] = {}
        self.hits = 0
        self.misses = 0

    @property
    def kT(self) -> float:
        return self.energy.kT

    @property
    def enc(self):
        return self.energy.enc

    @property
    def n(self) -> int:
        return self.energy.n

    def of(self, structure: Structure, length: int) -> float:
        key = (structure, length)
        found = self._values.get(key)
        if found is None:
            self.misses += 1
            found = self.energy.energy(helices_of(structure, length), length)
            self._values[key] = found
        else:
            self.hits += 1
        return found

    def __len__(self) -> int:
        return len(self._values)
