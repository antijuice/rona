"""The generator over a retained set of structures, and its leak.

The matrix built here is *deliberately defective*.  A state on the boundary of
the retained set has moves leading out of it, and those rates appear on the
diagonal as loss without appearing anywhere as gain.  That missing mass is not
an error to be hidden: it is the Finite State Projection error bound, and the
whole design rests on measuring it rather than discarding it silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from ..energy.evaluator import FoldingEnergy
from .moves import MoveModel, neighbours
from .structures import Structure


@dataclass(slots=True)
class StateIndex:
    """A bijection between retained structures and matrix rows."""

    states: list[Structure] = field(default_factory=list)
    position: dict[Structure, int] = field(default_factory=dict)

    @classmethod
    def of(cls, states) -> "StateIndex":
        index = cls()
        for state in states:
            index.add(state)
        return index

    def add(self, state: Structure) -> int:
        found = self.position.get(state)
        if found is None:
            found = len(self.states)
            self.states.append(state)
            self.position[state] = found
        return found

    def __len__(self) -> int:
        return len(self.states)

    def __contains__(self, state: Structure) -> bool:
        return state in self.position

    def __iter__(self):
        return iter(self.states)


@dataclass(slots=True)
class Generator:
    """The truncated generator, plus what it leaks and to where."""

    matrix: csr_matrix
    index: StateIndex
    #: Total rate out of the retained set, per retained state.
    leak: np.ndarray
    #: Structures just outside the retained set, with the flux reaching them.
    frontier: dict[Structure, float]


def build_generator(
    energy: FoldingEnergy,
    index: StateIndex,
    length: int,
    model: MoveModel,
) -> Generator:
    """Assemble ``A`` with ``dp/dt = A p`` over the retained set."""
    size = len(index)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(size)
    leak = np.zeros(size)
    frontier: dict[Structure, float] = {}

    for column, state in enumerate(index.states):
        for target, rate in neighbours(energy, state, length, model):
            diagonal[column] -= rate
            row = index.position.get(target)
            if row is None:
                leak[column] += rate
                frontier[target] = frontier.get(target, 0.0) + rate
                continue
            rows.append(row)
            cols.append(column)
            data.append(rate)

    rows.extend(range(size))
    cols.extend(range(size))
    data.extend(diagonal.tolist())
    matrix = csr_matrix((data, (rows, cols)), shape=(size, size))
    return Generator(matrix=matrix, index=index, leak=leak, frontier=frontier)
