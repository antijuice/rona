"""Adaptive Finite State Projection: propagate, and certify.

The guarantee, from Munsky & Khammash (2006) and sharpened by Dendukuri et al.
(2025): restrict the generator to a retained set, let probability leave and
never return, and the retained distribution is a pointwise *lower bound* on the
true one - so the mass that left is not merely a bound on the L1 error, it is
the L1 error.  Pruning adds to it by exactly the mass pruned.

The consequence worth stating plainly: every distribution this module reports
comes with a number saying how wrong it can be, and when the retained set is too
small the number says ``1.0`` rather than the answer looking plausible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from ..energy.evaluator import FoldingEnergy
from .cache import EnergyCache
from .integrate import integrate
from .moves import MoveModel, candidate_pairs, neighbours
from .structures import EMPTY, Structure, dotbracket


@dataclass(slots=True)
class Step:
    """What one time step cost and how wrong it may be."""

    escaped: float
    pruned: float
    states: int
    expansions: int

    @property
    def bound(self) -> float:
        return self.escaped + self.pruned


class Solver:
    """A distribution over structures, advanced by solving the master equation.

    The retained set persists between steps and is grown where probability is
    leaking out of it, which is the only place growing it can help.  Neighbour
    lists are memoised per ``(structure, length)``: a state that stays in the set
    is re-used every step, so the energy model is consulted once per state rather
    than once per step per state.
    """

    def __init__(
        self,
        energy: FoldingEnergy,
        model: MoveModel | None = None,
        *,
        length: int | None = None,
        tolerance: float = 1e-6,
        prune_below: float = 1e-10,
        max_states: int = 50_000,
        integration_tolerance: float = 1e-8,
        boundary: str = "reflecting",
    ) -> None:
        self.energy = energy if isinstance(energy, EnergyCache) else EnergyCache(energy)
        self.model = model or MoveModel()
        self.length = self.energy.n if length is None else length
        self.tolerance = tolerance
        self.prune_below = prune_below
        self.max_states = max_states
        self.integration_tolerance = integration_tolerance
        if boundary not in ("reflecting", "absorbing"):
            raise ValueError("boundary must be 'reflecting' or 'absorbing'")
        #: ``absorbing`` gives a rigorous bound on the mass that *ever* left the
        #: retained set.  That bound is correct and, for this system, useless:
        #: it accumulates at the gross boundary flux, so a reversible excursion
        #: to a high-energy structure that returns immediately is charged in
        #: full, and at 10^7 s^-1 the bound reaches 1 within microseconds.
        #: ``reflecting`` deletes the escaping transitions instead of charging
        #: for them, which conserves mass exactly and makes the retained chain a
        #: proper process whose equilibrium is the Boltzmann distribution
        #: *conditioned on the retained set*.  Its error is then governed by how
        #: much equilibrium weight lies outside, which is what ``leak_estimate``
        #: measures.  (Cao, Terebus & Liang, Bull Math Biol 2016.)
        self.boundary = boundary

        self.states: list[Structure] = [EMPTY]
        self.position: dict[Structure, int] = {EMPTY: 0}
        self.probability = np.array([1.0])
        #: Accumulated rigorous bound on the L1 error of ``probability``.
        self.bound = 0.0
        #: Estimated (not bounded) L1 error from time integration.
        self.integration_error = 0.0
        #: Latest boundary indicator (reflecting) or escaped mass (absorbing).
        self.leak = 0.0
        self.history: list[Step] = []
        self._neighbours: dict[tuple[Structure, int], list[tuple[Structure, float]]] = {}
        self._pair_cache: dict[int, list[tuple[int, int]]] = {}

    # ------------------------------------------------------------------
    def _moves(self, state: Structure) -> list[tuple[Structure, float]]:
        key = (state, self.length)
        found = self._neighbours.get(key)
        if found is None:
            found = neighbours(
                self.energy, state, self.length, self.model, self._pairs()
            )
            self._neighbours[key] = found
        return found

    def _pairs(self):
        found = self._pair_cache.get(self.length)
        if found is None:
            found = candidate_pairs(self.energy, self.length, self.model)
            self._pair_cache[self.length] = found
        return found

    def _add(self, state: Structure) -> int:
        found = self.position.get(state)
        if found is None:
            found = len(self.states)
            self.states.append(state)
            self.position[state] = found
        return found

    def _assemble(self) -> tuple[csr_matrix, list[tuple[Structure, int, float]]]:
        """The truncated generator, and the escapes, *unweighted*.

        The escapes are returned per source column rather than summed, because
        which of them matters depends on where the probability ends up during
        the step - not on where it started.  Weighting them by the initial
        vector, which an earlier version did, gives every newly admitted state a
        weight of zero, so its own escapes look harmless and the expansion
        stalls after a single round with most of the mass already gone.
        """
        size = len(self.states)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        diagonal = np.zeros(size)
        escapes: list[tuple[Structure, int, float]] = []
        for column, state in enumerate(self.states):
            for target, rate in self._moves(state):
                diagonal[column] -= rate
                row = self.position.get(target)
                if row is None:
                    escapes.append((target, column, rate))
                    if self.boundary == "reflecting":
                        # the move is refused rather than charged for, so the
                        # column still sums to zero and no mass is lost
                        diagonal[column] += rate
                    continue
                rows.append(row)
                cols.append(column)
                data.append(rate)
        rows.extend(range(size))
        cols.extend(range(size))
        data.extend(diagonal.tolist())
        return csr_matrix((data, (rows, cols)), shape=(size, size)), escapes

    def leak_estimate(self, escapes) -> float:
        """Equilibrium weight sitting one move outside the retained set.

        With a reflecting boundary the retained chain relaxes to the Boltzmann
        distribution conditioned on the retained set, so what the truncation
        costs is the weight the true distribution puts outside it.  The states
        one move out are the cheapest non-trivial probe of that: their summed
        Boltzmann weight, relative to the retained set's, estimates how much is
        missing.

        This is an *estimate*, not the certificate the absorbing boundary gives.
        It is reported as one, and validated by comparing against the error
        measured on fully enumerated state spaces.
        """
        if not escapes:
            return 0.0
        inside = 0.0
        reference = min(self.energy.of(s, self.length) for s in self.states)
        for state in self.states:
            inside += math.exp(-(self.energy.of(state, self.length) - reference)
                               / self.energy.kT)
        outside = 0.0
        for target in {state for state, _column, _rate in escapes}:
            outside += math.exp(-(self.energy.of(target, self.length) - reference)
                                / self.energy.kT)
        return outside / (inside + outside)

    def _padded(self) -> np.ndarray:
        vector = np.zeros(len(self.states))
        vector[: len(self.probability)] = self.probability
        return vector

    # ------------------------------------------------------------------
    def advance(self, dt: float, *, max_expansions: int = 24, per_round: int = 256) -> Step:
        """Advance by ``dt``, expanding the retained set until the leak fits."""
        expansions = 0
        while True:
            matrix, escapes = self._assemble()
            vector = self._padded()
            evolved, self.integration_error = integrate(
                matrix, vector, dt, tolerance=self.integration_tolerance
            )
            escaped = float(max(0.0, vector.sum() - evolved.sum()))
            if self.boundary == "reflecting":
                indicator = self.leak_estimate(escapes)
            else:
                indicator = escaped
            if (
                indicator <= self.tolerance
                or expansions >= max_expansions
                or len(self.states) >= self.max_states
                or not escapes
            ):
                break
            # weight each escape by the probability sitting on its source at the
            # *end* of the step, which is where the mass actually is
            frontier: dict[Structure, float] = {}
            if self.boundary == "reflecting":
                # what matters is which missing states carry equilibrium weight
                reference = min(self.energy.of(s, self.length) for s in self.states)
                for target in {state for state, _c, _r in escapes}:
                    frontier[target] = math.exp(
                        -(self.energy.of(target, self.length) - reference)
                        / self.energy.kT
                    )
            else:
                for target, column, rate in escapes:
                    share = rate * max(evolved[column], vector[column])
                    frontier[target] = frontier.get(target, 0.0) + share
            # Admit states in order of the probability actually arriving at
            # them, and stop once the flux left outside would not breach the
            # budget.  Admitting a fixed number instead - which an earlier
            # version did - pulls in thousands of structures that carry nothing:
            # on an 18 nt sequence, 99.99% of the equilibrium mass sits in 20 of
            # 80,232 states, so the retained set should look like the support of
            # the distribution, not like a breadth-first ball around it.
            ranked = sorted(frontier.items(), key=lambda kv: -kv[1])
            total = sum(flux for _state, flux in ranked)
            tail = total * (1.0 if self.boundary == "reflecting" else dt)
            added = 0
            for state, flux in ranked:
                if tail <= self.tolerance * 0.5:
                    break
                if added >= per_round or len(self.states) >= self.max_states:
                    break
                if state not in self.position:
                    self._add(state)
                    added += 1
                tail -= flux * (1.0 if self.boundary == "reflecting" else dt)
            if added == 0:
                break
            expansions += 1

        pruned = 0.0
        if self.prune_below > 0.0:
            small = evolved < self.prune_below
            # never prune everything: a state must survive to carry the mass
            if not small.all():
                pruned = float(evolved[small].sum())
                evolved = evolved.copy()
                evolved[small] = 0.0

        self.probability = evolved
        self.bound += escaped + pruned
        self.leak = indicator
        step = Step(escaped=escaped, pruned=pruned, states=len(self.states),
                    expansions=expansions)
        self.history.append(step)
        return step

    # ------------------------------------------------------------------
    def grow(self, new_length: int) -> None:
        """Transcribe one or more nucleotides.

        Appending sequence is an injective map on structures - every structure of
        the shorter prefix is a structure of the longer one - so the distribution
        carries across exactly.  Only the neighbour lists change, because new
        pairings become available; the memo is keyed by length, so they are
        simply recomputed.
        """
        self.length = min(new_length, self.energy.n)

    def distribution(self, *, top: int | None = None) -> list[tuple[str, float]]:
        pairs = [
            (dotbracket(state, self.length), float(p))
            for state, p in zip(self.states, self.probability)
            if p > 0.0
        ]
        pairs.sort(key=lambda kv: -kv[1])
        return pairs[:top] if top else pairs

    @property
    def retained_mass(self) -> float:
        return float(self.probability.sum())
