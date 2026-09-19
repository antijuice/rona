"""The v2 solver: reversibility, thermodynamics, integration, and the certificate.

These are the properties the whole approach rests on, and each is checked
against something independent rather than against the code's own opinion:
reversibility against the Boltzmann ratio, the long-time limit against the
Boltzmann distribution computed directly from the energy model, the integrator
against ``scipy``'s exact matrix exponential, and the FSP bound against the
error measured on a fully enumerated state space.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from rona.energy.evaluator import FoldingEnergy
from rona.master.cache import EnergyCache
from rona.master.fsp import Solver
from rona.master.integrate import integrate
from rona.master.moves import MoveModel, candidate_pairs, neighbours
from rona.master.structures import EMPTY, dotbracket, from_dotbracket

SMALL = "GCGCAAAAGCGC"


def chain(sequence, model=None):
    """Enumerate the whole reachable space and build its exact generator."""
    model = model or MoveModel()
    cache = EnergyCache(FoldingEnergy(sequence))
    n = len(sequence)
    pairs = candidate_pairs(cache, n, model)
    index = {EMPTY: 0}
    order = [EMPTY]
    stack = [EMPTY]
    while stack:
        state = stack.pop()
        for target, _rate in neighbours(cache, state, n, model, pairs):
            if target not in index:
                index[target] = len(order)
                order.append(target)
                stack.append(target)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(len(order))
    for column, state in enumerate(order):
        for target, rate in neighbours(cache, state, n, model, pairs):
            diagonal[column] -= rate
            rows.append(index[target])
            cols.append(column)
            data.append(rate)
    rows.extend(range(len(order)))
    cols.extend(range(len(order)))
    data.extend(diagonal.tolist())
    matrix = csr_matrix((data, (rows, cols)), shape=(len(order),) * 2)
    return cache, model, order, index, matrix, pairs


def boltzmann(cache, order, n):
    energies = np.array([cache.of(state, n) for state in order])
    weights = np.exp(-(energies - energies.min()) / cache.kT)
    return weights / weights.sum()


def test_structures_round_trip():
    for db in ("((((...))))", "..((...))..", "((.[[.))..]]"):
        assert dotbracket(from_dotbracket(db), len(db)) == db


def test_every_move_has_its_inverse_at_the_boltzmann_ratio():
    """Reversibility must be structural, not a property to be hoped for.

    Defining moves on maximal helices - which v1 and an early draft of v2 did -
    silently produces one-way edges: nucleating a block beside an existing helix
    yields one longer helix whose whole-melt is not an offered move.  The chain
    is then irreversible, its stationary distribution is not Boltzmann, and it
    drifts *away* from Boltzmann the longer it runs.  Single base pairs make
    add and remove inverses by construction.
    """
    cache, model, order, index, _matrix, pairs = chain(SMALL)
    n = len(SMALL)
    equilibrium = boltzmann(cache, order, n)
    checked = 0
    worst = 0.0
    for column, state in enumerate(order):
        forward = dict(neighbours(cache, state, n, model, pairs))
        for target, rate in forward.items():
            back = dict(neighbours(cache, target, n, model, pairs)).get(state)
            assert back is not None and back > 0.0, "one-way transition"
            left = rate * equilibrium[column]
            right = back * equilibrium[index[target]]
            worst = max(worst, abs(left - right) / max(left, right))
            checked += 1
    assert checked > 100
    assert worst < 1e-12


def test_long_time_limit_is_the_boltzmann_distribution():
    """The thermodynamic check: solve, wait, and land where statistics says."""
    cache, _model, order, _index, matrix, _pairs = chain(SMALL)
    equilibrium = boltzmann(cache, order, len(SMALL))
    start = np.zeros(len(order))
    start[0] = 1.0
    final, _error = integrate(matrix, start, 1.0, tolerance=1e-10)
    assert final.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(np.abs(final - equilibrium).sum()) < 1e-6


@pytest.mark.parametrize("t", [1e-7, 1e-5, 1e-3])
def test_integration_matches_the_exact_matrix_exponential(t):
    """And its error *estimate* is honest about how far off it is."""
    _cache, _model, order, _index, matrix, _pairs = chain(SMALL)
    start = np.zeros(len(order))
    start[0] = 1.0
    approx, estimate = integrate(matrix, start, t, tolerance=1e-9)
    exact = expm_multiply(matrix * t, start)
    true_error = float(np.abs(approx - exact).sum())
    assert approx.sum() == pytest.approx(1.0, abs=1e-9)
    assert (approx >= 0.0).all(), "backward Euler must keep probabilities positive"
    # the estimate should be the right size, not merely finite
    assert true_error <= max(10.0 * estimate, 1e-9)


def test_the_fsp_bound_is_never_violated_and_is_tight():
    """The certificate, against the error measured on the full state space.

    FSP's guarantee is not merely that the bound holds: because the truncated
    solution is a pointwise lower bound on the true one, the L1 error *equals*
    the escaped mass.  So a correct implementation reports a bound that is
    tight, and one that is loose is a sign the bookkeeping is wrong.
    """
    cache, model, order, index, _matrix, _pairs = chain(SMALL)
    horizon = 1e-4
    steps = 8

    def run(cap):
        solver = Solver(
            cache, model, tolerance=1e-14, prune_below=0.0, max_states=cap
        )
        for _ in range(steps):
            solver.advance(horizon / steps, per_round=32)
        vector = np.zeros(len(order))
        for state, value in zip(solver.states, solver.probability):
            vector[index[state]] = value
        return solver, vector

    # The reference is the *same integrator* on the untruncated space, so the
    # comparison isolates truncation error.  Comparing against an exact matrix
    # exponential instead would fold in the time-integration error, which the
    # certificate does not claim to cover and which is estimated separately.
    reference_solver, exact = run(10_000)
    assert reference_solver.bound < 1e-12, "the untruncated run must not truncate"

    seen_truncation = False
    for cap in (5, 12):
        solver, approx = run(cap)
        true_error = float(np.abs(exact - approx).sum())
        assert true_error <= solver.bound + 1e-9, (
            f"cap={cap}: error {true_error:.3e} exceeds certificate "
            f"{solver.bound:.3e}"
        )
        if solver.bound > 1e-6:
            seen_truncation = True
            # pointwise domination makes the bound tight, not merely valid
            assert true_error >= 0.5 * solver.bound
            assert (approx <= exact + 1e-12).all(), "FSP must under-estimate"
    assert seen_truncation, "the caps should have forced a real truncation"


def test_pruned_mass_is_added_to_the_certificate():
    cache, model, _order, _index, _matrix, _pairs = chain(SMALL)
    solver = Solver(cache, model, tolerance=1e-12, prune_below=1e-3)
    for _ in range(4):
        solver.advance(1e-5)
    assert solver.bound >= sum(step.pruned for step in solver.history)
    assert solver.retained_mass <= 1.0 + 1e-12
