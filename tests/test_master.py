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
    assert checked > 50
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
            cache, model, tolerance=1e-14, prune_below=0.0, max_states=cap,
            boundary="absorbing",
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
    solver = Solver(
        cache, model, tolerance=1e-12, prune_below=1e-3, boundary="absorbing"
    )
    for _ in range(4):
        solver.advance(1e-5)
    assert solver.bound >= sum(step.pruned for step in solver.history)
    assert solver.retained_mass <= 1.0 + 1e-12


def _enumerate(sequence, model=None):
    model = model or MoveModel()
    cache = EnergyCache(FoldingEnergy(sequence))
    n = len(sequence)
    pairs = candidate_pairs(cache, n, model)
    seen = {EMPTY}
    stack = [EMPTY]
    while stack:
        state = stack.pop()
        for target, _rate in neighbours(cache, state, n, model, pairs):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return cache, model, sorted(seen, key=sorted)


def test_reflecting_boundary_conserves_mass_where_absorbing_collapses():
    """The reason the default boundary is the reflecting one.

    Absorbing FSP charges for every excursion out of the retained set, including
    reversible ones that return microseconds later.  At 10^7 s^-1 that bound
    reaches 1.0 - valid, and saying nothing - before anything interesting has
    happened.
    """
    cache, model, _order = _enumerate(SMALL)
    horizon = 1e-3
    results = {}
    for boundary in ("absorbing", "reflecting"):
        solver = Solver(
            cache, model, tolerance=1e-3, prune_below=1e-12,
            max_states=20_000, boundary=boundary,
        )
        for _ in range(4):
            solver.advance(horizon / 4, per_round=512)
        results[boundary] = solver

    assert results["absorbing"].retained_mass < 0.5, (
        "the absorbing bound is expected to degenerate here; if it does not, "
        "the reflecting boundary has stopped being necessary"
    )
    assert results["reflecting"].retained_mass == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("sequence", ["GGCAUUGCAAGCAAU"])
def test_certificate_bounds_the_equilibrium_error(sequence):
    """The certificate, against the equilibrium error measured exactly.

    ``1 - Z_S/Z`` is computable without enumerating anything, because this
    domain has an exact partition function.  Here it is checked against the
    error measured on the fully enumerated space, and against ``Z`` recomputed
    by summing the enumerated Boltzmann weights - which also measures how far
    this package's energy model sits from the one supplying ``Z``.
    """
    pytest.importorskip("RNA")
    from rona.master.certify import certify, partition_function_energy

    cache, model, order = _enumerate(sequence)
    n = len(sequence)
    ensemble = partition_function_energy(sequence)
    summed = sum(math.exp(-(cache.of(s, n) - ensemble) / cache.kT) for s in order)
    # our energy model against ViennaRNA's partition function
    assert summed == pytest.approx(1.0, abs=5e-4)

    energies = np.array([cache.of(s, n) for s in order])
    weights = np.exp(-(energies - energies.min()) / cache.kT)
    equilibrium = weights / weights.sum()
    position = {state: i for i, state in enumerate(order)}

    previous = None
    for tolerance in (1e-2, 1e-3, 1e-4):
        solver = Solver(
            cache, model, tolerance=tolerance, prune_below=1e-12,
            max_states=20_000, integration_tolerance=1e-9,
        )
        for _ in range(4):
            solver.advance(0.1 / 4, per_round=2000, max_expansions=30)
        certificate = certify(solver)

        # the retained equilibrium, against the true one
        retained = np.zeros(len(order))
        for state in solver.states:
            retained[position[state]] = equilibrium[position[state]]
        retained = retained / retained.sum()
        measured = float(np.abs(retained - equilibrium).sum())
        assert measured <= certificate.l1 + 1e-3, (
            f"tol={tolerance}: equilibrium error {measured:.3e} exceeds "
            f"certificate {certificate.l1:.3e}"
        )
        # tightening the tolerance must not lose weight
        if previous is not None:
            assert certificate.outside <= previous + 1e-9
        previous = certificate.outside
    assert previous < 1e-3, "the tightest run should hold nearly all the weight"


@pytest.mark.parametrize(
    "sequence", ["GCGCAAAAGCGC", "GGCAUUGCAAGCC", "GGCGCUUGCGCAAAGCGCAAGCGCC"]
)
def test_loop_local_delta_equals_a_full_evaluation_everywhere(sequence):
    """Every rate in the generator, against two full energy evaluations.

    The move set scores a candidate with a loop-local delta rather than by
    evaluating the whole structure, which is what makes the solver fast enough
    to transcribe.  It is used as a pure function of the pair table - nothing is
    cached between calls, so there is no stale state - but it can still be
    *wrong*, and an incremental energy that silently disagrees with the real one
    is exactly the defect that dominated v1.  So it is checked exhaustively: for
    every state reachable in the chain and every move out of it, the rate must
    match the one implied by evaluating both structures in full.
    """
    model = MoveModel()
    cache = EnergyCache(FoldingEnergy(sequence))
    n = len(sequence)
    pairs = candidate_pairs(cache, n, model)
    # a bounded walk, not the whole space: for 25 nt the full space is millions
    # of structures and the point here is coverage of move *kinds*, not of states
    order = [EMPTY]
    seen = {EMPTY}
    frontier = [EMPTY]
    while frontier and len(order) < 300:
        state = frontier.pop(0)
        for target, _rate in neighbours(cache, state, n, model, pairs):
            if target not in seen:
                seen.add(target)
                order.append(target)
                frontier.append(target)
    worst = 0.0
    checked = 0
    for state in order:
        here = cache.of(state, n)
        for target, rate in neighbours(cache, state, n, model, pairs):
            expected = model.rate(cache.of(target, n) - here, cache.kT)
            if max(rate, expected) > 0.0:
                worst = max(worst, abs(rate - expected) / max(rate, expected))
            checked += 1
    assert checked > 80
    assert worst < 1e-9, f"loop-local delta disagrees with full evaluation: {worst:.2e}"
