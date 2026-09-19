"""The adaptive anchored solver, against the flat one it is meant to replace.

Factorising is an approximation, so the only question that matters is whether the
error it *reports* covers the error it *makes*.  Every test here is a comparison
against the flat solver on a problem small enough to do both ways, or an exact
identity the representation must satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest

from rona.energy.evaluator import FoldingEnergy
from rona.master.anchored import Anchored
from rona.master.fsp import Solver
from rona.master.moves import MoveModel
from rona.master.structures import dotbracket

SEQ = "GGCGCAAAAGCGCCAAAGGGAAACCC"
#: Long enough that the ensemble has relaxed and its dominant helix is certain,
#: which is when anchoring is meant to happen; at 1e-3 s nothing is certain yet
#: and every promotion is correctly refused, so the tests would pass vacuously.
DT = 0.05
TOLERANCE = 1e-3


def flat_after(sequence, length, dt, tolerance):
    solver = Solver(FoldingEnergy(sequence), length=length, tolerance=tolerance)
    solver.advance(dt, max_expansions=200, per_round=1024)
    return {
        dotbracket(state, length): float(p)
        for state, p in zip(solver.states, solver.probability)
        if p > 0.0
    }, solver


def anchored_after(sequence, length, dt, tolerance, *, reviews=3, budget=1e-2):
    ensemble = Anchored.of(sequence, length=length, tolerance=tolerance,
                           promote_budget=budget)
    ensemble.advance(dt, max_expansions=200, per_round=1024)
    for _ in range(reviews):
        if not ensemble.review():
            break
        ensemble.advance(dt, max_expansions=200, per_round=1024)
    return ensemble


def test_with_no_anchors_it_is_the_flat_solver():
    """The unanchored case must be the flat case, not merely close to it."""
    flat, _solver = flat_after(SEQ, len(SEQ), DT, TOLERANCE)
    ensemble = Anchored.of(SEQ, length=len(SEQ), tolerance=TOLERANCE)
    ensemble.advance(DT, max_expansions=200, per_round=1024)
    assert ensemble.anchors == frozenset()
    got = dict(ensemble.distribution(top=10_000))
    assert set(got) == set(flat)
    for key, value in flat.items():
        assert got[key] == pytest.approx(value, rel=1e-12, abs=1e-15)


def test_a_promote_release_round_trip_costs_exactly_what_it_reported():
    """Releasing is lossless; it is not an undo, and the difference is measurable.

    Multiplying the two factors back out reconstructs a joint distribution over
    the merged region exactly - but the *product* of the marginals, not the joint
    that was there before.  So a round trip leaves behind precisely the error the
    promotion reported, which makes ``Promotion.factorised`` checkable rather
    than merely plausible.  An earlier version of this test asserted the round
    trip was exact and was simply wrong about what the operation does.
    """
    ensemble = anchored_after(SEQ, len(SEQ), DT, TOLERANCE, reviews=0, budget=1.0)
    marginals = ensemble.marginals(None)
    pair = max(marginals, key=lambda q: marginals[q])
    before = dict(ensemble.distribution(top=10_000))
    record = ensemble.promote(None, pair)
    assert record is not None, "expected the most certain pair to be anchorable"
    assert record.factorised > 0.0, "a lossless split would not test anything"
    ensemble.demote(pair)
    assert ensemble.anchors == frozenset()
    after = dict(ensemble.distribution(top=10_000))
    error = sum(abs(before.get(k, 0.0) - after.get(k, 0.0))
                for k in set(before) | set(after))
    assert error == pytest.approx(record.total, rel=1e-9)


def test_the_reported_slack_covers_the_error_actually_made():
    """Anchored against flat, L1, versus what the promotions said they cost."""
    flat, solver = flat_after(SEQ, len(SEQ), DT, TOLERANCE)
    ensemble = anchored_after(SEQ, len(SEQ), DT, TOLERANCE)
    assert ensemble.anchors, "nothing was anchored, so nothing is being tested"
    got = dict(ensemble.distribution(top=10_000))
    keys = set(flat) | set(got)
    error = sum(abs(flat.get(k, 0.0) - got.get(k, 0.0)) for k in keys)
    # the flat run has its own truncation error, so the comparison is against
    # the reported slack plus what the flat solver admits to missing
    allowance = ensemble.slack + solver.certified_outside() + TOLERANCE
    assert error <= allowance, (error, ensemble.slack, allowance)


def test_a_promotion_that_would_cost_too_much_is_refused():
    """The budget has to bite, or 'measured error' means nothing."""
    ensemble = anchored_after(SEQ, len(SEQ), DT, TOLERANCE, reviews=0)
    marginals = ensemble.marginals(None)
    # the least certain pair: conditioning on it discards most of the ensemble
    pair = min(marginals, key=lambda q: marginals[q])
    assert marginals[pair] < 0.5
    assert ensemble.promote(None, pair) is None
    assert ensemble.anchors == frozenset()
    assert ensemble.slack == 0.0


def test_stored_states_are_the_sum_and_represented_are_the_product():
    ensemble = anchored_after(SEQ, len(SEQ), DT, TOLERANCE)
    stored = sum(len(s.states) for s in ensemble.parts.values())
    product = 1
    for solver in ensemble.parts.values():
        product *= len(solver.states)
    assert ensemble.states == stored
    assert ensemble.represented == product
    assert ensemble.represented >= ensemble.states


def test_the_mass_that_went_missing_is_the_mass_that_was_accounted_for():
    """Conditioning is visible, not renormalised away.

    The factors sum to less than one by exactly the probability that promotions
    discarded, so this is a check on the bookkeeping that does not go through the
    bookkeeping.
    """
    ensemble = anchored_after(SEQ, len(SEQ), DT, TOLERANCE)
    lost = 1.0 - ensemble.retained_mass
    assert lost > 0.0, "nothing was conditioned, so nothing is being checked"
    conditioned = sum(record.conditioned for record in ensemble.promotions)
    # Summing the promotions over-counts slightly and in the safe direction:
    # successive conditionings compound, so two losses of a and b leave
    # 1 - (1 - a)(1 - b) = a + b - ab, and pruning between them removes a little
    # more.  What must hold is that the running total never *under*-states what
    # actually went missing.
    assert lost <= conditioned + 1e-12
    assert lost == pytest.approx(conditioned, rel=1e-3)
    assert lost <= ensemble.slack + 1e-12


def test_a_growing_region_picks_up_the_new_nucleotides():
    """Transcription must reach the outermost region, or new pairs never appear."""
    ensemble = Anchored.of(SEQ, length=12, tolerance=TOLERANCE)
    before = len(ensemble.parts[None]._pairs())
    ensemble.grow(len(SEQ))
    assert max(ensemble.parts[None].within) == len(SEQ) - 1
    assert len(ensemble.parts[None]._pairs()) > before
