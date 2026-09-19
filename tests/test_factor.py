"""The anchored decomposition, checked as an exact identity.

Everything in :mod:`rona.master.factor` rests on one claim: conditional on a
nested set of anchors being formed, the generator is the Kronecker sum of
per-region generators.  If that is exact then a product distribution is carried
forward exactly at a cost that is the sum rather than the product of the region
supports; if it is off by anything at all then the factorised solver is
reporting a number it has not computed.

So it is checked against a fully enumerated flat generator, for equality up to
double rounding - the nearest-neighbour model is a sum over loops and a region
boundary is a loop boundary, so the identity is exact in the model and the only
slack is the order the evaluator happens to add its terms in (~1e-15 relative).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity, kron

from rona.energy.evaluator import FoldingEnergy
from rona.master.cache import EnergyCache
from rona.master.factor import (
    NotNested,
    check_nested,
    decompose,
    region_candidates,
    region_of,
    rejoin,
    restrict,
)
from rona.master.generator import StateIndex, build_generator
from rona.master.moves import MoveModel, candidate_pairs, neighbours
from rona.master.structures import Structure, occupied

SEQ = "GGCGCAAAAGCGCAAGGCCAAAAGGCC"
ANCHORS_ONE = [(1, 12)]
ANCHORS_TWO = [(1, 12), (15, 26)]


def crosses(pairs, pair) -> bool:
    i, j = pair
    return any((a < i < b < j) or (i < a < j < b) for a, b in pairs)


def enumerate_block(pairs, base: Structure) -> list[Structure]:
    """Every nested structure that adds pairs drawn from ``pairs`` to ``base``."""
    out = [base]
    frontier = [base]
    seen = {base}
    while frontier:
        nxt = []
        for state in frontier:
            used = occupied(state)
            for pair in pairs:
                if pair[0] in used or pair[1] in used or crosses(state, pair):
                    continue
                grown = state | {pair}
                if grown not in seen:
                    seen.add(grown)
                    nxt.append(grown)
                    out.append(grown)
        frontier = nxt
    return out


def flat_block(sequence, anchors, model):
    """The flat generator over {S : anchors subset of S}, anchor moves dropped.

    Dropping the anchor removals is not a fudge: they are the transitions that
    leave the block, and the factorisation only ever claims to describe the
    block's interior.
    """
    cache = EnergyCache(FoldingEnergy(sequence))
    n = len(sequence)
    anchors = frozenset(anchors)
    free = [p for p in candidate_pairs(cache, n, model)
            if p not in anchors and not crosses(anchors, p)
            and p[0] not in occupied(anchors) and p[1] not in occupied(anchors)]
    states = enumerate_block(free, anchors)
    index = StateIndex.of(states)
    size = len(index)
    matrix = np.zeros((size, size))
    for column, state in enumerate(index.states):
        for target, rate in neighbours(cache, state, n, model):
            if not anchors <= target:
                continue                      # leaves the block
            matrix[index.position[target], column] += rate
            matrix[column, column] -= rate
    return cache, index, matrix


def kronecker_block(cache, sequence, anchors, model):
    """The same operator assembled region by region."""
    n = len(sequence)
    regions = decompose(anchors, n)
    blocks = []
    indices = []
    for region in regions:
        pairs = region_candidates(cache, anchors, region, n, model)
        states = enumerate_block(pairs, frozenset())
        index = StateIndex.of(states)
        size = len(index)
        block = np.zeros((size, size))
        for column, part in enumerate(index.states):
            joined = rejoin(anchors, [part])
            for target, rate in neighbours(cache, joined, n, model, pairs=pairs):
                row = index.position[target - frozenset(anchors)]
                block[row, column] += rate
                block[column, column] -= rate
        blocks.append(csr_matrix(block))
        indices.append(index)
    total = blocks[0]
    for block in blocks[1:]:
        total = kron(total, identity(block.shape[0])) + kron(
            identity(total.shape[0]), block
        )
    return regions, indices, total.toarray()


@pytest.mark.parametrize("anchors", [ANCHORS_ONE, ANCHORS_TWO])
def test_generator_restricted_to_a_block_is_the_kronecker_sum(anchors):
    model = MoveModel()
    cache, index, flat = flat_block(SEQ, anchors, model)
    regions, indices, product = kronecker_block(cache, SEQ, anchors, model)

    # The Kronecker sum is indexed by tuples of per-region parts; map those onto
    # the flat rows so the two matrices can be compared entry by entry.
    sizes = [len(i) for i in indices]
    assert np.prod(sizes) == len(index), (sizes, len(index))
    order = []
    for combination in itertools.product(*[range(s) for s in sizes]):
        parts = [indices[k].states[c] for k, c in enumerate(combination)]
        order.append(index.position[rejoin(anchors, parts)])
    rebuilt = np.zeros_like(flat)
    rebuilt[np.ix_(order, order)] = product
    scale = np.abs(flat).max()
    assert np.abs(rebuilt - flat).max() <= 1e-12 * scale
    assert (rebuilt == 0.0).sum() == (flat == 0.0).sum()   # same sparsity


@pytest.mark.parametrize("anchors", [ANCHORS_ONE, ANCHORS_TWO])
def test_energy_is_additive_across_regions(anchors):
    """``G`` splits over regions to the last bit, which is why the rates do."""
    model = MoveModel()
    cache = EnergyCache(FoldingEnergy(SEQ))
    n = len(SEQ)
    regions = decompose(anchors, n)
    base = cache.of(frozenset(anchors), n)
    spaces = [
        enumerate_block(region_candidates(cache, anchors, r, n, model), frozenset())
        for r in regions
    ]
    rng = np.random.default_rng(3)
    for _ in range(200):
        parts = [space[rng.integers(len(space))] for space in spaces]
        whole = cache.of(rejoin(anchors, parts), n)
        alone = sum(cache.of(rejoin(anchors, [p]), n) - base for p in parts)
        assert whole == pytest.approx(base + alone, abs=1e-9)


def test_no_admissible_pair_straddles_two_regions():
    model = MoveModel()
    cache = EnergyCache(FoldingEnergy(SEQ))
    n = len(SEQ)
    anchors = frozenset(ANCHORS_TWO)
    taken = occupied(anchors)
    regions = decompose(anchors, n)
    owned = {p for r in regions for p in
             region_candidates(cache, anchors, r, n, model)}
    for pair in candidate_pairs(cache, n, model):
        if pair in anchors or crosses(anchors, pair):
            continue
        if pair[0] in taken or pair[1] in taken:
            continue
        assert pair in owned, pair
        assert region_of(anchors, pair) == next(
            r.owner for r in regions
            if pair in region_candidates(cache, anchors, r, n, model)
        )


def test_restrict_and_rejoin_are_inverse():
    anchors = frozenset(ANCHORS_TWO)
    regions = decompose(anchors, len(SEQ))
    state = anchors | {(2, 11), (3, 10), (16, 25)}
    parts = restrict(state, anchors, regions)
    assert rejoin(anchors, parts) == state
    assert sum(len(p) for p in parts) + len(anchors) == len(state)


def test_crossing_anchors_are_rejected():
    with pytest.raises(NotNested):
        check_nested([(1, 12), (5, 20)])
    with pytest.raises(NotNested):
        check_nested([(1, 12), (12, 20)])


def test_region_solvers_reproduce_the_block_evolution():
    """Independent per-region solvers, multiplied out, equal the joint answer.

    The Kronecker-sum identity is about the operator; this is about the thing the
    solver actually reports.  Evolve the enumerated block directly, evolve each
    region on its own with the general solver in region mode, take the product of
    the results, and require agreement - so a mistake in the region plumbing
    (wrong base, wrong candidate set, an energy read without its context) cannot
    hide behind a correct theorem.
    """
    from rona.master.fsp import Solver
    from rona.master.integrate import integrate
    from rona.master.structures import EMPTY

    model = MoveModel()
    anchors = frozenset(ANCHORS_ONE)
    cache, index, flat = flat_block(SEQ, anchors, model)
    n = len(SEQ)
    dt = 1e-5

    start = np.zeros(len(index))
    start[index.position[anchors]] = 1.0
    joint, _error = integrate(csr_matrix(flat), start, dt, tolerance=1e-12)

    regions = decompose(anchors, n)
    solvers = []
    for region in regions:
        pairs = region_candidates(cache, anchors, region, n, model)
        solver = Solver(
            cache, model, length=n, tolerance=1e-12, prune_below=0.0,
            integration_tolerance=1e-12, base=anchors, allowed=pairs,
        )
        solver.advance(dt, max_expansions=64, per_round=4096)
        solvers.append(solver)

    product: dict[Structure, float] = {EMPTY: 1.0}
    for solver in solvers:
        grown: dict[Structure, float] = {}
        for state, weight in product.items():
            for part, p in zip(solver.states, solver.probability):
                if p > 0.0:
                    grown[state | part] = grown.get(state | part, 0.0) + weight * p
        product = grown

    got = np.zeros(len(index))
    for state, p in product.items():
        got[index.position[rejoin(anchors, [state])]] = p
    assert float(np.abs(got - joint).sum()) < 1e-6


def test_a_regions_certificate_is_zero_when_it_holds_everything():
    """The constrained ``Z`` must be over the region's own state space, exactly.

    Regression test for a silent and expensive mistake: ViennaRNA's
    ``hc_add_bp`` without ``CONSTRAINT_CONTEXT_ENFORCE`` *permits* the pair
    rather than requiring it, so the "constrained" partition function still sums
    over structures that omit the anchors.  A region pinned so tightly that its
    state space holds a single structure then certified at 0.29 instead of 0 -
    the certificate was dividing by a partition function over the wrong set, and
    reported a fifth of the ensemble as missing when nothing was.

    Pin every nucleotide and the certificate has no room to be anything but
    zero, which is what makes this a test rather than a tolerance check.
    """
    pytest.importorskip("RNA")
    from rona.master.fsp import Solver

    model = MoveModel()
    cache = EnergyCache(FoldingEnergy(SEQ))
    n = len(SEQ)
    anchors = frozenset(ANCHORS_TWO)
    # a region with no free nucleotides at all: one state, nothing outside it
    solver = Solver(cache, model, length=n, base=anchors, allowed=[])
    assert len(solver.states) == 1
    assert solver.certified_outside() == 0.0

    # and one small enough to enumerate: still exactly everything
    regions = decompose(anchors, n)
    inner = max(
        (r for r in regions if r.owner is not None),
        key=lambda r: len(r.positions),
    )
    pairs = region_candidates(cache, anchors, inner, n, model)
    full = Solver(cache, model, length=n, base=anchors, allowed=pairs,
                  tolerance=0.0, prune_below=0.0)
    for state in enumerate_block(pairs, frozenset()):
        full._add(state)
    full.probability = np.zeros(len(full.states))
    full.probability[0] = 1.0
    assert full.certified_outside() == pytest.approx(0.0, abs=5e-4)
