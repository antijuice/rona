"""Basin coarse-graining: a reporting transform that must not invent anything.

It exists so this package can be compared against tools that report macrostates,
so its only job is to be a well-defined, order-independent function that conserves
probability.  Each of those is a way it could quietly go wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from rona.energy.evaluator import FoldingEnergy
from rona.master.cache import EnergyCache
from rona.master.coarse import basins, local_minimum
from rona.master.moves import MoveModel, neighbours
from rona.master.structures import EMPTY, from_dotbracket

SEQ = "GGCGCAAAAGCGCCAAAGGGAAACCC"


def walk(limit=400):
    """A bounded walk over the state space, for exhaustive-ish checks."""
    model = MoveModel()
    cache = EnergyCache(FoldingEnergy(SEQ))
    n = len(SEQ)
    order = [EMPTY]
    seen = {EMPTY}
    index = 0
    while index < len(order) and len(order) < limit:
        for target, _rate in neighbours(cache, order[index], n, model):
            if target not in seen:
                seen.add(target)
                order.append(target)
        index += 1
    return cache, model, n, order


def test_the_image_really_is_a_local_minimum():
    """Nothing downhill of it, in the same move set the dynamics uses."""
    cache, model, n, states = walk()
    memo = {}
    for state in states:
        bottom = local_minimum(cache, state, n, model, memo=memo)
        here = cache.of(bottom, n)
        for target, _rate in neighbours(cache, bottom, n, model):
            assert cache.of(target, n) >= here - 1e-9, (bottom, target)


def test_a_local_minimum_is_its_own_basin():
    cache, model, n, states = walk()
    memo = {}
    for state in states:
        bottom = local_minimum(cache, state, n, model, memo=memo)
        assert local_minimum(cache, bottom, n, model) == bottom


def test_the_map_does_not_depend_on_the_order_it_is_asked_in():
    """Ties are broken deterministically, not by set iteration order.

    v1 had a free energy that depended on the order helices happened to be stored
    in, which broke detailed balance silently.  A basin map with an arbitrary
    tie-break is the same class of defect, so it is checked rather than assumed.
    """
    cache, model, n, states = walk(200)
    forward = {s: local_minimum(cache, s, n, model) for s in states}
    backward = {s: local_minimum(cache, s, n, model) for s in reversed(states)}
    assert forward == backward


def test_grouping_conserves_probability():
    cache, model, n, states = walk(200)
    rng = np.random.default_rng(11)
    probability = rng.random(len(states))
    probability /= probability.sum()
    grouped = basins(cache, states, probability, n, model)
    assert sum(grouped.values()) == pytest.approx(1.0, abs=1e-12)
    assert len(grouped) <= len(states)


def test_the_memo_gives_the_same_answers_as_no_memo():
    """The walk caches every structure on the path, which is easy to get wrong."""
    cache, model, n, states = walk(200)
    memo = {}
    for state in states:
        assert (local_minimum(cache, state, n, model, memo=memo)
                == local_minimum(cache, state, n, model))


def test_a_truncated_helix_descends_to_the_full_one():
    cache, model, n, _states = walk(1)
    short = from_dotbracket(".((((....))))" + "." * (n - 13))
    bottom = local_minimum(cache, short, n, model)
    assert cache.of(bottom, n) < cache.of(short, n)
    assert len(bottom) > len(short)
