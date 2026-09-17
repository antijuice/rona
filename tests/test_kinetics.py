"""The simulator must reproduce the Boltzmann ensemble at fixed chain length.

This is the load-bearing test of the whole package: the move set and the rate
rule are only trustworthy if a long fixed-length run recovers the equilibrium
distribution that the energy model defines.  A cotranscriptional run is then
the *same* machinery with the chain growing underneath it.
"""

from __future__ import annotations

import random

import pytest

import math

from helpers import (
    boltzmann,
    enumerate_states,
    make_engine,
    outgoing_rates,
    sample_occupancy,
    set_state,
    state_energies,
    state_key,
    total_variation,
)
from rona.kinetics import Fenwick, RateModel


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6, 7, 8, 15, 16, 17, 100, 257])
def test_fenwick_total_and_sampling(size):
    rng = random.Random(size)
    tree = Fenwick(size)
    values = [0.0] * size
    for _ in range(200):
        index = rng.randrange(size)
        value = rng.random() * 10.0
        tree.set(index, value)
        values[index] = value
        assert tree.total() == pytest.approx(sum(values))
    total = sum(values)
    for _ in range(1000):
        target = rng.random() * total
        index = tree.find(target)
        low = sum(values[:index])
        assert low <= target < low + values[index] + 1e-12


@pytest.mark.parametrize(
    "sequence,mode,pk", [("GCGCAAAAGCGCAAAAGCGC", "helix", False)]
)
def test_stationary_distribution_is_boltzmann(sequence, mode, pk):
    engine = make_engine(sequence, mode=mode, pk=pk)
    states = enumerate_states(engine)
    expected, _energies = boltzmann(engine, states)
    # Zipping fires orders of magnitude more often than the moves that change
    # the coarse state, so convergence in *events* is slow.  The exact
    # per-transition check above is the rigorous statement and is cheap; this
    # is only an end-to-end check that the sampler uses those rates, so it runs
    # short and with a loose tolerance.
    observed = sample_occupancy(engine, steps=150_000, seed=17)

    # every sampled state must be one the enumeration found
    assert set(observed) <= set(states)
    # restrict the comparison to states carrying appreciable weight; rare states
    # are not sampled reliably in a finite run
    heavy = {k: v for k, v in expected.items() if v > 1e-3}
    scale = sum(heavy.values())
    heavy = {k: v / scale for k, v in heavy.items()}
    seen = {k: observed.get(k, 0.0) for k in heavy}
    seen_scale = sum(seen.values()) or 1.0
    seen = {k: v / seen_scale for k, v in seen.items()}
    assert total_variation(seen, heavy) < 0.15


@pytest.mark.parametrize(
    "sequence,mode,pk",
    [
        ("GCGCAAAAGCGCAAAAGCGC", "helix", False),
        ("GGCAUUGCAAGCAAUGCCAAAGGCAUU", "helix", False),
        ("GCGCAAAAGCGCAAAAGCGC", "breathe", False),
        ("GGCGAAAGCCAAAAGGCGAAAGCC", "helix", True),
    ],
)
def test_every_transition_satisfies_detailed_balance(sequence, mode, pk):
    """The exact statement: k(X->Y)/k(Y->X) = exp(-(G_Y - G_X)/RT), every pair.

    Far more sensitive than comparing sampled occupancies, and it localises any
    violation to a specific transition.  It is what caught the melt rule being
    a one-way door for a helix that had not zipped to its stem's full extent.
    """
    engine = make_engine(sequence, mode=mode, pk=pk)
    states = enumerate_states(engine)
    energies = state_energies(engine, states)
    kT = engine.energy.kT
    rates = {key: outgoing_rates(engine, key) for key in states}

    checked = 0
    for source in states:
        for target, forward in rates[source].items():
            if target not in energies:
                continue
            reverse = rates[target].get(source, 0.0)
            assert reverse > 0.0, (
                f"{mode}: transition exists in one direction only "
                f"({source} -> {target})"
            )
            expected = math.exp(-(energies[target] - energies[source]) / kT)
            assert (forward / reverse) == pytest.approx(expected, rel=1e-6)
            checked += 1
    assert checked > 0


def test_a_short_helix_can_always_grow():
    """A helix shorter than its stem's free window must have a growth move.

    Helices routinely nucleate while the 3' end is still inside the polymerase,
    so this situation is common - it arose in a third of visited states before
    zipping was added to the helix move set.  Without a growth move such a
    helix could never extend, which is wrong physically, and melting it would
    have no inverse, which breaks detailed balance.
    """
    import random

    from rona.moves import ZIP_IN, ZIP_OUT

    sequence = "GGCGCGGCACCGUCCGCGGAACAAACGGAGAAGGGGCCGCCGAAAGGCGGCCUUUUUU"
    engine = make_engine(sequence, mode="helix")
    rng = random.Random(7)
    short_states = 0
    for _ in range(3000):
        if engine.propensity() <= 0.0:
            break
        for stem_index, helix in engine.state.formed.items():
            window = engine._candidate(engine._stem_slot[stem_index], ignore_own=True)
            if window is None or window.length <= helix.length:
                continue
            short_states += 1
            grows = [
                m
                for m in engine._dyn
                if m.stem == stem_index and m.kind in (ZIP_IN, ZIP_OUT)
            ]
            assert grows, (
                f"helix {helix} can reach {window.length} bp but has no growth move"
            )
        move = engine.select(rng.random())
        if move is None:
            break
        engine.apply(move)
    assert short_states > 50, "the situation under test did not arise"


@pytest.mark.parametrize("scheme", ["metropolis", "kawasaki"])
def test_rate_rule_satisfies_detailed_balance(scheme):
    """k(A->B)/k(B->A) must equal exp(-dG/RT) for every move kind."""
    rates = RateModel(scheme=scheme)
    kT = 0.6156
    for dg in (-8.0, -2.5, -0.1, 0.0, 0.3, 4.0, 11.0):
        forward = rates.rate("form", dg, kT)
        reverse = rates.rate("melt", -dg, kT)
        assert forward / reverse == pytest.approx(pow(2.718281828459045, -dg / kT), rel=1e-9)
        zf = rates.rate("zip_in", dg, kT)
        zr = rates.rate("unzip_in", -dg, kT)
        assert zf / zr == pytest.approx(pow(2.718281828459045, -dg / kT), rel=1e-9)


def test_energy_tracking_matches_full_recomputation():
    """The incrementally tracked energy must not drift over a long run."""
    engine = make_engine("GGCGAAAGCCAAAAGGCGAAAGCCUUUUGCAUGCAAAGCAUGC", pk=True)
    rng = random.Random(4)
    for _ in range(4000):
        if engine.propensity() <= 0.0:
            break
        move = engine.select(rng.random())
        if move is None:
            break
        engine.apply(move)
        exact = engine.energy.energy(engine.state.helices(), engine.state.available)
        assert engine.state.energy == pytest.approx(exact, abs=1e-9)


def test_state_reconstruction_round_trips():
    engine = make_engine("GCGCAAAAGCGCAAAAGCGC")
    rng = random.Random(2)
    for _ in range(50):
        engine.propensity()
        move = engine.select(rng.random())
        if move is None:
            break
        engine.apply(move)
    key = state_key(engine)
    energy = engine.state.energy
    set_state(engine, key)
    assert state_key(engine) == key
    assert engine.state.energy == pytest.approx(energy)
