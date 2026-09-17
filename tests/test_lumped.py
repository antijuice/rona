"""The lumped chain: windows removed as a degree of freedom.

Lumped mode replaces a helix's window (where inside its stem the ladder sits,
and how long it is) with a deterministic function of the set of formed stems.
That is only legitimate if three things hold, and each is checked here:

* the window map really is a function of the *set*, not of the order the stems
  arrived in - otherwise the free energy is not a state function at all;
* the resulting chain is detailed-balanced, transition by transition;
* the distribution it converges to is the microscopic one, marginalised onto
  stem sets.  This is the only approximation in the mode, and it is measured
  rather than assumed.
"""

from __future__ import annotations

import math

import pytest

from helpers import (
    enumerate_states,
    make_engine,
    outgoing_rates,
    state_energies,
    state_key,
)
from rona.energy.evaluator import FoldingEnergy
from rona.energy.model import NearestNeighbourModel
from rona.energy.pseudoknot import PseudoknotModel
from rona.kinetics import FORM, MELT, RateModel
from rona.lumped import LumpedEngine, canonical_windows
from rona.moves import build_moveset
from rona.struct import Helix

SEQUENCES = [
    "GCGCAAAAGCGCAAAAGCGC",
    "GGCAUUGCAAGCAAUGCCAA",
    "GGGAAACCCAAAGGGAAACCCAAAA",
    "GCGGAUUUAGCUCAGUUGGGAGAGC",
]


def _reference(sequence, pk):
    """The reference engine, which scores every candidate by full evaluation."""
    model = NearestNeighbourModel()
    pk_model = PseudoknotModel(enabled=pk)
    energy = FoldingEnergy(sequence, model, pk_model)
    moveset = build_moveset(model, energy.enc)
    engine = LumpedEngine(energy, moveset, RateModel(k_zip=3.0e5))
    engine.grow(len(sequence), 0)
    return engine


def _reference_chain(engine):
    """``({state: {state: rate}}, {state: energy})`` over stem sets."""

    def load(stems):
        state = engine.state
        state.windows = engine._windows_for(stems)
        state.stems = set(state.windows)
        state.energy = engine._energy_of(state.windows)
        engine._dirty = True

    seen = {frozenset()}
    frontier = [frozenset()]
    rates: dict[frozenset, dict[frozenset, float]] = {}
    while frontier:
        current = frontier.pop()
        load(current)
        engine.propensity()
        out: dict[frozenset, float] = {}
        for move in engine._moves:
            if move.kind == FORM:
                target = frozenset(current | {move.stem})
            elif move.kind == MELT:
                target = frozenset(current - {move.stem})
            else:
                target = frozenset((current - {move.other}) | {move.stem})
            out[target] = out.get(target, 0.0) + move.rate
            if target not in seen:
                seen.add(target)
                frontier.append(target)
        rates[current] = out
    energies = {}
    for key in seen:
        load(key)
        energies[key] = engine.state.energy
    return rates, energies


def _stem_set(key):
    return frozenset(stem for stem, *_ in key)


def _marginal(distribution):
    out: dict[frozenset, float] = {}
    for key, value in distribution.items():
        label = _stem_set(key)
        out[label] = out.get(label, 0.0) + value
    return out


def _boltzmann(engine, states):
    energies = state_energies(engine, states)
    kT = engine.energy.kT
    weights = {k: math.exp(-e / kT) for k, e in energies.items()}
    total = math.fsum(weights.values())
    return {k: w / total for k, w in weights.items()}, energies, kT


# ----------------------------------------------------------------------
@pytest.mark.parametrize("sequence", SEQUENCES)
def test_windows_are_a_function_of_the_set_not_the_order(sequence):
    """Permuting the input must not move a single helix.

    If it did, the free energy would depend on folding history and the whole
    detailed-balance argument for this mode would collapse.
    """
    import random

    model = NearestNeighbourModel()
    energy = FoldingEnergy(sequence, model, PseudoknotModel(enabled=True))
    moveset = build_moveset(model, energy.enc)
    rng = random.Random(4)
    indices = list(range(len(moveset.stems)))
    for _ in range(40):
        size = rng.randint(1, min(4, len(indices)))
        subset = rng.sample(indices, size)
        reference = canonical_windows(moveset, subset, len(sequence), 3)
        for _ in range(3):
            rng.shuffle(subset)
            assert canonical_windows(moveset, subset, len(sequence), 3) == reference


@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.parametrize("pk", [False, True])
def test_lumped_transitions_satisfy_detailed_balance(sequence, pk):
    engine = make_engine(sequence, mode="lumped", pk=pk)
    states = enumerate_states(engine)
    energies = state_energies(engine, states)
    kT = engine.energy.kT
    rates = {key: outgoing_rates(engine, key) for key in states}
    checked = 0
    for source in states:
        for target, forward in rates[source].items():
            reverse = rates[target].get(source, 0.0)
            assert reverse > 0.0, f"one-way transition {source} -> {target}"
            expected = math.exp(-(energies[target] - energies[source]) / kT)
            assert (forward / reverse) == pytest.approx(expected, rel=1e-6)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.parametrize("pk", [False, True])
def test_fast_lumped_mode_matches_the_reference_engine(sequence, pk):
    """The incremental engine must reproduce the full-evaluation chain exactly.

    Lumped mode reuses the microscopic engine's loop-local energy deltas and
    invalidation machinery.  The reference engine in :mod:`rona.lumped` uses
    none of it - every candidate is scored by a full O(n) evaluation - so
    agreement here is a check on the fast path, not on the physics.
    """
    fast = make_engine(sequence, mode="lumped", pk=pk)
    states = enumerate_states(fast)
    fast_rates = {
        _stem_set(k): {_stem_set(t): r for t, r in outgoing_rates(fast, k).items()}
        for k in states
    }
    fast_energies = {_stem_set(k): e for k, e in state_energies(fast, states).items()}

    reference = _reference(sequence, pk)
    ref_rates, ref_energies = _reference_chain(reference)

    assert set(fast_rates) == set(ref_rates)
    for source, targets in fast_rates.items():
        assert set(targets) == set(ref_rates[source])
        for target, rate in targets.items():
            assert rate == pytest.approx(ref_rates[source][target], rel=1e-9)
    for key, value in fast_energies.items():
        assert value == pytest.approx(ref_energies[key], abs=1e-9)


@pytest.mark.parametrize("sequence", SEQUENCES)
@pytest.mark.parametrize("pk", [False, True])
def test_lumped_distribution_matches_the_microscopic_marginal(sequence, pk):
    """The cost of the approximation, in the only currency that matters.

    Both chains are enumerated exactly and their equilibrium distributions are
    compared after marginalising the microscopic one onto stem sets.  The
    discarded quantity is the window entropy of each helix; it is small because
    a helix end is bound too tightly to wander.
    """
    micro = make_engine(sequence, mode="helix", pk=pk)
    lumped = make_engine(sequence, mode="lumped", pk=pk)
    micro_p, _, _ = _boltzmann(micro, enumerate_states(micro))
    lumped_p, _, _ = _boltzmann(lumped, enumerate_states(lumped))
    a, b = _marginal(micro_p), _marginal(lumped_p)
    tv = 0.5 * math.fsum(
        abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)
    )
    assert tv < 0.02


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_no_helix_is_ever_stranded(sequence):
    """Every formed helix sits at its canonical window, after every event.

    Stranding at a stale window - a helix stuck at nucleation length because
    nothing ever re-examined it - was the defect that forced zipping into the
    microscopic move set.  Here it is structurally impossible, and this is the
    check on that claim.
    """
    import random

    rng = random.Random(11)
    engine = make_engine(sequence, mode="lumped", pk=True)
    for _ in range(400):
        rate = engine.propensity()
        if rate <= 0.0:
            break
        move = engine.select(rng.random())
        if move is None:
            break
        engine.apply(move)
        state = engine.state
        expected = canonical_windows(
            engine.moveset, state.formed.keys(), state.available, engine.min_helix
        )
        assert state.formed == expected
        # and the pair table agrees with the windows
        pt = [-1] * state.n_total
        for helix in state.formed.values():
            for i, j in helix.pairs:
                pt[i], pt[j] = j, i
        assert pt == state.pt
        assert state.energy == pytest.approx(
            engine.energy.energy(state.helices(), state.available), abs=1e-9
        )


def test_lumped_mode_respects_the_pseudoknot_switch():
    sequence = "GCGGAUUUAGCUCAGUUGGGAGAGC"
    for pk in (False, True):
        engine = make_engine(sequence, mode="lumped", pk=pk)
        crossing = 0
        for key in enumerate_states(engine):
            helices = [Helix(i, j, length) for _stem, i, j, length in key]
            crossing += any(
                a.crosses(b)
                for index, a in enumerate(helices)
                for b in helices[index + 1 :]
            )
        assert (crossing > 0) is pk


def test_fast_placement_agrees_with_the_definition():
    """The O(ladder) placement must equal a full canonical placement, always.

    ``_lumped_form_window`` reads a stem's window off an owner array rather
    than re-placing the whole structure, and reports whether anything else
    moved; ``_lumped_melt_local`` answers the same question for a melt.  Both
    are optimisations of :func:`canonical_windows`, and this is the check that
    they are only that.
    """
    import random

    sequence = (
        "GGCGCAAGCCAUUGGCUUAGCGCCAAAGGCAUUGCAAGCAAUGCCAAGGGAAACCCAAAG"
    )
    engine = make_engine(sequence, mode="lumped", pk=True)
    rng = random.Random(7)
    stems = range(len(engine.moveset.stems))
    checked = 0
    concerted = 0
    for _ in range(300):
        subset = rng.sample(sorted(stems), rng.randint(0, 5))
        windows = canonical_windows(
            engine.moveset, subset, len(sequence), engine.min_helix
        )
        # only reachable states have a definition to compare against
        if set(windows) != set(subset):
            continue
        engine.state.formed = dict(windows)
        engine.state.pt = [-1] * len(sequence)
        for helix in windows.values():
            for i, j in helix.pairs:
                engine.state.pt[i], engine.state.pt[j] = j, i
        engine._rebuild_owner()

        for candidate in stems:
            if candidate in windows:
                continue
            trial = canonical_windows(
                engine.moveset,
                list(subset) + [candidate],
                len(sequence),
                engine.min_helix,
            )
            found = engine._lumped_form_window(candidate)
            if candidate not in trial or len(trial) != len(windows) + 1:
                assert found is None
            else:
                moved = any(trial.get(k) != v for k, v in windows.items())
                assert found is not None
                helix, reported = found
                assert helix == trial[candidate]
                if moved:
                    concerted += 1
                    assert reported == trial
                else:
                    assert reported is None
            checked += 1
        for member in windows:
            rest = [k for k in subset if k != member]
            after = canonical_windows(
                engine.moveset, rest, len(sequence), engine.min_helix
            )
            expected = all(after.get(k) == v for k, v in windows.items() if k != member)
            assert engine._lumped_melt_local(member) is expected
            checked += 1
    assert checked > 1000
    # the concerted case is the interesting one; it must actually be exercised
    assert concerted > 0


def _master_equation(engine, times):
    """``P(t)`` for each t, from the open chain, over the enumerated states."""
    import numpy as np

    states = enumerate_states(engine)
    position = {key: i for i, key in enumerate(states)}
    size = len(states)
    generator = np.zeros((size, size))
    for source in states:
        for target, rate in outgoing_rates(engine, source).items():
            generator[position[target], position[source]] += rate
            generator[position[source], position[source]] -= rate
    p = np.zeros(size)
    p[position[frozenset()]] = 1.0
    out = []
    previous = 0.0
    for moment in times:
        step = moment - previous
        scale = np.abs(generator).max() * step
        squarings = (
            max(0, int(math.ceil(math.log2(scale / 0.25)))) if scale > 0.25 else 0
        )
        small = step / (2**squarings)
        matrix = np.eye(size)
        term = np.eye(size)
        for order in range(1, 18):
            term = term @ (generator * small) / order
            matrix = matrix + term
        for _ in range(squarings):
            matrix = matrix @ matrix
        p = matrix @ p
        out.append(p.copy())
        previous = moment
    return states, out


TIMES = (1e-4, 1e-2, 1.0, 10.0)


@pytest.mark.parametrize("sequence", SEQUENCES[:3])
def test_lumped_kinetics_track_the_microscopic_kinetics(sequence):
    """Not just the equilibrium: the whole time course.

    Both master equations are solved exactly and compared at every time, which
    is the only way to see whether lumping has changed a *rate*.  It is what
    caught the barrier being wrong twice - first missing the nucleation
    transition state altogether, then placing it at a full melt rather than at
    the saddle of a zipper trade.
    """
    micro = make_engine(sequence, mode="helix", pk=False)
    lumped = make_engine(sequence, mode="lumped", pk=False)
    micro_states, micro_p = _master_equation(micro, TIMES)
    lumped_states, lumped_p = _master_equation(lumped, TIMES)
    micro_labels = [_stem_set(k) for k in micro_states]
    lumped_labels = [_stem_set(k) for k in lumped_states]
    worst = 0.0
    for pm, pl in zip(micro_p, lumped_p):
        a: dict[frozenset, float] = {}
        b: dict[frozenset, float] = {}
        for label, value in zip(micro_labels, pm):
            a[label] = a.get(label, 0.0) + value
        for label, value in zip(lumped_labels, pl):
            b[label] = b.get(label, 0.0) + value
        worst = max(
            worst,
            0.5 * math.fsum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)),
        )
    # the transient lags, because a whole helix appears in one lumped event
    # where the microscopic chain has to zip it
    assert worst < 0.25
    # by the end both must agree closely
    assert (
        0.5
        * math.fsum(
            abs(x - y)
            for x, y in (
                (
                    sum(v for lab, v in zip(micro_labels, micro_p[-1]) if lab == key),
                    sum(v for lab, v in zip(lumped_labels, lumped_p[-1]) if lab == key),
                )
                for key in {*micro_labels, *lumped_labels}
            )
        )
        < 0.05
    )


def test_lumped_mode_escapes_a_kinetic_trap():
    """The load-bearing test of the barrier model.

    A designed trap: a local 8 bp hairpin forms first and has to be traded for
    an 11 bp helix that shares its nucleotides.  In the lumped chain the two
    cannot coexist, so if the only route between them were a full melt the trap
    would never resolve - which is exactly what the first two barrier models
    did, freezing the ensemble at its initial split forever.  The microscopic
    simulator resolves it because the incumbent retracts pair by pair while the
    challenger zips in; the barrier here has to reproduce that.
    """
    sequence = "GGACGCAGAAAACUGCGUCCUUAAUAAUAAUAAUAAGGACGCAG"
    times = (0.1, 1.0, 10.0)

    def trapped(engine):
        states, courses = _master_equation(engine, times)
        out = []
        for p in courses:
            total = 0.0
            for key, value in zip(states, p):
                pairs = [
                    (i + d, j - d)
                    for _stem, i, j, length in key
                    for d in range(length)
                ]
                if pairs and not any(i < 20 and j > 35 for i, j in pairs):
                    total += value
            out.append(total)
        return out

    micro = trapped(make_engine(sequence, mode="helix", pk=False))
    lumped = trapped(make_engine(sequence, mode="lumped", pk=False))
    # both start trapped, and both must get out
    assert micro[0] > 0.3 and lumped[0] > 0.3
    assert micro[-1] < 0.1 and lumped[-1] < 0.1
    for a, b in zip(micro, lumped):
        assert abs(a - b) < 0.15
