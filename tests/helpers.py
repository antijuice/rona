"""Shared helpers: exact state-space enumeration for the kinetic move set."""

from __future__ import annotations

import math

from rona.energy.evaluator import FoldingEnergy
from rona.energy.model import NearestNeighbourModel
from rona.energy.pseudoknot import PseudoknotModel
from rona.kinetics import KineticEngine, RateModel
from rona.moves import build_moveset
from rona.struct import Helix

StateKey = frozenset  # of (stem_index, i, j, length)


def make_engine(sequence, *, mode="helix", pk=False, rates=None, temperature=37.0):
    model = NearestNeighbourModel(temperature=temperature)
    pk_model = PseudoknotModel(enabled=pk)
    energy = FoldingEnergy(sequence, model, pk_model)
    moveset = build_moveset(model, energy.enc)
    engine = KineticEngine(
        energy,
        moveset,
        rates or RateModel(k_zip=3.0e5),
        pk_model=pk_model,
        mode=mode,
    )
    engine.grow(len(sequence), 0)
    return engine


def set_state(engine, key: StateKey) -> None:
    """Force the engine into a given state and invalidate every cached rate."""
    st = engine.state
    st.pt = [-1] * st.n_total
    st.formed = {}
    for stem_index, i, j, length in key:
        helix = Helix(i, j, length)
        for a, b in helix.pairs:
            st.pt[a], st.pt[b] = b, a
        st.formed[stem_index] = helix
    engine._dirty = set(range(engine.n_slots))
    engine._dyn_dirty = True
    # forcing an arbitrary state is not a move, so nothing the engine cached
    # about the previous one may be reused
    engine._dyn_cache.clear()
    engine._dyn_dirty_stems.clear()
    engine._update_crossings()
    # st.pt was replaced, so every derived table has to be rebuilt
    engine._rebuild_core()
    st.energy = engine.energy.energy(st.helices(), st.available)
    if getattr(engine, "mode", "") == "lumped":
        # in lumped mode the windows are a function of the stem set, so a
        # forced state has to be brought back onto that manifold
        engine._recanonicalise()


def state_key(engine) -> StateKey:
    return frozenset(
        (stem, h.i, h.j, h.length) for stem, h in engine.state.formed.items()
    )


def _move_destination(engine, key: StateKey, kind: str, ident: int, helix) -> StateKey | None:
    """Apply one move from ``key`` and read the state it lands in.

    ``ident`` is the *slot* for a form move and the stem index for every other
    kind; a stem may own several slots, and they are different moves reaching
    different states, so the slot is what identifies a form.

    The destination is never synthesised from the move's own helix: some modes
    adjust the rest of the structure when a move is applied, so the only
    reliable answer is the state the engine actually reaches.
    """
    set_state(engine, key)
    engine.propensity()
    if kind == "form":
        slot = ident
        if engine._fen.value(slot) <= 0.0:
            return None
        candidate = engine._slot_helix[slot]
        if candidate is None:
            return None
        from rona.kinetics import FORM, Move

        engine.apply(
            Move(
                FORM,
                candidate,
                engine._slot_stem[slot],
                engine._slot_dg[slot],
                engine._fen.value(slot),
            )
        )
        return state_key(engine)
    live = [
        m
        for m in engine._dyn
        if m.kind == kind and m.stem == ident and m.helix == helix
    ]
    if not live:
        return None
    engine.apply(live[0])
    return state_key(engine)


def neighbours(engine, key: StateKey) -> list[StateKey]:
    """Every state reachable in one move, using the engine's own move set."""
    set_state(engine, key)
    engine.propensity()
    proposals = [
        ("form", slot, engine._slot_helix[slot])
        for slot in range(engine.n_slots)
        if engine._fen.value(slot) > 0.0 and engine._slot_helix[slot] is not None
    ] + [(m.kind, m.stem, m.helix) for m in engine._dyn if m.rate > 0.0]

    out: list[StateKey] = []
    for kind, ident, helix in proposals:
        target = _move_destination(engine, key, kind, ident, helix)
        if target is not None:
            out.append(target)
    return out


def enumerate_states(engine, limit: int = 4000) -> list[StateKey]:
    """Breadth-first enumeration of every state reachable from the open chain."""
    start: StateKey = frozenset()
    seen = {start}
    frontier = [start]
    while frontier and len(seen) < limit:
        current = frontier.pop()
        for nxt in neighbours(engine, current):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return sorted(seen, key=lambda k: sorted(k))


def boltzmann(engine, states) -> dict:
    energies = {}
    for key in states:
        set_state(engine, key)
        energies[key] = engine.state.energy
    kT = engine.energy.kT
    weights = {k: math.exp(-e / kT) for k, e in energies.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}, energies


def sample_occupancy(engine, *, steps: int, seed: int) -> dict:
    """Time-weighted occupancy from a single long SSA run."""
    import random

    rng = random.Random(seed)
    set_state(engine, frozenset())
    dwell: dict[StateKey, float] = {}
    total_time = 0.0
    for _ in range(steps):
        rate = engine.propensity()
        if rate <= 0.0:
            break
        tau = rng.expovariate(rate)
        key = state_key(engine)
        dwell[key] = dwell.get(key, 0.0) + tau
        total_time += tau
        move = engine.select(rng.random())
        if move is None:
            break
        engine.apply(move)
    return {k: v / total_time for k, v in dwell.items()}


def outgoing_rates(engine, key: StateKey) -> dict:
    """Every transition out of ``key``, as ``{destination: rate}``.

    Reads the engine's own propensities rather than recomputing them, so the
    check below tests what the simulation actually uses.
    """
    out: dict[StateKey, float] = {}
    set_state(engine, key)
    engine.propensity()
    proposals = [
        ("form", slot, engine._slot_helix[slot], engine._fen.value(slot))
        for slot in range(engine.n_slots)
        if engine._fen.value(slot) > 0.0 and engine._slot_helix[slot] is not None
    ] + [(m.kind, m.stem, m.helix, m.rate) for m in engine._dyn if m.rate > 0.0]

    for kind, ident, helix, rate in proposals:
        target = _move_destination(engine, key, kind, ident, helix)
        if target is None:
            continue
        out[target] = out.get(target, 0.0) + rate
    return out


def state_energies(engine, states) -> dict:
    energies = {}
    for key in states:
        set_state(engine, key)
        energies[key] = engine.state.energy
    return energies


def total_variation(observed, expected) -> float:
    keys = set(observed) | set(expected)
    return 0.5 * sum(abs(observed.get(k, 0.0) - expected.get(k, 0.0)) for k in keys)
