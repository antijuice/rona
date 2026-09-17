"""Aggregation of many trajectories into a time-resolved kinetic ensemble.

A single trajectory is one realisation of a stochastic process and says very
little on its own.  The object of interest is the *ensemble*: the distribution
over structures at each moment during and after transcription.

:class:`Ensemble` stores the sampled structures of every trajectory on a shared
time grid and derives from them

* ``occupancy`` - the population of each distinct structure as a function of
  time, which is what a folding-pathway plot shows;
* ``pair_probabilities`` - ``P(i, j, t)``, the probability that ``i`` and ``j``
  are paired at time ``t``;
* ``dominant`` - the most populated structure at each time point;
* summary series such as mean free energy, pseudoknot fraction and the
  population of any user-supplied target structure.

None of these are equilibrium quantities.  ``occupancy`` at time ``t`` is the
distribution the *kinetics* has actually reached by ``t``, which for a
sequence with a folding trap differs sharply from the Boltzmann distribution of
the same chain length.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .cotrans import (
    Frame,
    SimulationConfig,
    Trajectory,
    simulate_trajectory,
    time_grid,
)
from .energy.evaluator import FoldingEnergy
from .moves import build_moveset
from .seq import normalise
from .struct import base_pair_distance, is_nested, iter_pairs, parse_dotbracket


@dataclass(slots=True)
class Ensemble:
    """A population of trajectories sampled on a common time grid."""

    seq: str
    times: list[float]
    #: ``structures[t][k]`` is trajectory ``k``'s dot-bracket at ``times[t]``.
    structures: list[list[str]]
    #: ``energies[t][k]`` in kcal/mol.
    energies: np.ndarray
    #: Transcript length at each time point.
    lengths: list[int]
    config: SimulationConfig | None = None
    events: list[int] = field(default_factory=list)
    #: Trajectories the event budget cut short.  A truncated trajectory holds
    #: its last structure for the rest of the run, so any observable after that
    #: point is frozen rather than simulated: the count belongs in any report.
    truncated: int = 0
    wall_time: float = 0.0

    # ------------------------------------------------------------------
    @property
    def n_trajectories(self) -> int:
        return len(self.structures[0]) if self.structures else 0

    @property
    def n_times(self) -> int:
        return len(self.times)

    def __len__(self) -> int:
        return self.n_trajectories

    # ------------------------------------------------------------------
    def occupancy(
        self, *, min_population: float = 0.0
    ) -> tuple[list[str], np.ndarray]:
        """Population of each distinct structure over time.

        Returns ``(labels, matrix)`` where ``matrix[s, t]`` is the fraction of
        trajectories in structure ``labels[s]`` at ``times[t]``.  Structures are
        ordered by the time at which they first become populated, which makes a
        stacked plot read left-to-right as a folding pathway.
        """
        counts: list[Counter] = []
        first_seen: dict[str, int] = {}
        for t, row in enumerate(self.structures):
            counter = Counter(row)
            counts.append(counter)
            for label in counter:
                first_seen.setdefault(label, t)

        n = self.n_trajectories or 1
        peak = {
            label: max(c.get(label, 0) for c in counts) / n for label in first_seen
        }
        labels = [
            label
            for label in sorted(first_seen, key=lambda s: (first_seen[s], s))
            if peak[label] >= min_population
        ]
        index = {label: k for k, label in enumerate(labels)}
        matrix = np.zeros((len(labels), self.n_times))
        for t, counter in enumerate(counts):
            for label, count in counter.items():
                k = index.get(label)
                if k is not None:
                    matrix[k, t] = count / n
        return labels, matrix

    def dominant(self) -> list[tuple[str, float]]:
        """Most populated structure at each time point, with its population."""
        out: list[tuple[str, float]] = []
        n = self.n_trajectories or 1
        for row in self.structures:
            label, count = Counter(row).most_common(1)[0]
            out.append((label, count / n))
        return out

    def pair_probabilities(self) -> np.ndarray:
        """``P[t, i, j]`` that ``i`` and ``j`` are paired, for ``i < j``."""
        size = len(self.seq)
        out = np.zeros((self.n_times, size, size))
        n = self.n_trajectories or 1
        for t, row in enumerate(self.structures):
            for db in row:
                for i, j in iter_pairs(parse_dotbracket(db)):
                    out[t, i, j] += 1.0
        return out / n

    def unpaired_probability(self) -> np.ndarray:
        """``P[t, i]`` that nucleotide ``i`` is unpaired *and* transcribed."""
        size = len(self.seq)
        out = np.zeros((self.n_times, size))
        n = self.n_trajectories or 1
        for t, row in enumerate(self.structures):
            for db in row:
                for i, ch in enumerate(db):
                    if ch == ".":
                        out[t, i] += 1.0
        return out / n

    def mean_energy(self) -> np.ndarray:
        return self.energies.mean(axis=1)

    def energy_spread(self) -> np.ndarray:
        return self.energies.std(axis=1)

    def pseudoknot_fraction(self) -> np.ndarray:
        """Fraction of trajectories carrying a crossing pair, over time."""
        out = np.zeros(self.n_times)
        n = self.n_trajectories or 1
        for t, row in enumerate(self.structures):
            out[t] = sum(
                1 for db in row if not is_nested(parse_dotbracket(db))
            ) / n
        return out

    def target_population(
        self, target: str, *, max_distance: int = 0
    ) -> np.ndarray:
        """Population of a target structure (optionally within a bp distance).

        ``target`` is a dot-bracket string over the full-length sequence; it is
        compared against each sampled structure padded to the same length.
        """
        target_pt = parse_dotbracket(target)
        target_pairs = set(iter_pairs(target_pt))
        out = np.zeros(self.n_times)
        n = self.n_trajectories or 1
        for t, row in enumerate(self.structures):
            hits = 0
            for db in row:
                pairs = set(iter_pairs(parse_dotbracket(db)))
                if len(pairs ^ target_pairs) <= max_distance:
                    hits += 1
            out[t] = hits / n
        return out

    def structure_at(self, t_index: int) -> list[str]:
        return self.structures[t_index]

    def final_distribution(self) -> list[tuple[str, float]]:
        """Structures present at the last time point, most populated first."""
        n = self.n_trajectories or 1
        return [
            (label, count / n)
            for label, count in Counter(self.structures[-1]).most_common()
        ]

    # ------------------------------------------------------------------
    def cluster(self, *, cutoff: int = 4) -> dict[str, str]:
        """Map each structure onto a representative within ``cutoff`` base pairs.

        A cheap leader clustering over base-pair distance: useful for collapsing
        the long tail of near-identical structures before plotting.
        """
        n = self.n_trajectories or 1
        weights: Counter = Counter()
        for row in self.structures:
            weights.update(row)
        representatives: list[tuple[str, list[int]]] = []
        mapping: dict[str, str] = {}
        for label, _count in weights.most_common():
            pt = parse_dotbracket(label)
            for rep, rep_pt in representatives:
                if len(rep_pt) == len(pt) and base_pair_distance(pt, rep_pt) <= cutoff:
                    mapping[label] = rep
                    break
            else:
                representatives.append((label, pt))
                mapping[label] = label
        return mapping

    def apply_clustering(self, mapping: dict[str, str]) -> "Ensemble":
        """A copy with every structure replaced by its cluster representative."""
        return Ensemble(
            seq=self.seq,
            times=list(self.times),
            structures=[[mapping.get(s, s) for s in row] for row in self.structures],
            energies=self.energies.copy(),
            lengths=list(self.lengths),
            config=self.config,
            events=list(self.events),
            truncated=self.truncated,
            wall_time=self.wall_time,
        )

    # ------------------------------------------------------------------
    def to_dict(self, *, min_population: float = 0.01) -> dict:
        """JSON-ready summary, used by the HTML player and the movie renderer."""
        labels, matrix = self.occupancy(min_population=min_population)
        return {
            "sequence": self.seq,
            "times": list(self.times),
            "lengths": list(self.lengths),
            "n_trajectories": self.n_trajectories,
            "structures": labels,
            "occupancy": matrix.tolist(),
            "dominant": [s for s, _p in self.dominant()],
            "dominant_population": [p for _s, p in self.dominant()],
            "mean_energy": self.mean_energy().tolist(),
            "energy_spread": self.energy_spread().tolist(),
            "pseudoknot_fraction": self.pseudoknot_fraction().tolist(),
            "wall_time": self.wall_time,
            "events": list(self.events),
            "truncated": self.truncated,
        }

    def to_json(self, path: str | os.PathLike, *, min_population: float = 0.01) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(min_population=min_population), handle, indent=1)


# ----------------------------------------------------------------------
# simulation drivers
# ----------------------------------------------------------------------
_WORKER: dict = {}


def _worker_init(seq: str, config: SimulationConfig) -> None:
    energy = config.build_energy(seq)
    _WORKER["seq"] = seq
    _WORKER["config"] = config
    _WORKER["energy"] = energy
    _WORKER["moveset"] = build_moveset(
        energy.model,
        energy.enc,
        min_loop=config.min_loop,
        min_helix=config.min_helix,
        nucleation_size=config.nucleation_size,
        max_span=config.max_span,
        max_stem_energy=config.max_stem_energy,
    )


def _worker_run(args: tuple[int, list[float]]) -> Trajectory:
    seed, grid = args
    return simulate_trajectory(
        _WORKER["seq"],
        _WORKER["config"],
        seed=seed,
        moveset=_WORKER["moveset"],
        energy=_WORKER["energy"],
        grid=grid,
    )


def simulate_ensemble(
    seq: str,
    config: SimulationConfig | None = None,
    *,
    n_trajectories: int = 100,
    seed: int = 0,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    grid: Sequence[float] | None = None,
) -> Ensemble:
    """Run ``n_trajectories`` independent cotranscriptional folding simulations.

    Trajectories are independent, so they parallelise perfectly; ``workers``
    defaults to the machine's CPU count.  All trajectories share one time grid
    so that structures can be compared across the ensemble at each instant;
    ``grid`` overrides the default spacing, which is how the probing benchmark
    asks for exactly one sample per transcript length.
    """
    import time as _time

    started = _time.perf_counter()
    config = config or SimulationConfig()
    seq = normalise(seq)
    n = len(seq)

    schedule = config.transcription
    t_end = (
        schedule.total_time(n) if schedule is not None else config.duration
    )
    grid = list(grid) if grid is not None else time_grid(
        t_end, config.frames, config.grid
    )

    seeds = [seed + k for k in range(n_trajectories)]
    trajectories: list[Trajectory] = []

    if workers is None:
        workers = min(os.cpu_count() or 1, n_trajectories)
    workers = max(1, workers)

    if workers == 1:
        energy = config.build_energy(seq)
        moveset = build_moveset(
            energy.model,
            energy.enc,
            min_loop=config.min_loop,
            min_helix=config.min_helix,
            nucleation_size=config.nucleation_size,
            max_span=config.max_span,
            max_stem_energy=config.max_stem_energy,
        )
        for k, s in enumerate(seeds):
            trajectories.append(
                simulate_trajectory(
                    seq, config, seed=s, moveset=moveset, energy=energy, grid=grid
                )
            )
            if progress:
                progress(k + 1, n_trajectories)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
        with ctx.Pool(
            processes=workers, initializer=_worker_init, initargs=(seq, config)
        ) as pool:
            for k, traj in enumerate(
                pool.imap(_worker_run, [(s, grid) for s in seeds], chunksize=1)
            ):
                trajectories.append(traj)
                if progress:
                    progress(k + 1, n_trajectories)

    return build_ensemble(seq, trajectories, config, _time.perf_counter() - started)


def build_ensemble(
    seq: str,
    trajectories: Sequence[Trajectory],
    config: SimulationConfig | None = None,
    wall_time: float = 0.0,
) -> Ensemble:
    """Collate trajectories that share a time grid into an :class:`Ensemble`."""
    if not trajectories:
        raise ValueError("no trajectories to collate")
    times = list(trajectories[0].times)
    n_times = len(times)
    for traj in trajectories:
        if len(traj.frames) != n_times:
            raise ValueError("trajectories were sampled on different grids")

    structures = [
        [traj.frames[t].structure for traj in trajectories] for t in range(n_times)
    ]
    energies = np.array(
        [[traj.frames[t].energy for traj in trajectories] for t in range(n_times)]
    )
    lengths = [trajectories[0].frames[t].length for t in range(n_times)]
    return Ensemble(
        seq=seq,
        times=times,
        structures=structures,
        energies=energies,
        lengths=lengths,
        config=config,
        events=[traj.events for traj in trajectories],
        truncated=sum(1 for traj in trajectories if traj.truncated),
        wall_time=wall_time,
    )
