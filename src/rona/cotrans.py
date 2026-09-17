"""Cotranscriptional folding: the growing chain, and trajectory sampling.

A trajectory is produced by interleaving two processes:

* **Transcription** - a deterministic (or stochastically paused) schedule that
  adds nucleotides to the 3' end at the elongation rate, keeping the last
  ``footprint`` nucleotides sequestered inside the polymerase where they cannot
  pair.
* **Folding** - the Gillespie SSA of :mod:`rona.kinetics`, run exactly between
  elongation events.

Because propensities only change when a move fires or a nucleotide is added,
interleaving is exact: after an elongation event a fresh exponential waiting
time is drawn, which is legitimate by the memorylessness of the exponential
distribution.

This is what makes the result a *kinetic* ensemble rather than a sequence of
equilibria.  Nothing is ever re-equilibrated at any chain length; a structure
that forms early persists until the kinetics melt it, which is precisely the
trapping that lets riboswitches and other cotranscriptional folders work.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Sequence

from .energy.evaluator import FoldingEnergy
from .energy.model import NearestNeighbourModel
from .energy.pseudoknot import PseudoknotModel
from .kinetics import KineticEngine, RateModel
from .moves import build_moveset
from .seq import encode, normalise


@dataclass(frozen=True, slots=True)
class Pause:
    """A transcriptional pause site.

    ``position`` is the transcript length (in nucleotides) at which the
    polymerase halts; ``duration`` is the dwell time in seconds.  With
    ``stochastic=True`` the dwell is drawn from an exponential distribution
    with that mean, which is the usual description of pause escape.
    """

    position: int
    duration: float
    stochastic: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptionSchedule:
    """How the chain grows."""

    #: Elongation rate in nucleotides per second.
    rate: float = 30.0
    #: Nucleotides held inside the polymerase exit channel and unable to pair.
    footprint: int = 10
    #: Transcript length already present when the simulation starts.
    start_length: int = 10
    pauses: tuple[Pause, ...] = ()
    #: Seconds of continued folding after the full transcript is released.
    post_time: float = 10.0

    def arrival_times(self, n: int, rng: random.Random | None = None) -> list[float]:
        """Time at which the transcript first reaches each length ``0..n``.

        Index ``L`` holds the time at which length ``L`` is reached; lengths at
        or below ``start_length`` are present from ``t = 0``.
        """
        if self.rate <= 0:
            raise ValueError("elongation rate must be positive")
        pause_at = {p.position: p for p in self.pauses}
        times = [0.0] * (n + 1)
        t = 0.0
        step = 1.0 / self.rate
        for length in range(1, n + 1):
            if length > self.start_length:
                t += step
                pause = pause_at.get(length)
                if pause is not None:
                    if pause.stochastic and rng is not None:
                        t += rng.expovariate(1.0 / max(pause.duration, 1e-12))
                    else:
                        t += pause.duration
            times[length] = t
        return times

    def total_time(self, n: int) -> float:
        return self.arrival_times(n)[n] + self.post_time


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Everything needed to reproduce a run."""

    temperature: float = 37.0
    dangles: int = 2
    rates: RateModel = field(default_factory=RateModel)
    pseudoknots: PseudoknotModel = field(default_factory=PseudoknotModel)
    transcription: TranscriptionSchedule | None = field(
        default_factory=TranscriptionSchedule
    )
    min_helix: int = 3
    nucleation_size: int = 3
    min_loop: int = 3
    max_span: int | None = None
    max_stem_energy: float = -1.0
    max_events: int = 2_000_000
    #: Number of sampled time points per trajectory.
    frames: int = 200
    #: ``linear`` or ``log`` spacing of the sampling grid.
    grid: str = "linear"
    #: For fixed-length refolding runs (``transcription=None``), the duration.
    duration: float = 10.0

    def build_energy(self, seq: str) -> FoldingEnergy:
        model = NearestNeighbourModel(
            temperature=self.temperature,
            dangles=self.dangles,
            min_loop=self.min_loop,
        )
        return FoldingEnergy(seq, model, self.pseudoknots)


@dataclass(slots=True)
class Frame:
    """The state of one trajectory at one sampled time."""

    time: float
    length: int
    structure: str
    energy: float


@dataclass(slots=True)
class Trajectory:
    """A single stochastic folding pathway."""

    seq: str
    seed: int
    times: list[float]
    frames: list[Frame]
    events: int
    wall_time: float = 0.0

    @property
    def final(self) -> Frame:
        return self.frames[-1]

    def structure_at(self, t: float) -> str:
        """Structure at the sampled time closest to (but not after) ``t``."""
        best = self.frames[0]
        for frame in self.frames:
            if frame.time <= t:
                best = frame
            else:
                break
        return best.structure


#: Decades of dynamic range covered by the logarithmic sampling grid.
LOG_GRID_DECADES = 3.0


def time_grid(t_end: float, frames: int, mode: str = "linear") -> list[float]:
    """Sampling grid over ``[0, t_end]``.

    The logarithmic grid spans :data:`LOG_GRID_DECADES` decades below
    ``t_end``.  A much lower floor would spend most of the frames on times
    before the first nucleotide is even added, where nothing has happened.
    """
    if frames < 2:
        return [0.0, t_end]
    if mode == "log":
        lo = max(t_end * 10.0 ** (-LOG_GRID_DECADES), 1e-9)
        step = (math.log(t_end) - math.log(lo)) / (frames - 2)
        return [0.0] + [math.exp(math.log(lo) + step * k) for k in range(frames - 1)]
    return [t_end * k / (frames - 1) for k in range(frames)]


def simulate_trajectory(
    seq: str,
    config: SimulationConfig | None = None,
    *,
    seed: int = 0,
    moveset=None,
    energy: FoldingEnergy | None = None,
    grid: Sequence[float] | None = None,
) -> Trajectory:
    """Run one cotranscriptional folding trajectory.

    ``moveset`` and ``energy`` may be supplied to avoid re-enumerating stems
    when running many trajectories over the same sequence.
    """
    import time as _time

    started = _time.perf_counter()
    config = config or SimulationConfig()
    seq = normalise(seq)
    n = len(seq)
    rng = random.Random(seed)

    energy = energy or config.build_energy(seq)
    moveset = moveset or build_moveset(
        energy.model,
        energy.enc,
        min_loop=config.min_loop,
        min_helix=config.min_helix,
        nucleation_size=config.nucleation_size,
        max_span=config.max_span,
        max_stem_energy=config.max_stem_energy,
    )

    schedule = config.transcription
    if schedule is None:
        arrivals = [0.0] * (n + 1)
        t_end = config.duration
        footprint = 0
        initial_length = n
    else:
        arrivals = schedule.arrival_times(n, rng)
        t_end = arrivals[n] + schedule.post_time
        footprint = schedule.footprint
        initial_length = min(n, schedule.start_length)

    grid_times = list(grid) if grid is not None else time_grid(
        t_end, config.frames, config.grid
    )

    engine = KineticEngine(energy, moveset, config.rates, pk_model=config.pseudoknots)
    state = engine.state
    length = initial_length
    engine.grow(length, footprint if length < n else 0)

    frames: list[Frame] = []
    grid_index = 0
    t = 0.0
    events = 0

    def record_until(now: float) -> None:
        nonlocal grid_index
        while grid_index < len(grid_times) and grid_times[grid_index] <= now:
            frames.append(
                Frame(
                    time=grid_times[grid_index],
                    length=state.length,
                    structure=state.dotbracket(),
                    energy=state.energy,
                )
            )
            grid_index += 1

    record_until(0.0)

    while t < t_end and events < config.max_events:
        total = engine.propensity()
        tau = float("inf") if total <= 0.0 else rng.expovariate(total)
        t_next_growth = arrivals[length + 1] if length < n else float("inf")

        if t + tau >= t_next_growth:
            # the polymerase acts before any folding move fires
            t = t_next_growth
            record_until(t)
            length += 1
            engine.grow(length, footprint if length < n else 0)
            continue

        if t + tau >= t_end:
            break

        t += tau
        record_until(t)
        move = engine.select(rng.random())
        if move is None:
            break
        engine.apply(move)
        events += 1

    record_until(t_end)
    while len(frames) < len(grid_times):
        frames.append(
            Frame(
                time=grid_times[len(frames)],
                length=state.length,
                structure=state.dotbracket(),
                energy=state.energy,
            )
        )

    return Trajectory(
        seq=seq,
        seed=seed,
        times=grid_times,
        frames=frames,
        events=events,
        wall_time=_time.perf_counter() - started,
    )
