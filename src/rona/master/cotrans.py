"""Transcription: a sequence of master equations, carried across exactly.

The chain grows, so the state space grows with it, and every tool in this area
has to say how a distribution over structures of an ``n``-nucleotide prefix
becomes one over structures of ``n+1``.  BarMap builds explicit maps between
consecutive coarse-grained landscapes; DrTransformer re-derives representative
local minima at each step and repopulates them.

Here it is not a modelling choice at all.  A structure is a set of base pairs,
and every structure of a prefix *is* a structure of the longer prefix - the map
is the identity, and probability carries across untouched.  What changes is only
the energy of each structure (a new nucleotide dangles off the exterior loop)
and which further pairs are available; both are keyed by length and recomputed,
and the retained set then grows where the new sequence makes it grow.

Nothing is approximated by the elongation itself, which is worth saying because
it is the step where other methods lose their accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .certify import Certificate, certify
from .fsp import Solver


@dataclass(frozen=True, slots=True)
class Schedule:
    """How the polymerase moves."""

    #: Elongation rate in nucleotides per second.
    rate: float = 30.0
    #: Nucleotides held inside the polymerase and unable to pair.
    footprint: int = 0
    #: Transcript length at which folding starts being simulated.
    start: int = 10
    #: Seconds simulated after the last nucleotide arrives.
    post_time: float = 0.0

    def interval(self) -> float:
        return 1.0 / self.rate


@dataclass(slots=True)
class Frame:
    """The ensemble at one moment, and how much of it is accounted for."""

    time: float
    transcript: int
    available: int
    states: int
    #: Equilibrium weight outside the retained set, or ``nan`` without ViennaRNA.
    certificate: Certificate | None
    #: Boundary indicator from the dynamics.
    leak: float
    #: Estimated L1 error from time integration.
    integration_error: float
    distribution: list[tuple[str, float]] = field(default_factory=list)

    @property
    def dominant(self) -> tuple[str, float]:
        return self.distribution[0] if self.distribution else ("", 0.0)


def transcribe(
    solver: Solver,
    schedule: Schedule | None = None,
    *,
    steps_per_nucleotide: int = 2,
    top: int = 8,
    certify_every: int = 1,
) -> Iterator[Frame]:
    """Fold while the chain grows, yielding one frame per nucleotide.

    ``steps_per_nucleotide`` splits each elongation interval, which costs a
    little and lets the retained set be re-examined before the sequence moves
    under it again.
    """
    schedule = schedule or Schedule()
    total = solver.energy.n
    interval = schedule.interval()
    clock = 0.0
    transcript = max(1, min(schedule.start, total))
    solver.grow(max(0, transcript - schedule.footprint))

    index = 0
    while True:
        for _ in range(steps_per_nucleotide):
            solver.advance(interval / steps_per_nucleotide)
        clock += interval
        certificate = (
            certify(solver)
            if certify_every and index % certify_every == 0
            else None
        )
        yield Frame(
            time=clock,
            transcript=transcript,
            available=solver.length,
            states=len(solver.states),
            certificate=certificate,
            leak=solver.leak,
            integration_error=solver.integration_error,
            distribution=solver.distribution(top=top),
        )
        index += 1
        if transcript >= total:
            break
        transcript += 1
        # the identity map: the retained structures are structures of the longer
        # prefix too, so only the length they are scored at changes
        solver.grow(max(0, transcript - schedule.footprint))

    if schedule.post_time > 0.0:
        remaining = schedule.post_time
        block = max(remaining / 8.0, 1e-9)
        while remaining > 0.0:
            span = min(block, remaining)
            solver.advance(span)
            clock += span
            remaining -= span
        yield Frame(
            time=clock,
            transcript=transcript,
            available=solver.length,
            states=len(solver.states),
            certificate=certify(solver),
            leak=solver.leak,
            integration_error=solver.integration_error,
            distribution=solver.distribution(top=top),
        )
