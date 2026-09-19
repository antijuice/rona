"""v2: the ensemble as a solved master equation, with a certified error bound.

v1 sampled trajectories, which costs one step per *event* - and the event count
is set by the fastest mode in the system (base-pair zipping, ~10^6 s^-1) while
the answer lives on the transcription timescale of seconds.  This package
integrates the master equation instead, which costs one step per *time step*,
and pays nothing for fast modes beyond the states they occupy.

The truncation that makes that finite is not a heuristic here.  Finite State
Projection (Munsky & Khammash 2006) restricts the generator to a retained set
and returns a certificate: the probability mass that leaves the set bounds the
L1 error of what remains.  Every distribution this package reports carries that
bound.  See ``docs/v2-design.md``.
"""

from .structures import Structure, dotbracket, from_dotbracket, helices_of
from .moves import MoveModel, neighbours
from .generator import StateIndex, build_generator
from .fsp import Solver, Step

__all__ = [
    "Structure",
    "dotbracket",
    "from_dotbracket",
    "helices_of",
    "MoveModel",
    "neighbours",
    "StateIndex",
    "build_generator",
    "Solver",
    "Step",
]
