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

Past about 50 nt the limit stops being arithmetic and becomes representational:
the retained set has to cover a *product* over structural domains that fold
independently.  :mod:`rona.master.factor` shows that conditional on a nested set
of anchor pairs the generator is exactly the Kronecker sum of per-region
generators, so the product can be stored as a product; :mod:`rona.master.anchored`
decides when to do that and measures what it costs.
"""

from .structures import Structure, dotbracket, from_dotbracket, helices_of
from .moves import MoveModel, neighbours
from .generator import StateIndex, build_generator
from .certify import Certificate, certify
from .cotrans import Frame, Schedule, transcribe
from .fsp import Solver, Step
from .factor import Region, decompose, region_candidates, rejoin, restrict
from .anchored import Anchored, AnchoredFrame, Promotion
from .coarse import basins, local_minimum

__all__ = [
    "Structure",
    "dotbracket",
    "from_dotbracket",
    "helices_of",
    "MoveModel",
    "neighbours",
    "StateIndex",
    "build_generator",
    "Certificate",
    "certify",
    "Frame",
    "Schedule",
    "transcribe",
    "Solver",
    "Step",
    "Region",
    "decompose",
    "region_candidates",
    "rejoin",
    "restrict",
    "Anchored",
    "AnchoredFrame",
    "Promotion",
    "basins",
    "local_minimum",
]
