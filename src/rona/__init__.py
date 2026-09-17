"""rona - cotranscriptional RNA folding kinetics.

Simulates how an RNA transcript folds *while it is being synthesised*, as a
stochastic kinetic process rather than a sequence of equilibrium calculations.
The result is a time-resolved ensemble: the distribution over structures at each
moment during and after transcription.

Quick start
-----------
>>> from rona import simulate_ensemble, SimulationConfig
>>> ensemble = simulate_ensemble("GGGAAACCCAAAGGGAAACCC", n_trajectories=20)
>>> structure, population = ensemble.dominant()[-1]

The command line offers the same thing plus rendering::

    rona fold sequence.fa -n 200 --html --svg --movie folding.mp4
"""

__version__ = "0.1.0"

from .cotrans import (  # noqa: F401
    Pause,
    SimulationConfig,
    TranscriptionSchedule,
    Trajectory,
    simulate_trajectory,
)
from .energy.evaluator import FoldingEnergy  # noqa: F401
from .energy.model import NearestNeighbourModel  # noqa: F401
from .energy.pseudoknot import PseudoknotModel  # noqa: F401
from .ensemble import Ensemble, simulate_ensemble  # noqa: F401
from .kinetics import RateModel  # noqa: F401
from .struct import Helix, parse_dotbracket, to_dotbracket  # noqa: F401

__all__ = [
    "__version__",
    "Ensemble",
    "FoldingEnergy",
    "Helix",
    "NearestNeighbourModel",
    "Pause",
    "PseudoknotModel",
    "RateModel",
    "SimulationConfig",
    "Trajectory",
    "TranscriptionSchedule",
    "parse_dotbracket",
    "simulate_ensemble",
    "simulate_trajectory",
    "to_dotbracket",
]
