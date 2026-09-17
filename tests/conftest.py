import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Tests exercise logic, not physics.  Zipping is a futile fast mode that costs
# ~99% of events at the physical rate, so the suite runs it only modestly
# faster than nucleation; nothing under test depends on the exact ratio.
TEST_RATES_KWARGS = {"k_zip": 3.0e5}


def fast_rates(**overrides):
    from rona.kinetics import RateModel

    return RateModel(**{**TEST_RATES_KWARGS, **overrides})
