"""Benchmark against public cotranscriptional SHAPE-seq data.

Downloads the *B. cereus* crcB fluoride riboswitch probing matrix from the RNA
Mapping Database (108 transcript lengths, 127 nt, 7,938 measured reactivities)
and scores three predictions against it:

* **rona** - the kinetic ensemble's probability that each nucleotide is unpaired;
* **stepwise equilibrium** - ViennaRNA's equilibrium unpaired probability for
  each prefix, the standard "fold every prefix" approximation;
* **DrTransformer** - if a ``.drf`` time course is supplied with ``--drf``.

Caveats worth keeping in mind are in ``docs/validation.md``; the important one
is that roadblock protocols probe *stalled* complexes, which is closer to
per-length equilibrium than to free elongation, so this dataset does not
strongly separate kinetic from equilibrium models.

Runtime: about 55 minutes on four cores for 16 trajectories. Use ``-n 4`` for a
quick look (~14 minutes).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

RDAT_URL = (
    "https://github.com/DasLab/rmdb.github.io/releases/download/"
    "data-general/CRCBFL_BZCN_0001.rdat"
)
#: Polymerase footprint: sequestered nucleotides are unreactive whatever the
#: structure, so they are excluded from the comparison.
FOOTPRINT = 14
#: 3' primer-binding cassette added by the probing protocol.
CASSETTE = 35


def fetch(path: Path) -> Path:
    if not path.exists():
        print(f"downloading {RDAT_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RDAT_URL, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--trajectories", type=int, default=16)
    parser.add_argument("--rate", type=float, default=30.0, help="nt/s")
    parser.add_argument("--k-zip", type=float, default=1e5)
    parser.add_argument("--rdat", default="data/CRCBFL_BZCN_0001.rdat")
    parser.add_argument("--drf", default=None, help="a DrTransformer .drf file")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    from rona.cotrans import SimulationConfig
    from rona.kinetics import RateModel
    from rona.validation import shape
    from rona.validation.rdat import read_rdat

    entry = read_rdat(fetch(Path(args.rdat)))
    sequence = entry.sequence
    measured, lengths = entry.matrix()
    print(f"{entry.name}")
    print(
        f"{len(sequence)} nt, {len(lengths)} transcript lengths, "
        f"{int(measured.size - (measured != measured).sum()):,} reactivities\n"
    )

    started = time.perf_counter()
    predicted, ensemble = shape.run_rona(
        sequence,
        lengths,
        rate=args.rate,
        footprint=FOOTPRINT,
        post_time=0.5,
        n_trajectories=args.trajectories,
        seed=1,
        config=SimulationConfig(rates=RateModel(k_zip=args.k_zip)),
        workers=args.workers,
    )
    print(
        f"rona: {time.perf_counter() - started:.0f} s, "
        f"{sum(ensemble.events) / len(ensemble):,.0f} events/trajectory\n"
    )

    rows = [("rona (kinetic SSA)", predicted)]
    if args.drf:
        from rona.render.drforna import read_drf

        rows.append(
            (
                "DrTransformer",
                read_drf(args.drf).unpaired_matrix(lengths, len(sequence)),
            )
        )
    rows.append(
        (
            "stepwise equilibrium",
            shape.equilibrium_matrix(sequence, lengths, footprint=FOOTPRINT),
        )
    )

    print("agreement with measured reactivity "
          "(footprint and 3' cassette excluded):")
    for name, matrix in rows:
        result = shape.compare(
            name, matrix, measured, lengths,
            footprint_mask=FOOTPRINT, trim_3prime=CASSETTE,
        )
        print("  " + result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
