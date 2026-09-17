"""Does the coarse folding result depend on how fast zipping is?

Zipping a base pair onto a helix end is a *fast* mode: it fires at ~10^6 s^-1
against ~10^5 s^-1 for nucleation, and on a 58 nt transcript it accounts for
99.7% of all simulated events, with forward and reverse counts equal to three
significant figures.  The helix length is already at internal equilibrium and
merely jittering.

That means the exact value of ``k_zip`` should not change the coarse kinetics,
only how much compute is spent reaching the same answer - the same reasoning
behind DrTransformer's ``--t-fast``.  This script checks that rather than
assuming it.  The event count should scale linearly with ``k_zip`` while the
final distribution does not move.
"""

from __future__ import annotations

import time

import numpy as np

from rona import SimulationConfig, TranscriptionSchedule, simulate_ensemble
from rona.kinetics import RateModel

SEQUENCE = "GGACGCAGAAAACUGCGUCCUUAAUAAUAAUAAUAAGGACGCAG"


def main(trajectories: int = 24) -> None:
    schedule = TranscriptionSchedule(rate=30.0, post_time=6.0)
    results = {}
    print(f"sequence {SEQUENCE} ({len(SEQUENCE)} nt), {trajectories} trajectories\n")
    for k_zip in (1e5, 1e6):
        config = SimulationConfig(
            frames=30, rates=RateModel(k_zip=k_zip), transcription=schedule
        )
        started = time.perf_counter()
        ensemble = simulate_ensemble(
            SEQUENCE, config, n_trajectories=trajectories, seed=11
        )
        elapsed = time.perf_counter() - started
        results[k_zip] = ensemble
        structure, population = ensemble.final_distribution()[0]
        events = sum(ensemble.events) / len(ensemble)
        print(
            f"  k_zip={k_zip:7.0e}  {events:>10,.0f} events/trajectory  "
            f"{elapsed:6.1f} s   most populated {population:5.1%}"
        )
        print(f"      {structure}")

    reference = results[max(results)]
    print("\n  agreement with the fastest setting:")
    for k_zip, ensemble in results.items():
        a = dict(ensemble.final_distribution())
        b = dict(reference.final_distribution())
        distance = 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))
        unpaired = float(
            np.mean(
                np.abs(
                    ensemble.unpaired_probability()
                    - reference.unpaired_probability()
                )
            )
        )
        print(
            f"    k_zip={k_zip:7.0e}   total-variation {distance:5.3f}   "
            f"mean |dP(unpaired)| {unpaired:6.4f}"
        )
    print(
        "\n  Event count scales with k_zip; the distribution should not.\n"
        "  With few trajectories the comparison is noisy - raise `trajectories`\n"
        "  before drawing a conclusion."
    )


if __name__ == "__main__":
    main()
