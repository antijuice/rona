"""Pseudoknot formation during transcription.

A hairpin loop is transcribed first; the sequence that can pair with it arrives
later and has to thread through the existing helix.  The simulation tracks the
crossing helix explicitly, so the pseudoknotted fraction of the ensemble can be
followed through time.

The pseudoknot topology penalty is a tunable, transparent functional form rather
than a fitted parameter set - see the README - so treat the populations as
qualitative and vary --pk-init to see how sensitive they are.
"""

from __future__ import annotations

from rona import (
    PseudoknotModel,
    SimulationConfig,
    TranscriptionSchedule,
    simulate_ensemble,
)

SEQUENCE = "GGCGCGGCACCGUCCGCGGAACAAACGGAGAAGGGGCCGCCGAAAGGCGGCC"


def main() -> None:
    print(f"sequence  {SEQUENCE}  ({len(SEQUENCE)} nt)\n")
    for init in (5.0, 7.0, 10.0):
        ensemble = simulate_ensemble(
            SEQUENCE,
            SimulationConfig(
                frames=60,
                transcription=TranscriptionSchedule(rate=30.0, post_time=15.0),
                pseudoknots=PseudoknotModel(init=init),
            ),
            n_trajectories=48,
            seed=3,
        )
        fraction = ensemble.pseudoknot_fraction()
        print(
            f"  pk-init {init:4.1f} kcal/mol:  "
            f"peak pseudoknotted {fraction.max():5.1%}, "
            f"final {fraction[-1]:5.1%}"
        )
        knotted = [
            (s, p)
            for s, p in ensemble.final_distribution()
            if "[" in s
        ]
        if knotted:
            structure, population = knotted[0]
            print(f"              {population:5.1%}  {structure}")


if __name__ == "__main__":
    main()
