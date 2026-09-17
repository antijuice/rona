"""A transcriptional pause changes what the RNA folds into.

Pausing gives the nascent transcript time to reach a structure it would
otherwise skip past.  This is the mechanism behind several riboswitches, and it
is invisible to any equilibrium calculation: the sequence, the temperature and
the energy model are identical in both runs here.  Only the elongation schedule
differs.
"""

from __future__ import annotations

from rona import Pause, SimulationConfig, TranscriptionSchedule, simulate_ensemble

SEQUENCE = "GGACGCAGAAAACUGCGUCCUUAAUAAUAAUAAUAAGGACGCAG"


def run(label: str, schedule: TranscriptionSchedule) -> None:
    ensemble = simulate_ensemble(
        SEQUENCE,
        SimulationConfig(frames=60, transcription=schedule),
        n_trajectories=48,
        seed=5,
    )
    print(f"\n  {label}")
    for structure, population in ensemble.final_distribution()[:3]:
        print(f"    {population:6.1%}  {structure}")


def main() -> None:
    print(f"sequence  {SEQUENCE}  ({len(SEQUENCE)} nt)")
    run(
        "fast elongation, 100 nt/s, no pause",
        TranscriptionSchedule(rate=100.0, post_time=10.0),
    )
    run(
        "slow elongation, 5 nt/s, no pause",
        TranscriptionSchedule(rate=5.0, post_time=10.0),
    )
    run(
        "100 nt/s with a 5 s pause once 20 nt are out",
        TranscriptionSchedule(
            rate=100.0, post_time=10.0,
            pauses=(Pause(position=20, duration=5.0),),
        ),
    )


if __name__ == "__main__":
    main()
