"""Cotranscriptional folding, compared against equilibrium.

A designed folding trap: segment A pairs with the nearby A' the moment A' is
transcribed - a local, fast, 8 bp hairpin.  The global free-energy minimum
instead pairs A' with the *later* A'', an 11 bp helix that is 2.2 kcal/mol
better.

The interesting part is that the answer depends on whether helices can zip.
With the trap forced to melt all at once it is inescapable; letting it unzip one
pair at a time - which is what really happens - most of the ensemble escapes.
The barrier that matters is the stepwise one, not the all-at-once one.  Vary
``rate`` below and watch the balance shift: the faster the polymerase, the less
time the trap has to resolve.

Run with ViennaRNA installed to see the equilibrium comparison as well.
"""

from __future__ import annotations

from rona import SimulationConfig, TranscriptionSchedule, simulate_ensemble

SEQUENCE = "GGACGCAGAAAACUGCGUCCUUAAUAAUAAUAAUAAGGACGCAG"
TRAP = "((((((((....))))))))" + "." * 24


def main() -> None:
    print(f"sequence  {SEQUENCE}  ({len(SEQUENCE)} nt)\n")

    try:
        import RNA

        fold = RNA.fold_compound(SEQUENCE)
        mfe_structure, mfe_energy = fold.mfe()
        print(f"  equilibrium MFE   {mfe_structure}  {mfe_energy:6.2f} kcal/mol")
        print(f"  local trap        {TRAP}  {fold.eval_structure(TRAP):6.2f} kcal/mol")
    except ImportError:
        print("  (install ViennaRNA for the equilibrium comparison)")
    print()

    cotranscriptional = simulate_ensemble(
        SEQUENCE,
        SimulationConfig(
            frames=80,
            transcription=TranscriptionSchedule(rate=30.0, post_time=20.0),
        ),
        n_trajectories=48,
        seed=1,
    )
    print("  folding while being transcribed at 30 nt/s:")
    for structure, population in cotranscriptional.final_distribution()[:3]:
        print(f"    {population:6.1%}  {structure}")

    refolded = simulate_ensemble(
        SEQUENCE,
        SimulationConfig(frames=80, transcription=None, duration=30.0),
        n_trajectories=48,
        seed=1,
    )
    print("\n  refolding the same chain from the open state:")
    for structure, population in refolded.final_distribution()[:3]:
        print(f"    {population:6.1%}  {structure}")

    print(
        "\n  Same sequence, same energy model, two different answers, because\n"
        "  the order in which the chain appears decides which helix nucleates\n"
        "  first and how long it has to resolve."
    )


if __name__ == "__main__":
    main()
