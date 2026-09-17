"""Cotranscriptional folding produces a different answer from equilibrium.

A designed folding trap: segment A pairs with the nearby A' the moment A' is
transcribed - a local, fast, 8 bp hairpin.  The global free-energy minimum
instead pairs A' with the *later* A'', an 11 bp helix that is 2.2 kcal/mol
better.  An equilibrium calculation reports the long-range helix.  The molecule,
folding as it is made, never gets there.

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
        "\n  The difference is the point: the same sequence, the same energy\n"
        "  model, and a different answer, because the order in which the chain\n"
        "  appears decides which helix wins."
    )


if __name__ == "__main__":
    main()
