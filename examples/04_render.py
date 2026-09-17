"""Render a kinetic ensemble: figures, an interactive player and a movie."""

from __future__ import annotations

from pathlib import Path

from rona import SimulationConfig, TranscriptionSchedule, simulate_ensemble
from rona.render import plots
from rona.render.player import write_player
from rona.render.svg import DrawOptions, structure_svg

SEQUENCE = "GGACGCAGAAAACUGCGUCCUUAAUAAUAAUAAUAAGGACGCAG"


def main(out: str = "out") -> None:
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)

    ensemble = simulate_ensemble(
        SEQUENCE,
        SimulationConfig(
            frames=90,
            transcription=TranscriptionSchedule(rate=30.0, post_time=15.0),
        ),
        n_trajectories=48,
        seed=1,
    )

    (directory / "occupancy.svg").write_text(plots.occupancy_plot(ensemble))
    (directory / "energy.svg").write_text(plots.energy_plot(ensemble))
    (directory / "pairprob.svg").write_text(plots.pair_probability_plot(ensemble, -1))

    structure, population = ensemble.dominant()[-1]
    (directory / "final.svg").write_text(
        structure_svg(
            SEQUENCE,
            structure,
            options=DrawOptions(
                title="dominant structure at the end of the run",
                subtitle=f"population {population:.0%}",
            ),
        )
    )
    write_player(ensemble, str(directory / "player.html"))
    print(f"wrote figures and player.html to {directory}/")

    try:
        from rona.render.movie import MovieOptions, render_movie

        render_movie(
            ensemble,
            str(directory / "folding.mp4"),
            options=MovieOptions(mode="ensemble", fps=20, interpolation=3),
        )
        print(f"wrote {directory}/folding.mp4")
    except RuntimeError as error:
        print(f"skipping the movie: {error}")


if __name__ == "__main__":
    main()
