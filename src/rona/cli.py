"""Command-line interface.

``rona fold`` is the main entry point: it runs an ensemble of cotranscriptional
folding trajectories and writes whichever outputs are asked for - an interactive
HTML player, a movie, SVG figures and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .cotrans import (
    Pause,
    SimulationConfig,
    TranscriptionSchedule,
    simulate_trajectory,
)
from .energy.pseudoknot import PseudoknotModel
from .ensemble import simulate_ensemble
from .kinetics import HELIX_MODE, MOVE_SETS, RateModel
from .moves import build_moveset
from .seq import normalise, read_fasta


def _read_sequence(value: str) -> tuple[str, str]:
    """Accept a literal sequence, a FASTA path, or ``-`` for stdin."""
    if value == "-":
        records = read_fasta(sys.stdin.read())
        if not records:
            raise SystemExit("no sequence found on stdin")
        return records[0]
    path = Path(value)
    if path.exists():
        records = read_fasta(path.read_text(encoding="utf-8"))
        if not records:
            raise SystemExit(f"no sequence found in {value}")
        return records[0]
    return (path.stem or "seq", normalise(value))


def _parse_pause(text: str) -> Pause:
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "pause must be POSITION:SECONDS[:stochastic]"
        )
    position = int(parts[0])
    duration = float(parts[1])
    stochastic = len(parts) == 3 and parts[2].lower() in ("s", "stochastic", "1", "true")
    return Pause(position=position, duration=duration, stochastic=stochastic)


def _build_config(args) -> SimulationConfig:
    if args.equilibrium:
        schedule = None
    else:
        schedule = TranscriptionSchedule(
            rate=args.rate,
            footprint=args.footprint,
            start_length=args.start_length,
            pauses=tuple(args.pause or ()),
            post_time=args.post_time,
        )
    pk = PseudoknotModel(
        init=args.pk_init,
        per_unpaired=args.pk_unpaired,
        per_branch=args.pk_branch,
        min_helix=args.pk_min_helix,
        max_helices=args.max_pk,
        enabled=not args.no_pseudoknots,
    )
    return SimulationConfig(
        temperature=args.temperature,
        dangles=args.dangles,
        rates=RateModel(
            k_nucleate=args.k_nucleate, k_zip=args.k_zip, scheme=args.scheme
        ),
        pseudoknots=pk,
        transcription=schedule,
        min_helix=args.min_helix,
        nucleation_size=args.nucleation,
        max_span=args.max_span,
        max_stem_energy=args.max_stem_energy,
        max_events=args.max_events,
        frames=args.frames,
        grid=args.grid,
        duration=args.duration,
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    model = parser.add_argument_group("energy model")
    model.add_argument("--temperature", type=float, default=37.0, metavar="C",
                       help="folding temperature in degrees Celsius (default 37)")
    model.add_argument("--dangles", type=int, choices=(0, 2), default=2,
                       help="dangling-end model (default 2, as in ViennaRNA)")

    kinetics = parser.add_argument_group("kinetics")
    kinetics.add_argument("--mode", choices=MOVE_SETS, default=HELIX_MODE,
                          help="move set: whole-helix (default) or base-pair zipping")
    kinetics.add_argument("--k-nucleate", type=float, default=1e5, metavar="HZ",
                          help="helix nucleation attempt frequency (default 1e5/s)")
    kinetics.add_argument("--k-zip", type=float, default=1e7, metavar="HZ",
                          help="base-pair zipping attempt frequency (default 1e7/s)")
    kinetics.add_argument("--scheme", choices=("metropolis", "kawasaki"),
                          default="metropolis", help="detailed-balance rate rule")
    kinetics.add_argument("--min-helix", type=int, default=3, metavar="BP",
                          help="shortest helix that may form (default 3)")
    kinetics.add_argument("--nucleation", type=int, default=3, metavar="BP",
                          help="nucleation window size (default 3)")
    kinetics.add_argument("--max-span", type=int, default=None, metavar="NT",
                          help="cap on base-pair span; prunes long-range pairs")
    kinetics.add_argument("--max-stem-energy", type=float, default=-1.0,
                          metavar="KCAL",
                          help="discard stems weaker than this (default -1.0)")
    kinetics.add_argument("--max-events", type=int, default=2_000_000,
                          help="safety cap on events per trajectory")

    transcription = parser.add_argument_group("transcription")
    transcription.add_argument("--rate", type=float, default=30.0, metavar="NT_S",
                               help="elongation rate in nt/s (default 30)")
    transcription.add_argument("--footprint", type=int, default=10, metavar="NT",
                               help="nucleotides held inside the polymerase (default 10)")
    transcription.add_argument("--start-length", type=int, default=10, metavar="NT",
                               help="transcript length at t=0 (default 10)")
    transcription.add_argument("--pause", type=_parse_pause, action="append",
                               metavar="POS:SEC[:s]",
                               help="pause site; repeatable. Append ':s' to draw "
                                    "the dwell from an exponential distribution")
    transcription.add_argument("--post-time", type=float, default=10.0, metavar="S",
                               help="folding time after release (default 10 s)")
    transcription.add_argument("--equilibrium", action="store_true",
                               help="no transcription: refold the full-length chain, "
                                    "for comparison against the cotranscriptional run")
    transcription.add_argument("--duration", type=float, default=10.0, metavar="S",
                               help="run length for --equilibrium (default 10 s)")

    pk = parser.add_argument_group("pseudoknots")
    pk.add_argument("--no-pseudoknots", action="store_true",
                    help="forbid crossing helices entirely")
    pk.add_argument("--pk-init", type=float, default=7.0, metavar="KCAL",
                    help="pseudoknot initiation penalty (default 7.0)")
    pk.add_argument("--pk-unpaired", type=float, default=0.1, metavar="KCAL",
                    help="penalty per unpaired nt in a pseudoknot region (default 0.1)")
    pk.add_argument("--pk-branch", type=float, default=0.2, metavar="KCAL",
                    help="penalty per crossed helix (default 0.2)")
    pk.add_argument("--pk-min-helix", type=int, default=2, metavar="BP",
                    help="shortest helix allowed to form a pseudoknot (default 2)")
    pk.add_argument("--max-pk", type=int, default=None, metavar="N",
                    help="cap on simultaneous pseudoknot helices")

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--frames", type=int, default=160,
                          help="sampled time points per trajectory (default 160)")
    sampling.add_argument("--grid", choices=("linear", "log"), default="linear",
                          help="spacing of the sampling grid (default linear)")
    sampling.add_argument("--seed", type=int, default=0, help="base random seed")


# ----------------------------------------------------------------------
def cmd_fold(args) -> int:
    name, sequence = _read_sequence(args.sequence)
    config = _build_config(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or name

    quiet = args.quiet
    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if quiet:
            return
        width = 28
        filled = int(width * done / total)
        bar = "#" * filled + "." * (width - filled)
        elapsed = time.perf_counter() - started
        eta = elapsed / done * (total - done) if done else 0.0
        sys.stderr.write(
            f"\r  [{bar}] {done}/{total} trajectories  "
            f"{elapsed:5.1f}s elapsed  ~{eta:5.1f}s left"
        )
        sys.stderr.flush()

    if not quiet:
        print(f"rona {__version__}")
        print(f"  sequence  {name}  ({len(sequence)} nt)")
        energy = config.build_energy(sequence)
        moveset = build_moveset(
            energy.model,
            energy.enc,
            min_loop=config.min_loop,
            min_helix=config.min_helix,
            nucleation_size=config.nucleation_size,
            max_span=config.max_span,
            max_stem_energy=config.max_stem_energy,
        )
        print(f"  move set  {len(moveset.stems)} stems, {len(moveset)} nucleation sites")
        if config.transcription is not None:
            total = config.transcription.total_time(len(sequence))
            print(
                f"  schedule  {config.transcription.rate:g} nt/s, "
                f"footprint {config.transcription.footprint} nt, "
                f"{len(config.transcription.pauses)} pause site(s), "
                f"{total:.2f} s total"
            )
        else:
            print(f"  schedule  none (fixed-length refolding for {config.duration} s)")

    ensemble = simulate_ensemble(
        sequence,
        config,
        n_trajectories=args.trajectories,
        seed=args.seed,
        workers=args.workers,
        progress=progress,
    )
    if not quiet:
        sys.stderr.write("\n")

    written: list[str] = []
    if args.json:
        path = out_dir / f"{prefix}.json"
        ensemble.to_json(path)
        written.append(str(path))
    if args.html:
        from .render.player import PlayerOptions, write_player

        path = out_dir / f"{prefix}.html"
        write_player(
            ensemble,
            str(path),
            options=PlayerOptions(title=f"{name} — cotranscriptional folding"),
        )
        written.append(str(path))
    if args.svg:
        from .render import plots
        from .render.plots import PlotOptions

        for label, content in (
            ("occupancy", plots.occupancy_plot(ensemble)),
            ("energy", plots.energy_plot(ensemble)),
            ("pairprob", plots.pair_probability_plot(ensemble, -1)),
            ("pseudoknots", plots.pseudoknot_plot(ensemble)),
        ):
            path = out_dir / f"{prefix}.{label}.svg"
            path.write_text(content, encoding="utf-8")
            written.append(str(path))

        from .render.svg import DrawOptions, structure_svg

        final, population = ensemble.dominant()[-1]
        path = out_dir / f"{prefix}.final.svg"
        path.write_text(
            structure_svg(
                sequence,
                final,
                options=DrawOptions(
                    title=f"{name}: dominant structure at t = {ensemble.times[-1]:.2f} s",
                    subtitle=f"population {population:.0%}",
                ),
            ),
            encoding="utf-8",
        )
        written.append(str(path))
    if args.movie:
        from .render.movie import MovieOptions, render_movie

        path = out_dir / args.movie if not os.path.isabs(args.movie) else Path(args.movie)
        if not quiet:
            print("  rendering movie ...", file=sys.stderr)
        render_movie(
            ensemble,
            str(path),
            options=MovieOptions(
                mode=args.movie_mode,
                fps=args.fps,
                interpolation=args.interpolation,
                title=f"{name} — cotranscriptional folding kinetics",
            ),
        )
        written.append(str(path))

    if not quiet:
        _report(ensemble, args)
        for path in written:
            print(f"  wrote     {path}")
    return 0


def _report(ensemble, args) -> None:
    print()
    print(f"  {len(ensemble)} trajectories, "
          f"{sum(ensemble.events) / max(len(ensemble), 1):,.0f} events each on average, "
          f"{ensemble.wall_time:.1f} s wall clock")
    print()
    print("  final ensemble (top 5):")
    for structure, population in ensemble.final_distribution()[:5]:
        print(f"    {population:6.1%}  {structure}")
    pk = ensemble.pseudoknot_fraction()[-1]
    if pk > 0:
        print(f"\n  pseudoknotted at the end: {pk:.1%} of the ensemble")
    if args.target:
        series = ensemble.target_population(args.target, max_distance=args.target_distance)
        print(f"\n  target population: final {series[-1]:.1%}, peak {series.max():.1%}")


def cmd_trajectory(args) -> int:
    name, sequence = _read_sequence(args.sequence)
    config = _build_config(args)
    trajectory = simulate_trajectory(sequence, config, seed=args.seed)
    print(f"# {name}  {len(sequence)} nt  seed {args.seed}  "
          f"{trajectory.events} events  {trajectory.wall_time:.2f} s")
    print(f"# {'time':>9}  {'len':>4}  {'G':>8}  structure")
    previous = None
    for frame in trajectory.frames:
        if args.changes_only and frame.structure == previous:
            continue
        previous = frame.structure
        print(f"{frame.time:11.4f}  {frame.length:4d}  {frame.energy:8.2f}  {frame.structure}")
    return 0


def cmd_info(args) -> int:
    name, sequence = _read_sequence(args.sequence)
    config = _build_config(args)
    energy = config.build_energy(sequence)
    moveset = build_moveset(
        energy.model,
        energy.enc,
        min_loop=config.min_loop,
        min_helix=config.min_helix,
        nucleation_size=config.nucleation_size,
        max_span=config.max_span,
        max_stem_energy=config.max_stem_energy,
    )
    print(f"sequence      {name}  ({len(sequence)} nt)")
    print(f"temperature   {config.temperature} C   dangles {config.dangles}")
    print(f"stems         {len(moveset.stems)}")
    print(f"sites         {len(moveset)}")
    if moveset.stems:
        best = sorted(moveset.stems, key=lambda s: s.energy)[:8]
        print("strongest stems (context-free stacking energy):")
        for stem in best:
            print(f"   {stem.i + 1:4d}-{stem.j + 1:4d}  {stem.length:2d} bp  {stem.energy:7.2f} kcal/mol")
    if config.transcription is not None:
        print(f"transcription {config.transcription.total_time(len(sequence)):.2f} s total")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rona",
        description=(
            "Cotranscriptional RNA folding kinetics: stochastic simulation of "
            "how a transcript folds while it is being made."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rona {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fold = subparsers.add_parser(
        "fold", help="simulate an ensemble and write figures, a movie and a player"
    )
    fold.add_argument("sequence", help="RNA sequence, a FASTA file, or - for stdin")
    fold.add_argument("-n", "--trajectories", type=int, default=100,
                      help="number of independent trajectories (default 100)")
    fold.add_argument("-o", "--out", default="out", help="output directory")
    fold.add_argument("--prefix", default=None, help="output file prefix")
    fold.add_argument("--workers", type=int, default=None,
                      help="parallel worker processes (default: CPU count)")
    fold.add_argument("--html", action="store_true", help="write the interactive player")
    fold.add_argument("--svg", action="store_true", help="write SVG figures")
    fold.add_argument("--json", action="store_true", help="write the JSON summary")
    fold.add_argument("--movie", default=None, metavar="FILE",
                      help="write a movie (.mp4 or .gif)")
    fold.add_argument("--movie-mode", choices=("ensemble", "dominant", "trajectory"),
                      default="ensemble", help="what the movie shows (default ensemble)")
    fold.add_argument("--fps", type=int, default=24, help="movie frame rate")
    fold.add_argument("--interpolation", type=int, default=4,
                      help="tweened frames between sampled time points")
    fold.add_argument("--target", default=None, metavar="DOTBRACKET",
                      help="report the population of this structure over time")
    fold.add_argument("--target-distance", type=int, default=0, metavar="BP",
                      help="count structures within this base-pair distance of --target")
    fold.add_argument("-q", "--quiet", action="store_true")
    _add_model_arguments(fold)
    fold.set_defaults(func=cmd_fold)

    traj = subparsers.add_parser("trajectory", help="print one folding pathway")
    traj.add_argument("sequence", help="RNA sequence, a FASTA file, or - for stdin")
    traj.add_argument("--changes-only", action="store_true",
                      help="only print time points where the structure changed")
    _add_model_arguments(traj)
    traj.set_defaults(func=cmd_trajectory)

    info = subparsers.add_parser("info", help="describe the move set for a sequence")
    info.add_argument("sequence", help="RNA sequence, a FASTA file, or - for stdin")
    _add_model_arguments(info)
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
