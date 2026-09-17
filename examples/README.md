# Examples

```bash
python examples/01_kinetic_trap.py    # cotranscriptional vs equilibrium
python examples/02_pause_site.py      # elongation rate and pausing change the outcome
python examples/03_pseudoknot.py      # pseudoknot formation and its parameter sensitivity
python examples/04_render.py          # figures, interactive player and a movie
```

Or from the command line:

```bash
rona fold examples/sequences/kinetic_trap.fa -n 200 --html --svg --movie trap.mp4
rona fold examples/sequences/trna_phe_yeast.fa -n 100 --html --svg --rate 20
rona trajectory examples/sequences/kinetic_trap.fa --changes-only
```

`01_kinetic_trap.py` is the one to run first. It is the shortest demonstration
that this tool answers a different question from an equilibrium folder, and the
claim it makes is checkable against ViennaRNA.

`05_timescale_separation.py` checks that the coarse folding result does not
depend on how fast zipping is — the assumption that lets `k_zip` be lowered for
tractability.

`06_shape_benchmark.py` scores rona, stepwise equilibrium and (optionally)
DrTransformer against public cotranscriptional SHAPE-seq data. It downloads the
data itself. About 55 minutes on four cores; use `-n 4` for a quick look.
