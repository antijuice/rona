# Validating against public data

Three levels of check, from the one that must hold exactly to the one that
actually decides whether the model is useful.

---

## 1. The energy model, against ViennaRNA — *exact*

The Turner 2004 implementation is checked against ViennaRNA 2.7.2 on ~3000
random structures at 37 °C and on MFE structures from 4–90 °C with both dangle
models. Worst absolute deviation **0.009 kcal/mol** — dekacalorie rounding and
nothing else. `tests/test_energy.py`.

## 2a. The tracked energy, against full recomputation — *exact*

The simulator carries the free energy incrementally, adding each move's ΔG
rather than re-deriving it. After every move that sum must still equal a full
recomputation. This is the cheapest test in the suite and it has been the most
productive one: it caught **three** defects that no other test saw, all in
pseudoknotted states, all silently corrupting energies rather than crashing.

| defect | worst error |
|---|---|
| `delta_full` handed a zipped base pair rather than a whole helix removed nothing and returned zero | 15.6 kcal/mol |
| candidates crossing the structure kept a cached score across changes they depend on | 0.6 kcal/mol |
| the pseudoknot fast path assumed a zip inherits its parent helix's crossing status | — |

After the fixes, worst drift over 6000 events on the fluoride riboswitch is
**2.8 × 10⁻¹⁴ kcal/mol**. The regression test runs on a sequence that provably
reaches pseudoknotted states and asserts that it did, so it cannot pass by
avoiding the path.

## 2. The kinetics, against its own equilibrium — *exact*

For every pair of states in the enumerated reachable space, the simulator's own
propensities must satisfy

```
k(X→Y) / k(Y→X) = exp(-(G_Y - G_X) / RT)
```

`tests/test_kinetics.py` checks this transition by transition, to a relative
tolerance of 1e-6, on four systems including pseudoknots and base-pair-resolution
mode. This is what caught a real defect: helices froze at their nucleation
length, so melting one had no inverse. Six of eighteen transitions violated
balance on one system before the fix, zero after.

## 2b. The lumped chain, against the microscopic one — *measured*

`--mode lumped` replaces a helix's window with a function of the set of formed
stems. That is an approximation, so it is checked three ways, all on exactly
enumerated state spaces (`tests/test_lumped.py`):

1. **Against the definition.** The incremental engine's placement, read off an
   owner array, must equal a full canonical placement — and the whole fast
   engine must reproduce the reference engine in `rona.lumped`, which scores
   every candidate by full `O(n)` evaluation: the same states, the same edges,
   the same energies, rates to 1 part in 10⁹.
2. **Detailed balance**, transition by transition, as in §2 — including the
   exchange moves, where one helix replaces a competitor.
3. **Against the microscopic chain**, both master equations solved exactly and
   compared at every time, not just at equilibrium. Equilibrium total variation
   is ≤ 0.0013; the time course stays within 0.18 in the sub-millisecond
   transient (a whole helix appears in one lumped event) and within 0.05 at
   long times.

Check 3 is the one that earns its keep. It caught two barrier defects that no
equilibrium test can see, because both had the *right* stationary distribution:
rates with no nucleation barrier at all, and then a barrier placed at a full
melt rather than at the saddle of a helix-for-helix trade — which froze the
designed trap in `examples/01_kinetic_trap.py` at its initial 50/50 split
forever. The numbers, and the mathematics, are in `docs/lumping.md`.

## 3. Against experiment — *this is the one that matters*

### The data

Cotranscriptional SHAPE-seq measures, for every transcript length, how reactive
each nucleotide is — a proxy for being unpaired and flexible. It is a
two-dimensional observable, length × position, which is exactly the shape of
what a cotranscriptional folding simulation predicts.

`rona.validation.rdat` reads the RDAT format used by the
[RNA Mapping Database](https://rmdb.stanford.edu/). Two entries are directly
usable:

| RMDB ID | RNA | lengths | nt | source |
|---|---|---|---|---|
| `CRCBFL_BZCN_0001` | *B. cereus* crcB fluoride riboswitch, 0 mM fluoride | 108 | 127 | TECprobe-ML, BzCN, 37 °C |
| `ECOSRP_BZCN_0001` | *E. coli* SRP RNA | 186 | 205 | TECprobe-ML, BzCN, 37 °C |

The fluoride riboswitch entry carries **7,938 measured reactivities**.

```bash
curl -L -o CRCBFL_BZCN_0001.rdat \
  https://github.com/DasLab/rmdb.github.io/releases/download/data-general/CRCBFL_BZCN_0001.rdat
```

### Results

Predicted probability that each nucleotide is unpaired, against measured
reactivity, over the whole length x position matrix. The polymerase footprint
(14 nt, sequestered and therefore unreactive whatever the structure) and the
35 nt 3' primer-binding cassette are excluded; `n = 6,195` points.

| method | Spearman ρ | median per-length ρ | AUROC |
|---|---|---|---|
| **rona** (kinetic SSA, 16 trajectories) | **+0.289** | +0.311 | **0.642** |
| stepwise equilibrium (ViennaRNA per prefix) | +0.249 | **+0.313** | 0.633 |
| DrTransformer 2.x | +0.223 | +0.239 | 0.575 |

rona improves monotonically with sampling — ρ = +0.264 at 4 trajectories,
+0.289 at 16 — so what limits it here is Monte Carlo noise rather than the
model.

Read honestly:

* **All three are close.** rona comes out ahead on overall rank correlation
  and on AUROC and is level on the per-length median, but the margins are
  small and this is one RNA under one condition. It is a fair result, not a
  strong one.
* **The correlations are modest for everything**, and that is expected.
  Reactivity reports 2'-OH flexibility, not base pairing; an unpaired but
  stacked nucleotide reads as unreactive. Values around 0.25 are not evidence
  that a model is wrong.
* **This dataset is a weak discriminator between kinetic and equilibrium
  models.** TECprobe stalls the polymerase with roadblocks and probes the
  *stalled* complex, so the RNA has had far longer at each length than free
  elongation would give it — much closer to per-length equilibrium than to
  cotranscriptional folding. Equilibrium doing well here is what one should
  expect, not a surprise.
* **More trajectories would sharpen it further**, since the trend with
  sampling has not flattened.

A fair test of the kinetic claim needs a system where kinetics and equilibrium
are *known* to give different answers: the fluoride riboswitch **with** ligand,
where the published analysis shows a ligand-dependent bifurcation that delays or
promotes terminator formation. That is the obvious next experiment.

### Cost, and a caveat on the numbers above

The 127 nt riboswitch, on four cores: **842 s** for 4 trajectories, **3,328 s**
for 16, at roughly 710,000 events per trajectory on average.

That average hides something that only came to light later, and it belongs with
the results: the default event budget is 2 million per trajectory, and covering
this schedule in `helix` mode takes about **9 million**. Trajectories that fold
into a low-propensity state early finish comfortably; those that do not stop
partway and hold their last structure for the rest of the run. Late transcript
lengths and the post-transcriptional equilibration are therefore frozen rather
than simulated in some fraction of the ensemble, which will if anything have cost
`rona` accuracy here rather than flattered it. `Trajectory.truncated` and
`Ensemble.truncated` now record this and the CLI warns about it; `--mode lumped`
needs about 15× fewer events and does not hit the cap on this input.

`docs/lumping.md` has the per-mode cost measurements. DrTransformer did the same
input in seconds; `rona` is the slower tool by a wide margin, and
`docs/methods.md` explains why and what would fix it.

### Reproducing

```bash
python examples/06_shape_benchmark.py -n 16     # ~55 min on four cores
python examples/06_shape_benchmark.py -n 4      # ~14 min
python examples/05_timescale_separation.py
```

The script downloads the RDAT file itself. Pass ``--drf`` a DrTransformer time
course to include it in the comparison:

```bash
pip install drtransformer
printf '>crcb\n%s\n' "$SEQ" | DrTransformer --name crcb --outdir drt --t-ext 0.0333
python examples/06_shape_benchmark.py --drf drt/crcb.drf
```
