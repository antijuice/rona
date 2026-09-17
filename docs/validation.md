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

### Results so far

Predicted probability that each nucleotide is unpaired, against measured
reactivity, over the whole length × position matrix. The polymerase footprint
(14 nt, sequestered and therefore unreactive whatever the structure) and the
35 nt 3' primer-binding cassette are excluded; `n = 6,195` points.

| method | Spearman ρ | median per-length ρ | AUROC |
|---|---|---|---|
| stepwise equilibrium (ViennaRNA per prefix) | **+0.249** | **+0.313** | **0.633** |
| DrTransformer 2.x | +0.223 | +0.239 | 0.575 |
| rona | *pending — see below* | | |

Read honestly:

* **The correlations are modest for everything.** Reactivity reports 2'-OH
  flexibility, not base pairing; an unpaired but stacked nucleotide reads as
  unreactive. Values around 0.25 are not evidence that a model is wrong.
* **Stepwise equilibrium is not beaten here**, and there is a good reason to
  expect that: TECprobe stalls the polymerase with roadblocks and probes the
  *stalled* complex, so the RNA has had far longer at each length than free
  elongation would give it. That is much closer to per-length equilibrium than
  to free cotranscriptional folding. This dataset is therefore a weak
  discriminator between kinetic and equilibrium models, and a fair test needs
  a system where the two are known to differ — a riboswitch decision under
  ligand, or the fluoride riboswitch *with* fluoride, where the published
  analysis shows a ligand-dependent bifurcation.
* **rona's own number is not in yet.** A 127 nt transcript does not finish in
  reasonable time with the corrected move set; see below.

### The blocker

Fixing the zipping defect (§2) made the simulator correct and much slower.
Measured on a 58 nt transcript: **99.7% of all events are zip/unzip**, with
forward and reverse counts equal to three significant figures — the helix
length is at internal equilibrium and merely jittering.

Profiling the 127 nt case found a second cost that *was* fixable: any move near
a crossing helix fell back to full re-evaluation, at 40 pseudoknot conflict
graphs per event. Replacing that with an exact O(1) correction took throughput
from 184 to 336 events/s — but the fast mode still dominates, and a 127 nt
trajectory needs of the order of 10⁶ events.

Removing the fast mode properly means lumping the helix-length degree of
freedom, and that is genuinely hard here: helices compete for nucleotides, so
their window distributions are not independent and the lumped free energy does
not factorise. It is a research-scale task, not an optimisation.
`docs/methods.md` sets out the options. Until then `rona` is practical to
roughly 60–80 nt, and DrTransformer is the better choice above that for nested
structures.

### Reproducing

```bash
python -m rona.validation.shape --help   # module API
python examples/05_timescale_separation.py
```
