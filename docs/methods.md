# What to borrow from the other cotranscriptional folding tools

An assessment of methods used by Kinefold, DrTransformer, Kinfold, CoStochFold
and DrForna, and whether `rona` should adopt them. Ordered by expected value.

Status key: **adopted** · **recommended** · **considered and rejected**

---

## 1. Handling fast modes — the binding constraint · *recommended*

**The problem, measured.** `rona` simulates every elementary event. A helix
with an in-context ΔG near −2 kcal/mol genuinely opens and closes thousands of
times a second, and adding base-pair zipping (§2) made this worse: zipping fires
at ~10⁷ s⁻¹ against ~10⁵ s⁻¹ for nucleation, so the overwhelming majority of
events do not change the coarse structure at all. A 127 nt riboswitch at 30 nt/s
did not finish a single trajectory in ten minutes. This is *the* reason `rona`
is slower than every tool it is compared against.

**What the others do.**

* **Kinefold** uses what its authors call *exactly clustered stochastic
  simulation*: rapidly interconverting states are grouped and the group is
  sampled as a unit, so the fast mode costs one event instead of thousands.
* **DrTransformer** exposes `--t-fast`: transitions faster than a cutoff are
  assumed equilibrated and the states they connect are lumped into one
  macrostate with a Boltzmann-weighted interior. Because it then integrates the
  master equation deterministically, it pays nothing per event at all — it
  finished the same 127 nt riboswitch in seconds.

A second, separate cost was found by profiling and *is* fixed: every move near
a crossing helix fell back to full re-evaluation, costing 40 pseudoknot conflict
graphs per event. A helix that crosses nothing cannot change the core/pseudoknot
split or any region's extent, so only the per-unpaired term of the topology
penalty moves — an O(1) correction. Throughput on the 127 nt riboswitch went
from 184 to 336 events/s, and three energy bugs surfaced on the way (see
`docs/validation.md` §2a).

**Assessment.** Kinefold's and DrTransformer's answers are the same idea:
exploit timescale separation by lumping. The honest ranking of options for
`rona`:

1. *Rate ceiling on the fast mode* (cheap, testable). The zipping rate only has
   to be fast relative to the slow modes; its exact value should not change the
   coarse kinetics. This is checkable by running the same system at several
   `k_zip` values and comparing coarse observables — see
   `examples/05_timescale_separation.py`. It does not disturb detailed balance,
   because the stationary distribution does not depend on a rate prefactor.
2. *Proper lumping of helix length* (correct, hard). Treat a helix's window as
   an internal degree of freedom at local equilibrium and give the coarse state
   its lumped free energy `-RT ln Σ exp(-G_i/RT)`. This removes the fast mode
   entirely rather than slowing it down. The difficulty is that helices compete
   for nucleotides, so their window distributions are *not* independent and the
   sum does not factorise.
3. *Full ECS / macrostate lumping across all fast transitions* (the Kinefold
   answer; substantial work).

Option 1 is implemented and its validity is tested rather than assumed.

Option 2 is **implemented** as `--mode lumped`, with the block free energy taken
at a single canonical window assignment rather than summed — a ground-state
lumping, whose error is then measured exactly rather than argued about: total
variation against the microscopic distribution ≤ 0.0013, and the neglected
window entropy 0.003–0.7 kcal/mol against a worst case of 1.8. The full
mathematics, including the exact route by belief propagation over the loop tree
that was *not* taken and why, is in `docs/lumping.md`.

What that write-up makes clear is that the interesting part was not the
partition function at all. It was the kinetics: lumping the window away also
lumps away the transition state, so the rates have to be rebuilt around explicit
barriers — a nucleus for an ordinary nucleation, and a minimum-bottleneck saddle
for a helix-for-helix trade — or the chain reaches the right equilibrium by the
wrong route and freezes in kinetic traps. It also shows that option 2 was not
where the remaining cost lived: with zipping gone, the events are dominated by
futile *nucleation*, which is untouched by it. The recommended next step is now
option 4 below.

4. *Integrate the lumped master equation instead of sampling it* (DrTransformer's
   approach, now feasible). Sampling spends its events on transitions that
   change nothing; a deterministic propagation over the lumped graph spends none.
   The obstacle used to be that the microscopic state space is far too large to
   enumerate — 336 states for a 44 nt trap. The lumped graph for the same system
   has **9**, small enough to build on the fly and integrate directly, which is
   exactly the representation DrTransformer needs and did not previously exist
   here.

---

## 2. Base-pair zipping within a helix · *adopted*

Kinfold and CoStochFold work at single-base-pair resolution; Kinefold forms
helices as units. `rona` originally did the latter, and it was wrong in a way
that only showed up when tested: a helix that nucleated while its 3' end was
still inside the polymerase could never extend afterwards. That state arose in
**33% of visited states**, and it also broke detailed balance, because melting
such a helix had no inverse — re-forming it would produce the longer helix
instead.

Zipping moves (`ZIP_IN`/`ZIP_OUT`, reversed by `UNZIP_*`) now exist in the
default move set, and melting is only offered when a helix is at the extent that
re-forming it would produce. `tests/test_kinetics.py` checks every transition of
the enumerated state space against `k(X→Y)/k(Y→X) = exp(-(G_Y-G_X)/RT)`; there
are now zero violations, where the previous move set had six out of eighteen on
one system.

The cost is §1.

---

## 3. Deterministic master-equation integration · *recommended*

DrTransformer does not sample trajectories at all. It builds a coarse-grained
landscape for each transcription step and integrates the master equation, which
gives smooth populations with no sampling noise and no per-event cost. It is
orders of magnitude faster than `rona` on the same input.

**Assessment.** Worth adding as an alternative engine rather than a replacement.
`rona` already has everything needed — the move set, the rates, and an exact
energy model — so the missing piece is: enumerate a pruned state set per length,
assemble the rate matrix, and integrate. The stochastic engine would remain
useful for what it is genuinely better at: sampling individual pathways, and
handling pseudoknots, which the coarse-grained nested landscape machinery does
not cover.

The honest summary is that for *nested* cotranscriptional folding on sequences
over ~100 nt, DrTransformer is currently the better tool and `rona` should say
so.

---

## 4. Polymer-entropy loop energies for pseudoknots · *recommended*

`rona`'s pseudoknot term is a tunable penalty of the Dirks–Pierce shape bolted
onto the Turner model. Kinefold instead derives loop entropies from polymer
physics (the Isambert–Siggia treatment), which extends naturally to crossing
helices and even to knots, and is not a fitted constant.

**Assessment.** The right long-term answer for the pseudoknot term specifically.
It would replace `topology_penalty` with a computed loop entropy while leaving
the nested core on exact Turner energies, so the agreement with ViennaRNA on
nested structures would be preserved. Moderate work, and it would remove the
most hand-waved part of the current model.

---

## 5. Helix shift moves · *considered*

Kinfold includes shift moves, which slide a helix by one register without fully
melting it, lowering a barrier that is genuinely lower than melting.

**Assessment.** Partly obtained for free: zipping (§2) lets a helix change
length one pair at a time, and overlapping stems give alternative registers. A
true atomic shift — melt stem A and form overlapping stem B in one move — is
implementable with a unique inverse (the reverse shift), so detailed balance is
safe. Worth adding, but it is a refinement rather than a fix, and it should come
after §1.

---

## 6. Landscape pruning by occupancy · *partly adopted*

DrTransformer's `--o-prune` drops states below an occupancy threshold.
`rona` prunes at the candidate level instead (`--max-stem-energy` discards
stems that cannot survive any loop context), which is cheaper but blunter. An
occupancy-based prune only makes sense for an ensemble-level engine, so it comes
with §3.

---

## 7. DrForna's ensemble visualisation · *adopted*

Colouring base pairs by their "imaginary centre" `(i+j)/2` so a helix keeps one
colour for its lifetime; structures drawn as rectangles with area proportional
to occupancy; a time overview of nucleotide against time; a time axis that is
linear during transcription and logarithmic afterwards. All four are in
`rona.render.overview` and appear in the movie, the SVG figures and the player.

`rona` also reads and writes DrForna's `.drf` time-course format
(`rona.render.drforna`), so a `rona` ensemble opens in DrForna and a
DrTransformer run can be read back for comparison.

---

## 8. Experimental validation against chemical probing · *adopted*

Not a modelling method, but the thing that decides whether any of the above
matters. See `docs/validation.md`.
