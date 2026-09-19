# v2: a certified cotranscriptional kinetic ensemble

## Why v1 is being replaced

v1 is a Gillespie SSA over secondary structures with helix-level moves,
cotranscription as a growing chain, and detailed balance against a Turner 2004
energy model. That is the same mathematical object Kinefold has been since
Isambert & Siggia (2000), with a weaker pseudoknot model, no knot topology, no
exactly-clustered stochastic simulation, and worse performance. The "lumped"
mode added late in v1 - eliminating the base-pair zipping degree of freedom -
independently rediscovered Kinefold's founding assumption, that stacking and
unstacking are quasi-equilibrated relative to the transitions between visited
secondary structures.

The failure is not implementation quality. Parts of v1 are good: the energy
model reproduces ViennaRNA to 0.009 kcal/mol, detailed balance is verified
transition by transition, and the SHAPE benchmark is honest. The failure is
architectural, and it is this:

**Trajectory sampling is the wrong algorithm for this question.**

The question is "what is the distribution over structures at each time during
transcription". An SSA answers it by simulating individual histories and
averaging. Its cost scales with the number of *events*, and the event count is
set by the fastest mode in the system - base-pair zipping at ~10^6 s^-1 - while
the answer lives on the transcription timescale of seconds. So one pays ~10^6
events per second of simulated time to learn about a process that changes on a
1-second timescale, and 99.7% of those events change nothing about the answer.
Running more trajectories reduces variance; it does not touch this. No amount of
engineering fixes a mismatch between what the algorithm costs and what the
question needs.

## What the field does instead

Every cotranscriptional tool built since Kinfold has moved away from sampling
elementary events, towards *deterministic integration of the master equation on
a coarse-grained state space*:

| tool | state space | integration |
|---|---|---|
| BarMap | barrier tree per transcription step, mapped between steps | Treekin |
| DrTransformer | representative local minima, found per step | numeric, then prune |
| StraD | constrained local flooding | flooding-based |
| landscape-zooming (Xu & Chen) | partitions by long stable helices | inter-partition network |

Cost per unit of *simulated time* is essentially zero for all of them; the whole
cost is in choosing which states to keep. That is the right trade for this
question.

## The gap, and what v2 is

All of them prune heuristically. DrTransformer removes part of the ensemble
after each simulation and repopulates from the remainder; none of these tools
reports how much probability mass it discarded, or what that discarding does to
the answer. The output is a distribution with no error bar.

Meanwhile the numerical-analysis literature has solved exactly this problem for
the chemical master equation, and has for twenty years:

* **Finite State Projection** (Munsky & Khammash 2006) truncates the state space
  and returns *a certificate*: the probability mass that has left the retained
  set is a rigorous upper bound on the L1 error of the retained distribution.
* **Adaptive FSP with quantile pruning** (Dendukuri et al. 2025) proves the
  truncation error is bounded by the pruned mass at each step, is user-controlled
  and **does not propagate forward in time**.
* **Krylov-FSP** (Burrage et al. 2006; Kormann & Loehner 2016) approximates the
  matrix exponential on the truncated generator, giving intermediate times for
  free and error estimates compatible with the Krylov approximation.

This is standard practice in systems biology and appears to be absent from the
RNA folding literature entirely.

**v2 is cotranscriptional folding posed as a time-inhomogeneous CME over
secondary structures, solved by adaptive Finite State Projection, reporting a
certified bound on the error of every distribution it outputs.**

The deliverable is not "a prediction". It is "a prediction, and a number that
says how wrong it can be". Nothing in the field currently offers that, and it is
what makes the ensemble honest rather than merely plausible.

## The construction

**State.** A secondary structure, represented as a set of helices. The retained
set `S(t)` is explicit and finite.

**Generator.** `A` is the sparse rate matrix over `S(t)`, built from the move set
and the energy model, with `k(x->y)/k(y->x) = exp(-(G_y - G_x)/RT)` exactly. This
part carries over from v1, which verifies it transition by transition.

**Propagation.** `p(t + dt) = exp(A dt) p(t)` by Krylov (Arnoldi) on the sparse
`A` - never forming the matrix exponential, never sampling. Stiffness is handled
by the exponential integrator rather than avoided: a 10^6 s^-1 mode costs
nothing here, because it is integrated rather than enumerated.

**Truncation and its certificate.** `A` is defective by construction: rows for
states whose neighbours are outside `S` do not conserve probability. That leak
is the error bound. Each step:
1. propagate on `S`;
2. measure the escaped mass `eps_step`;
3. expand `S` along the leaking edges and re-propagate until `eps_step < tol`;
4. prune the smallest-probability states by quantile, adding the pruned mass to
   the certificate.
The reported bound is the accumulated `sum(eps)` - a real number attached to
every output distribution.

**Transcription.** Each elongation is a new, larger state space and a new
generator. Mapping `p` from step `n` to step `n+1` is *exact*, not a heuristic:
appending a nucleotide is an injective map on structures, so probability is
carried across unchanged. This is where BarMap needs its landscape maps and
where v2 needs nothing.

## Physics to fix at the same time

These are independent of the numerics and each is a known defect of v1.

1. **Branch migration as an elementary move.** R2D2 (Yu et al. 2021, *Mol Cell*)
   showed E. coli SRP RNA rearranges by internal toehold-mediated strand
   displacement; ANNaMo (Guerra et al. 2024) reproduces displacement rate vs
   toehold length from a coarse-grained model. v1 had no such move, so a helix
   could only be replaced by melting it first - which is why v1 needed a
   minimum-bottleneck saddle search to get trap escape approximately right.
   With branch migration in the move set the barrier is the mechanism, not a
   correction to it.
2. **Pseudoknot energetics from polymer theory.** v1 uses three hand-tuned
   constants. Kinefold uses stiff rods for helices and polymer springs for single
   strands with an explicit confinement factor; Vfold2D-MC derives loop entropies
   by virtual-bond Monte Carlo. Either is a model; neither is three numbers
   chosen to make examples work.
3. **Loop entropy for large loops.** Same source, same reason.

## What carries over from v1

Kept, because it is verified and rebuilding it would be waste, not virtue:

* `rona.energy` - Turner 2004, 0.009 kcal/mol against ViennaRNA across 4-90 C
  and both dangle models, with the pair-type padding and truncation subtleties
  already found and fixed;
* `rona.struct` - structure representation, dot-bracket I/O, helix extraction;
* `rona.validation` - RDAT reader and the SHAPE benchmark harness;
* `rona.render` - the rendering and the player.

Discarded: `rona.kinetics`, `rona.lumped`, `rona.cotrans` - the SSA engine and
everything built on it. That is the part that was a worse Kinefold.

## How it gets validated

The certificate is the first thing to test, and it is testable *exactly*:

1. On systems small enough to enumerate completely, solve the full master
   equation and the FSP-truncated one. The true L1 error must be at or below the
   reported bound, always. A certificate that is ever violated is a bug, and this
   is a hard pass/fail, not a correlation.
2. The bound must be tight enough to be useful, not merely valid - measure the
   ratio of true error to reported bound.
3. Against the field's benchmarks: E. coli SRP RNA (R2D2, landscape-zooming,
   Badelt's guide and Sun & Chen all use it) and the pbuE riboswitch with and
   without ligand, where kinetics and equilibrium are known to give different
   answers. The crcB TECprobe data from v1 stays as a regression check.
4. Against DrTransformer on the same inputs, which is the tool this most
   directly competes with.

## What would make this fail

Stated in advance, so it is not rationalised later:

* If the retained set needed to hold `eps` below a useful tolerance grows
  exponentially with length for real sequences, the method is exact and useless.
  This is the central empirical risk and the first thing milestone 2 measures.
* If FSP's bound is technically valid but astronomically loose, the certificate
  is decoration.
* If branch migration explodes the move set's branching factor, the generator
  becomes too dense to propagate.

Each of these is measurable early, and each is a reason to stop rather than to
keep building.

---

## Measured, milestone 1

What is built and verified (`rona.master`, `tests/test_master.py`):

| property | checked against | result |
|---|---|---|
| reversibility | Boltzmann ratio on every edge | 0 one-way of 52,976; worst violation 2e-14 |
| long-time limit | Boltzmann distribution from the energy model | L1 = 3e-11 |
| time integration | scipy's exact matrix exponential | matches; error *estimate* accurate to ~7% |
| positivity, mass | - | non-negative, mass 1.000000000 |
| FSP certificate | error measured on the fully enumerated space | never violated, and exactly tight |

Three defects were found by those checks rather than by reasoning, and each is
worth keeping in view:

1. **Helix-level moves are not self-inverse.** Nucleating a block beside an
   existing helix yields one longer helix whose whole-melt is never offered.
   119 of 1155 edges were one-way on a 25 nt sequence; the chain was
   irreversible and drifted *away* from Boltzmann the longer it ran. Moves are
   now single base pairs, where add and remove are inverses by construction.
   This is the same defect class as v1's stranded helices, and the lesson is
   the same: a move set defined on helices has to have its reversibility
   proved case by case, and will not survive it.
2. **Stiffness moves into the linear algebra.** It does not go away by
   abandoning sampling. `expm_multiply` does not finish at `||A|| t ~ 10^7`;
   explicit Krylov overflows. Backward Euler works because it is L-stable and
   `(I - dt A)` is an M-matrix, so probabilities stay non-negative.
3. **Truncation error and integration error are different things.** Conflating
   them reported a bound of 3e-11 on an answer that was 7e-2 wrong. The
   certificate covers truncation only.

### The result that matters

The full state space is exponential, as expected:

| length | structures |
|---|---|
| 12 nt | 49 |
| 18 nt | 80,232 |

The mass-carrying support is not:

| | full space | 99.99% of mass | 99.9999% |
|---|---|---|---|
| 12 nt | 49 | 9 | 18 |
| 18 nt | 80,232 | 20 | 84 |

The space grew 1,637x; the support grew 4.7x. **84 structures out of 80,232
carry all but one part in a million of the equilibrium distribution.** This is
the regime FSP exists for, and it is the central premise of v2 holding up.

### The obstacle, and where its answer lives

Loss-only FSP is too conservative for this system. With rates of 10^7 s^-1, a
2 ms step lets probability make ~10^4 hops, and the method counts *every*
excursion out of the retained set as permanent loss - including a brief visit to
a high-energy structure that would return immediately. The bound stays valid and
becomes useless: 1.0 at any step long enough to be interesting.

This is a known property, and the literature's answer is a **reflecting
boundary**: return escaped mass to the boundary rather than discarding it, and
bound the error by the probability resident on the boundary instead. Cao, Terebus
& Liang (*Bull Math Biol* 2016) prove the bound for that construction and show
it is asymptotically tight. That is milestone 2, and it is a change to the
boundary condition, not to anything already verified above.

---

## Measured, milestone 2

### The boundary condition decides whether the method works

Both boundaries on a 15 nt sequence with 4,344 reachable structures, against the
exact solution on the full space:

| boundary | tolerance | states | mass retained | true L1 at t = 0.1 s |
|---|---|---|---|---|
| absorbing | 1e-2 | 27 | 0.000 | 1.000 |
| absorbing | 1e-4 | 27 | 0.000 | 1.000 |
| reflecting | 1e-2 | 20 | 1.000000 | 7.1e-3 |
| reflecting | 1e-3 | 49 | 1.000000 | 4.6e-4 |
| reflecting | 1e-4 | **81** | 1.000000 | **7.8e-5** |

The absorbing bound degenerates to 1.0 beyond a few microseconds, exactly as the
gross-flux argument predicts. The reflecting chain gives 1e-4 accuracy from 81
of 4,344 structures - 1.9% of the space - and the answer does not drift between
1 ms and 0.1 s, because the retained chain is a proper process rather than a
leaking one.

### The certificate, made rigorous by something only this domain has

The reflecting chain's equilibrium is the Boltzmann distribution conditioned on
the retained set, so

```
|| pi|_S - pi ||_1  =  2 (1 - Z_S / Z)
```

`Z_S` is a sum over states held. `Z` is the partition function over *every*
secondary structure, and McCaskill's algorithm computes it exactly in `O(n^3)`
without enumerating anything. The generic chemical-master-equation setting has
no such thing; RNA does, and it converts the boundary indicator into a real
number:

| tolerance | states | equilibrium weight outside | measured L1 |
|---|---|---|---|
| 1e-2 | 20 | 3.49e-3 | 7.1e-3 |
| 1e-3 | 49 | 1.82e-4 | 4.6e-4 |
| 1e-4 | 81 | < 1e-12 | 7.8e-5 |

Summing the enumerated Boltzmann weights against McCaskill's `Z` gives 1.000047,
so this package's energy model and the one supplying `Z` agree to 5e-5 - the
certificate rests on that agreement and is no better than it.

Two limits, stated rather than buried. It certifies the **equilibrium**
component; a transient distribution can be wrong in ways it does not see, which
is why the dynamic indicator is reported alongside it. At t = 0.1 s the measured
L1 of 7.1e-3 slightly exceeds the equilibrium bound of 6.98e-3, and the
difference is the transient that has not yet decayed plus the model disagreement
above - which is the bound behaving correctly, not failing.
