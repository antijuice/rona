# Lumping the fast mode

## The problem, stated precisely

The simulator's state is a set of helices, each occupying a *window* — an
offset and a length — inside its maximal stem. Two move classes act on it:

| class | rate | what it changes |
|---|---|---|
| nucleate / melt | `k_nucleate` ≈ 10⁵ s⁻¹ | *which* stems are formed |
| zip / unzip | `k_zip` ≈ 10⁶–10⁷ s⁻¹ | one helix's window |

Measured on a 58 nt transcript, **99.7% of all events are zip/unzip**, and
forward and reverse counts agree to three significant figures. The window
variable is at internal equilibrium and merely jittering. Every one of those
events is simulated, and none of them changes the answer.

So: eliminate the window variable, keeping its equilibrium contribution.

## Why the obvious lumping is not obvious

Markov-chain lumping is exact in the limit of infinite timescale separation.
Partition the microstates into blocks `B`, and the process on blocks is Markov
with

```
k(B → B') = Σ_{x∈B} π_B(x) Σ_{y∈B'} k(x→y),      π_B(x) = e^(−G(x)/RT) / Z_B
```

and macrostate free energy `G(B) = −RT ln Z_B`. Detailed balance is inherited
exactly, because

```
k(B→B') Z_B = Σ_{x,y} e^(−G(x)/RT) k(x→y) = Σ_{x,y} e^(−G(y)/RT) k(y→x) = k(B'→B) Z_{B'}
```

The difficulty is `Z_B`. The natural block is "this set of stems is formed, with
any windows", so

```
Z_B = Σ_{window assignments} e^(−G(assignment)/RT)
```

and that sum does **not** factorise over helices. Windows compete for
nucleotides, and loop energies couple every helix bounding a loop. Naively it is
exponential in the number of helices.

## Two ways out

**Exact, by message passing.** The nearest-neighbour free energy is a sum over
loops, and each loop's energy depends only on the *ends* of the helices bounding
it — the inner end of its closing helix and the outer end of each branch. The
loop decomposition of a nested structure is a **tree**. So `Z_B` is a
tree-structured sum and is computable exactly by belief propagation, with one
factor per loop over its incident end-variables. Cost per factor is the product
of the incident domains: `O(L)` for a hairpin, `O(L²)` for an interior loop,
`O(L^(k+1))` for a `k`-branch multiloop.

This is the right answer and it is not intractable. It is, however, expensive
*per candidate move*, and a move needs `ΔG`, not `G` — so it only pays with
incremental message updates. That is a substantial build.

**Ground-state lumping, measured.** Take the block's free energy to be that of
a single canonical window assignment:

```
G(B) := G(windows(B))
```

where `windows(·)` is a deterministic, pure function of the stem set. This
discards the window entropy, at most `RT ln|W|` per helix, and much of that
cancels between states because it appears on both sides of every `ΔG`.

Crucially, because `windows` is a pure function of the stem set:

* `G(B)` is a genuine state function, so the Metropolis rule imposes detailed
  balance on the lumped chain **exactly**;
* `FORM(s)` and `MELT(s)` are exact inverses — `windows(S ∪ {s})` and
  `windows(S)` are each determined by their argument alone, with no history
  dependence;
* no helix can be stranded at a stale window, because the windows are
  recomputed from the stem set after every move. That was the defect that
  forced zipping into the microscopic move set in the first place.

The approximation is therefore *only* in the free energy, not in the
reversibility or the connectivity — and it is directly measurable, by comparing
the lumped stationary distribution over stem sets against the microscopic one
marginalised over windows.

## What is implemented

The second one, with the error measured rather than assumed. `--mode lumped`
runs it.

`windows(S)` (`rona.lumped.canonical_windows`) places the stems of `S` in
increasing stem index, each taking the longest run of its ladder not already
occupied and already transcribed. Fixing the order is what makes it a function
of the set rather than of the history; a stem that cannot reach `min_helix`
under that placement is not a member of the state.

### Rates need a barrier, not just a ΔG

This is the part that is easy to get wrong, and getting it wrong does not show
up in any equilibrium check.

Lumping the window away also lumps away the **transition state**. A helix does
not appear whole: it nucleates `min_helix` pairs and then zips. A chain whose
`FORM` rate is `k min(1, e^(−ΔG_full/RT))` has no nucleation barrier at all, so
it reaches the right equilibrium by the wrong route — it is a fast equilibrium
sampler, which is the one thing this tool exists not to be.

So every rate is built from a transition state:

```
k(A→B) = A exp(−(G_top − G_A)/RT),   k(B→A) = A exp(−(G_top − G_B)/RT)
G_top  = max(G_A, G_B, G_saddle)
```

The ratio is `exp(−ΔG/RT)` for **any** `G_saddle` that is a function of the
unordered pair `{A, B}`, so detailed balance stays exact whatever the barrier
model; and with no barrier it reduces to Metropolis. What `G_saddle` is:

| transition | saddle |
|---|---|
| nucleate or melt one helix, nothing else moving | the helix's **nucleus** — its most stable `min_helix`-pair run |
| anything where a helix also *moves* | the lowest saddle over monotone microscopic paths |

The first is what a microscopic nucleation followed by fast zipping commits at,
`k_nucleate min(1, e^(−ΔG_nuc/RT))`.

### Competing helices, and the trap that exposed them

Two helices that want the same nucleotides cannot both be in a lumped state. If
the only route between them is "melt one, then form the other", the barrier is
the full melt, and a kinetic trap never resolves. That is not an abstraction:
solved exactly, the designed trap in `examples/01_kinetic_trap.py` sat at its
initial 50/50 split **forever**, against a microscopic ensemble that escapes to
96% native within a second.

Two things were needed.

**An exchange move.** One helix replacing a competitor in a single transition.
Candidates come from competitor lists (which stems share a nucleotide) computed
once from the sequence, and the edge exists only when a predicate on *both*
states holds — each blocked by the other and by nothing else — so the two
directions cannot disagree about whether the move is there.

**An honest saddle.** Microscopically the incumbent retracts pair by pair while
the challenger zips into the nucleotides that frees, and neither is ever far
from full length at the top. `rona.lumped.slide_barrier` finds that saddle
exactly, as a **minimum-bottleneck path** (Dijkstra with `max` in place of `+`)
over the microscopic move set — nucleate `min_helix`, zip or unzip one pair,
melt at nucleation length — restricted to the pairs the two endpoints disagree
about, which is what bounds the search. It is memoised on the pair of window
assignments, and the lumped chain revisits the same transitions constantly, so
the cost is amortised to about 15%.

A greedy walk is *not* good enough, and this is worth stating because it is the
obvious thing to write: greedy retracts the incumbent from whichever end is
cheaper, which is precisely the end that does not unblock the challenger. It
overestimated this saddle by about 4 kcal/mol — a factor of 10³ in the rate.

### Two implementations, checked against each other

| | scoring | used for |
|---|---|---|
| `KineticEngine(mode="lumped")` | incremental, loop-local ΔG | simulation |
| `rona.lumped.LumpedEngine` | full `O(n)` evaluations per candidate | reference |

The reference engine carries none of the invalidation apparatus, so it cannot
have the class of bug that apparatus produces. `tests/test_lumped.py` enumerates
both chains exactly and requires the same states, the same edges, the same
energies, and rates agreeing to 1 part in 10⁹.

### What it costs, measured

Both chains enumerated exactly at full length; the microscopic distribution is
marginalised onto stem sets. `max ΔG_window` is the largest gap between the
exact block free energy `−RT ln Z_B` and the canonical-window energy the lumped
chain uses.

| sequence | states (micro → lumped) | equilibrium TV | max ΔG_window |
|---|---|---|---|
| `GCGCAAAAGCGCAAAAGCGC` | 10 → 4 | 0.0013 | 0.016 |
| `GGCAUUGCAAGCAAUGCCAA` | 19 → 4 | 0.0000 | 0.010 |
| `GGGAAACCCAAAGGGAAACCCAAAA` | 7 → 7 | 0.0000 | 0.000 |
| `GCGGAUUUAGCUCAGUUGGGAGAGC` | 58 → 15 | 0.0013 | 0.705 |
| `GGCGCUUGCGCAAAGCGCAAGCGCC` | 133 → 11 | 0.0000 | 0.024 |

The worst-case bound `RT ln|W|` is around 1.8 kcal/mol per helix. The realised
error is one to two orders of magnitude below it: helix ends are bound too
tightly to wander, so the window sum is dominated by one assignment, which is
exactly the condition under which ground-state lumping is accurate.

Equilibrium is the easy half. The **time course**, from both master equations
solved exactly, total variation over stem sets:

| sequence | 10⁻⁴ s | 10⁻² s | 1 s | 10 s |
|---|---|---|---|---|
| `GCGCAAAAGCGCAAAAGCGC` | 0.009 | 0.149 | 0.001 | 0.001 |
| `GGCAUUGCAAGCAAUGCCAA` | 0.163 | 0.000 | 0.000 | 0.000 |
| `GGGAAACCCAAAGGGAAACCCAAAA` | 0.008 | 0.136 | 0.000 | 0.000 |
| `GCGGAUUUAGCUCAGUUGGGAGAGC` | 0.182 | 0.001 | 0.001 | 0.001 |
| `GGCGCUUGCGCAAAGCGCAAGCGCC` | 0.165 | 0.183 | 0.144 | 0.045 |
| trap (44 nt, `examples/01`) | 0.005 | 0.008 | 0.077 | 0.005 |

The transient lags, and it must: a whole helix appears in one lumped event where
the microscopic chain zips it pair by pair, so lumped folding runs slightly
ahead below a millisecond. By the time anything is observable the two agree.

The exception is the fifth row, and it says where the accuracy limit actually
lies. That sequence has a fork: an early state decays either into a long-lived
competing pair of helices or into the eventual ground state, and the two chains
apportion it differently — 0.70/0.25 lumped against 0.49/0.48 microscopic, held
for four decades until both relax to the same equilibrium. Nothing is frozen and
no state is missing; the **branching ratio** at the fork is off, by the ~0.6
kcal/mol that separates two barrier estimates. That is the expected size of
error: the saddle is the lowest over *monotone* paths, which is a lower bound on
the true one, and a few tenths of a kcal/mol at a fork is a factor of 2–3 in a
commitment probability. Read lumped mode as reliable for equilibrium, for
whether a trap resolves and on what timescale, and as approximate for how a
population divides between two competing pathways.

The trap, in populations rather than distances — 9 lumped states against 336
microscopic:

| trapped fraction | 0.1 s | 1 s | 10 s | 100 s |
|---|---|---|---|---|
| microscopic | 0.430 | 0.133 | 0.040 | 0.040 |
| lumped | 0.458 | 0.210 | 0.045 | 0.045 |

### Speed

Measured on the 127 nt crcB riboswitch at full length, starting from the open
chain — the worst case, where every candidate is live. What matters is not events
per second but **simulated time per second of compute**:

| mode | pseudoknots | events/s | µs/event | simulated s per wall s |
|---|---|---|---|---|
| lumped | on | 60 | 16,600 | **0.00091** |
| helix | on | 376 | 2,660 | 0.00022 |
| lumped | off | 200 | 4,990 | **0.00315** |
| helix | off | 3,613 | 277 | 0.00220 |

So lumped buys **4× more simulated time per unit compute with pseudoknots on,
1.4× with them off**, by needing about 15× fewer events at 6–18× the cost each.
It is also, on this system, the only mode that *finishes*: the microscopic chain
needs roughly 9 million events to cover the riboswitch's 13.6 s schedule, against
a default budget of 2 million, so it stops a quarter of the way through and holds
its last structure — which is now recorded and warned about rather than silently
reported as a prediction.

Where the per-event cost goes, counted by call site over 60 events at full length
(full `O(n)` energy evaluations per event):

| | pseudoknots on | off |
|---|---|---|
| saddle search (`slide_barrier`) | 165 | 43 |
| exchange candidates | 46 | 16 |
| crossing form candidates | 87 | 2 |
| everything else | ~2 | ~1 |
| **microscopic mode, for comparison** | **19** | **0** |

Two things stand out. The pseudoknot path is expensive in *both* modes — a
candidate that crosses the structure cannot use a loop-local delta, and with
them switched off the microscopic engine is fully incremental. And lumped mode's
own overhead is the saddle search: it is memoised on the pair of window
assignments, but that key changes whenever any helix moves, so the cache misses
more than it should. Keying it on the local neighbourhood of the trade instead is
the obvious next optimisation, and it is bounded work.

What lumping does *not* fix is worth stating plainly. The events it removes are
the futile zips; what remains is dominated by futile **nucleation** — marginal
helices flickering on and off at ~10⁵ s⁻¹ and changing nothing. The next real
reduction is not another lumping but to stop sampling the lumped chain and
integrate its master equation directly, which is now feasible because the lumped
state space is small enough to enumerate: 9 states where the microscopic chain
has 336.
