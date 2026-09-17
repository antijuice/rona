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

### Which transitions exist

`FORM(s)` from `S`, and `MELT(s)` from `S ∪ {s}`, exist only when

```
windows(S ∪ {s}) restricted to S  ==  windows(S)
```

— that is, when toggling `s` leaves every other helix exactly where it is. The
condition is literally the same expression on both sides, so the two moves are
exact inverses by construction, and the energy change is confined to one loop,
which keeps the incremental `ΔG` valid.

Nothing is lost by the restriction. A stem's canonical window depends only on
stems of *lower* index, so any valid set is still reachable by forming its stems
in increasing index order; and the highest-index member of any set can always
melt, so every state still has a path back to the open chain. What the rule
forbids is a single event that both nucleates one helix and retracts another —
which would need a concerted move, and would break reversibility if allowed.

### Two implementations, checked against each other

| | scoring | used for |
|---|---|---|
| `KineticEngine(mode="lumped")` | incremental, loop-local `ΔG` | simulation |
| `rona.lumped.LumpedEngine` | two full `O(n)` evaluations per candidate | reference |

The reference engine carries none of the invalidation apparatus, so it cannot
have the class of bug that apparatus produces. `tests/test_lumped.py`
enumerates both chains exactly and requires the same states, the same edges,
the same energies, and rates agreeing to 1 part in 10⁹.

### The measured size of the approximation

Both chains enumerated exactly at full length; the microscopic distribution is
marginalised onto stem sets. `max ΔG_window` is the largest discrepancy between
the exact block free energy `−RT ln Z_B` and the canonical-window energy the
lumped chain uses.

| sequence | states (micro → lumped) | total variation | max ΔG_window |
|---|---|---|---|
| `GCGCAAAAGCGCAAAAGCGC` | 10 → 4 | 0.0013 | 0.016 |
| `GGCAUUGCAAGCAAUGCCAA` | 19 → 4 | 0.0000 | 0.010 |
| `GGGAAACCCAAAGGGAAACCCAAAA` | 7 → 7 | 0.0000 | 0.000 |
| `GCGGAUUUAGCUCAGUUGGGAGAGC` | 58 → 15 | 0.0013 | 0.705 |
| `GGCGCUUGCGCAAAGCGCAAGCGCC` | 133 → 11 | 0.0000 | 0.024 |

The worst-case bound `RT ln|W|` is around 1.8 kcal/mol per helix. The realised
error is one to two orders of magnitude below it, and the one state where the
block free energy is off by 0.7 kcal/mol carries little enough population that
the distribution still matches to 0.0013. Helix ends are bound too tightly to
wander: the window sum is dominated by one assignment, which is exactly the
condition under which ground-state lumping is accurate.

Cost, and the cotranscriptional benchmark in this mode, are in
`docs/validation.md`.
