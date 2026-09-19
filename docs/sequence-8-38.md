# The supplied 316 nt sequence, and what happens to nucleotides 8–38

A question asked of a specific sequence, answered as far as the method honestly
reaches and no further.

```
GUCGAUAGGAGUGUUUGUUCCUUGAAAACUACAUAUAGACAAAAAAAGCGCCUUGACGAAUGAUCAUCAAG…  (316 nt)
        ^------------- 8 to 38 -------------^
window 8–38 = GGAGUGUUUGUUCCUUGAAAACUACAUAUAG
```

## What the kinetic solver can reach

Not 316 nt on this sequence, and the reason is physical rather than a limitation
of the arithmetic. The sequence is AU-rich with many competing marginal helices,
so nothing in it becomes certain enough to anchor: at 40 nt the ensemble holds
6,725 structures, still 1.8e-2 of the equilibrium weight short of its target, with
zero anchors promoted after 581 s. The anchored representation of
`rona.master.anchored` correctly buys nothing here — there is no product structure
to exploit, because there are no confidently formed separating pairs. A method
reporting a speedup on this sequence would be reporting an error it had not
measured.

So the kinetic result covers the first ~40 nt, which happens to be exactly where
the window of interest does its folding.

## The kinetic result: the window's hairpin is a trap

The window folds a hairpin on itself almost immediately, and then holds it past
the point where equilibrium says it should have let go.

| transcript | pair 9–20, kinetic | pair 9–20, exact equilibrium |
|---|---|---|
| 30 nt | 0.994 | 0.990 |
| 40 nt | 0.772 | 0.473 |

At 30 nt the two agree, which is the check that the kinetic number means
something: the hairpin has had time to equilibrate and it has. By 40 nt the
downstream sequence has arrived and equilibrium has moved on — the hairpin's share
falls to 0.47 — but the molecule is still holding it at 0.77. That gap, 0.30, is
well outside the 1.8e-2 the certificate admits to missing, so it is a result and
not noise: **the 8–38 hairpin is metastable, retained above its equilibrium
occupancy while the transcript grows past it.**

The hairpin itself, at 30 nt, is

```
.(((..(((((......))))))))....
```

with pairs 8–21, 9–20, 10–19 all above 0.98 and 11–18 at 0.63. Note that it is
*not* self-contained: pairs 7–22, 3–24 and 2–25 run from the window into the
upstream 5' end at probabilities 0.57, 0.49 and 0.43. The window is the inner part
of a longer stem that starts at nucleotide 2.

## The equilibrium picture, which does reach 316 nt

McCaskill is `O(n^3)` regardless of how heterogeneous the ensemble is, so the
equilibrium fate of the window can be followed the whole way. Expected pairs, by
where the partner sits:

| prefix | pairs inside 8–38 | partner 5′ of window | partner 3′ of window | dominant pair |
|---|---|---|---|---|
| 20 nt | 0.64 | 2.31 | 0.00 | 8–15 (0.31) |
| 30 nt | 3.70 | 2.99 | 0.00 | 9–20 (0.99) |
| 40 nt | 5.15 | 3.11 | 0.00 | 9–20 (0.47) |
| 50 nt | 5.76 | 3.20 | 0.07 | 11–29 (0.69) |
| 60 nt | 2.33 | 0.02 | 8.61 | 19–38 (0.48) |
| 120 nt | 2.35 | 0.02 | 8.67 | 19–38 (0.47) |
| 160 nt | 0.27 | 0.51 | 17.35 | 9–20 (0.04) |
| 200 nt | 5.52 | 3.07 | 0.41 | 11–29 (0.63) |
| 280 nt | 4.53 | 2.46 | 2.93 | 11–29 (0.49) |
| 316 nt | 1.65 | 0.69 | 10.06 | 29–38 (0.44) |

Read down the last two columns and the story is a takeover. Up to 50 nt the window
pairs with itself and with the 5′ end. From 60 nt onwards downstream sequence
competes for it and wins: at full length about 10 of the window's 31 nucleotides
are paired to positions outside it and downstream, and the local 9–20 hairpin is
gone (0.04 at 160 nt).

The non-monotonicity is worth flagging rather than smoothing over. 50 nt and 80 nt
give nearly identical numbers (5.762 and 5.763 internal pairs), as do 60 nt and
120 nt (2.329 and 2.354), and 200 nt returns to the 50 nt configuration. The window
is bistable between a local fold and a long-range one, and which wins depends on
which downstream partner the polymerase has produced. That is a real feature of
the sequence, but it also means any single-length equilibrium prediction for this
window is fragile.

## What this does and does not say

It says the window folds a strong local hairpin during the first 30 nucleotides of
transcription, holds it metastably at least to 40 nt, and that equilibrium at full
length wants it unfolded and paired downstream instead. That combination —
early local fold, later long-range partner — is the setup for a kinetic trap that
persists on biological timescales, and it is the interesting case.

It does not say whether the trap actually survives to 316 nt, because that is a
kinetic question at a length this solver cannot reach on this sequence. Answering
it needs either a representation that works without confident anchors, or a
coarse-graining accepted as an approximation with its error stated — not the
certified microstate ensemble used here.
