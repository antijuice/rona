"""Benchmark predicted folding against cotranscriptional chemical probing.

Cotranscriptional SHAPE-seq (and its TECprobe variants) measure, for every
transcript length, how reactive each nucleotide is - a proxy for how unpaired
and flexible it is.  That is a *two-dimensional* observable, length x position,
and it is exactly the shape of what a cotranscriptional folding simulation
predicts.  It is therefore the most direct public test available.

What is compared
----------------
For each transcript length ``L`` in the dataset:

* **measured** - reactivity of nucleotides ``1..L``;
* **rona** - the probability that each nucleotide is unpaired in the kinetic
  ensemble, sampled at the moment just before the next nucleotide is added;
* **stepwise equilibrium** - ViennaRNA's equilibrium unpaired probability for
  the prefix, the standard "fold every prefix" approximation.

Caveats, which matter
---------------------
* Reactivity reports 2'-OH flexibility, not base pairing as such.  An unpaired
  but stacked or protein-bound nucleotide can read as unreactive.
* The nucleotides inside the polymerase footprint are sequestered, so they are
  unreactive regardless of structure while the model calls them unpaired.
  ``footprint_mask`` excludes them; results are reported both ways.
* Roadblock protocols probe *stalled* complexes, so the RNA has had longer at
  each length than free elongation would give it.  The elongation rate is
  therefore not directly transferable, which is why
  :func:`scan_elongation_rate` exists - and why its output is a sensitivity
  analysis, not a fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..cotrans import SimulationConfig, TranscriptionSchedule
from ..ensemble import Ensemble, simulate_ensemble
from ..struct import parse_dotbracket


# ----------------------------------------------------------------------
# statistics (no scipy dependency)
# ----------------------------------------------------------------------
def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, robust to the unknown reactivity scale."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return pearson(_rank(a[mask]), _rank(b[mask]))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a positive outranks a negative (ties count as a half)."""
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[mask], labels[mask].astype(bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rank(scores)
    return float((ranks[labels].sum() - positives * (positives - 1) / 2) / (positives * negatives))


# ----------------------------------------------------------------------
# predictions
# ----------------------------------------------------------------------
def length_sampling_grid(
    schedule: TranscriptionSchedule, n: int, lengths: Sequence[int]
) -> tuple[list[float], list[int]]:
    """One sample per requested transcript length, taken just before elongation.

    Probing captures the chain as it is at that length, so the informative
    moment is the end of the dwell at ``L``, not the instant it was reached.
    """
    arrivals = schedule.arrival_times(n)
    grid: list[float] = []
    kept: list[int] = []
    for length in lengths:
        if length < 1 or length > n:
            continue
        if length < n:
            time = arrivals[length + 1] - 1e-9
        else:
            time = arrivals[n] + schedule.post_time
        grid.append(max(time, 0.0))
        kept.append(length)
    order = np.argsort(grid, kind="mergesort")
    return [grid[k] for k in order], [kept[k] for k in order]


def unpaired_matrix(ensemble: Ensemble, lengths: Sequence[int]) -> np.ndarray:
    """``P(unpaired)[row, position]`` for the ensemble, one row per length."""
    size = len(ensemble.seq)
    out = np.full((len(lengths), size), np.nan)
    by_length: dict[int, int] = {}
    for index, length in enumerate(ensemble.lengths):
        by_length[length] = index  # last sample at that length wins
    for row, length in enumerate(lengths):
        index = by_length.get(length)
        if index is None:
            continue
        counts = np.zeros(size)
        structures = ensemble.structures[index]
        for db in structures:
            pt = parse_dotbracket(db)
            for position, partner in enumerate(pt):
                if partner < 0:
                    counts[position] += 1.0
        out[row, :length] = counts[:length] / max(len(structures), 1)
    return out


def equilibrium_matrix(
    sequence: str, lengths: Sequence[int], *, footprint: int = 0
) -> np.ndarray:
    """Stepwise-equilibrium baseline: fold each prefix to its Boltzmann ensemble.

    Requires ViennaRNA.  ``footprint`` folds only the part of the prefix that
    has left the polymerase, which is what practitioners actually do and makes
    this the stronger version of the baseline.
    """
    import RNA

    size = len(sequence)
    out = np.full((len(lengths), size), np.nan)
    for row, length in enumerate(lengths):
        usable = max(0, length - footprint)
        if usable < 2:
            out[row, :length] = 1.0
            continue
        prefix = sequence[:usable]
        fold = RNA.fold_compound(prefix)
        fold.pf()
        probabilities = fold.bpp()
        paired = np.zeros(usable)
        for i in range(1, usable + 1):
            for j in range(i + 1, usable + 1):
                p = probabilities[i][j]
                if p:
                    paired[i - 1] += p
                    paired[j - 1] += p
        out[row, :usable] = 1.0 - np.clip(paired, 0.0, 1.0)
        out[row, usable:length] = 1.0  # inside the polymerase: treated as unpaired
    return out


# ----------------------------------------------------------------------
@dataclass(slots=True)
class BenchmarkResult:
    """Agreement between a prediction and a probing dataset."""

    name: str
    overall_spearman: float
    per_length_spearman: np.ndarray
    auroc: float
    n_points: int
    lengths: list[int]

    @property
    def median_per_length(self) -> float:
        values = self.per_length_spearman[np.isfinite(self.per_length_spearman)]
        return float(np.median(values)) if len(values) else float("nan")

    def summary(self) -> str:
        return (
            f"{self.name:28s} rho={self.overall_spearman:+.3f}  "
            f"median per-length rho={self.median_per_length:+.3f}  "
            f"AUROC={self.auroc:.3f}  (n={self.n_points:,})"
        )


def compare(
    name: str,
    predicted: np.ndarray,
    measured: np.ndarray,
    lengths: Sequence[int],
    *,
    footprint_mask: int = 0,
    trim_3prime: int = 0,
    reactive_quantile: float = 0.75,
) -> BenchmarkResult:
    """Score a prediction against measured reactivity.

    ``footprint_mask`` drops the given number of nucleotides at the 3' end of
    each row (they sit inside the polymerase).  ``trim_3prime`` drops a fixed
    number of positions at the 3' end of the construct, for removing a cloning
    or primer-binding cassette.
    """
    predicted = predicted.copy()
    measured = measured.copy()
    size = measured.shape[1]

    if trim_3prime:
        predicted[:, size - trim_3prime :] = np.nan
        measured[:, size - trim_3prime :] = np.nan
    if footprint_mask:
        for row, length in enumerate(lengths):
            lo = max(0, length - footprint_mask)
            predicted[row, lo:length] = np.nan
            measured[row, lo:length] = np.nan

    mask = np.isfinite(predicted) & np.isfinite(measured)
    flat_p = predicted[mask]
    flat_m = measured[mask]

    per_length = np.array(
        [spearman(predicted[row], measured[row]) for row in range(len(lengths))]
    )
    # label the most reactive nucleotides as "accessible" and ask whether the
    # predicted unpaired probability ranks them above the rest
    if len(flat_m) > 10:
        threshold = np.quantile(flat_m, reactive_quantile)
        labels = (flat_m >= threshold).astype(float)
        area = auroc(flat_p, labels)
    else:
        area = float("nan")

    return BenchmarkResult(
        name=name,
        overall_spearman=spearman(flat_p, flat_m),
        per_length_spearman=per_length,
        auroc=area,
        n_points=int(mask.sum()),
        lengths=list(lengths),
    )


def run_rona(
    sequence: str,
    lengths: Sequence[int],
    *,
    rate: float = 30.0,
    footprint: int = 14,
    post_time: float = 10.0,
    n_trajectories: int = 100,
    seed: int = 0,
    config: SimulationConfig | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, Ensemble]:
    """Simulate and return the predicted unpaired-probability matrix."""
    schedule = TranscriptionSchedule(
        rate=rate, footprint=footprint, start_length=min(20, len(sequence)),
        post_time=post_time,
    )
    grid, kept = length_sampling_grid(schedule, len(sequence), lengths)
    base = config or SimulationConfig()
    cfg = SimulationConfig(
        temperature=base.temperature,
        dangles=base.dangles,
        rates=base.rates,
        pseudoknots=base.pseudoknots,
        transcription=schedule,
        mode=base.mode,
        min_helix=base.min_helix,
        nucleation_size=base.nucleation_size,
        min_loop=base.min_loop,
        max_span=base.max_span,
        max_stem_energy=base.max_stem_energy,
        max_events=base.max_events,
        frames=len(grid),
        grid="linear",
        duration=base.duration,
    )
    ensemble = simulate_ensemble(
        sequence, cfg, n_trajectories=n_trajectories, seed=seed,
        workers=workers, grid=grid,
    )
    return unpaired_matrix(ensemble, kept), ensemble
