"""Interoperability with DrForna's time-course format (``.drf``).

DrForna (Tang et al., 2023) is the established viewer for cotranscriptional
folding trajectories, and DrTransformer writes this format.  Supporting it both
ways means a rona ensemble can be opened in DrForna, and a DrTransformer run
can be read back for a like-for-like comparison.

The format is whitespace-separated with a header line::

    id time occupancy structure energy

One row per (structure, time) with non-negligible occupancy.  ``id`` groups
rows belonging to the same structure so a viewer can track it over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER = "id time occupancy structure energy"


@dataclass(slots=True)
class DrfCourse:
    """A time course read from a ``.drf`` file."""

    #: ``(time, occupancy, structure, energy)`` rows in file order.
    rows: list[tuple[float, float, str, float]] = field(default_factory=list)

    @property
    def times(self) -> list[float]:
        seen: list[float] = []
        for time, *_rest in self.rows:
            if not seen or time != seen[-1]:
                seen.append(time)
        return sorted(set(seen))

    def lengths(self) -> list[int]:
        return sorted({len(structure) for _t, _o, structure, _e in self.rows})

    def unpaired_matrix(self, lengths, n: int) -> np.ndarray:
        """``P(unpaired)[row, position]``, one row per requested length.

        For each length the *latest* time at which the transcript had that
        length is used, which is the moment just before the next nucleotide is
        added - matching how probing data is collected.
        """
        by_length: dict[int, float] = {}
        for time, _occ, structure, _e in self.rows:
            length = len(structure)
            by_length[length] = max(by_length.get(length, -1.0), time)

        out = np.full((len(lengths), n), np.nan)
        for row, length in enumerate(lengths):
            target = by_length.get(length)
            if target is None:
                continue
            weight = np.zeros(n)
            total = 0.0
            for time, occupancy, structure, _e in self.rows:
                if len(structure) != length or time != target:
                    continue
                total += occupancy
                for position, character in enumerate(structure[:n]):
                    if character == ".":
                        weight[position] += occupancy
            if total > 0:
                out[row, :length] = weight[:length] / total
        return out


def read_drf(path: str | Path) -> DrfCourse:
    """Read a ``.drf`` time course."""
    course = DrfCourse()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5 or parts[0] == "id":
                continue
            try:
                time = float(parts[1])
                occupancy = float(parts[2])
                energy = float(parts[4])
            except ValueError:
                continue
            course.rows.append((time, occupancy, parts[3], energy))
    return course


def write_drf(
    ensemble, path: str | Path, *, min_occupancy: float = 0.001
) -> str:
    """Write an ensemble as a ``.drf`` time course, openable in DrForna."""
    labels, occupancy = ensemble.occupancy(min_population=0.0)
    ids = {label: index + 1 for index, label in enumerate(labels)}
    energies = ensemble.mean_energy()
    lines = [HEADER]
    for t, time in enumerate(ensemble.times):
        for row, label in enumerate(labels):
            value = float(occupancy[row, t])
            if value < min_occupancy:
                continue
            lines.append(
                f"{ids[label]} {time:.4f} {value:.4f} {label} {energies[t]:7.2f}"
            )
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return str(path)
