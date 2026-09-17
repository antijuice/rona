"""Reader for the RDAT format used by the RNA Mapping Database.

RDAT is a tab-separated text format; the fields this module needs are

``SEQUENCE``
    The full construct sequence.
``SEQPOS``
    Column labels, one per nucleotide position.
``DATA_ANNOTATION:k``
    Per-row metadata.  In a *cotranscriptional* dataset each row is one
    transcript length, tagged ``ID:Length<n>`` and carrying the prefix
    sequence.
``DATA:k``
    The measurements for row ``k`` - for cotranscriptional probing, the
    reactivity of each nucleotide of that transcript.

A cotranscriptional entry therefore holds a ragged matrix: row ``L`` has ``L``
values, because only the transcribed part of the chain exists to be probed.
:meth:`RdatEntry.matrix` pads it to a rectangle with NaN so it can be compared
against a prediction of the same shape.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LENGTH_TAG = re.compile(r"Length(\d+)", re.I)


@dataclass(slots=True)
class RdatEntry:
    """One RMDB entry."""

    name: str = ""
    sequence: str = ""
    structure: str = ""
    offset: int = 0
    seqpos: list[int] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    #: ``(row_annotations, values)`` in file order.
    rows: list[tuple[dict[str, str], list[float]]] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def lengths(self) -> list[int]:
        """Transcript length of each row, or ``-1`` where it is not declared."""
        out: list[int] = []
        for meta, values in self.rows:
            match = _LENGTH_TAG.search(meta.get("ID", ""))
            if match:
                out.append(int(match.group(1)))
            elif "sequence" in meta:
                out.append(len(meta["sequence"]))
            else:
                out.append(len(values))
        return out

    def is_cotranscriptional(self) -> bool:
        """True when the rows form a ladder of increasing transcript lengths."""
        lengths = self.lengths
        return len(set(lengths)) > 1 and sorted(lengths) == lengths

    def matrix(self, n: int | None = None) -> tuple[np.ndarray, list[int]]:
        """Reactivities as a ``(n_rows, n)`` array padded with NaN.

        Returns the array and the transcript length of each row.
        """
        size = n or len(self.sequence)
        lengths = self.lengths
        out = np.full((len(self.rows), size), np.nan)
        for index, (_meta, values) in enumerate(self.rows):
            take = min(len(values), size)
            out[index, :take] = values[:take]
        return out, lengths


def _parse_annotations(fields: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in fields:
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        out[key] = value
    return out


def read_rdat(path: str | Path) -> RdatEntry:
    """Parse an RDAT file."""
    entry = RdatEntry()
    indexed: dict[int, list[float]] = {}
    meta: dict[int, dict[str, str]] = {}

    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            tag = fields[0]
            key = tag.split(":", 1)[0]

            if key == "NAME":
                entry.name = fields[1] if len(fields) > 1 else ""
            elif key == "SEQUENCE" and ":" not in tag:
                entry.sequence = (fields[1] if len(fields) > 1 else "").strip().upper()
            elif key == "STRUCTURE" and ":" not in tag:
                entry.structure = fields[1] if len(fields) > 1 else ""
            elif key == "OFFSET":
                entry.offset = int(float(fields[1])) if len(fields) > 1 else 0
            elif key == "SEQPOS":
                entry.seqpos = [
                    int(re.sub(r"[^0-9-]", "", tok) or 0) for tok in fields[1:] if tok
                ]
            elif key == "ANNOTATION":
                entry.annotations.update(_parse_annotations(fields[1:]))
            elif key == "COMMENT":
                entry.comments.append("\t".join(fields[1:]))
            elif key == "DATA_ANNOTATION":
                index = int(tag.split(":", 1)[1])
                meta[index] = _parse_annotations(fields[1:])
            elif key in ("DATA", "REACTIVITY"):
                index = int(tag.split(":", 1)[1])
                values = []
                for tok in fields[1:]:
                    tok = tok.strip()
                    if not tok:
                        continue
                    try:
                        values.append(float(tok))
                    except ValueError:
                        values.append(math.nan)
                indexed[index] = values

    for index in sorted(indexed):
        entry.rows.append((meta.get(index, {}), indexed[index]))
    return entry
