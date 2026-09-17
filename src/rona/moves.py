"""The elementary move set of the kinetic simulation.

Folding is modelled at the level of **helices**, not individual base pairs,
because that is what produces the right separation of timescales:

* **Nucleation** - forming the first few pairs of a helix is slow, because the
  two strands have to be brought together against the loop-closure entropy.
* **Zipping** - once nucleated, a helix extends one pair at a time, orders of
  magnitude faster.
* **Melting** - the reverse of both, with rates fixed by detailed balance.

Candidate helices live inside *maximal stems*: runs of consecutive pairs that
cannot be extended without hitting a mismatch.  A nucleation site is a
``nucleation_size``-pair window inside such a stem; once formed, the helix can
zip outwards or inwards but never past the boundary of its stem, so every
structure the simulation visits is built from real stacked ladders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .seq import PAIR_TYPE
from .struct import Helix


@dataclass(frozen=True, slots=True)
class Stem:
    """A maximal run of stacked pairs ``(i+k, j-k)`` for ``k in range(length)``."""

    i: int
    j: int
    length: int
    energy: float = 0.0  # context-free stacking energy, kcal/mol

    def window(self, offset: int, size: int) -> Helix:
        """The sub-ladder of ``size`` pairs starting ``offset`` pairs in."""
        return Helix(self.i + offset, self.j - offset, size)

    @property
    def max_position(self) -> int:
        return self.j


@dataclass(frozen=True, slots=True)
class MoveSet:
    """Precomputed, sequence-wide candidate structure elements."""

    stems: tuple[Stem, ...]
    #: ``(stem_index, offset)`` for every nucleation window, ordered by the
    #: 3'-most nucleotide they need so cotranscriptional activation is a
    #: prefix scan.
    sites: tuple[tuple[int, int], ...]
    #: For each site, the 3'-most nucleotide index it requires.
    site_max_pos: tuple[int, ...]
    nucleation_size: int

    def helix(self, site: int) -> Helix:
        stem_index, offset = self.sites[site]
        return self.stems[stem_index].window(offset, self.nucleation_size)

    def stem_of(self, site: int) -> Stem:
        return self.stems[self.sites[site][0]]

    def __len__(self) -> int:
        return len(self.sites)


def _pairable(enc: Sequence[int], i: int, j: int, min_loop: int) -> bool:
    if i < 0 or j >= len(enc) or j - i <= min_loop:
        return False
    return PAIR_TYPE[enc[i]][enc[j]] != 0


def enumerate_stems(
    enc: Sequence[int],
    *,
    min_loop: int = 3,
    min_length: int = 3,
    max_span: int | None = None,
) -> list[Stem]:
    """All maximal stems of at least ``min_length`` base pairs.

    ``max_span`` optionally caps ``j - i``; very long-range pairs are rarely
    reachable on transcription timescales and dropping them keeps the candidate
    list small for long transcripts.
    """
    n = len(enc)
    out: list[Stem] = []
    for i in range(n):
        jmax = n - 1 if max_span is None else min(n - 1, i + max_span)
        for j in range(i + min_loop + 1, jmax + 1):
            if not _pairable(enc, i, j, min_loop):
                continue
            if _pairable(enc, i - 1, j + 1, min_loop):
                continue  # not the outermost pair of its ladder
            length = 0
            while _pairable(enc, i + length, j - length, min_loop):
                length += 1
            if length >= min_length:
                out.append(Stem(i, j, length))
    return out


def score_stems(model, enc: Sequence[int], stems: Sequence[Stem]) -> list[Stem]:
    """Attach each stem's context-free stacking energy (kcal/mol)."""
    from .energy.pseudoknot import helix_stack_energy

    return [
        Stem(
            s.i,
            s.j,
            s.length,
            helix_stack_energy(model, enc, Helix(s.i, s.j, s.length)),
        )
        for s in stems
    ]


def build_moveset(
    model,
    enc: Sequence[int],
    *,
    min_loop: int = 3,
    min_helix: int = 3,
    nucleation_size: int = 3,
    max_span: int | None = None,
    max_stem_energy: float = -1.0,
) -> MoveSet:
    """Enumerate stems and nucleation windows for a sequence.

    ``max_stem_energy`` prunes stems whose isolated stacking energy is weaker
    than the threshold; such stems cannot survive even the most favourable loop
    context, so carrying them would only slow the simulation down.
    """
    if nucleation_size < 1:
        raise ValueError("nucleation_size must be >= 1")
    min_helix = max(min_helix, nucleation_size)
    stems = enumerate_stems(
        enc, min_loop=min_loop, min_length=min_helix, max_span=max_span
    )
    stems = [s for s in score_stems(model, enc, stems) if s.energy <= max_stem_energy]

    sites: list[tuple[int, int]] = []
    for index, stem in enumerate(stems):
        for offset in range(stem.length - nucleation_size + 1):
            sites.append((index, offset))
    # activation order: a site becomes usable once its 3'-most base exists
    sites.sort(key=lambda t: stems[t[0]].j - t[1])
    site_max_pos = tuple(stems[k].j - off for k, off in sites)
    return MoveSet(
        stems=tuple(stems),
        sites=tuple(sites),
        site_max_pos=site_max_pos,
        nucleation_size=nucleation_size,
    )


#: Move kinds emitted by the kinetic engine.
FORM = "form"
MELT = "melt"
ZIP_OUT = "zip_out"
ZIP_IN = "zip_in"
UNZIP_OUT = "unzip_out"
UNZIP_IN = "unzip_in"

MOVE_KINDS = (FORM, MELT, ZIP_OUT, ZIP_IN, UNZIP_OUT, UNZIP_IN)
