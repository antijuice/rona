"""A secondary structure as a frozen set of base pairs.

Deliberately not a dot-bracket string: the state space here contains
pseudoknotted structures, for which dot-bracket needs an arbitrary choice of
bracket level, and two spellings of one structure would be two states.  A set of
pairs is canonical, hashable and topology-agnostic, which is what a state in a
master equation has to be.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from ..struct import Helix, helices_from_pairtable, to_dotbracket

#: A structure is the set of its base pairs, 0-based, ``i < j``.
Structure = frozenset  # of tuple[int, int]

EMPTY: Structure = frozenset()


def pair_table(structure: Structure, n: int) -> list[int]:
    """``pt[i]`` is the partner of ``i``, or -1."""
    pt = [-1] * n
    for i, j in structure:
        pt[i], pt[j] = j, i
    return pt


def dotbracket(structure: Structure, n: int) -> str:
    return to_dotbracket(pair_table(structure, n))


def from_dotbracket(db: str) -> Structure:
    from ..struct import parse_dotbracket

    pt = parse_dotbracket(db)
    return frozenset((i, pt[i]) for i in range(len(pt)) if pt[i] > i)


def helices_of(structure: Structure, n: int) -> list[Helix]:
    """The structure's maximal stacked runs, which is what the energy model wants."""
    return helices_from_pairtable(pair_table(structure, n))


def occupied(structure: Structure) -> set[int]:
    out: set[int] = set()
    for i, j in structure:
        out.add(i)
        out.add(j)
    return out


def is_valid(structure: Structure, n: int, min_loop: int) -> bool:
    """Every nucleotide in at most one pair, and no hairpin below ``min_loop``."""
    seen: set[int] = set()
    for i, j in structure:
        if not (0 <= i < j < n) or j - i - 1 < min_loop:
            return False
        if i in seen or j in seen:
            return False
        seen.add(i)
        seen.add(j)
    return True


def iter_helix_pairs(helix: Helix) -> Iterator[tuple[int, int]]:
    yield from helix.pairs
