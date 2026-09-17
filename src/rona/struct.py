"""Secondary-structure representation that is pseudoknot-aware from the start.

The central objects are:

``PairTable``
    A mutable ``list[int]`` wrapper where ``pt[i]`` is the partner of ``i`` or
    ``-1``.  Crossing pairs are perfectly legal here - nothing in this module
    assumes a nested structure.

``Helix``
    A contiguous ladder of stacked pairs ``(i+k, j-k)`` for ``k in range(n)``.
    The kinetic engine works at this granularity: helices nucleate, zip and
    unzip, which is what gives the simulation its physical timescale
    separation.

Dot-bracket strings use the extended alphabet ``() [] {} <> Aa Bb ...`` so a
pseudoknotted structure round-trips losslessly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

#: Bracket pairs used when writing dot-bracket, in priority order.
BRACKETS: tuple[tuple[str, str], ...] = (
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
) + tuple((chr(ord("A") + k), chr(ord("a") + k)) for k in range(26))

_OPEN = {o: idx for idx, (o, _c) in enumerate(BRACKETS)}
_CLOSE = {c: idx for idx, (_o, c) in enumerate(BRACKETS)}
UNPAIRED = ".-:_,"


class StructureError(ValueError):
    """Raised for malformed dot-bracket strings or illegal pair operations."""


# --------------------------------------------------------------------------
# dot-bracket <-> pair table
# --------------------------------------------------------------------------
def parse_dotbracket(db: str) -> list[int]:
    """Parse an (optionally pseudoknotted) dot-bracket string to a pair table.

    >>> parse_dotbracket("((..[[))..]]")[0]
    7
    """
    pt = [-1] * len(db)
    stacks: dict[int, list[int]] = {}
    for i, ch in enumerate(db):
        if ch in UNPAIRED:
            continue
        if ch in _OPEN:
            stacks.setdefault(_OPEN[ch], []).append(i)
        elif ch in _CLOSE:
            k = _CLOSE[ch]
            stack = stacks.get(k)
            if not stack:
                raise StructureError(f"unbalanced {ch!r} at position {i}")
            j = stack.pop()
            pt[i] = j
            pt[j] = i
        else:
            raise StructureError(f"illegal dot-bracket character {ch!r} at {i}")
    for k, stack in stacks.items():
        if stack:
            raise StructureError(
                f"unbalanced {BRACKETS[k][0]!r} at position {stack[0]}"
            )
    return pt


def to_dotbracket(pt: Sequence[int]) -> str:
    """Render a pair table, assigning bracket levels so crossings stay legible.

    Pairs are greedily assigned the lowest bracket level that does not conflict
    with a pair already at that level, which reproduces the conventional
    ``()``/``[]``/``{}`` layering for H-type and kissing-hairpin pseudoknots.
    """
    pairs = sorted((i, j) for i, j in iter_pairs(pt))
    levels: list[list[tuple[int, int]]] = []
    assignment: dict[tuple[int, int], int] = {}
    for pair in pairs:
        for lvl, members in enumerate(levels):
            if not any(_crosses(pair, other) for other in members):
                members.append(pair)
                assignment[pair] = lvl
                break
        else:
            levels.append([pair])
            assignment[pair] = len(levels) - 1
    if len(levels) > len(BRACKETS):
        raise StructureError(
            f"structure needs {len(levels)} bracket levels, only "
            f"{len(BRACKETS)} available"
        )
    out = ["."] * len(pt)
    for (i, j), lvl in assignment.items():
        out[i], out[j] = BRACKETS[lvl]
    return "".join(out)


def _crosses(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if pairs ``a`` and ``b`` cross (form a pseudoknot)."""
    (i, j), (k, l) = a, b
    return i < k < j < l or k < i < l < j


def iter_pairs(pt: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Yield each pair once, as ``(i, j)`` with ``i < j``."""
    for i, j in enumerate(pt):
        if j > i:
            yield i, j


def is_nested(pt: Sequence[int]) -> bool:
    """True when the pair table contains no crossing pairs."""
    stack: list[int] = []
    for i, j in enumerate(pt):
        if j < 0:
            continue
        if j > i:
            stack.append(j)
        else:
            if not stack or stack.pop() != i:
                return False
    return not stack


def crossing_pairs(pt: Sequence[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All unordered pairs-of-pairs that cross each other."""
    pairs = sorted(iter_pairs(pt))
    return [
        (a, b)
        for idx, a in enumerate(pairs)
        for b in pairs[idx + 1 :]
        if _crosses(a, b)
    ]


# --------------------------------------------------------------------------
# helices
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Helix:
    """A contiguous ladder of ``length`` stacked pairs anchored at ``(i, j)``.

    ``i`` is the 5'-most and ``j`` the 3'-most position, so the constituent
    pairs are ``(i + k, j - k)`` for ``k`` in ``range(length)``.
    """

    i: int
    j: int
    length: int

    def __post_init__(self) -> None:
        if self.length < 1:
            raise StructureError("helix length must be >= 1")
        if self.j - self.i < 2 * self.length - 1:
            raise StructureError(f"helix {self!r} overlaps itself")

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((self.i + k, self.j - k) for k in range(self.length))

    @property
    def inner(self) -> tuple[int, int]:
        """Innermost pair of the ladder."""
        return (self.i + self.length - 1, self.j - self.length + 1)

    @property
    def outer(self) -> tuple[int, int]:
        """Outermost (closing) pair of the ladder."""
        return (self.i, self.j)

    @property
    def span(self) -> int:
        return self.j - self.i + 1

    @property
    def loop_size(self) -> int:
        """Number of nucleotides enclosed by the innermost pair."""
        a, b = self.inner
        return b - a - 1

    def positions(self) -> Iterator[int]:
        for a, b in self.pairs:
            yield a
            yield b

    def occupies(self) -> frozenset[int]:
        return frozenset(self.positions())

    def crosses(self, other: "Helix") -> bool:
        """True if any pair of ``self`` crosses any pair of ``other``."""
        return any(_crosses(p, q) for p in self.pairs for q in other.pairs)

    def conflicts(self, other: "Helix") -> bool:
        """True if the two helices would need to share a nucleotide."""
        return bool(self.occupies() & other.occupies())

    def shrunk(self, *, outer: int = 0, inner: int = 0) -> "Helix":
        """Return this helix with base pairs peeled from either end."""
        length = self.length - outer - inner
        if length < 1:
            raise StructureError("cannot shrink helix below one base pair")
        return Helix(self.i + outer, self.j - outer, length)

    def grown_outer(self) -> "Helix":
        return Helix(self.i - 1, self.j + 1, self.length + 1)

    def grown_inner(self) -> "Helix":
        return Helix(self.i, self.j, self.length + 1)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.i}-{self.j}:{self.length}"


def helices_from_pairtable(pt: Sequence[int]) -> list[Helix]:
    """Decompose a pair table into maximal stacked ladders.

    A ladder is broken by any bulge or interior loop, so each returned
    :class:`Helix` is a run of perfectly stacked pairs.
    """
    seen: set[tuple[int, int]] = set()
    out: list[Helix] = []
    for i, j in sorted(iter_pairs(pt)):
        if (i, j) in seen:
            continue
        # only start a ladder at its outermost pair
        if i > 0 and j + 1 < len(pt) and pt[i - 1] == j + 1:
            continue
        length = 0
        a, b = i, j
        while a < b and pt[a] == b:
            seen.add((a, b))
            length += 1
            a, b = a + 1, b - 1
        out.append(Helix(i, j, length))
    return out


def pairtable_from_helices(helices: Iterable[Helix], n: int) -> list[int]:
    """Build a pair table of length ``n`` from a set of helices."""
    pt = [-1] * n
    for h in helices:
        for a, b in h.pairs:
            if pt[a] != -1 or pt[b] != -1:
                raise StructureError(f"helices conflict at {a}/{b}")
            pt[a], pt[b] = b, a
    return pt


def base_pair_distance(pt_a: Sequence[int], pt_b: Sequence[int]) -> int:
    """Symmetric difference of the two pair sets."""
    a = set(iter_pairs(pt_a))
    b = set(iter_pairs(pt_b))
    return len(a ^ b)


def pairs_of(db: str) -> set[tuple[int, int]]:
    """Convenience: the pair set of a dot-bracket string."""
    return set(iter_pairs(parse_dotbracket(db)))
