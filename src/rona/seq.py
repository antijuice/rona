"""Sequence handling, alphabets and base-pair typing.

The pair-type encoding follows the convention used by the Turner-parameter
tables shipped in ``rona/params`` (which use the ViennaRNA ``.par`` layout):

    0 = NP (no pair), 1 = CG, 2 = GC, 3 = GU, 4 = UG, 5 = AU, 6 = UA, 7 = NS

Energies in the tables are indexed by the *type of the closing pair* ``(i, j)``
and, where a second pair is involved, by the type of the enclosed pair read in
the *reversed* orientation ``(j-1, i+1)``.  Keeping that convention lets us use
the published tables verbatim; :func:`rona.energy.model.NearestNeighbourModel`
has unit tests pinning it against the Xia et al. (1998) Watson-Crick doublets.
"""

from __future__ import annotations

from typing import Iterable, Sequence

#: Canonical RNA alphabet, in the index order used by the parameter tables.
ALPHABET = "NACGU"
BASES = "ACGU"

#: ``ENCODE[base] -> 0..4`` where 0 is the "any/unknown" slot ``N``.
ENCODE = {c: i for i, c in enumerate(ALPHABET)}

NP, CG, GC, GU, UG, AU, UA, NS = range(8)

PAIR_NAMES = ("NP", "CG", "GC", "GU", "UG", "AU", "UA", "NS")

# pair_type[a][b] for encoded bases a, b in 0..4
_PT = [[NP] * 5 for _ in range(5)]
_PT[ENCODE["C"]][ENCODE["G"]] = CG
_PT[ENCODE["G"]][ENCODE["C"]] = GC
_PT[ENCODE["G"]][ENCODE["U"]] = GU
_PT[ENCODE["U"]][ENCODE["G"]] = UG
_PT[ENCODE["A"]][ENCODE["U"]] = AU
_PT[ENCODE["U"]][ENCODE["A"]] = UA
PAIR_TYPE = tuple(tuple(row) for row in _PT)

#: Pairs that carry the terminal-AU/GU penalty (everything except C-G / G-C).
TERMINAL_PENALTY_TYPES = frozenset({GU, UG, AU, UA})

_TRANSLATE = str.maketrans("tTn ", "UUN_")


class SequenceError(ValueError):
    """Raised for sequences that cannot be interpreted as RNA."""


def normalise(seq: str) -> str:
    """Upper-case, convert DNA ``T`` to ``U`` and validate the alphabet.

    Characters outside ``ACGU`` are collapsed to ``N``, which never pairs.
    Whitespace is stripped.
    """
    if seq is None:
        raise SequenceError("sequence is None")
    s = "".join(seq.split()).translate(_TRANSLATE).upper()
    if not s:
        raise SequenceError("empty sequence")
    out = []
    for ch in s:
        if ch in BASES:
            out.append(ch)
        elif ch in "N_-":
            out.append("N")
        elif ch.isalpha():
            # IUPAC ambiguity codes degrade to N rather than failing outright.
            out.append("N")
        else:
            raise SequenceError(f"illegal character {ch!r} in sequence")
    return "".join(out)


def encode(seq: str) -> list[int]:
    """Map a normalised sequence onto the 0..4 index alphabet."""
    return [ENCODE[c] for c in seq]


def pair_type(enc: Sequence[int], i: int, j: int) -> int:
    """Pair type of positions ``i`` and ``j`` (0-based) in an encoded sequence."""
    return PAIR_TYPE[enc[i]][enc[j]]


def can_pair(enc: Sequence[int], i: int, j: int, min_loop: int = 3) -> bool:
    """Whether ``i`` and ``j`` may form a canonical pair with a legal hairpin."""
    if j - i <= min_loop:
        return False
    return PAIR_TYPE[enc[i]][enc[j]] != NP


def read_fasta(text: str) -> list[tuple[str, str]]:
    """Minimal FASTA reader returning ``(name, sequence)`` pairs.

    The record name is the first whitespace-delimited token of the header, as
    is conventional; anything after it is a free-text description and is
    dropped.  A bare sequence with no header is accepted and named ``"seq"``.
    """
    records: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith(">"):
            identifier = line[1:].split(None, 1)[0] if line[1:].strip() else ""
            records.append((identifier or f"seq{len(records) + 1}", []))
        else:
            if not records:
                records.append(("seq", []))
            records[-1][1].append(line)
    return [(name, normalise("".join(chunks))) for name, chunks in records if chunks]


def gc_content(seq: str) -> float:
    """Fraction of G/C in a normalised sequence."""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in "GC") / len(seq)


def reverse_complement(seq: str) -> str:
    """Reverse complement under the Watson-Crick alphabet (G pairs with C)."""
    table = str.maketrans("ACGUN", "UGCAN")
    return normalise(seq).translate(table)[::-1]


def windows(seq: Iterable[str], size: int):
    """Yield consecutive windows of ``size`` items; helper for diagnostics."""
    buf: list[str] = []
    for item in seq:
        buf.append(item)
        if len(buf) == size:
            yield "".join(buf)
            buf.pop(0)
