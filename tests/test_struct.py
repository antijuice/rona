import random

import pytest

from rona import seq as S
from rona.struct import (
    Helix,
    StructureError,
    base_pair_distance,
    helices_from_pairtable,
    is_nested,
    pairtable_from_helices,
    parse_dotbracket,
    to_dotbracket,
    _crosses,
)


@pytest.mark.parametrize(
    "db",
    [
        "....",
        "((..))",
        "((((...))))",
        "((..[[))..]]",
        "((((...[[[[))))...]]]]",
        ".((.[[.{{.)).]].}}.",
        "(([[{{<<))]]}}>>",
    ],
)
def test_dotbracket_round_trip(db):
    """Canonical strings round-trip exactly."""
    assert to_dotbracket(parse_dotbracket(db)) == db


@pytest.mark.parametrize("db", [".((.{{.[[.)).}}.]].", "((([[[)))]]]"])
def test_noncanonical_bracket_levels_preserve_pairing(db):
    """Bracket *levels* are re-canonicalised, but the pairing is preserved."""
    pt = parse_dotbracket(db)
    assert parse_dotbracket(to_dotbracket(pt)) == pt


def test_unbalanced_brackets_rejected():
    with pytest.raises(StructureError):
        parse_dotbracket("((.")
    with pytest.raises(StructureError):
        parse_dotbracket(".))")
    with pytest.raises(StructureError):
        parse_dotbracket("(%)")


def test_nestedness_detection():
    assert is_nested(parse_dotbracket("((..))"))
    assert not is_nested(parse_dotbracket("((..[[))..]]"))


def test_helix_decomposition_splits_at_bulges():
    # a one-nucleotide bulge breaks the ladder into two helices
    helices = helices_from_pairtable(parse_dotbracket("(((.(((...))))))"))
    assert sorted((h.i, h.j, h.length) for h in helices) == [(0, 15, 3), (4, 12, 3)]


def test_pairtable_from_helices_rejects_conflicts():
    with pytest.raises(StructureError):
        pairtable_from_helices([Helix(0, 9, 3), Helix(1, 8, 2)], 10)


def test_helix_geometry():
    h = Helix(2, 20, 4)
    assert h.pairs == ((2, 20), (3, 19), (4, 18), (5, 17))
    assert h.outer == (2, 20) and h.inner == (5, 17)
    assert h.loop_size == 11
    assert h.shrunk(outer=1).pairs == ((3, 19), (4, 18), (5, 17))
    assert h.grown_outer() == Helix(1, 21, 5)
    with pytest.raises(StructureError):
        Helix(0, 3, 3)


def test_fast_crossing_test_matches_brute_force():
    rng = random.Random(5)
    for _ in range(20000):
        i = rng.randrange(0, 30)
        l1 = rng.randint(1, 4)
        j = i + 2 * l1 + rng.randint(3, 20)
        k = rng.randrange(0, 30)
        l2 = rng.randint(1, 4)
        m = k + 2 * l2 + rng.randint(3, 20)
        if j >= 60 or m >= 60:
            continue
        a, b = Helix(i, j, l1), Helix(k, m, l2)
        if a.occupies() & b.occupies():
            continue
        brute = any(_crosses(p, q) for p in a.pairs for q in b.pairs)
        assert a.crosses(b) is brute
        assert b.crosses(a) is brute


def test_base_pair_distance():
    a = parse_dotbracket("((((...))))")
    b = parse_dotbracket(".(((...))).")
    assert base_pair_distance(a, a) == 0
    # the two structures differ by exactly the outermost pair
    assert base_pair_distance(a, b) == 1


def test_fasta_header_uses_the_first_token_as_the_name():
    records = S.read_fasta(">id123  free text description\nGGGAAACCC\n")
    assert records == [("id123", "GGGAAACCC")]
    assert S.read_fasta("GGGAAACCC") == [("seq", "GGGAAACCC")]
    multi = S.read_fasta(">a\nGGG\nAAA\n>b desc\nCCC\n")
    assert multi == [("a", "GGGAAA"), ("b", "CCC")]


def test_sequence_normalisation():
    assert S.normalise("ggu cat") == "GGUCAU"
    assert S.normalise("acgRy") == "ACGNN"
    assert S.reverse_complement("GGGAAACCC") == "GGGUUUCCC"
    with pytest.raises(S.SequenceError):
        S.normalise("")
