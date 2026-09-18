"""The energy model is checked against ViennaRNA where available, and against
pinned reference values otherwise, so the suite is meaningful either way."""

from __future__ import annotations

import random

import pytest

from rona import seq as S
from rona.energy import paramfile
from rona.energy.evaluator import FoldingEnergy, enclosing_pair
from rona.energy.model import NearestNeighbourModel
from rona.energy.pseudoknot import PseudoknotModel, split_crossing
from rona.struct import Helix, helices_from_pairtable, is_nested, parse_dotbracket

try:  # optional cross-check against the reference implementation
    import RNA  # type: ignore

    HAVE_VIENNA = True
except Exception:  # pragma: no cover - ViennaRNA is not a hard dependency
    RNA = None
    HAVE_VIENNA = False


# Xia et al. (1998) Watson-Crick nearest-neighbour free energies, kcal/mol.
XIA_DOUBLETS = {
    ("AU", "UA"): -0.93,  # 5'AA3'/3'UU5'
    ("AU", "AU"): -1.10,  # 5'AU3'/3'UA5'
    ("UA", "UA"): -1.33,  # 5'UA3'/3'AU5'
    ("CG", "AU"): -2.08,  # 5'CU3'/3'GA5'
    ("CG", "UA"): -2.11,  # 5'CA3'/3'GU5'
    ("GC", "AU"): -2.24,  # 5'GU3'/3'CA5'
    ("GC", "UA"): -2.35,  # 5'GA3'/3'CU5'
    ("CG", "CG"): -2.36,  # 5'CG3'/3'GC5'
    ("GC", "CG"): -3.26,  # 5'GG3'/3'CC5'
    ("GC", "GC"): -3.42,  # 5'GC3'/3'CG5'
}


def test_parameter_file_tables_are_complete():
    params = paramfile.default_params()
    assert params["stack"].shape == (8, 8)
    assert params["int22"].shape == (8, 8, 5, 5, 5, 5)
    assert len(params.tetraloops) == 16
    assert params.terminal_au == 50.0


def test_stacking_table_matches_published_doublets():
    """Pins the pair-type indexing convention to measured values."""
    model = NearestNeighbourModel()
    names = {n: i for i, n in enumerate(S.PAIR_NAMES)}
    for (outer, inner), expected in XIA_DOUBLETS.items():
        value = model.stack[names[outer]][names[inner]] / 100.0
        assert value == pytest.approx(expected, abs=0.06)


@pytest.mark.parametrize(
    "sequence,structure,expected",
    [
        ("GGGAAACCC", "(((...)))", -1.20),
        ("GCGCUUCGGCGC", "((((....))))", -5.50),
        ("GGGGAAAACCCCAAAGGGGAAAACCCC", "((((....))))...((((....))))", -12.70),
    ],
)
def test_reference_energies(sequence, structure, expected):
    model = NearestNeighbourModel()
    got = model.eval_nested(S.encode(sequence), sequence, parse_dotbracket(structure))
    assert got == pytest.approx(expected, abs=0.01)


def _random_nested(sequence, rng):
    enc = S.encode(sequence)
    n = len(sequence)
    pt = [-1] * n
    for _ in range(n * 3):
        i, j = sorted((rng.randrange(n), rng.randrange(n)))
        length = rng.randint(1, 5)
        ok = all(
            j - k - (i + k) > 3
            and pt[i + k] == -1
            and pt[j - k] == -1
            and S.PAIR_TYPE[enc[i + k]][enc[j - k]] != 0
            for k in range(length)
        )
        if not ok:
            continue
        saved = list(pt)
        for k in range(length):
            pt[i + k], pt[j - k] = j - k, i + k
        if not is_nested(pt):
            pt = saved
    return pt


@pytest.mark.skipif(not HAVE_VIENNA, reason="ViennaRNA not installed")
@pytest.mark.parametrize("temperature,dangles", [(37.0, 2), (37.0, 0), (10.0, 2), (70.0, 2)])
def test_matches_viennarna(temperature, dangles):
    rng = random.Random(7)
    model = NearestNeighbourModel(temperature=temperature, dangles=dangles)
    md = RNA.md()
    md.temperature = temperature
    md.dangles = dangles
    worst = 0.0
    for _ in range(300):
        n = rng.randint(15, 90)
        sequence = "".join(rng.choice("ACGU") for _ in range(n))
        pt = _random_nested(sequence, rng)
        from rona.struct import to_dotbracket

        db = to_dotbracket(pt)
        if "(" not in db:
            continue
        mine = model.eval_nested(S.encode(sequence), sequence, pt)
        theirs = RNA.fold_compound(sequence, md).eval_structure(db)
        worst = max(worst, abs(mine - theirs))
    assert worst < 0.02, f"worst deviation {worst} kcal/mol"


def test_incremental_delta_matches_full_evaluation():
    rng = random.Random(3)
    worst = 0.0
    for _ in range(120):
        n = rng.randint(25, 80)
        sequence = "".join(rng.choice("ACGU") for _ in range(n))
        fe = FoldingEnergy(sequence)
        pt = [-1] * n
        helices: list[Helix] = []
        for _ in range(20):
            i, j = sorted((rng.randrange(n), rng.randrange(n)))
            length = rng.randint(2, 5)
            try:
                helix = Helix(i, j, length)
            except Exception:
                continue
            if helix.loop_size < 3:
                continue
            if any(pt[a] != -1 or pt[b] != -1 for a, b in helix.pairs):
                continue
            if any(S.PAIR_TYPE[fe.enc[a]][fe.enc[b]] == 0 for a, b in helix.pairs):
                continue
            probe = list(pt)
            for a, b in helix.pairs:
                probe[a], probe[b] = b, a
            if not is_nested(probe):
                continue
            before = fe.energy(helices, n)
            delta = fe.delta_add(pt, helix, n, nested=True)
            for a, b in helix.pairs:
                pt[a], pt[b] = b, a
            helices = helices_from_pairtable(pt)
            after = fe.energy(helices, n)
            if before == float("inf") or after == float("inf"):
                break
            worst = max(worst, abs(delta - (after - before)))
            worst = max(
                worst,
                abs(fe.delta_remove(pt, helix, n, nested=True) - (before - after)),
            )
    assert worst < 1e-9


def test_enclosing_pair():
    pt = parse_dotbracket("..((..((..))..))..")
    assert enclosing_pair(pt, 0, len(pt)) is None
    # the *innermost* enclosing pair, not the outermost
    assert enclosing_pair(pt, 5, len(pt)) == (3, 14)
    assert enclosing_pair(pt, 9, len(pt)) == (7, 10)
    assert enclosing_pair(pt, 13, len(pt)) == (3, 14)


def test_pseudoknot_split_and_penalty():
    fe = FoldingEnergy("GGGAAACCCAAAGGGAAACCC", pk_model=PseudoknotModel())
    nested = [Helix(0, 8, 3), Helix(12, 20, 3)]
    core, pk = split_crossing(nested)
    assert not pk and len(core) == 2

    crossing = [Helix(0, 14, 3), Helix(6, 20, 3)]
    core, pk = split_crossing(crossing)
    assert len(core) == 1 and len(pk) == 1

    # a pseudoknot must cost more than the same helices evaluated apart
    with_pk = fe.energy(crossing, 21)
    disabled = FoldingEnergy(
        "GGGAAACCCAAAGGGAAACCC", pk_model=PseudoknotModel(enabled=False)
    )
    assert with_pk > disabled.energy([Helix(0, 14, 3)], 21)


def test_pseudoknot_energy_is_path_independent():
    """Energy must be a state function or detailed balance is meaningless."""
    fe = FoldingEnergy("GGCGAAAGCCAAAAGGCGAAAGCCUUUU")
    helices = [Helix(0, 9, 3), Helix(4, 24, 3)]
    assert fe.energy(helices, 28) == pytest.approx(fe.energy(helices[::-1], 28))


def test_energy_does_not_depend_on_the_order_helices_are_listed_in():
    """A free energy that is not a function of the state breaks everything.

    The nested-core / pseudoknot partition is greedy, and a greedy rule with an
    order-dependent tie-break makes the topology penalty depend on the order the
    caller happened to store its helices in.  The simulator stores them in a
    dict keyed by stem, whose order follows the history of the run, so the same
    state could be scored two ways and a move's dG could disagree with its
    reverse by kcal/mol.
    """
    import itertools
    import random

    from rona.energy.evaluator import FoldingEnergy
    from rona.energy.model import NearestNeighbourModel
    from rona.energy.pseudoknot import PseudoknotModel
    from rona.struct import Helix

    sequence = "GGCGCGGCACCGUCCGCGGAACAAACGGAGAAGGGGCCGCCGAAAGGCGGCC"
    energy = FoldingEnergy(
        sequence, NearestNeighbourModel(), PseudoknotModel(enabled=True)
    )
    # a pseudoknotted state: (9,18) threads through (12,28)
    states = [
        # two crossing helices of equal length: the greedy partition has to
        # choose one, and the choice must not come from the list order
        [Helix(0, 51, 3), Helix(5, 37, 3), Helix(9, 18, 3), Helix(13, 27, 3),
         Helix(38, 47, 3)],
        [Helix(0, 51, 3), Helix(5, 37, 3), Helix(9, 18, 3), Helix(12, 28, 4),
         Helix(38, 47, 3)],
        [Helix(0, 51, 3), Helix(9, 18, 3), Helix(13, 27, 3)],
    ]
    rng = random.Random(5)
    for helices in states:
        reference = energy.energy(helices, len(sequence))
        core, pk = energy.split(helices)
        core_key = sorted((h.i, h.j, h.length) for h in core)
        for _ in range(12):
            shuffled = list(helices)
            rng.shuffle(shuffled)
            assert energy.energy(shuffled, len(sequence)) == pytest.approx(
                reference, abs=1e-12
            )
            again, _pk = energy.split(shuffled)
            assert sorted((h.i, h.j, h.length) for h in again) == core_key
    # and exhaustively on the smallest one
    smallest = states[2]
    reference = energy.energy(smallest, len(sequence))
    for order in itertools.permutations(smallest):
        assert energy.energy(list(order), len(sequence)) == pytest.approx(
            reference, abs=1e-12
        )
