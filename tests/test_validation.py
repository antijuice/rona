"""Tests for the probing-data readers and the benchmark statistics."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import fast_rates
from rona.render.drforna import DrfCourse, read_drf, write_drf
from rona.validation import shape
from rona.validation.rdat import read_rdat

RDAT = """RDAT_VERSION\t0.4
NAME\ttoy cotranscriptional set
SEQUENCE\tGGGAAACCCUUU
STRUCTURE\t............
OFFSET\t0
SEQPOS\tG1\tG2\tG3\tA4\tA5\tA6\tC7\tC8\tC9\tU10\tU11\tU12

ANNOTATION\texperimentType:StandardState\tmodifier:BzCN\ttemperature:37C
COMMENT\ta comment
DATA_ANNOTATION:1\tsequence:GGGAAAC\tdatatype:REACTIVITY\tID:Length7
DATA:1\t0.1\t0.2\t0.3\t0.9\t0.8\t0.7\t0.1
DATA_ANNOTATION:2\tsequence:GGGAAACCC\tdatatype:REACTIVITY\tID:Length9
DATA:2\t0.1\t0.1\t0.1\t0.9\t0.9\t0.9\t0.1\t0.1\t0.1
"""


def test_rdat_parses_a_cotranscriptional_ladder(tmp_path):
    path = tmp_path / "toy.rdat"
    path.write_text(RDAT)
    entry = read_rdat(path)
    assert entry.sequence == "GGGAAACCCUUU"
    assert entry.annotations["modifier"] == "BzCN"
    assert entry.comments == ["a comment"]
    assert entry.lengths == [7, 9]
    assert entry.is_cotranscriptional()
    matrix, lengths = entry.matrix()
    assert matrix.shape == (2, 12)
    assert lengths == [7, 9]
    # rows are padded with NaN beyond the transcript
    assert np.isnan(matrix[0, 7:]).all()
    assert matrix[1, 3] == pytest.approx(0.9)


def test_rdat_handles_missing_values(tmp_path):
    path = tmp_path / "x.rdat"
    path.write_text(RDAT.replace("\t0.9\t0.8", "\tNaN\t0.8"))
    entry = read_rdat(path)
    matrix, _ = entry.matrix()
    assert np.isnan(matrix[0, 3])


# ----------------------------------------------------------------------
def test_spearman_against_known_values():
    assert shape.spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert shape.spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    assert abs(shape.spearman(a, rng.normal(size=400))) < 0.2


def test_spearman_is_monotone_invariant():
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    straight = shape.spearman(a, b)
    # any increasing transform must leave a rank correlation alone
    assert shape.spearman(np.exp(a), b) == pytest.approx(straight)
    assert shape.spearman(a, b * 3.0 + 7.0) == pytest.approx(straight)


def test_spearman_ignores_nans():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    b = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    # only the four positions finite in both are used, and they agree exactly
    assert shape.spearman(a, b) == pytest.approx(1.0)


def test_spearman_needs_enough_overlap():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.0, 3.0, np.nan])
    assert np.isnan(shape.spearman(a, b))


def test_auroc_bounds():
    assert shape.auroc(np.array([1.0, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert shape.auroc(np.array([4.0, 3, 2, 1]), np.array([0, 0, 1, 1])) == 0.0
    assert shape.auroc(np.array([1.0, 1, 1, 1]), np.array([0, 0, 1, 1])) == 0.5


def _ragged(lengths, width, value=0.5):
    """A length x position matrix padded with NaN past each transcript end."""
    out = np.full((len(lengths), width), np.nan)
    for row, length in enumerate(lengths):
        out[row, :length] = value
    return out


def test_compare_masks_the_footprint():
    lengths = [6, 8]
    measured = _ragged(lengths, 8)
    predicted = _ragged(lengths, 8)
    result = shape.compare("x", predicted, measured, lengths, footprint_mask=3)
    # the three nucleotides inside the polymerase are dropped from each row
    assert result.n_points == (6 - 3) + (8 - 3)


def test_compare_trims_the_3_prime_cassette():
    lengths = [8]
    measured = _ragged(lengths, 8)
    predicted = _ragged(lengths, 8)
    result = shape.compare("x", predicted, measured, lengths, trim_3prime=3)
    assert result.n_points == 5


def test_length_sampling_grid_is_ordered_and_per_length():
    from rona.cotrans import TranscriptionSchedule

    schedule = TranscriptionSchedule(rate=10.0, start_length=5, post_time=2.0)
    grid, kept = shape.length_sampling_grid(schedule, 20, [5, 10, 15, 20])
    assert kept == [5, 10, 15, 20]
    assert all(b >= a for a, b in zip(grid, grid[1:]))
    # the last sample is taken after the post-transcriptional window
    assert grid[-1] > schedule.arrival_times(20)[20]


# ----------------------------------------------------------------------
DRF = """id time occupancy structure energy
1 0.0000 1.0000 .... 0.00
1 1.0000 0.6000 ((.)) -1.00
2 1.0000 0.4000 ..... 0.00
"""


def test_drf_round_trip(tmp_path):
    path = tmp_path / "x.drf"
    path.write_text(DRF)
    course = read_drf(path)
    assert len(course.rows) == 3
    assert course.lengths() == [4, 5]
    matrix = course.unpaired_matrix([5], 5)
    # at t=1 the two structures split 0.6 / 0.4; position 0 is paired in one
    assert matrix[0, 0] == pytest.approx(0.4)
    assert matrix[0, 2] == pytest.approx(1.0)


def test_write_drf_is_readable_back(tmp_path):
    from rona.cotrans import SimulationConfig, TranscriptionSchedule
    from rona.ensemble import simulate_ensemble

    ensemble = simulate_ensemble(
        "GGCGCGGCACCGUCCGCGGAACAAACGG",
        SimulationConfig(
            frames=6, rates=fast_rates(),
            transcription=TranscriptionSchedule(rate=60.0, post_time=0.5),
        ),
        n_trajectories=4,
        seed=1,
        workers=1,
    )
    path = tmp_path / "e.drf"
    write_drf(ensemble, path)
    back = read_drf(path)
    assert back.rows
    assert all(0.0 <= occupancy <= 1.0 for _t, occupancy, _s, _e in back.rows)
