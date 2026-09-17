import numpy as np
import pytest

from rona.cotrans import SimulationConfig, TranscriptionSchedule
from rona.ensemble import simulate_ensemble
from rona.struct import iter_pairs, parse_dotbracket

SEQ = "GGCGCGGCACCGUCCGCGGAACAAACGGAGAAGGGGCCGCCG"


@pytest.fixture(scope="module")
def ensemble():
    config = SimulationConfig(
        frames=20, transcription=TranscriptionSchedule(rate=40.0, post_time=2.0)
    )
    return simulate_ensemble(SEQ, config, n_trajectories=12, seed=2, workers=1)


def test_shape_and_alignment(ensemble):
    assert ensemble.n_trajectories == 12
    assert ensemble.n_times == 20
    assert ensemble.energies.shape == (20, 12)
    assert len(ensemble.lengths) == 20
    assert ensemble.lengths[-1] == len(SEQ)


def test_occupancy_is_a_distribution(ensemble):
    _labels, matrix = ensemble.occupancy()
    assert np.allclose(matrix.sum(axis=0), 1.0)
    assert (matrix >= 0).all()


def test_occupancy_filter_only_removes_mass(ensemble):
    _all_labels, full = ensemble.occupancy()
    _few_labels, filtered = ensemble.occupancy(min_population=0.5)
    assert (filtered.sum(axis=0) <= full.sum(axis=0) + 1e-9).all()


def test_dominant_matches_occupancy(ensemble):
    labels, matrix = ensemble.occupancy()
    for t, (structure, population) in enumerate(ensemble.dominant()):
        assert population == pytest.approx(matrix[:, t].max())
        assert labels[int(np.argmax(matrix[:, t]))] == structure


def test_pair_probabilities_are_consistent(ensemble):
    probabilities = ensemble.pair_probabilities()
    assert probabilities.shape == (20, len(SEQ), len(SEQ))
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    # a nucleotide cannot be paired with probability above 1
    for t in range(ensemble.n_times):
        totals = probabilities[t].sum(axis=0) + probabilities[t].sum(axis=1)
        assert (totals <= 1.0 + 1e-9).all()
    # cross-check one time point against the raw structures
    counts = {}
    for db in ensemble.structures[-1]:
        for pair in iter_pairs(parse_dotbracket(db)):
            counts[pair] = counts.get(pair, 0) + 1
    for (i, j), count in counts.items():
        assert probabilities[-1, i, j] == pytest.approx(count / 12)


def test_target_population(ensemble):
    dominant = ensemble.dominant()[-1][0]
    padded = dominant + "." * (len(SEQ) - len(dominant))
    series = ensemble.target_population(padded)
    assert series[-1] == pytest.approx(ensemble.dominant()[-1][1])
    loose = ensemble.target_population(padded, max_distance=6)
    assert (loose >= series - 1e-9).all()


def test_clustering_collapses_near_identical_structures(ensemble):
    mapping = ensemble.cluster(cutoff=4)
    clustered = ensemble.apply_clustering(mapping)
    before, _ = ensemble.occupancy()
    after, _ = clustered.occupancy()
    assert len(after) <= len(before)
    assert clustered.n_trajectories == ensemble.n_trajectories


def test_summary_series(ensemble):
    assert ensemble.mean_energy().shape == (20,)
    assert (ensemble.energy_spread() >= 0).all()
    fraction = ensemble.pseudoknot_fraction()
    assert ((fraction >= 0) & (fraction <= 1)).all()


def test_serialisation_round_trips(ensemble, tmp_path):
    import json

    path = tmp_path / "e.json"
    ensemble.to_json(path)
    payload = json.loads(path.read_text())
    assert payload["sequence"] == SEQ
    assert len(payload["times"]) == 20
    assert len(payload["dominant"]) == 20


def test_parallel_matches_serial():
    config = SimulationConfig(
        frames=8, transcription=TranscriptionSchedule(rate=50.0, post_time=1.0)
    )
    serial = simulate_ensemble(SEQ, config, n_trajectories=4, seed=7, workers=1)
    parallel = simulate_ensemble(SEQ, config, n_trajectories=4, seed=7, workers=2)
    assert serial.structures == parallel.structures
