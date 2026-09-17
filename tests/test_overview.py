"""Tests for the ensemble-level views and the PNG encoder."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import fast_rates
from rona.cotrans import SimulationConfig, TranscriptionSchedule
from rona.ensemble import simulate_ensemble
from rona.render import colors, overview, png

# short on purpose: these tests exercise plumbing, and simulation
# cost grows steeply with length
SEQ = "GGCGCGGCACCGUCCGCGGAACAAACGG"


@pytest.fixture(scope="module")
def ensemble():
    config = SimulationConfig(
        frames=14, rates=fast_rates(),
        transcription=TranscriptionSchedule(rate=50.0, post_time=0.3),
    )
    return simulate_ensemble(SEQ, config, n_trajectories=8, seed=1, workers=1)


# ----------------------------------------------------------------------
def test_every_pair_of_a_helix_shares_one_colour():
    """The point of centre colouring: a helix is one colour, for its lifetime."""
    n = 60
    stem = [(5 + k, 25 - k) for k in range(5)]
    seen = {colors.pair_center_color(i, j, n) for i, j in stem}
    assert len(seen) == 1


def test_different_helices_get_different_colours():
    n = 60
    shades = {
        colors.pair_center_color(i, j, n)
        for i, j in [(5, 25), (10, 30), (20, 40), (30, 50)]
    }
    assert len(shades) == 4


def test_pair_colour_is_a_valid_hex_triple():
    for i, j in [(0, 5), (3, 40), (17, 18)]:
        value = colors.pair_center_color(i, j, 50)
        assert len(value) == 7 and value[0] == "#"
        int(value[1:], 16)


# ----------------------------------------------------------------------
def test_png_round_trips_through_an_independent_reader(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(7, 11, 4), dtype=np.uint8)
    path = tmp_path / "x.png"
    path.write_bytes(png.encode_rgba(pixels))
    assert np.array_equal(imageio.imread(path), pixels)


def test_png_rejects_wrong_shape():
    with pytest.raises(ValueError):
        png.encode_rgba(np.zeros((4, 4, 3), dtype=np.uint8))


def test_data_uri_prefix():
    uri = png.data_uri(np.zeros((2, 2, 4), dtype=np.uint8))
    assert uri.startswith("data:image/png;base64,")


# ----------------------------------------------------------------------
def test_kymograph_shape_and_transparency(ensemble):
    rgb, alpha = overview.kymograph_array(ensemble)
    assert rgb.shape == (len(SEQ), ensemble.n_times, 3)
    assert alpha.shape == (len(SEQ), ensemble.n_times)
    # untranscribed nucleotides must be fully transparent
    for t, length in enumerate(ensemble.lengths):
        assert np.all(alpha[length:, t] == 0.0)
        if length:
            assert np.all(alpha[:length, t] > 0.0)


def test_kymograph_rgba_is_uint8(ensemble):
    image = overview.kymograph_rgba(ensemble)
    assert image.shape[2] == 4 and image.dtype == np.uint8


@pytest.mark.parametrize("mode", ["dominant", "ensemble"])
def test_kymograph_modes_run(ensemble, mode):
    rgb, alpha = overview.kymograph_array(ensemble, mode=mode)
    assert np.isfinite(alpha).all()


def test_tiles_partition_the_unit_interval(ensemble):
    labels, occupancy = ensemble.occupancy(min_population=0.02)
    for t in range(ensemble.n_times):
        tiles = overview.tile_spans(labels, occupancy[:, t], min_width=0.0)
        total = sum(tile.width for tile in tiles)
        assert total == pytest.approx(1.0, abs=1e-9)
        for a, b in zip(tiles, tiles[1:]):
            assert a.start + a.width <= b.start + 1e-9


def test_tiles_keep_their_order_across_time(ensemble):
    """Tiles must not swap places, or the movie jitters."""
    labels, occupancy = ensemble.occupancy(min_population=0.02)
    orders = []
    for t in range(ensemble.n_times):
        tiles = overview.tile_spans(labels, occupancy[:, t], min_width=0.01)
        orders.append([tile.index for tile in tiles if tile.index >= 0])
    for order in orders:
        assert order == sorted(order)


def test_open_chain_merge():
    labels = ["....", "......", "((..))", "(....)"]
    matrix = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.7], [0.0, 0.3]])
    merged_labels, merged = overview.merge_open_chain(labels, matrix)
    assert merged_labels[0] == overview.OPEN_CHAIN
    assert len(merged_labels) == 3
    assert merged[0].tolist() == [1.5, 0.0]


def test_top_bands_prefers_persistent_over_transient():
    labels = ["((..))", "(....)", "((...))"]
    times = [0.0, 1.0, 2.0]
    # band 0 spikes to 1.0 for an instant; band 2 is occupied throughout
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.5, 0.6]])
    kept, _ = overview.top_bands(labels, matrix, times, 1)
    assert kept == ["((...))"]


def test_hybrid_time_axis_is_monotone_and_splits_at_transcription_end(ensemble):
    positions = overview.hybrid_time_positions(ensemble, split=0.75)
    assert np.all(np.diff(positions) >= -1e-12)
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)
    cut = overview.transition_index(ensemble)
    assert positions[cut] == pytest.approx(0.75, abs=1e-9)
    assert ensemble.lengths[cut] == len(SEQ)
