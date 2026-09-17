"""Rendering tests: geometry, well-formed output, and movie-frame coherence."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import fast_rates
from rona.cotrans import SimulationConfig, TranscriptionSchedule
from rona.ensemble import simulate_ensemble
from rona.render import plots
from rona.render.layout import (
    LayoutOptions,
    bounding_box,
    camera_path,
    fit_aspect,
    interpolate,
    kabsch_align,
    layout_series,
    layout_structure,
)
from rona.render.player import player_html
from rona.render.svg import DrawOptions, legend_svg, structure_svg
from rona.struct import iter_pairs, parse_dotbracket

# short on purpose: these tests exercise plumbing, and simulation
# cost grows steeply with length
SEQ = "GGCGCGGCACCGUCCGCGGAACAAACGG"

NESTED = [
    "((((((....))))))",
    "(((((((..((((........)))).(((((.......))))).....(((((.......))))))))))))....",
    "((((...((((....))))...((((....))))...))))",
]
KNOTTED = [
    "((((((((....[[[[[[))))))))........]]]]]]..",
    "(((((....[[[[[)))))....]]]]].........",
]


@pytest.mark.parametrize("structure", NESTED)
def test_nested_layout_has_exact_geometry(structure):
    """Nested structures need no relaxation, so the geometry must be exact."""
    options = LayoutOptions()
    coords = layout_structure(structure, options=options)
    assert coords.shape == (len(structure), 2)
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    assert np.allclose(steps, options.bond, atol=1e-6)
    pt = parse_dotbracket(structure)
    for i, j in iter_pairs(pt):
        assert np.linalg.norm(coords[i] - coords[j]) == pytest.approx(
            options.pair, abs=1e-6
        )


@pytest.mark.parametrize("structure", NESTED + KNOTTED)
def test_layout_keeps_nucleotides_apart(structure):
    coords = layout_structure(structure)
    for i in range(len(structure)):
        for j in range(i + 2, len(structure)):
            assert np.linalg.norm(coords[i] - coords[j]) > 0.35


@pytest.mark.parametrize("structure", KNOTTED)
def test_pseudoknot_pairs_are_brought_together(structure):
    """Crossing partners must end up roughly a pair-width apart, not splayed."""
    options = LayoutOptions()
    coords = layout_structure(structure, options=options)
    pt = parse_dotbracket(structure)
    distances = [
        float(np.linalg.norm(coords[i] - coords[j])) for i, j in iter_pairs(pt)
    ]
    assert max(distances) < options.pair * 1.6
    assert min(distances) > options.pair * 0.55


def test_kabsch_align_recovers_a_rigid_motion():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(24, 2))
    angle = 0.9
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    moved = reference @ rotation.T + np.array([3.0, -2.0])
    assert np.allclose(kabsch_align(moved, reference), reference, atol=1e-9)


def test_kabsch_align_never_reflects():
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mirrored = reference * np.array([1.0, -1.0])
    aligned = kabsch_align(mirrored, reference)
    # a reflection would superpose exactly; a proper rotation cannot
    assert not np.allclose(aligned, reference, atol=1e-6)


def test_layout_series_is_frame_to_frame_coherent():
    """Consecutive frames must not jump when the structure barely changes."""
    series = ["((((((....))))))" for _ in range(3)] + [".(((((....)))))."]
    frames = layout_series(series)
    for a, b in zip(frames, frames[1:3]):
        assert np.allclose(a, b, atol=1e-6)
    drift = np.linalg.norm(frames[3] - frames[2], axis=1).mean()
    assert drift < 1.5  # a redraw without alignment moves far more than this


def test_interpolation_is_smooth_and_endpoint_exact():
    frames = [np.zeros((4, 2)), np.ones((4, 2))]
    tweened = interpolate(frames, 5, 4)
    assert len(tweened) == 6
    assert np.allclose(tweened[0], frames[0])
    assert np.allclose(tweened[-1], frames[1])
    values = [float(f[0, 0]) for f in tweened]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_camera_never_clips_the_molecule():
    """The view box must always contain every nucleotide of its frame."""
    rng = np.random.default_rng(3)
    frames = [rng.normal(scale=1.0 + k * 0.2, size=(30, 2)) for k in range(25)]
    boxes = camera_path(frames)
    assert boxes.shape == (25, 4)
    for frame, box in zip(frames, boxes):
        assert box[0] <= frame[:, 0].min() and frame[:, 0].max() <= box[2]
        assert box[1] <= frame[:, 1].min() and frame[:, 1].max() <= box[3]


def test_camera_moves_smoothly():
    """Consecutive view boxes must not jump, or the movie judders."""
    frames = []
    for k in range(40):
        scale = 1.0 + 3.0 * k / 39.0
        frames.append(np.array([[-scale, -scale], [scale, scale], [0.0, 0.5]]))
    boxes = camera_path(frames)
    widths = boxes[:, 2] - boxes[:, 0]
    steps = np.abs(np.diff(widths))
    assert steps.max() < 0.25 * widths.mean()


def test_fit_aspect_reaches_the_target_ratio():
    for aspect in (0.5, 1.0, 1.75):
        box = fit_aspect((0.0, 0.0, 4.0, 1.0), aspect)
        assert (box[2] - box[0]) / (box[3] - box[1]) == pytest.approx(aspect)
        # only ever grows, never crops
        assert box[0] <= 0.0 and box[2] >= 4.0
        assert box[1] <= 0.0 and box[3] >= 1.0


def test_bounding_box():
    box = bounding_box([np.array([[0.0, 1.0], [2.0, -1.0]])])
    assert box == (0.0, -1.0, 2.0, 1.0)


def _parse(svg: str):
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    return root


@pytest.mark.parametrize("structure", NESTED + KNOTTED)
def test_structure_svg_is_well_formed(structure):
    sequence = ("GCAU" * 40)[: len(structure)]
    svg = structure_svg(
        sequence, structure, options=DrawOptions(title="t", subtitle="s")
    )
    root = _parse(svg)
    circles = [e for e in root.iter() if e.tag.endswith("circle")]
    assert len(circles) == len(structure)
    lines = [e for e in root.iter() if e.tag.endswith("line")]
    assert len(lines) == len(list(iter_pairs(parse_dotbracket(structure))))


def test_legend_svg_is_well_formed():
    _parse(legend_svg())


@pytest.fixture(scope="module")
def ensemble():
    config = SimulationConfig(
        frames=12, rates=fast_rates(),
        transcription=TranscriptionSchedule(rate=50.0, post_time=0.3),
    )
    return simulate_ensemble(SEQ, config, n_trajectories=8, seed=1, workers=1)


def test_plots_are_well_formed(ensemble):
    for svg in (
        plots.occupancy_plot(ensemble),
        plots.energy_plot(ensemble),
        plots.pair_probability_plot(ensemble, -1),
        plots.pseudoknot_plot(ensemble),
    ):
        _parse(svg)


def test_player_is_self_contained(ensemble):
    html = player_html(ensemble)
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # no external resources of any kind
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "src=" not in html
    assert '<script id="payload"' in html
    import json
    import re

    payload = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>', html, re.S
    )
    data = json.loads(payload.group(1))
    assert data["sequence"] == SEQ
    assert len(data["frames"]) == ensemble.n_times
    assert len(data["frames"][0]["xy"]) == len(SEQ)


def test_movie_frames_render(ensemble):
    matplotlib = pytest.importorskip("matplotlib")
    from rona.render.movie import MovieOptions, render_frames

    frames = list(
        render_frames(ensemble, options=MovieOptions(interpolation=1, dpi=50))
    )
    assert len(frames) == ensemble.n_times
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3
    assert frames[0].dtype == np.uint8
    # consecutive frames should differ somewhere: the cursor moves at minimum
    assert any(not np.array_equal(frames[0], f) for f in frames[1:])
