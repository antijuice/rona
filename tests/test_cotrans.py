import pytest

from conftest import fast_rates
from rona.cotrans import (
    Pause,
    SimulationConfig,
    TranscriptionSchedule,
    simulate_trajectory,
    time_grid,
)

# short on purpose: these tests exercise plumbing, and simulation
# cost grows steeply with length
SEQ = "GGCGCGGCACCGUCCGCGGAACAAACGG"


def test_arrival_times_respect_rate_and_start_length():
    schedule = TranscriptionSchedule(rate=10.0, start_length=5, post_time=0.0)
    arrivals = schedule.arrival_times(20)
    assert arrivals[5] == pytest.approx(0.0)
    assert arrivals[6] == pytest.approx(0.1)
    assert arrivals[20] == pytest.approx(1.5)


def test_pause_sites_delay_everything_downstream():
    plain = TranscriptionSchedule(rate=10.0, start_length=0, post_time=0.0)
    paused = TranscriptionSchedule(
        rate=10.0, start_length=0, post_time=0.0,
        pauses=(Pause(position=5, duration=2.0),),
    )
    assert paused.arrival_times(10)[4] == pytest.approx(plain.arrival_times(10)[4])
    assert paused.arrival_times(10)[10] == pytest.approx(
        plain.arrival_times(10)[10] + 2.0
    )


def test_zero_rate_rejected():
    with pytest.raises(ValueError):
        TranscriptionSchedule(rate=0.0).arrival_times(10)


def test_time_grid_shapes():
    assert time_grid(5.0, 6)[0] == 0.0
    assert time_grid(5.0, 6)[-1] == pytest.approx(5.0)
    assert len(time_grid(5.0, 40)) == 40
    log = time_grid(5.0, 20, "log")
    assert log[0] == 0.0 and log[-1] == pytest.approx(5.0)
    assert all(b >= a for a, b in zip(log, log[1:]))


def test_trajectory_is_reproducible_and_grid_aligned():
    config = SimulationConfig(
        frames=25, rates=fast_rates(),
        transcription=TranscriptionSchedule(post_time=0.5),
    )
    a = simulate_trajectory(SEQ, config, seed=3)
    b = simulate_trajectory(SEQ, config, seed=3)
    c = simulate_trajectory(SEQ, config, seed=4)
    assert [f.structure for f in a.frames] == [f.structure for f in b.frames]
    assert len(a.frames) == 25
    assert a.times == b.times == c.times
    # different seeds should explore different pathways
    assert any(
        x.structure != y.structure for x, y in zip(a.frames, c.frames)
    ) or a.events != c.events


def test_polymerase_footprint_keeps_the_3_prime_end_unpaired():
    """Nothing inside the footprint may pair while transcription is running."""
    footprint = 12
    config = SimulationConfig(
        frames=40, rates=fast_rates(),
        transcription=TranscriptionSchedule(
            rate=20.0, footprint=footprint, post_time=0.0
        ),
    )
    trajectory = simulate_trajectory(SEQ, config, seed=8)
    for frame in trajectory.frames:
        if frame.length >= len(SEQ):
            continue  # released: the whole chain is free
        tail = frame.structure[max(0, frame.length - footprint) : frame.length]
        assert set(tail) <= {"."}, f"pairing inside the footprint: {frame.structure}"


def test_structures_only_span_the_transcribed_prefix():
    config = SimulationConfig(
        frames=30, rates=fast_rates(),
        transcription=TranscriptionSchedule(post_time=0.3),
    )
    trajectory = simulate_trajectory(SEQ, config, seed=5)
    for frame in trajectory.frames:
        assert len(frame.structure) == frame.length
        assert frame.length <= len(SEQ)


def test_equilibrium_mode_has_the_whole_chain_from_the_start():
    config = SimulationConfig(
        frames=10, rates=fast_rates(), transcription=None, duration=1.0
    )
    trajectory = simulate_trajectory(SEQ, config, seed=1)
    assert all(f.length == len(SEQ) for f in trajectory.frames)
