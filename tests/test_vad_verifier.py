"""Tests for VAD-based post-verification and speech snapping."""

from __future__ import annotations

import pytest

from src.subtitle_aligner.vad_verifier import VADVerifier


@pytest.fixture
def verifier():
    """Create a default VAD verifier instance."""
    return VADVerifier(min_vad_ratio=0.25, padding_seconds=0.150)


@pytest.fixture
def sample_vad_intervals():
    """Return mock VAD intervals on a timeline."""
    return [
        {"start": 1.0, "end": 3.0},
        {"start": 5.0, "end": 8.0},
        {"start": 12.0, "end": 15.0},
    ]


def test_calculate_vad_ratio_no_speech(verifier, sample_vad_intervals):
    """Calculate ratio when segment sits completely in a silent gap."""
    ratio = verifier.calculate_vad_ratio(8.5, 11.5, sample_vad_intervals)
    assert ratio == 0.0


def test_calculate_vad_ratio_full_speech(verifier, sample_vad_intervals):
    """Calculate ratio when segment is fully inside speech."""
    ratio = verifier.calculate_vad_ratio(1.5, 2.5, sample_vad_intervals)
    assert ratio == pytest.approx(1.0)


def test_calculate_vad_ratio_partial_speech(verifier, sample_vad_intervals):
    """Calculate ratio with partial speech overlap."""
    # Overlap with (5.0, 8.0) -> overlap duration = 1.0s (5.0 to 6.0)
    ratio = verifier.calculate_vad_ratio(4.0, 6.0, sample_vad_intervals)
    assert ratio == pytest.approx(0.5)


def test_verify_segment_valid(verifier, sample_vad_intervals):
    """Validate segment when speech ratio is above the 25% threshold."""
    assert verifier.verify_segment(4.0, 6.0, sample_vad_intervals) is True


def test_verify_segment_hallucination(verifier, sample_vad_intervals):
    """Discard segment when speech ratio is below the 25% threshold."""
    # Segment from 7.5 to 11.5 (duration 4.0s)
    # Overlaps with (5.0, 8.0) -> overlap is (7.5 to 8.0) = 0.5s
    # Ratio = 0.5 / 4.0 = 0.125 (12.5% < 25%)
    assert verifier.verify_segment(7.5, 11.5, sample_vad_intervals) is False


def test_snap_and_pad_segment_normal(verifier, sample_vad_intervals):
    """Verify boundaries snap to overlapping speech and add 150ms padding."""
    # Segment from 4.5 to 9.0 (duration 4.5s)
    # Overlapping VAD interval: (5.0, 8.0)
    # Snapped bounds: start=max(4.5, 5.0)=5.0, end=min(9.0, 8.0)=8.0
    # Padded bounds: start=5.0-0.15=4.85, end=8.0+0.15=8.15
    p_start, p_end = verifier.snap_and_pad_segment(4.5, 9.0, sample_vad_intervals)
    assert p_start == pytest.approx(4.85)
    assert p_end == pytest.approx(8.15)


def test_snap_and_pad_segment_multi_overlaps(verifier, sample_vad_intervals):
    """Verify snapping with multiple overlapping VAD intervals."""
    # Overlapping intervals: (1.0, 3.0) and (5.0, 8.0)
    # Snapped bounds: start=max(0.5, 1.0)=1.0, end=min(6.0, 8.0)=6.0
    p_start, p_end = verifier.snap_and_pad_segment(0.5, 6.0, sample_vad_intervals)
    assert p_start == pytest.approx(0.85)
    assert p_end == pytest.approx(6.15)


def test_snap_and_pad_segment_no_overlap(verifier, sample_vad_intervals):
    """Fallback to original timestamps when no speech overlap is found."""
    p_start, p_end = verifier.snap_and_pad_segment(8.5, 11.5, sample_vad_intervals)
    assert p_start == 8.5
    assert p_end == 11.5
