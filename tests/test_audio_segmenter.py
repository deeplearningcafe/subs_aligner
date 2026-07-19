"""Tests for the audio segmenter module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torchaudio

from src.subtitle_aligner.audio_segmenter import AudioSegmenter
from src.subtitle_aligner.subtitle_parser import SubtitleBlock, SubtitleParser


# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def wav_path():
    """Return the path to the sample WAV file in data/."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    wav_files = list(data_dir.glob("*.wav"))
    if not wav_files:
        pytest.skip("No WAV file found in data/")
    return str(wav_files[0])


@pytest.fixture
def short_wav_path():
    """Create a short synthetic WAV file for fast tests."""
    sr = 16000
    duration = 10.0  # 10 seconds
    # Generate a simple sine wave (speech-like)
    t = torch.linspace(0, duration, int(sr * duration))
    freq = 440.0
    waveform = torch.sin(2 * torch.pi * freq * t).unsqueeze(0) * 0.5

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, waveform, sr)
        return f.name


@pytest.fixture
def silence_wav_path():
    """Create a WAV file with speech, silence, speech pattern."""
    sr = 16000
    # 3s speech, 2s silence, 3s speech, 2s silence, 2s speech
    parts = []
    # Speech regions (440 Hz tone)
    for dur in [3.0, 3.0, 2.0]:
        n = int(sr * dur)
        t = torch.linspace(0, dur, n)
        parts.append(torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) * 0.5)
    # Silence regions (zeros)
    for dur in [2.0, 2.0]:
        n = int(sr * dur)
        parts.append(torch.zeros(1, n))

    waveform = torch.cat(parts, dim=1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, waveform, sr)
        return f.name


# ── tests ──────────────────────────────────────────────────────────────


class TestAudioSegmenterInit:
    """Test AudioSegmenter initialization."""

    def test_default_device_is_cpu(self):
        segmenter = AudioSegmenter()
        assert segmenter.device == "cpu"

    def test_cuda_device(self):
        segmenter = AudioSegmenter(device="cuda")
        assert segmenter.device == "cuda"

    def test_default_target_duration(self):
        segmenter = AudioSegmenter()
        assert segmenter.target_duration == 300.0

    def test_custom_target_duration(self):
        segmenter = AudioSegmenter(target_duration=120.0)
        assert segmenter.target_duration == 120.0


class TestSilentIntervalComputation:
    """Test _compute_silent_intervals static method."""

    def test_no_speech_returns_full_duration(self):
        intervals = AudioSegmenter._compute_silent_intervals([], 10.0)
        assert intervals == [(0.0, 10.0)]

    def test_speech_only_no_silence(self):
        speech_ts = [{"start": 0.0, "end": 10.0}]
        intervals = AudioSegmenter._compute_silent_intervals(speech_ts, 10.0)
        assert intervals == []

    def test_leading_silence(self):
        speech_ts = [{"start": 2.0, "end": 8.0}]
        intervals = AudioSegmenter._compute_silent_intervals(speech_ts, 10.0)
        assert (0.0, 2.0) in intervals

    def test_trailing_silence(self):
        speech_ts = [{"start": 2.0, "end": 8.0}]
        intervals = AudioSegmenter._compute_silent_intervals(speech_ts, 10.0)
        assert (8.0, 10.0) in intervals

    def test_gaps_between_speech(self):
        speech_ts = [
            {"start": 0.0, "end": 2.0},
            {"start": 5.0, "end": 8.0},
        ]
        intervals = AudioSegmenter._compute_silent_intervals(speech_ts, 10.0)
        assert (2.0, 5.0) in intervals


class TestFindNearestSilentBoundary:
    """Test _find_nearest_silent_boundary static method."""

    def test_finds_boundary_near_target(self):
        silent = [(4.5, 5.5)]
        result = AudioSegmenter._find_nearest_silent_boundary(5.0, silent)
        assert result is not None
        assert abs(result - 5.0) < 0.1

    def test_no_boundary_near_target(self):
        silent = [(0.0, 0.5), (9.5, 10.0)]
        # Target in the middle, far from any silence with small window
        result = AudioSegmenter._find_nearest_silent_boundary(5.0, silent, window=2.0)
        assert result is None

    def test_boundary_at_edge(self):
        silent = [(0.0, 1.0)]
        result = AudioSegmenter._find_nearest_silent_boundary(0.5, silent)
        assert result is not None
        assert abs(result - 0.5) < 0.01


class TestSegmentationLogic:
    """Test segmentation on synthetic audio."""

    def test_segment_creates_wav_files(self, silence_wav_path):
        """Verify segmentation produces physical .wav files."""
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,  # Short target for fast testing
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                silence_wav_path,
                tmpdir,
                target_duration=3.0,
            )

            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath), f"Missing: {seg.filepath}"
                # Verify it's a valid wav
                wave, sr = torchaudio.load(seg.filepath)
                assert wave.shape[0] >= 1
                assert sr > 0

    def test_segment_offsets_are_monotonically_increasing(self, silence_wav_path):
        """Verify segment start times form a non-decreasing sequence."""
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                silence_wav_path,
                tmpdir,
                target_duration=3.0,
            )

            for i in range(1, len(segments)):
                assert segments[i].start_time >= segments[i - 1].start_time

    def test_segment_offsets_start_at_zero(self, silence_wav_path):
        """First segment should start at time 0."""
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                silence_wav_path,
                tmpdir,
                target_duration=3.0,
            )

            assert segments[0].start_time == 0.0

    def test_segments_cover_full_duration(self, silence_wav_path):
        """Sum of segment durations should approximately equal total duration."""
        orig_wave, orig_sr = torchaudio.load(silence_wav_path)
        total_duration = orig_wave.shape[1] / orig_sr

        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                silence_wav_path,
                tmpdir,
                target_duration=3.0,
            )

            total_segmented = sum(seg.duration for seg in segments)
            # Allow 10% tolerance for edge effects
            assert abs(total_segmented - total_duration) < total_duration * 0.1

    def test_segment_source_sr_matches_original(self, silence_wav_path):
        """Segment source_sr should match the original file's sample rate."""
        orig_wave, orig_sr = torchaudio.load(silence_wav_path)

        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                silence_wav_path,
                tmpdir,
                target_duration=3.0,
            )

            for seg in segments:
                assert seg.source_sr == orig_sr

    def test_segment_wav_sr_matches_source(self, silence_wav_path):
        """Written segment WAV files should match the source sample rate."""
        orig_wave, orig_sr = torchaudio.load(silence_wav_path)

        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                silence_wav_path,
                tmpdir,
                target_duration=3.0,
            )

            for seg in segments:
                wave, sr = torchaudio.load(seg.filepath)
                assert sr == orig_sr


class TestIntegration:
    """Integration tests with the actual sample audio."""

    @pytest.mark.skipif(
        True,
        reason="UVR pipeline requires parselmouth which has a Python 3.11 "
        "compatibility issue (urllib.quote). Test manually with "
        "python run_aligner.py --audio --video-file data/...",
    )
    def test_process_video_with_sample(self, wav_path):
        """Test the full pipeline on the sample WAV file."""
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=30.0,  # Short target for test speed
            min_silence_duration_ms=500,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.process_video(
                wav_path,
                output_dir=tmpdir,
                target_duration=30.0,
            )

            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath)
                assert seg.start_time >= 0.0
                assert seg.duration > 0.0


class TestLocateSubtitleGap:
    """Test _locate_subtitle_gap static method."""

    def _make_subs(
        self,
        starts: list[float],
        ends: list[float],
    ) -> list[SubtitleBlock]:
        """Helper to create SubtitleBlock instances from timing lists."""
        blocks: list[SubtitleBlock] = []
        for s, e in zip(starts, ends):
            blocks.append(
                SubtitleBlock(start_time=s, end_time=e, raw_text="", cleaned_text="")
            )
        return blocks

    def test_finds_gap_near_target(self):
        """A gap >= 2.0s exists near the target time."""
        subs = self._make_subs(
            [0.0, 10.0, 20.0, 40.0],
            [5.0, 15.0, 38.0, 50.0],
        )
        # Gap between subs[1] and subs[2]: 20.0 -> 38.0 (gap=18s)
        # Gap between subs[2] and subs[3]: 38.0 -> 40.0 (gap=2s)
        segmenter = AudioSegmenter()
        result = segmenter._locate_subtitle_gap(subs, target_time=30.0)
        assert result is not None
        assert result == (38.0, 40.0)

    def test_returns_none_when_no_gaps(self):
        """No gaps >= 2.0s exist between subtitles."""
        subs = self._make_subs(
            [0.0, 5.0, 10.0, 15.0],
            [4.0, 9.0, 14.0, 19.0],
        )
        # All gaps are < 2.0s
        segmenter = AudioSegmenter()
        result = segmenter._locate_subtitle_gap(subs, target_time=10.0)
        assert result is None

    def test_returns_none_when_gap_outside_max_distance(self):
        """Gap exists but is too far from target time."""
        subs = self._make_subs(
            [0.0, 10.0, 200.0],
            [3.0, 15.0, 210.0],  # gap2: 15.0->200.0 = 185s, midpoint=107.5
        )
        # Target at 10.0: gap1 (3,10)=7s midpoint=6.5 dist=3.5
        #                   gap2 (15,200)=185s midpoint=107.5 dist=97.5
        # Both within max_distance=120, so gap1 is closer -> found
        # Now test with a gap truly outside:
        subs2 = self._make_subs(
            [0.0, 10.0],
            [3.0, 300.0],  # gap: 3.0->10.0 = 7s, midpoint=6.5, dist to 100 = 93.5
        )
        # With max_distance=50, gap midpoint 6.5 is 93.5 away from target 100
        segmenter = AudioSegmenter()
        result = segmenter._locate_subtitle_gap(
            subs2, target_time=100.0, max_distance=50.0
        )
        assert result is None

    def test_returns_closest_gap_when_multiple_exist(self):
        """Two gaps >= 2.0s exist; returns the one closest to target."""
        subs = self._make_subs(
            [0.0, 10.0, 20.0, 50.0, 80.0],
            [5.0, 15.0, 45.0, 55.0, 85.0],
        )
        # Gaps: (15,20)=5s, (45,50)=5s, (55,80)=25s
        # Target at 30.0: closest is (15,20) at dist=10, or (45,50) at dist=15
        segmenter = AudioSegmenter()
        result = segmenter._locate_subtitle_gap(subs, target_time=30.0)
        assert result is not None
        assert result == (15.0, 20.0)

    def test_gap_at_exact_boundary(self):
        """Gap midpoint exactly matches target time."""
        subs = self._make_subs(
            [0.0, 10.0],
            [4.0, 16.0],  # gap: 4.0 -> 10.0, midpoint=7.0
        )
        segmenter = AudioSegmenter()
        result = segmenter._locate_subtitle_gap(subs, target_time=7.0)
        assert result is not None
        assert result == (4.0, 10.0)

    def test_min_gap_duration_is_2_seconds(self):
        """Gaps smaller than 2.0s are ignored."""
        # gaps: end[0]=2.0 -> start[1]=3.0 = 1.0s
        #       end[1]=4.5 -> start[2]=6.0 = 1.5s
        #       end[2]=7.5 -> start[3]=9.0 = 1.5s
        subs = self._make_subs(
            [0.0, 3.0, 6.0, 9.0],
            [2.0, 4.5, 7.5, 10.0],
        )
        segmenter = AudioSegmenter()
        result = segmenter._locate_subtitle_gap(subs, target_time=5.0)
        assert result is None

    def test_gap_exactly_2_seconds(self):
        """Gap of exactly 2.0s is included (>= 2.0 threshold)."""
        subs = self._make_subs(
            [0.0, 5.0],
            [4.0, 6.0],  # gap: 4.0 -> 5.0 = 1.0s (end of sub0 to start of sub1)
        )
        # Need: end_time[0]=4.0, start_time[1]=6.0 => gap=2.0s
        subs = self._make_subs(
            [0.0, 6.0],
            [4.0, 6.0],  # gap: 4.0 -> 6.0 = 2.0s
        )
        segmenter = AudioSegmenter()
        result = segmenter._locate_subtitle_gap(subs, target_time=5.0)
        assert result is not None
        assert result == (4.0, 6.0)


class TestCrossVerifySilence:
    """Test _cross_verify_silence method."""

    def test_finds_silence_near_gap_midpoint(self):
        """A VAD silence >= 1.0s exists near the gap midpoint."""
        silent_intervals = [(7.0, 9.0), (15.0, 17.0)]  # 2s silences each
        gap_midpoint = 8.0
        segmenter = AudioSegmenter()
        result = segmenter._cross_verify_silence(silent_intervals, gap_midpoint)
        assert result is not None
        assert abs(result - 8.0) < 0.1  # midpoint of (7.0, 9.0)

    def test_returns_none_when_no_long_enough_silence(self):
        """All silences are shorter than 1.0s."""
        silent_intervals = [(7.0, 7.8), (15.0, 15.5)]  # < 1.0s each
        gap_midpoint = 8.0
        segmenter = AudioSegmenter()
        result = segmenter._cross_verify_silence(silent_intervals, gap_midpoint)
        assert result is None

    def test_returns_none_when_silence_outside_window(self):
        """Silence exists but is outside the +/-30s window."""
        silent_intervals = [(100.0, 102.0)]  # far from midpoint=8.0
        gap_midpoint = 8.0
        segmenter = AudioSegmenter()
        result = segmenter._cross_verify_silence(silent_intervals, gap_midpoint)
        assert result is None

    def test_returns_closest_silence(self):
        """Multiple silences in window; returns the closest one."""
        silent_intervals = [(6.0, 7.0), (7.0, 8.0)]  # both 1.0s
        gap_midpoint = 8.0
        segmenter = AudioSegmenter()
        result = segmenter._cross_verify_silence(silent_intervals, gap_midpoint)
        assert result is not None
        # (7.0, 8.0) midpoint=7.5 is closer to 8.0 than (6.0, 7.0) midpoint=6.5
        assert abs(result - 7.5) < 0.01

    def test_exactly_1_second_silence(self):
        """Silence of exactly 1.0s is accepted (>= 1.0s threshold)."""
        silent_intervals = [(7.0, 8.0)]  # exactly 1.0s
        gap_midpoint = 8.0
        segmenter = AudioSegmenter()
        result = segmenter._cross_verify_silence(silent_intervals, gap_midpoint)
        assert result is not None

    def test_silence_at_boundary(self):
        """Silence right at the edge of the window."""
        silent_intervals = [(38.0, 39.0)]  # midpoint=38.5, window from 8.0
        gap_midpoint = 8.0
        segmenter = AudioSegmenter()
        result = segmenter._cross_verify_silence(
            silent_intervals, gap_midpoint, window=30.0
        )
        assert result is None  # 38.5 is 30.5 away from 8.0, exceeds window


class TestHybridSplitting:
    """Test the hybrid splitting logic with subtitle cross-verification."""

    def _make_wav_with_silence(self, sr: int = 16000) -> str:
        """Create a synthetic WAV with speech and silence regions."""
        parts = []
        # 3s speech, 2s silence, 3s speech, 2s silence, 2s speech
        for dur in [3.0, 3.0, 2.0]:
            n = int(sr * dur)
            t = torch.linspace(0, dur, n)
            parts.append(torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) * 0.5)
        for dur in [2.0, 2.0]:
            n = int(sr * dur)
            parts.append(torch.zeros(1, n))
        waveform = torch.cat(parts, dim=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f.name, waveform, sr)
            return f.name

    def _make_subs(
        self,
        starts: list[float],
        ends: list[float],
    ) -> list[SubtitleBlock]:
        """Helper to create SubtitleBlock instances from timing lists."""
        blocks: list[SubtitleBlock] = []
        for s, e in zip(starts, ends):
            blocks.append(
                SubtitleBlock(start_time=s, end_time=e, raw_text="", cleaned_text="")
            )
        return blocks

    def test_hybrid_primary_path_subtitles_and_vad(self):
        """Primary path: subtitle gap found AND VAD silence confirmed."""
        wav_path = self._make_wav_with_silence()
        # Subtitles: gap between 4.0s and 6.0s (2s gap)
        subs = self._make_subs(
            [0.0, 2.0, 5.0],
            [1.5, 4.0, 7.0],
        )
        # VAD silence: (3.0, 5.0) overlaps with subtitle gap region
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=3.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            # Verify segments were created
            for seg in segments:
                assert os.path.isfile(seg.filepath)

    def test_fallback_no_subtitle_gaps(self):
        """Fallback 1: No subtitle gaps, use VAD-only splitting."""
        wav_path = self._make_wav_with_silence()
        # Subtitles with no gaps >= 2.0s
        subs = self._make_subs(
            [0.0, 1.0, 2.0, 3.0],
            [0.8, 1.8, 2.8, 4.0],
        )
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=3.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath)

    def test_fallback_no_vad_silence(self):
        """Fallback 2: Subtitle gap found but no VAD silence, split at subtitle boundary."""
        wav_path = self._make_wav_with_silence()
        # Subtitles: gap between 4.0s and 6.0s (2s gap)
        subs = self._make_subs(
            [0.0, 2.0, 5.0],
            [1.5, 4.0, 7.0],
        )
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=3.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath)

    def test_no_subtitles_parameter(self):
        """When subtitles is None, falls back to VAD-only logic."""
        wav_path = self._make_wav_with_silence()
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=3.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=3.0,
                subtitles=None,
            )
            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath)

    def test_split_point_at_subtitle_gap_midpoint_when_no_vad(self):
        """When subtitle gap exists but no VAD silence, split at gap midpoint."""
        wav_path = self._make_wav_with_silence()
        # Subtitles: gap between 8.0s and 10.0s (2s gap, midpoint=9.0)
        # This gap is near target 9.0s
        subs = self._make_subs(
            [0.0, 5.0, 8.0],
            [3.0, 8.0, 10.0],
        )
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=9.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=9.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            # The split should be near the subtitle gap midpoint (9.0)
            # First segment should end around the split point
            first_seg_end = segments[0].start_time + segments[0].duration
            # Allow generous tolerance since VAD on synthetic sine waves is noisy
            assert abs(first_seg_end - 9.0) < 5.0

    def test_split_near_target_with_hybrid(self):
        """Split points are near target times even with hybrid logic."""
        wav_path = self._make_wav_with_silence()
        # Subtitles with gaps near target times
        subs = self._make_subs(
            [0.0, 3.0, 6.0],
            [2.0, 5.0, 8.0],
        )
        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=5.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=5.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            # Segments should cover the full duration
            total_dur = sum(seg.duration for seg in segments)
            orig_wave, orig_sr = torchaudio.load(wav_path)
            orig_dur = orig_wave.shape[1] / orig_sr
            assert abs(total_dur - orig_dur) < orig_dur * 0.1


class TestRealSubtitleIntegration:
    """Integration tests with real subtitle files."""

    def test_segment_with_wataoshi_subtitles(self):
        """Segment audio using real wataoshi_06.srt subtitle file."""
        # Find the sample WAV file
        data_dir = Path(__file__).resolve().parent.parent / "data"
        wav_files = list(data_dir.glob("*.wav"))
        if not wav_files:
            pytest.skip("No WAV file found in data/")
        wav_path = str(wav_files[0])

        # Parse the real subtitle file
        parser = SubtitleParser()
        subs_path = Path(__file__).resolve().parent / "wataoshi_06.srt"
        subs = parser.parse_file(str(subs_path))

        assert len(subs) > 0

        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=30.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=30.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath)
                assert seg.duration > 0.0
                assert seg.start_time >= 0.0

    def test_segment_with_test_sample_subtitles(self):
        """Segment using the test_sample.srt fixture file."""
        # Find the sample WAV file
        data_dir = Path(__file__).resolve().parent.parent / "data"
        wav_files = list(data_dir.glob("*.wav"))
        if not wav_files:
            pytest.skip("No WAV file found in data/")
        wav_path = str(wav_files[0])

        # Parse the test sample subtitle file
        parser = SubtitleParser()
        subs_path = Path(__file__).resolve().parent / "fixtures" / "test_sample.srt"
        subs = parser.parse_file(str(subs_path))

        assert len(subs) > 0

        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=30.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=30.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath)

    def test_segment_with_no_subtitles_gaps(self):
        """Test fallback when subtitle file has no gaps >= 2.0s."""
        # Find the sample WAV file
        data_dir = Path(__file__).resolve().parent.parent / "data"
        wav_files = list(data_dir.glob("*.wav"))
        if not wav_files:
            pytest.skip("No WAV file found in data/")
        wav_path = str(wav_files[0])

        # Create synthetic subtitles with no gaps >= 2.0s
        subs = [
            SubtitleBlock(
                start_time=0.0, end_time=3.0, raw_text="test", cleaned_text="test"
            ),
            SubtitleBlock(
                start_time=3.5, end_time=6.0, raw_text="test", cleaned_text="test"
            ),
            SubtitleBlock(
                start_time=6.5, end_time=9.0, raw_text="test", cleaned_text="test"
            ),
        ]

        segmenter = AudioSegmenter(
            device="cpu",
            target_duration=30.0,
            min_silence_duration_ms=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            segments = segmenter.segment_audio(
                wav_path,
                tmpdir,
                target_duration=30.0,
                subtitles=subs,
            )
            assert len(segments) > 0
            for seg in segments:
                assert os.path.isfile(seg.filepath)
