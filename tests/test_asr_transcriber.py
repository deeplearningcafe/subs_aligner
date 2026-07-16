"""Tests for the ASR transcriber module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchaudio

from src.subtitle_aligner.asr_transcriber import ASRTranscriber, TranscriptionSegment
from src.subtitle_aligner.audio_segmenter import AudioSegment


# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def short_wav_path():
    """Create a short synthetic WAV file for fast tests."""
    sr = 16000
    duration = 2.0
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) * 0.5

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, waveform, sr)
        return f.name


@pytest.fixture
def sample_segments(short_wav_path):
    """Create a couple of mock AudioSegment instances."""
    return [
        AudioSegment(
            filepath=short_wav_path,
            start_time=0.0,
            duration=2.0,
            source_sr=16000,
        ),
    ]


# ── ASRTranscriber init tests ──────────────────────────────────────────


class TestASRTranscriberInit:
    """Test ASRTranscriber initialization."""

    def test_default_device_is_cpu(self):
        transcriber = ASRTranscriber()
        assert transcriber.device == "cpu"

    def test_cuda_device(self):
        transcriber = ASRTranscriber(device="cuda")
        assert transcriber.device == "cuda"

    def test_model_not_loaded_initially(self):
        transcriber = ASRTranscriber()
        assert transcriber._model is None


# ── Katakana normalization in transcription ────────────────────────────


class TestTranscriptionSegmentData:
    """Test TranscriptionSegment dataclass."""

    def test_creates_segment(self):
        seg = TranscriptionSegment(
            start_time=1.5,
            end_time=3.2,
            text="こんにちは",
            katakana="コンニチハ",
            char_timings=[1.5, 1.8, 2.1, 2.4, 2.7],
        )
        assert seg.start_time == 1.5
        assert seg.end_time == 3.2
        assert seg.text == "こんにちは"
        assert seg.katakana == "コンニチハ"
        assert seg.char_timings == [1.5, 1.8, 2.1, 2.4, 2.7]


class TestKatakanaNormalization:
    """Test that transcription outputs are Katakana-normalized."""

    @pytest.mark.parametrize(
        "input_text,expected_has_katakana",
        [
            ("こんにちは", True),
            ("よくぞ集った精鋭たちよ", True),
            ("学院騎士団は諸君らを歓迎する", True),
            ("", False),
        ],
    )
    def test_transcriber_normalizes_katakana(self, input_text, expected_has_katakana):
        """Verify that the transcriber's internal normalizer produces Katakana."""
        transcriber = ASRTranscriber()

        if not input_text:
            katakana = transcriber._text_processor.text_to_katakana(input_text)
            assert katakana == ""
            return

        katakana = transcriber._text_processor.text_to_katakana(input_text)

        if expected_has_katakana:
            assert katakana != ""
            for ch in katakana:
                assert "\u30a0" <= ch <= "\u30ff"


# ── Timestamp mapping tests ────────────────────────────────────────────


class TestTimestampMapping:
    """Test that relative timestamps are correctly mapped to absolute times."""

    def test_offset_added_to_start_time(self):
        """Relative start_seconds + segment offset = absolute start_time."""
        transcriber = ASRTranscriber()
        mock_audio = MagicMock()

        with patch.object(transcriber, "_ensure_model"):
            with patch(
                "src.subtitle_aligner.asr_transcriber.audio_from_path",
                return_value=mock_audio,
            ):
                with patch(
                    "src.subtitle_aligner.asr_transcriber.transcribe"
                ) as mock_transcribe:
                    mock_result = MagicMock()
                    mock_result.segments = [
                        MagicMock(
                            start_seconds=0.5,
                            end_seconds=1.2,
                            text="テスト",
                        )
                    ]
                    mock_transcribe.return_value = mock_result

                    segments = transcriber.transcribe_segment("/dev/null", offset=10.0)

                    assert len(segments) == 1
                    assert segments[0].start_time == pytest.approx(10.5)
                    assert segments[0].end_time == pytest.approx(11.2)

    def test_offset_zero(self):
        """When offset is 0, absolute time equals relative time."""
        transcriber = ASRTranscriber()
        mock_audio = MagicMock()

        with patch.object(transcriber, "_ensure_model"):
            with patch(
                "src.subtitle_aligner.asr_transcriber.audio_from_path",
                return_value=mock_audio,
            ):
                with patch(
                    "src.subtitle_aligner.asr_transcriber.transcribe"
                ) as mock_transcribe:
                    mock_result = MagicMock()
                    mock_result.segments = [
                        MagicMock(
                            start_seconds=0.0,
                            end_seconds=2.5,
                            text="音声",
                        )
                    ]
                    mock_transcribe.return_value = mock_result

                    segments = transcriber.transcribe_segment("/dev/null", offset=0.0)

                    assert len(segments) == 1
                    assert segments[0].start_time == 0.0
                    assert segments[0].end_time == 2.5

    def test_multiple_segments_with_offset(self):
        """Multiple ASR segments all get the same offset applied."""
        transcriber = ASRTranscriber()
        mock_audio = MagicMock()

        with patch.object(transcriber, "_ensure_model"):
            with patch(
                "src.subtitle_aligner.asr_transcriber.audio_from_path",
                return_value=mock_audio,
            ):
                with patch(
                    "src.subtitle_aligner.asr_transcriber.transcribe"
                ) as mock_transcribe:
                    mock_result = MagicMock()
                    mock_result.segments = [
                        MagicMock(start_seconds=0.1, end_seconds=0.8, text="あ"),
                        MagicMock(start_seconds=1.0, end_seconds=1.9, text="い"),
                        MagicMock(start_seconds=2.0, end_seconds=3.0, text="う"),
                    ]
                    mock_transcribe.return_value = mock_result

                    segments = transcriber.transcribe_segment("/dev/null", offset=100.0)

                    assert len(segments) == 3
                    assert segments[0].start_time == pytest.approx(100.1)
                    assert segments[1].start_time == pytest.approx(101.0)
                    assert segments[2].start_time == pytest.approx(102.0)


class TestTranscribeSegments:
    """Test transcribe_segments (batch transcription)."""

    def test_batches_multiple_segments(self, sample_segments):
        """transcribe_segments processes all input segments."""
        transcriber = ASRTranscriber()

        with patch.object(transcriber, "_ensure_model"):
            with patch(
                "src.subtitle_aligner.asr_transcriber.transcribe"
            ) as mock_transcribe:
                mock_result = MagicMock()
                mock_result.segments = [
                    MagicMock(
                        start_seconds=0.0,
                        end_seconds=2.0,
                        text="テスト",
                    )
                ]
                mock_transcribe.return_value = mock_result

                all_segments = transcriber.transcribe_segments(sample_segments)

                assert len(all_segments) == 1
                assert all_segments[0].start_time == pytest.approx(0.0)
                assert all_segments[0].katakana != ""

    def test_flat_list_from_multiple_audio_segments(self):
        """transcribe_segments returns a flat list, not nested."""
        transcriber = ASRTranscriber()

        # Create two mock segments
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sr = 16000
            t = torch.linspace(0, 1.0, sr)
            torchaudio.save(
                f.name,
                torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) * 0.5,
                sr,
            )
            seg_path = f.name

        segments = [
            AudioSegment(
                filepath=seg_path, start_time=0.0, duration=1.0, source_sr=16000
            ),
            AudioSegment(
                filepath=seg_path, start_time=1.0, duration=1.0, source_sr=16000
            ),
        ]

        try:
            with patch.object(transcriber, "_ensure_model"):
                with patch(
                    "src.subtitle_aligner.asr_transcriber.transcribe"
                ) as mock_transcribe:
                    mock_result = MagicMock()
                    mock_result.segments = [
                        MagicMock(start_seconds=0.0, end_seconds=1.0, text="音声"),
                    ]
                    mock_transcribe.return_value = mock_result

                    all_segments = transcriber.transcribe_segments(segments)

                    assert len(all_segments) == 2
                    assert all_segments[0].start_time == pytest.approx(0.0)
                    assert all_segments[1].start_time == pytest.approx(1.0)
        finally:
            os.unlink(seg_path)


class TestKatakanaInTranscriptionOutput:
    """Test that transcription text is Katakana-normalized."""

    def test_output_katakana_is_pure(self):
        """All Katakana output contains only Katakana characters."""
        transcriber = ASRTranscriber()
        mock_audio = MagicMock()

        with patch.object(transcriber, "_ensure_model"):
            with patch(
                "src.subtitle_aligner.asr_transcriber.audio_from_path",
                return_value=mock_audio,
            ):
                with patch(
                    "src.subtitle_aligner.asr_transcriber.transcribe"
                ) as mock_transcribe:
                    mock_result = MagicMock()
                    mock_result.segments = [
                        MagicMock(
                            start_seconds=0.0,
                            end_seconds=1.0,
                            text="学院騎士団",
                        ),
                        MagicMock(
                            start_seconds=1.0,
                            end_seconds=2.0,
                            text="諸君らを歓迎する！",
                        ),
                    ]
                    mock_transcribe.return_value = mock_result

                    segments = transcriber.transcribe_segment("/dev/null", offset=0.0)

                    assert len(segments) == 2
                    for seg in segments:
                        assert seg.katakana != ""
                        for ch in seg.katakana:
                            assert "\u30a0" <= ch <= "\u30ff", (
                                f"Non-Katakana character '{ch}' in: {seg.katakana}"
                            )
                        assert " " not in seg.katakana
