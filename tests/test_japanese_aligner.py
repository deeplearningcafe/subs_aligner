"""Tests for the Japanese forced aligner using pydomino."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchaudio

from src.subtitle_aligner.japanese_aligner import JapaneseForcedAligner


@pytest.fixture
def synthetic_wav():
    """Create a temporary 16kHz mono WAV file."""
    sr = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) * 0.5

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, waveform, sr)
        wav_path = Path(f.name)

    yield wav_path

    if wav_path.exists():
        wav_path.unlink()


def test_normalize_and_tokenize_standard_text():
    """Verify standard text normalization and JEIDA conversion."""
    with patch("src.subtitle_aligner.japanese_aligner.pydomino") as mock_pyd:
        mock_pyd.Aligner = MagicMock()
        aligner = JapaneseForcedAligner(
            model_path=Path(__file__),  # Dummy path
        )

        result = aligner.normalize_and_tokenize("こんにちは")
        assert result.startswith("pau ")
        assert result.endswith(" pau")

        tokens = result.split()
        for t in tokens:
            assert t in aligner.VALID_PHONEMES


def test_normalize_and_tokenize_consecutive_pauses():
    """Ensure consecutive pauses and duplicate pau are cleaned."""
    with patch("src.subtitle_aligner.japanese_aligner.pydomino") as mock_pyd:
        mock_pyd.Aligner = MagicMock()
        aligner = JapaneseForcedAligner(model_path=Path(__file__))

        with patch("pyopenjtalk.g2p", return_value="pau pau k a pau pau"):
            result = aligner.normalize_and_tokenize("か")
            assert result == "pau k a pau"


def test_align_waveform_mono_conversion(synthetic_wav):
    """Test loading and mono conversion of audio files."""
    with patch("src.subtitle_aligner.japanese_aligner.pydomino") as mock_pyd:
        mock_aligner_inst = MagicMock()
        mock_pyd.Aligner.return_value = mock_aligner_inst
        mock_pyd.Aligner.__name__ = "Aligner"

        mock_aligner_inst.align.return_value = [
            (0.0, 0.2, "pau"),
            (0.2, 0.8, "a"),
            (0.8, 1.0, "pau"),
        ]

        aligner = JapaneseForcedAligner(model_path=Path(__file__))

        results = aligner.align(
            audio=synthetic_wav,
            transcript="あ",
            min_frames=3,
        )

        assert len(results) == 3
        assert results[0] == {
            "char": "pau",
            "start": 0.0,
            "end": 0.2,
            "score": 1.0,
        }

        mock_aligner_inst.align.assert_called_once()
        args = mock_aligner_inst.align.call_args[0]
        assert args[2] == 3  # min_frames parameter
