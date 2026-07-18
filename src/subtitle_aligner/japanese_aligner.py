"""Japanese Forced Aligner wrapper using pydomino."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import torch
from .audio_loader import AudioLoader

try:
    import pydomino
except ImportError:
    pydomino = None

import pyopenjtalk

logger = logging.getLogger(__name__)


class JapaneseForcedAligner:
    """Handles temporal phonetic alignment of Japanese audio to text."""

    VALID_PHONEMES = {
        "pau",
        "ry",
        "r",
        "my",
        "m",
        "ny",
        "n",
        "j",
        "z",
        "by",
        "b",
        "dy",
        "d",
        "gy",
        "g",
        "ky",
        "k",
        "ch",
        "ts",
        "sh",
        "s",
        "hy",
        "h",
        "v",
        "f",
        "py",
        "p",
        "t",
        "y",
        "w",
        "N",
        "a",
        "i",
        "u",
        "e",
        "o",
        "I",
        "U",
        "cl",
    }

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        """Initialize the Japanese Forced Aligner.

        Args:
            model_path: Path to the pydomino ONNX model file.
            device: Computing device ('cpu' or 'cuda').
        """
        self.model_path = Path(model_path)
        self.device = device
        self.sample_rate = 16000

        if pydomino is None:
            raise ImportError(
                "The 'pydomino' library is required to run forced "
                "alignment, but it is not installed."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"pydomino ONNX model not found at: {self.model_path}"
            )

        logger.info(f"Loading pydomino model: {self.model_path}...")
        self.aligner = pydomino.Aligner(str(self.model_path))

    def normalize_and_tokenize(self, text: str) -> str:
        """Normalize Japanese text and convert it to standard JEIDA phonemes.

        Args:
            text: Raw transcript.

        Returns:
            Space-separated string of 39 JEIDA phoneme tokens wrapped by pau.
        """
        # Strip all whitespace
        clean_text = text.replace(" ", "").replace("　", "")
        if not clean_text:
            return "pau"

        # Grapheme-to-Phoneme tokenization
        raw_ph = pyopenjtalk.g2p(clean_text).split()

        # Retain only valid 39 JEIDA phonetic tokens
        phonemes = [p for p in raw_ph if p in self.VALID_PHONEMES]

        # Compress consecutive identical 'pau' tokens to clean the sequence
        compressed = []
        for p in phonemes:
            if not compressed or p != "pau" or compressed[-1] != "pau":
                compressed.append(p)

        # Trim leading/trailing 'pau' to avoid double-pau wrapping
        while compressed and compressed[0] == "pau":
            compressed.pop(0)
        while compressed and compressed[-1] == "pau":
            compressed.pop()

        # Wrap with a single leading and trailing pau token
        final_phonemes = ["pau"] + compressed + ["pau"]
        return " ".join(final_phonemes)

    def align(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        transcript: str,
        min_frames: int = 3,
    ) -> list[dict[str, str | float]]:
        """Align a Japanese transcript to a WAV audio waveform.

        Args:
            audio: WAV path, or pre-loaded (waveform, sample_rate) tuple.
            transcript: Japanese subtitle or ASR string.
            min_frames: Min Viterbi frame allocation (default 3, approx 30ms).

        Returns:
            List of aligned phonemes with start/end timestamps and score.
        """
        # Load audio from file path or unpack pre-loaded tuple
        if isinstance(audio, (str, Path)):
            loader = AudioLoader(audio)
            waveform, sr = loader.load_torchaudio(
                sampling_rate=16000,
                mono=True,
            )
        else:
            # assume already mono and 16K sr
            waveform, sr = audio

        y = waveform.squeeze(0).numpy().astype(np.float32)

        # Generate standard JEIDA G2P sequence wrapped in pau
        phoneme_sequence = self.normalize_and_tokenize(transcript)
        logger.info(
            "[pydomino] Aligning phonetic sequence: %s",
            phoneme_sequence,
        )

        # Run pydomino forced alignment
        raw_alignment = self.aligner.align(y, phoneme_sequence, min_frames)

        # Process raw output into standard segment structures
        alignment_data = []
        for start, end, ph in raw_alignment:
            alignment_data.append(
                {
                    "char": ph,
                    "start": start,
                    "end": end,
                    "score": 1.0,
                }
            )

        return alignment_data
