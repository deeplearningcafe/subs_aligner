"""ASR transcriber — ReazonSpeech transcription with timestamp mapping and CTC character timings.

Wraps the ReazonSpeech (espnet-asr) pipeline to transcribe audio segments
and convert relative timestamps into absolute video timeline positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from reazonspeech.espnet.asr import transcribe, audio_from_path
from reazonspeech.espnet.asr.ctc import get_timings

from .text_processing import TextProcessor
from .subtitle_writer import SubtitleWriter


@dataclass
class TranscriptionSegment:
    """A single transcription result with absolute timing."""

    start_time: float  # absolute time in seconds on the video timeline
    end_time: float
    text: str  # original transcribed text
    katakana: str  # phonetic Katakana representation
    char_timings: list[float]  # Absolute timestamp for each char in text


class ASRTranscriber:
    """Transcribes audio segments using ReazonSpeech (espnet-asr) with CTC character-level timings..

    1. Loads the ReazonSpeech model (lazy, single instance).
    2. Transcribes each audio segment, producing timestamped text.
    3. Converts relative ASR timestamps to absolute video timestamps
       by adding the segment's start-time offset.
    4. Normalizes all text to pure Katakana via ``TextProcessor``.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the ASR transcriber.

        Args:
            device: Device to run inference on (``"cpu"`` or ``"cuda"``).
        """
        self.device = device
        self._model = None
        self._text_processor = TextProcessor()

    def _ensure_model(self):
        """Lazily load the ReazonSpeech model."""
        if self._model is None:
            from reazonspeech.espnet.asr import load_model

            print(f"[ASRTranscriber] Loading ReazonSpeech model on {self.device}...")
            self._model = load_model(device=self.device)
            print("[ASRTranscriber] Model loaded.")

    def transcribe_segment(
        self,
        audio_path: str,
        offset: float = 0.0,
    ) -> list[TranscriptionSegment]:
        """Transcribe a single audio segment and map timestamps to video time with CTC timings.

        Args:
            audio_path: Path to the audio file (must be .wav).
            offset: Absolute start time of this segment on the video timeline
                    (seconds).

        Returns:
            List of ``TranscriptionSegment`` with absolute timestamps and
            Katakana-normalized text.
        """
        self._ensure_model()

        audio = audio_from_path(audio_path)
        result = transcribe(self._model, audio)

        samples = audio.waveform

        segments: list[TranscriptionSegment] = []
        for seg in result.segments:
            absolute_start = seg.start_seconds + offset
            absolute_end = seg.end_seconds + offset
            katakana = self._text_processor.text_to_katakana(seg.text)

            # Compute CTC character timings relative to the segment
            char_timings: list[float] = []
            try:
                # get_timings returns timings in audio samples
                raw_timings = get_timings(self._model, samples, seg.text)
                # to absolute seconds
                char_timings = [
                    (sample_idx / 16000.0) + offset for sample_idx in raw_timings
                ]
            except Exception as e:
                # distribute timings uniformly
                print(f"[ASR Warning] CTC alignment failed: {e}")
                num_chars = len(seg.text)
                if num_chars > 0:
                    step = (absolute_end - absolute_start) / num_chars
                    char_timings = [
                        absolute_start + (i * step) for i in range(num_chars)
                    ]

            segments.append(
                TranscriptionSegment(
                    start_time=absolute_start,
                    end_time=absolute_end,
                    text=seg.text,
                    katakana=katakana,
                    char_timings=char_timings,
                )
            )

        return segments

    def transcribe_segments(
        self,
        segments: list,
    ) -> list[TranscriptionSegment]:
        """Transcribe a list of audio segments (with ``start_time`` attribute).

        Args:
            segments: List of objects with ``filepath`` and ``start_time``
                      attributes (e.g. ``AudioSegment`` instances).

        Returns:
            Flat list of all ``TranscriptionSegment`` instances across
            every audio segment.
        """
        all_segments: list[TranscriptionSegment] = []

        for seg in segments:
            audio_path = seg.filepath
            offset = seg.start_time

            print(
                f"[ASRTranscriber] Transcribing: {Path(audio_path).name} "
                f"(offset={offset:.2f}s)"
            )
            transcribed = self.transcribe_segment(audio_path, offset=offset)
            print(f"[ASRTranscriber]   -> {len(transcribed)} transcription segment(s)")
            all_segments.extend(transcribed)

        return all_segments

    def write_aligned(
        self,
        blocks: list[TranscriptionSegment],
        output_path: str | Path,
        fmt: str = "srt",
    ) -> None:
        """Write aligned subtitle blocks to a file.

        Args:
            blocks: Aligned SubtitleBlock list.
            output_path: Destination file path.
            fmt: Output format — ``"srt"`` or ``"vtt"``.
        """
        # create raw_text for writter
        for block in blocks:
            setattr(block, "raw_text", block.text)
        SubtitleWriter.write_blocks(blocks, output_path, fmt=fmt)
