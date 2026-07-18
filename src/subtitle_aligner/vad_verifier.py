"""VAD-Based Post-Verification & Speech Hypothesis Snapping."""

from __future__ import annotations

import logging
from silero_vad import load_silero_vad, get_speech_timestamps
from .audio_loader import AudioLoader
from .audio_segmenter import AudioSegment

logger = logging.getLogger(__name__)


class VADVerifier:
    """Filters ASR hallucinations and refines segment boundaries using VAD.

    Enforces minimum voice activity ratio constraints and snaps the rough ASR
    container start/end times to the outer boundaries of overlapping speech
    intervals.
    """

    def __init__(
        self,
        min_vad_ratio: float = 0.25,
        padding_seconds: float = 0.150,
    ) -> None:
        """Initialize the VAD verifier with logic thresholds.

        Args:
            min_vad_ratio: Threshold ratio below which ASR is discarded.
            padding_seconds: Safe padding buffer in seconds (exactly 150ms).
        """
        self.min_vad_ratio = min_vad_ratio
        self.padding_seconds = padding_seconds

    def calculate_vad_ratio(
        self,
        asr_start: float,
        asr_end: float,
        vad_intervals: list[dict[str, float]],
    ) -> float:
        """Calculate the ratio of active speech duration within the segment.

        Args:
            asr_start: ASR segment start timestamp (seconds).
            asr_end: ASR segment end timestamp (seconds).
            vad_intervals: List of speech intervals from VAD.

        Returns:
            The ratio of active speech duration to total segment duration.
        """
        segment_duration = asr_end - asr_start
        if segment_duration <= 0.0:
            return 0.0

        speech_duration = 0.0
        for interval in vad_intervals:
            v_start = interval["start"]
            v_end = interval["end"]

            # Calculate overlap of VAD interval with the ASR segment
            overlap_start = max(asr_start, v_start)
            overlap_end = min(asr_end, v_end)
            overlap = max(0.0, overlap_end - overlap_start)
            speech_duration += overlap

        return speech_duration / segment_duration

    def is_hallucination(
        self,
        segment_start: float,
        segment_end: float,
        vad_intervals: list[dict[str, float]],
        threshold: float = 0.25,
    ) -> bool:
        """Verify if the ASR segment overlaps with VAD speech.

        Args:
            segment_start: ASR segment start timestamp.
            segment_end: ASR segment end timestamp.
            vad_intervals: List of active VAD intervals.
            threshold: Minimum active speech ratio (default 25%).
        """
        if not vad_intervals:
            return True
        overlap = 0.0
        for val in vad_intervals:
            s = max(segment_start, val["start"])
            e = min(segment_end, val["end"])
            if s < e:
                overlap += e - s
        total_duration = segment_end - segment_start
        if total_duration <= 0:
            return True
        return (overlap / total_duration) < threshold

    def snap_and_pad_segment(
        self,
        segment_start: float,
        segment_end: float,
        vad_intervals: list[dict[str, float]],
    ) -> tuple[float, float]:
        """Snap ASR boundaries to overlapping VAD speech and apply padding.

        Args:
            segment_start: Raw ASR segment start time.
            segment_end: Raw ASR segment end time.
            vad_intervals: Active speech intervals.
        """
        # start:1.222, end:2.444(same); start:3.233, end:4.111 (latter); start:0.233, end:1.111 (prev)
        # asr_s:1.002, asr_e:3.111
        # algo:
        # 1. max(s)=1.222, min(2.444) -> accepted
        # 2. 3.233, 2.444 -> Rejected
        # 3. 1.002, 1.111 -> accepted
        # for the snap: s->1.002 & e->2.444
        overlapping = []
        for val in vad_intervals:
            s = max(segment_start, val["start"])
            e = min(segment_end, val["end"])
            if s < e:
                overlapping.append(val)

        if not overlapping:
            return segment_start, segment_end

        v_min_start = min(val["start"] for val in overlapping)
        v_max_end = max(val["end"] for val in overlapping)

        snapped_start = max(segment_start, v_min_start)
        snapped_end = min(segment_end, v_max_end)

        padded_start = max(0.0, snapped_start - self.padding_seconds)
        padded_end = snapped_end + self.padding_seconds

        return padded_start, padded_end

    def _get_absolute_vad_intervals(
        self,
        audio_segments: list[AudioSegment],
    ) -> list[dict[str, float]]:
        """Compute absolute VAD speech intervals across all segments."""

        vad_model = load_silero_vad(onnx=True)
        absolute_intervals = []

        for seg in audio_segments:
            loader = AudioLoader(seg.filepath)
            waveform, sr = loader.load_torchaudio(
                sampling_rate=16000,
                mono=True,
            )
            waveform_1d = waveform.squeeze(0)

            relative_ts = get_speech_timestamps(
                waveform_1d,
                vad_model,
                sampling_rate=sr,
                return_seconds=True,
            )

            for ts in relative_ts:
                absolute_intervals.append(
                    {
                        "start": ts["start"] + seg.start_time,
                        "end": ts["end"] + seg.start_time,
                    }
                )

        return absolute_intervals
