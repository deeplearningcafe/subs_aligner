"""VAD-Based Post-Verification & Speech Hypothesis Snapping."""

from __future__ import annotations

import logging

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

    def verify_segment(
        self,
        asr_start: float,
        asr_end: float,
        vad_intervals: list[dict[str, float]],
    ) -> bool:
        """Determine if the ASR segment contains sufficient speech.

        Args:
            asr_start: ASR segment start timestamp (seconds).
            asr_end: ASR segment end timestamp (seconds).
            vad_intervals: List of active speech intervals from VAD.

        Returns:
            True if VAD ratio meets or exceeds the threshold.
        """
        ratio = self.calculate_vad_ratio(asr_start, asr_end, vad_intervals)
        is_valid = ratio >= self.min_vad_ratio
        if not is_valid:
            logger.warning(
                "[VAD] Segment [%.2fs - %.2fs] discarded (ratio: %.2f < %.2f)",
                asr_start,
                asr_end,
                ratio,
                self.min_vad_ratio,
            )
        return is_valid

    def snap_and_pad_segment(
        self,
        asr_start: float,
        asr_end: float,
        vad_intervals: list[dict[str, float]],
    ) -> tuple[float, float]:
        """Snap segment boundaries to overlapping VAD intervals and pad.

        Args:
            asr_start: ASR segment start timestamp (seconds).
            asr_end: ASR segment end timestamp (seconds).
            vad_intervals: List of active speech intervals from VAD.

        Returns:
            A tuple of (padded_start, padded_end) representing optimized
            container bounds.
        """
        # Collect VAD intervals that have active overlap with the segment
        overlapping = []
        for interval in vad_intervals:
            overlap_start = max(asr_start, interval["start"])
            overlap_end = min(asr_end, interval["end"])
            if overlap_start < overlap_end:
                overlapping.append(interval)

        if not overlapping:
            # Fallback if no overlap is detected; retain original timestamps
            return asr_start, asr_end

        # Snap exactly to the outer boundaries of the overlapping intervals
        v_min_start = min(inter["start"] for inter in overlapping)
        v_max_end = max(inter["end"] for inter in overlapping)

        # Crop the start and end boundaries (prevent expanding past raw ASR)
        snapped_start = max(asr_start, v_min_start)
        snapped_end = min(asr_end, v_max_end)

        # Apply safe padding buffer to protect starting and ending consonants
        padded_start = max(0.0, snapped_start - self.padding_seconds)
        padded_end = snapped_end + self.padding_seconds

        return padded_start, padded_end
