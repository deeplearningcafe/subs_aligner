"""Subtitle aligner — local sliding-window timeline alignment engine.

Performs local window searches to pair original subtitles with their
corresponding ASR speech segments, adjusting timestamps based on a
3-tiered threshold logic.

Workflow:
    1. For each subtitle at start time T, define a ±5-minute window.
    2. Extract candidate ASR segments whose start times fall inside the window.
    3. Compare Katakana representations using difflib.SequenceMatcher.
    4. Apply 3-tier timing adjustments:
       - Difference < 0.2s → keep original timing
       - Difference 0.2–5.0s → overwrite with ASR timing
       - Difference > 5.0s → overwrite with ASR timing + log shift event
    5. Fallback offset propagation: unmatched subtitles inherit the shift
       offset from the last successfully matched "shift" line if they
       overlap with that matched ASR region.
    6. ASR insertion: unmatched ASR segments in subtitle-free gaps are
       inserted as new subtitle lines with original ASR text.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .subtitle_parser import SubtitleBlock
from .subtitle_writer import SubtitleWriter
from .text_processing import TextProcessor
from .asr_transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class AlignmentMatch:
    """Result of matching a subtitle to an ASR segment.

    Attributes:
        subtitle_index: Position of the subtitle in the original list
            (``-1`` for inserted ASR scenes).
        asr_segment: The matched ASR segment, or None if no match.
        similarity: Katakana similarity ratio (0.0–1.0).
        timing_difference: Absolute time difference between subtitle and ASR
            start times in seconds.
        action: One of ``"keep"``, ``"adjust"``, ``"shift"``, or
            ``"inserted"``.
        original_start: Subtitle's original start time (seconds).
        original_end: Subtitle's original end time (seconds).
        new_start: Aligned start time (seconds).
        new_end: Aligned end time (seconds).
    """

    subtitle_index: int
    asr_segment: Optional[TranscriptionSegment]
    similarity: float
    timing_difference: float
    action: str
    original_start: float
    original_end: float
    new_start: float
    new_end: float


class SubtitleAligner:
    """Aligns original subtitles with ASR transcription using sliding windows.

    For each subtitle, searches a ±5-minute window for candidate ASR segments,
    matches them by Katakana phonetic similarity, and applies 3-tier timing
    adjustments.  Unmatched subtitles inherit shift offsets from neighbours.
    Unmatched ASR segments in blank gaps are inserted as new subtitle lines.

    Args:
        device: Device for any computation (``"cpu"`` or ``"cuda"``).
    """

    WINDOW_SECONDS: float = 300.0  # ±5 minutes
    SIMILARITY_THRESHOLD: float = 0.70  # 70%
    SMALL_SHIFT_THRESHOLD: float = 0.2  # <0.2s: keep original
    LARGE_SHIFT_THRESHOLD: float = 5.0  # >5.0s: shift + log
    INSERTION_GAP_THRESHOLD: float = 2.0  # seconds: min gap edge tolerance
    MIN_ASR_DURATION: float = 1.0  # seconds: minimum ASR segment to insert

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the aligner.

        Args:
            device: Device for computation (cpu/cuda).
        """
        self.device = device
        self._text_processor = TextProcessor()

    # ── helpers ─────────────────────────────────────────────────────────

    def _get_katakana(self, text: str) -> str:
        """Convert text to Katakana for phonetic comparison.

        Args:
            text: Input text (preferably already cleaned by TextProcessor).

        Returns:
            Pure Katakana string, or empty string if input is empty.
        """
        if not text:
            return ""
        return self._text_processor.text_to_katakana(text)

    def _find_candidates(
        self,
        subtitle: SubtitleBlock,
        asr_segments: list[TranscriptionSegment],
        window: float = WINDOW_SECONDS,
    ) -> list[TranscriptionSegment]:
        """Find ASR segments within the sliding window of a subtitle.

        Args:
            subtitle: The subtitle to search around.
            asr_segments: All available ASR transcription segments.
            window: Window half-width in seconds (default ±5 min).

        Returns:
            ASR segments whose start_time falls within
            ``[subtitle.start_time - window, subtitle.start_time + window]``.
        """
        t = subtitle.start_time
        candidates: list[TranscriptionSegment] = []
        for seg in asr_segments:
            if t - window <= seg.start_time <= t + window:
                candidates.append(seg)
        return candidates

    @staticmethod
    def _compute_similarity(sub_kana: str, asr_kana: str) -> float:
        """Compute Katakana similarity ratio between subtitle and ASR text.

        Args:
            sub_kana: Katakana representation of the subtitle text.
            asr_kana: Katakana representation of the ASR transcription.

        Returns:
            Similarity ratio from 0.0 to 1.0 via
            ``difflib.SequenceMatcher.ratio()``.
        """
        return difflib.SequenceMatcher(None, sub_kana, asr_kana).ratio()

    def _apply_timing_adjustment(
        self,
        original_start: float,
        original_end: float,
        asr_start: float,
        asr_end: float,
    ) -> tuple[float, float, str]:
        """Apply the 3-tier timing adjustment logic.

        Args:
            original_start: Subtitle's original start time.
            original_end: Subtitle's original end time.
            asr_start: Matched ASR segment's start time.
            asr_end: Matched ASR segment's end time.

        Returns:
            Tuple of ``(new_start, new_end, action)`` where ``action`` is
            one of ``"keep"``, ``"adjust"``, or ``"shift"``.
        """
        diff = abs(original_start - asr_start)

        if diff < self.SMALL_SHIFT_THRESHOLD:
            return original_start, original_end, "keep"
        elif diff <= self.LARGE_SHIFT_THRESHOLD:
            return asr_start, asr_end, "adjust"
        else:
            return asr_start, asr_end, "shift"

    def _subtitle_overlaps_matched_asr(self, subtitle: SubtitleBlock) -> bool:
        """Check if a subtitle overlaps with the last matched ASR segment.

        Args:
            subtitle: The subtitle to check.

        Returns:
            True if the subtitle's time range overlaps with the last
            matched ASR segment's time range.
        """
        if self._last_shift_asr is None:
            return False
        asr = self._last_shift_asr
        return subtitle.start_time < asr.end_time and subtitle.end_time > asr.start_time

    def _insert_asr_scenes(
        self,
        aligned_blocks: list[SubtitleBlock],
        asr_segments: list[TranscriptionSegment],
        matched_asr_indices: set[int],
    ) -> tuple[list[SubtitleBlock], list[AlignmentMatch]]:
        """Insert unmatched ASR segments that fall in subtitle-free gaps.

        Identifies ASR segments that were not matched to any subtitle and
        inserts them as new subtitle lines when they occur in temporal gaps
        where no subtitles exist.

        Args:
            aligned_blocks: Already-aligned subtitle blocks.
            asr_segments: All ASR transcription segments.
            matched_asr_indices: Set of indices into ``asr_segments`` that
                were already matched.

        Returns:
            Tuple of ``(inserted_blocks, inserted_matches)``.
        """
        inserted_blocks: list[SubtitleBlock] = []
        inserted_matches: list[AlignmentMatch] = []

        # Collect unmatched ASR segments
        unmatched_asr: list[TranscriptionSegment] = [
            seg
            for idx, seg in enumerate(asr_segments)
            if idx not in matched_asr_indices
        ]

        if not unmatched_asr:
            return inserted_blocks, inserted_matches

        unmatched_asr.sort(key=lambda s: s.start_time)

        # Build sorted subtitle time ranges
        sub_ranges = [(b.start_time, b.end_time) for b in aligned_blocks]
        sub_ranges.sort(key=lambda r: r[0])

        for asr_seg in unmatched_asr:
            if (asr_seg.end_time - asr_seg.start_time) < self.MIN_ASR_DURATION:
                continue

            if self._is_in_subtitle_gap(asr_seg, sub_ranges):
                new_block = SubtitleBlock(
                    start_time=asr_seg.start_time,
                    end_time=asr_seg.end_time,
                    raw_text=asr_seg.text,
                    cleaned_text=asr_seg.text,
                )
                inserted_blocks.append(new_block)
                inserted_matches.append(
                    AlignmentMatch(
                        subtitle_index=-1,
                        asr_segment=asr_seg,
                        similarity=1.0,
                        timing_difference=0.0,
                        action="inserted",
                        original_start=asr_seg.start_time,
                        original_end=asr_seg.end_time,
                        new_start=asr_seg.start_time,
                        new_end=asr_seg.end_time,
                    )
                )
                logger.info(
                    f"[Aligner] Inserted ASR scene at "
                    f"{asr_seg.start_time:.1f}s–{asr_seg.end_time:.1f}s: "
                    f"{asr_seg.text[:60]}"
                )

        return inserted_blocks, inserted_matches

    def _is_in_subtitle_gap(
        self,
        asr_seg: TranscriptionSegment,
        sub_ranges: list[tuple[float, float]],
    ) -> bool:
        """Check if an ASR segment falls entirely in a subtitle-free gap.

        A gap is a region where no subtitle overlaps the ASR segment,
        with a tolerance of ``INSERTION_GAP_THRESHOLD`` on each side.

        Args:
            asr_seg: The ASR segment to check.
            sub_ranges: Sorted list of (start, end) subtitle time ranges.

        Returns:
            True if the ASR segment does not overlap any subtitle
            (within the tolerance margin).
        """
        margin = self.INSERTION_GAP_THRESHOLD

        for sub_start, sub_end in sub_ranges:
            if not (
                asr_seg.end_time < sub_start - margin
                or asr_seg.start_time > sub_end + margin
            ):
                return False
        return True

    # ── public API ──────────────────────────────────────────────────────

    def align(
        self,
        subtitles: list[SubtitleBlock],
        asr_segments: list[TranscriptionSegment],
        window: float = WINDOW_SECONDS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> tuple[list[SubtitleBlock], list[AlignmentMatch]]:
        """Align subtitles with ASR transcription.

        For each subtitle, searches a local window for candidate ASR segments,
        matches by Katakana similarity, and applies 3-tier timing adjustments.
        Unmatched subtitles inherit shift offsets from neighbours.  Unmatched
        ASR segments in blank gaps are inserted as new subtitle lines.

        Args:
            subtitles: List of parsed subtitle blocks.
            asr_segments: List of ASR transcription segments.
            window: Search window half-width in seconds (default ±5 min).
            similarity_threshold: Minimum similarity ratio to accept a match.

        Returns:
            Tuple of ``(aligned_blocks, matches)`` where:
            - ``aligned_blocks``: Updated SubtitleBlock list with adjusted
              timestamps plus any inserted ASR segments.
            - ``matches``: List of AlignmentMatch objects for logging.
        """
        aligned_blocks: list[SubtitleBlock] = []
        matches: list[AlignmentMatch] = []

        # Track shift offset propagation and matched ASR indices
        self._last_shift_offset: Optional[float] = None
        self._last_shift_asr: Optional[TranscriptionSegment] = None
        matched_asr_indices: set[int] = set()

        for i, sub in enumerate(subtitles):
            new_block = SubtitleBlock(
                start_time=sub.start_time,
                end_time=sub.end_time,
                raw_text=sub.raw_text,
                cleaned_text=sub.cleaned_text,
            )

            sub_kana = self._get_katakana(sub.cleaned_text)
            if not sub_kana:
                aligned_blocks.append(new_block)
                matches.append(
                    AlignmentMatch(
                        subtitle_index=i,
                        asr_segment=None,
                        similarity=0.0,
                        timing_difference=0.0,
                        action="keep",
                        original_start=sub.start_time,
                        original_end=sub.end_time,
                        new_start=sub.start_time,
                        new_end=sub.end_time,
                    )
                )
                continue

            candidates = self._find_candidates(sub, asr_segments, window)

            if not candidates:
                # No candidates in window — keep original timing
                aligned_blocks.append(new_block)
                matches.append(
                    AlignmentMatch(
                        subtitle_index=i,
                        asr_segment=None,
                        similarity=0.0,
                        timing_difference=0.0,
                        action="keep",
                        original_start=sub.start_time,
                        original_end=sub.end_time,
                        new_start=sub.start_time,
                        new_end=sub.end_time,
                    )
                )
                continue

            # Evaluate similarity for each candidate and pick the best
            best_match: Optional[TranscriptionSegment] = None
            best_score: float = -1.0
            for candidate in candidates:
                asr_kana = candidate.katakana
                if not asr_kana:
                    continue
                score = self._compute_similarity(sub_kana, asr_kana)
                if score > best_score:
                    best_score = score
                    best_match = candidate

            if best_match is None or best_score < similarity_threshold:
                # No good match — apply fallback offset if available
                # and subtitle overlaps with a previously matched ASR region
                fallback_offset = self._last_shift_offset
                if (
                    fallback_offset is not None
                    and self._last_shift_asr is not None
                    and self._subtitle_overlaps_matched_asr(sub)
                ):
                    new_start = sub.start_time + fallback_offset
                    new_end = sub.end_time + fallback_offset
                    logger.info(
                        f"[Aligner] Subtitle {i + 1} unmatched (sim={best_score:.1%}) "
                        f"shifted by fallback offset {fallback_offset:.2f}s"
                    )
                    new_block.start_time = new_start
                    new_block.end_time = new_end
                else:
                    new_start, new_end = sub.start_time, sub.end_time

                aligned_blocks.append(new_block)
                matches.append(
                    AlignmentMatch(
                        subtitle_index=i,
                        asr_segment=None,
                        similarity=best_score if best_score >= 0 else 0.0,
                        timing_difference=0.0,
                        action="keep",
                        original_start=sub.start_time,
                        original_end=sub.end_time,
                        new_start=new_start,
                        new_end=new_end,
                    )
                )
                continue

            # Apply 3-tier timing adjustment
            new_start, new_end, action = self._apply_timing_adjustment(
                sub.start_time,
                sub.end_time,
                best_match.start_time,
                best_match.end_time,
            )

            diff = abs(sub.start_time - best_match.start_time)

            # Track shift offset for fallback propagation (only "shift" action)
            if action == "shift":
                self._last_shift_offset = best_match.start_time - sub.start_time
                self._last_shift_asr = best_match
                logger.warning(
                    f"[Aligner] Subtitle {i + 1} shifted by {diff:.2f}s "
                    f"(similarity={best_score:.1%})"
                )

            # Record that this ASR segment was matched
            for idx, seg in enumerate(asr_segments):
                if seg is best_match:
                    matched_asr_indices.add(idx)
                    break

            new_block.start_time = new_start
            new_block.end_time = new_end

            aligned_blocks.append(new_block)
            matches.append(
                AlignmentMatch(
                    subtitle_index=i,
                    asr_segment=best_match,
                    similarity=best_score,
                    timing_difference=diff,
                    action=action,
                    original_start=sub.start_time,
                    original_end=sub.end_time,
                    new_start=new_start,
                    new_end=new_end,
                )
            )

        # ── ASR insertion for unrecorded scenes ─────────────────────────
        inserted_blocks, inserted_matches = self._insert_asr_scenes(
            aligned_blocks,
            asr_segments,
            matched_asr_indices,
        )
        aligned_blocks.extend(inserted_blocks)
        matches.extend(inserted_matches)

        # Sort final output by start time
        combined = list(zip(aligned_blocks, matches))
        combined.sort(key=lambda x: x[0].start_time)
        aligned_blocks = [b for b, _ in combined]
        matches = [m for _, m in combined]

        return aligned_blocks, matches

    def write_aligned(
        self,
        blocks: list[SubtitleBlock],
        output_path: str | Path,
        fmt: str = "srt",
    ) -> None:
        """Write aligned subtitle blocks to a file.

        Args:
            blocks: Aligned SubtitleBlock list.
            output_path: Destination file path.
            fmt: Output format — ``"srt"`` or ``"vtt"``.
        """
        SubtitleWriter.write_blocks(blocks, output_path, fmt=fmt)
