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
from .text_processing import TextProcessor
from .asr_transcriber import TranscriptionSegment
from .japanese_aligner import JapaneseForcedAligner
from .vad_verifier import VADVerifier
from .audio_segmenter import AudioSegment
from .audio_loader import AudioLoader

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

    Supports global phrase similarity matching and localized character-level
    CTC alignment.

    Args:
        device: Device for computation (cpu/cuda).
        mode: The active alignment strategy ("local_ctc" or "original_global").
    """

    WINDOW_SECONDS: float = 300.0  # ±5 minutes
    SIMILARITY_THRESHOLD: float = 0.80  # 70%
    SMALL_SHIFT_THRESHOLD: float = 0.2  # <0.2s: keep original
    LARGE_SHIFT_THRESHOLD: float = 5.0  # >5.0s: shift + log
    INSERTION_GAP_THRESHOLD: float = 2.0  # seconds: min gap edge tolerance
    MIN_ASR_DURATION: float = 1.0  # seconds: minimum ASR segment to insert

    def __init__(
        self,
        device: str = "cpu",
        mode: str = "global",
        model_path: str | None = None,
        padding_seconds: float = 0.100,
    ) -> None:
        """Initialize the aligner.

        Args:
            device: Device for computation (cpu/cuda).
            mode: Active alignment strategy ("local_ctc" or "global").
            model_path: Optional path to the pydomino ONNX model file.
        """
        self.device = device
        self.mode = mode
        self.padding_seconds = padding_seconds
        self._text_processor = TextProcessor()
        self._validator = VADVerifier()
        print(f"Using mode: {mode}")
        self.jp_aligner = None

        if self.mode == "local_ctc":
            import os

            m_path = model_path or os.getenv("ALIGNER_MODEL_PATH")
            if not m_path:
                m_path = "models/pydomino.onnx"
            try:
                self.jp_aligner = JapaneseForcedAligner(m_path, device=self.device)
            except Exception as e:
                logger.warning("Could not load pydomino forced aligner: %s", e)

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
        diff_start = abs(original_start - asr_start)
        diff_end = abs(original_end - asr_end)

        start = original_start
        end = original_end
        action = "keep"
        if diff_start > self.SMALL_SHIFT_THRESHOLD:
            start = asr_start
            action = "adjust"
        elif diff_start >= self.LARGE_SHIFT_THRESHOLD:
            start = asr_start
            action = "shift"

        if diff_end > self.SMALL_SHIFT_THRESHOLD:
            end = asr_end
            action = "adjust"
        elif diff_end >= self.LARGE_SHIFT_THRESHOLD:
            end = asr_end
            action = "shift"

        return start, end, action

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

    def _create_keep_match(
        self,
        index: int,
        subtitle: SubtitleBlock,
        new_start: float,
        new_end: float,
        similarity: float = 0.0,
    ) -> AlignmentMatch:
        """Helper to construct a standardized AlignmentMatch with a keep action."""
        return AlignmentMatch(
            subtitle_index=index,
            asr_segment=None,
            similarity=similarity,
            timing_difference=0.0,
            action="keep",
            original_start=subtitle.start_time,
            original_end=subtitle.end_time,
            new_start=new_start,
            new_end=new_end,
        )

    def _apply_fallback_offset(self, subtitle: SubtitleBlock) -> tuple[float, float]:
        """Apply fallback shift offset if the subtitle overlaps matched ASR."""
        if (
            self._last_shift_offset is not None
            and self._last_shift_asr is not None
            and self._subtitle_overlaps_matched_asr(subtitle)
        ):
            return (
                subtitle.start_time + self._last_shift_offset,
                subtitle.end_time + self._last_shift_offset,
            )
        return subtitle.start_time, subtitle.end_time

    def _update_shift_state(
        self,
        action: str,
        subtitle: SubtitleBlock,
        best_match: TranscriptionSegment,
        new_start: float,
    ) -> None:
        """Track shift state variations to resolve propagation later."""
        if action == "shift":
            self._last_shift_offset = new_start - subtitle.start_time
            self._last_shift_asr = best_match
            logger.warning(
                f"[Aligner] Subtitle shifted by "
                f"{abs(new_start - subtitle.start_time):.2f}s"
            )

    def _mark_asr_consumed(
        self,
        best_match: TranscriptionSegment,
        asr_segments: list[TranscriptionSegment],
    ) -> None:
        """Mark a matched ASR segment index as consumed to prevent duplicate matches."""
        for idx, seg in enumerate(asr_segments):
            if seg is best_match:
                self._matched_asr_indices.add(idx)
                break

    def _align_global(
        self,
        subtitles: list[SubtitleBlock],
        asr_segments: list[TranscriptionSegment],
        window: float,
        similarity_threshold: float,
    ) -> tuple[list[SubtitleBlock], list[AlignmentMatch]]:
        """Perform full-phrase global phonetic alignment."""
        aligned_blocks: list[SubtitleBlock] = []
        matches: list[AlignmentMatch] = []

        for i, sub in enumerate(subtitles):
            new_block = SubtitleBlock(
                start_time=sub.start_time,
                end_time=sub.end_time,
                raw_text=sub.raw_text,
                cleaned_text=sub.cleaned_text,
                bom=sub.bom,
                line_ending=sub.line_ending,
                trailing_blank=sub.trailing_blank,
                ass_header=sub.ass_header,
                ass_metadata=sub.ass_metadata,
            )

            sub_kana = self._get_katakana(sub.cleaned_text)
            if not sub_kana:
                aligned_blocks.append(new_block)
                matches.append(
                    self._create_keep_match(
                        i, sub, new_block.start_time, new_block.end_time
                    )
                )
                continue

            candidates = self._find_candidates(sub, asr_segments, window)
            if not candidates:
                aligned_blocks.append(new_block)
                matches.append(
                    self._create_keep_match(
                        i, sub, new_block.start_time, new_block.end_time
                    )
                )
                continue

            # Evaluate global similarity on candidate list
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
                new_start, new_end = self._apply_fallback_offset(sub)
                new_block.start_time = new_start
                new_block.end_time = new_end
                aligned_blocks.append(new_block)
                matches.append(
                    self._create_keep_match(
                        i, sub, new_start, new_end, max(best_score, 0.0)
                    )
                )
                continue

            # Apply 3-tier global timing adjustments
            new_start, new_end, action = self._apply_timing_adjustment(
                sub.start_time,
                sub.end_time,
                best_match.start_time,
                best_match.end_time,
            )

            # log the diff with the asr model
            # TODO: log the start and end difference
            diff = max(
                abs(sub.start_time - best_match.start_time),
                abs(sub.end_time - best_match.end_time),
            )
            self._update_shift_state(action, sub, best_match, new_start)
            self._mark_asr_consumed(best_match, asr_segments)

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

        return aligned_blocks, matches

    def _find_local_ctc_bounds(
        self,
        subtitle: SubtitleBlock,
        candidate: TranscriptionSegment,
        similarity_threshold: float,
    ) -> tuple[float, float, float] | None:
        """Find phonetic substring bounds inside an ASR segment using CTC.

        Args:
            subtitle: The subtitle block to match.
            candidate: ASR segment containing character-level timings.
            similarity_threshold: Minimum match ratio.

        Returns:
            Tuple of (new_start, new_end, similarity) or None if unmatched.
        """
        sub_text = subtitle.cleaned_text
        asr_text = candidate.text

        if not sub_text or not asr_text or not candidate.char_timings:
            return None

        matcher = difflib.SequenceMatcher(None, sub_text, asr_text)
        matching_blocks = matcher.get_matching_blocks()

        valid_blocks = [b for b in matching_blocks if b.size > 0]
        if not valid_blocks:
            return None

        # Determine indices within ASR characters
        match_start_idx = min(b.b for b in valid_blocks)
        match_end_idx = max(b.b + b.size for b in valid_blocks)

        match_end_idx = min(match_end_idx, len(candidate.char_timings) - 1)
        if match_start_idx >= match_end_idx:
            return None

        matched_substring = asr_text[match_start_idx : match_end_idx + 1]
        similarity = difflib.SequenceMatcher(None, sub_text, matched_substring).ratio()

        if similarity < similarity_threshold:
            return None

        # Extract absolute timestamps via CTC
        new_start = candidate.char_timings[match_start_idx]
        new_end = candidate.char_timings[match_end_idx]

        return new_start, new_end, similarity

    def _align_local_ctc(
        self,
        subtitles: list[SubtitleBlock],
        asr_segments: list[TranscriptionSegment],
        window: float,
        similarity_threshold: float,
        audio_segments: list[AudioSegment] | None = None,
        vad_intervals: list[dict[str, float]] | None = None,
    ) -> tuple[list[SubtitleBlock], list[AlignmentMatch]]:
        """Perform localized substring phonetic matching using pydomino."""
        aligned_blocks: list[SubtitleBlock] = []
        matches: list[AlignmentMatch] = []
        vad_list = vad_intervals or []

        for i, sub in enumerate(subtitles):
            new_block = SubtitleBlock(
                start_time=sub.start_time,
                end_time=sub.end_time,
                raw_text=sub.raw_text,
                cleaned_text=sub.cleaned_text,
                bom=sub.bom,
                line_ending=sub.line_ending,
                trailing_blank=sub.trailing_blank,
                ass_header=sub.ass_header,
                ass_metadata=sub.ass_metadata,
            )

            sub_kana = self._get_katakana(sub.cleaned_text)
            if not sub_kana:
                aligned_blocks.append(new_block)
                matches.append(
                    self._create_keep_match(
                        i, sub, new_block.start_time, new_block.end_time
                    )
                )
                continue

            candidates = self._find_candidates(sub, asr_segments, window)
            if not candidates:
                aligned_blocks.append(new_block)
                matches.append(
                    self._create_keep_match(
                        i, sub, new_block.start_time, new_block.end_time
                    )
                )
                continue

            best_match: Optional[TranscriptionSegment] = None
            best_score: float = -1.0
            best_times: Optional[tuple[float, float]] = None

            for candidate in candidates:
                # Filter out hallucinations
                if self._validator.is_hallucination(
                    candidate.start_time,
                    candidate.end_time,
                    vad_list,
                ):
                    continue

                score = difflib.SequenceMatcher(
                    None, sub_kana, candidate.katakana
                ).ratio()

                if score >= similarity_threshold and score > best_score:
                    # Snap and pad the candidate container bounds
                    c_start, c_end = self._validator.snap_and_pad_segment(
                        candidate.start_time,
                        candidate.end_time,
                        vad_list,
                    )

                    # TODO: the ctc is flawed, simple vad works better
                    # the end times are completely broken +3 secs
                    # segments are the subs ones, not the clean ones
                    # from the asr+vad pred
                    pydomino_aligned = False
                    if audio_segments and self.jp_aligner:
                        # search for segment containing corrected asr
                        matching_seg = None
                        for seg in audio_segments:
                            seg_end = seg.start_time + seg.duration
                            if (
                                seg.start_time <= c_start + 0.1
                                and c_end <= seg_end + 0.1
                            ):
                                matching_seg = seg
                                break

                        if matching_seg:
                            try:
                                loader = AudioLoader(matching_seg.filepath)
                                waveform, sr = loader.load_torchaudio(
                                    sampling_rate=16000,
                                    mono=True,
                                )
                                # start:1.222, end:2.444(same); start:3.233, end:4.111 (latter); start:0.233, end:1.111 (prev)
                                # asr_s:1.002, asr_e:3.111
                                # algo:
                                # rel_start = max(0.0, 1.002-1.222=-0.200)=0
                                # rel_end = min(2.444, 3.111-1.222?) = 2.444?

                                rel_start = max(
                                    0.0,
                                    c_start,
                                )
                                seg_end = (
                                    matching_seg.start_time + matching_seg.duration
                                )
                                rel_end = min(
                                    seg_end,
                                    c_end,
                                )

                                start_sample = int(rel_start * sr)
                                end_sample = int(rel_end * sr)
                                # slide segment to get corrected asr
                                slice_wf = waveform[:, start_sample:end_sample]

                                offset = matching_seg.start_time + rel_start
                                local_alignment = self.jp_aligner.align(
                                    (slice_wf, sr), sub.cleaned_text
                                )

                                non_pau = [
                                    p for p in local_alignment if p["char"] != "pau"
                                ]
                                if non_pau:
                                    refined_start = non_pau[0]["start"] + offset
                                    refined_end = non_pau[-1]["end"] + offset
                                    # TODO: check if next sub would overlap if padding
                                    padded_start = max(
                                        0.0, refined_start - self.padding_seconds
                                    )
                                    padded_end = min(
                                        seg_end, refined_end + self.padding_seconds
                                    )

                                    best_times = (padded_start, padded_end)
                                    print(
                                        f"VAD align {c_start, c_end} and pydomino {best_times}"
                                    )
                                    pydomino_aligned = True
                            except Exception as e:
                                logger.error("[pydomino] Refinement failed: %s", e)

                    if not pydomino_aligned:
                        best_times = (c_start, c_end)

                    best_score = score
                    best_match = candidate

            if best_match is None or best_times is None:
                new_start, new_end = self._apply_fallback_offset(sub)
                new_block.start_time = new_start
                new_block.end_time = new_end
                aligned_blocks.append(new_block)
                matches.append(
                    self._create_keep_match(
                        i, sub, new_start, new_end, max(best_score, 0.0)
                    )
                )
                continue

            new_start, new_end, action = self._apply_timing_adjustment(
                sub.start_time,
                sub.end_time,
                best_times[0],
                best_times[1],
            )

            self._update_shift_state(action, sub, best_match, new_start)
            self._mark_asr_consumed(best_match, asr_segments)

            new_block.start_time = new_start
            new_block.end_time = new_end
            aligned_blocks.append(new_block)
            matches.append(
                AlignmentMatch(
                    subtitle_index=i,
                    asr_segment=best_match,
                    similarity=best_score,
                    timing_difference=abs(sub.start_time - new_start),
                    action=action,
                    original_start=sub.start_time,
                    original_end=sub.end_time,
                    new_start=new_start,
                    new_end=new_end,
                )
            )

        return aligned_blocks, matches

    def align(
        self,
        subtitles: list[SubtitleBlock],
        asr_segments: list[TranscriptionSegment],
        window: float = WINDOW_SECONDS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        audio_segments: list[AudioSegment] | None = None,
        vad_intervals: list[dict[str, float]] | None = None,
    ) -> tuple[list[SubtitleBlock], list[AlignmentMatch]]:
        """Align subtitles with ASR transcription.

        Routes logic dynamically to the configured alignment algorithm, executes
        post-alignment scene insertions, and sorts results chronologically.
        """
        self._last_shift_offset = None
        self._last_shift_asr = None
        self._matched_asr_indices = set()

        if vad_intervals is None and audio_segments is not None:
            logger.info("[Align] Pre-computing active VAD intervals...")
            vad_intervals = self._validator._get_absolute_vad_intervals(audio_segments)

        if self.mode == "global":
            aligned_blocks, matches = self._align_global(
                subtitles, asr_segments, window, similarity_threshold
            )
        elif self.mode == "local_ctc":
            aligned_blocks, matches = self._align_local_ctc(
                subtitles,
                asr_segments,
                window,
                similarity_threshold,
                audio_segments=audio_segments,
                vad_intervals=vad_intervals,
            )

        inserted_blocks, inserted_matches = self._insert_asr_scenes(
            aligned_blocks,
            asr_segments,
            self._matched_asr_indices,
        )
        aligned_blocks.extend(inserted_blocks)
        matches.extend(inserted_matches)

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
        # SubtitleWriter.write_blocks(blocks, output_path, fmt=fmt)
        pass
