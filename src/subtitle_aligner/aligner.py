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
    TAIL_PAD_SECONDS: float = 0.100

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

    def _find_best_local_phonetic_span(
        self,
        sub_kana: str,
        asr_kana: str,
        similarity_threshold: float = 0.75,
    ) -> tuple[float, int, int] | None:
        """Find the optimal fuzzy phonetic sub-span in an ASR transcription.

        Performs bounded local alignment allowing small mora substitutions
        or contractions (e.g. 'し' vs 'す') while rejecting low-similarity
        matches.

        Args:
            sub_kana: Katakana representation of the subtitle card.
            asr_kana: Katakana representation of the candidate ASR sentence.
            similarity_threshold: Minimum local similarity ratio required.

        Returns:
            Tuple of (similarity, start_idx, end_idx) in asr_kana, or None.
        """
        sub_len = len(sub_kana)
        asr_len = len(asr_kana)
        if sub_len == 0 or asr_len == 0:
            return None

        # Direct containment shortcut for exact matches
        if sub_kana in asr_kana:
            idx = asr_kana.index(sub_kana)
            return 1.0, idx, idx + sub_len

        best_score = 0.0
        best_span: tuple[int, int] | None = None

        # Search window allowing +/- 2 moras variation for edits
        min_w = max(1, sub_len - 2)
        max_w = min(asr_len, sub_len + 2)

        for w in range(min_w, max_w + 1):
            for i in range(asr_len - w + 1):
                candidate_slice = asr_kana[i : i + w]
                score = difflib.SequenceMatcher(None, sub_kana, candidate_slice).ratio()

                if score > best_score:
                    best_score = score
                    best_span = (i, i + w)

        if best_span is not None and best_score >= similarity_threshold:
            return best_score, best_span[0], best_span[1]

        return None

    def _align_asr_container_phonemes(
        self,
        candidate: TranscriptionSegment,
        container_start: float,
        container_end: float,
        audio_segments: list[AudioSegment],
    ) -> list[dict[str, float | str]] | None:
        """Run pydomino once on the entire continuous ASR audio container.

        Aligns the full ASR transcript over the VAD-sanitized audio window,
        generating continuous, gapless phoneme timestamps for the sentence.

        Args:
            candidate: ASR segment being aligned.
            container_start: VAD-sanitized speech start timestamp (seconds).
            container_end: VAD-sanitized speech end timestamp (seconds).
            audio_segments: Loaded physical audio segments.

        Returns:
            List of aligned non-pau phoneme timing dictionaries, or None.
        """
        if not self.jp_aligner or not audio_segments:
            return None

        # Find the physical chunk containing the continuous container
        matching_seg: Optional[AudioSegment] = None
        for seg in audio_segments:
            seg_end = seg.start_time + seg.duration
            if (
                seg.start_time <= container_start + 0.200
                and container_end <= seg_end + 0.200
            ):
                matching_seg = seg
                break

        if matching_seg is None:
            return None

        try:
            loader = AudioLoader(matching_seg.filepath)
            waveform, sr = loader.load_torchaudio(sampling_rate=16000, mono=True)

            rel_start = max(0.0, container_start - matching_seg.start_time)
            rel_end = min(
                matching_seg.duration,
                container_end - matching_seg.start_time,
            )

            start_sample = int(rel_start * sr)
            end_sample = int(rel_end * sr)

            if end_sample <= start_sample:
                return None

            slice_wf = waveform[:, start_sample:end_sample]
            slice_offset = matching_seg.start_time + rel_start

            # Align the full ASR sentence text
            alignment = self.jp_aligner.align((slice_wf, sr), candidate.text)

            non_pau: list[dict[str, float | str]] = []
            for p in alignment:
                if p["char"] != "pau":
                    non_pau.append(
                        {
                            "char": p["char"],
                            "start": float(p["start"]) + slice_offset,
                            "end": float(p["end"]) + slice_offset,
                        }
                    )
            return non_pau if non_pau else None
        except Exception as err:
            logger.debug("[pydomino] Full container alignment failed: %s", err)
            return None

    def _extract_mora_span_timings(
        self,
        k_start: int,
        k_end: int,
        asr_kana_len: int,
        phonemes: list[dict[str, float | str]],
        is_phrase_final: bool,
    ) -> tuple[float, float]:
        """Map Katakana mora indices to phoneme timestamps with zero bleed.

        Args:
            k_start: Starting mora index in the ASR Katakana sequence.
            k_end: Ending mora index in the ASR Katakana sequence.
            asr_kana_len: Total length of ASR Katakana string.
            phonemes: Continuous phonemes from full-container alignment.
            is_phrase_final: True if this card is the final card of the ASR.

        Returns:
            Tuple of (card_start_time, card_end_time) in absolute seconds.
        """
        num_ph = len(phonemes)
        # Interpolate mora indices to phoneme indices
        p_start_idx = int((k_start / asr_kana_len) * num_ph)
        p_end_idx = int((k_end / asr_kana_len) * num_ph) - 1

        p_start_idx = max(0, min(p_start_idx, num_ph - 1))
        p_end_idx = max(p_start_idx, min(p_end_idx, num_ph - 1))

        # Direct, unpadded acoustic boundary extraction
        start_time = float(phonemes[p_start_idx]["start"])
        end_time = float(phonemes[p_end_idx]["end"])

        # Tail-trimming applied strictly to phrase-final cards
        if is_phrase_final:
            end_time += self.TAIL_PAD_SECONDS

        return start_time, end_time

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
        """Partition continuous ASR containers into non-bleeding subtitle cards.

        Args:
            subtitles: Subtitle cards to be aligned.
            asr_segments: Continuous ASR transcription blocks.
            window: Temporal candidate search window in seconds.
            similarity_threshold: Minimum local phonetic similarity.
            audio_segments: Loaded audio chunk references.
            vad_intervals: Validated voice activity intervals.

        Returns:
            Tuple of (aligned_blocks, match_details).
        """

        aligned_blocks: list[SubtitleBlock] = []
        matches: list[AlignmentMatch] = []
        vad_list = vad_intervals or []

        # Cache full container phoneme alignments to avoid redundant runs
        container_phoneme_cache: dict[int, list[dict[str, float | str]]] = {}

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

            best_candidate: Optional[TranscriptionSegment] = None
            best_cand_idx: int = -1
            best_score: float = -1.0
            best_span: Optional[tuple[int, int]] = None

            for cand_idx, candidate in enumerate(candidates):
                # Filter out hallucinations
                if self._validator.is_hallucination(
                    candidate.start_time,
                    candidate.end_time,
                    vad_list,
                ):
                    continue

                res = self._find_best_local_phonetic_span(
                    sub_kana, candidate.katakana, similarity_threshold
                )
                if res is None:
                    continue

                score, k_start, k_end = res
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_cand_idx = cand_idx
                    best_span = (k_start, k_end)

            if best_candidate is None or best_span is None or best_cand_idx < 0:
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

            c_start, c_end = self._validator.snap_and_pad_segment(
                best_candidate.start_time,
                best_candidate.end_time,
                vad_list,
            )

            # Lazy-compute pydomino alignment on full ASR container
            if best_cand_idx not in container_phoneme_cache:
                aligned_ph = None
                if audio_segments and self.jp_aligner:
                    aligned_ph = self._align_asr_container_phonemes(
                        best_candidate, c_start, c_end, audio_segments
                    )
                if aligned_ph:
                    container_phoneme_cache[best_cand_idx] = aligned_ph

            cand_kana = best_candidate.katakana or self._get_katakana(
                best_candidate.text
            )
            kana_len = max(1, len(cand_kana))
            k_start, k_end = best_span
            is_final = k_end >= (kana_len - 1)

            # Partition using pydomino phonemes or CTC timings
            if best_cand_idx in container_phoneme_cache:
                phonemes = container_phoneme_cache[best_cand_idx]
                p_start, p_end = self._extract_mora_span_timings(
                    k_start, k_end, kana_len, phonemes, is_final
                )
            elif best_candidate.char_timings:
                # Proportional character CTC fallback
                num_ctc = len(best_candidate.char_timings)
                c_s = max(0, min(int((k_start / kana_len) * num_ctc), num_ctc - 1))
                c_e = max(c_s, min(int((k_end / kana_len) * num_ctc) - 1, num_ctc - 1))
                p_start = best_candidate.char_timings[c_s]
                p_end = best_candidate.char_timings[c_e]
                if is_final:
                    p_end += self.TAIL_PAD_SECONDS
            else:
                # Linear ratio fallback within snapped container
                dur = c_end - c_start
                p_start = c_start + (k_start / kana_len) * dur
                p_end = c_start + (k_end / kana_len) * dur

            new_start, new_end, action = self._apply_timing_adjustment(
                sub.start_time, sub.end_time, p_start, p_end
            )

            self._update_shift_state(action, sub, best_candidate, new_start)
            self._mark_asr_consumed(best_candidate, asr_segments)

            new_block.start_time = new_start
            new_block.end_time = new_end
            aligned_blocks.append(new_block)
            matches.append(
                AlignmentMatch(
                    subtitle_index=i,
                    asr_segment=best_candidate,
                    similarity=best_score,
                    timing_difference=abs(sub.start_time - new_start),
                    action=action,
                    original_start=sub.start_time,
                    original_end=sub.end_time,
                    new_start=new_start,
                    new_end=new_end,
                )
            )

        # Monotonic safety clamp: prevent adjacent card collisions
        for i in range(len(aligned_blocks) - 1):
            if aligned_blocks[i].end_time > aligned_blocks[i + 1].start_time:
                safe_end = max(
                    aligned_blocks[i].start_time + 0.200,
                    aligned_blocks[i + 1].start_time - 0.020,
                )
                aligned_blocks[i].end_time = safe_end
                matches[i].new_end = safe_end

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
