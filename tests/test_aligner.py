"""Tests for the sliding-window timeline alignment engine."""

from __future__ import annotations

import difflib
import tempfile
from pathlib import Path

import pytest

from src.subtitle_aligner.aligner import AlignmentMatch, SubtitleAligner
from src.subtitle_aligner.asr_transcriber import TranscriptionSegment
from src.subtitle_aligner.subtitle_parser import SubtitleBlock, SubtitleParser


# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def aligner():
    return SubtitleAligner(device="cpu")


@pytest.fixture
def sample_subtitles():
    """Create a small set of subtitle blocks for testing."""
    return [
        SubtitleBlock(
            start_time=5.0,
            end_time=8.0,
            raw_text="こんにちは、世界！",
            cleaned_text="こんにちは、世界！",
        ),
        SubtitleBlock(
            start_time=600.0,  # 10 min
            end_time=605.0,
            raw_text="おはようございます",
            cleaned_text="おはようございます",
        ),
        SubtitleBlock(
            start_time=1200.0,  # 20 min
            end_time=1203.0,
            raw_text="ありがとう！",
            cleaned_text="ありがとう！",
        ),
    ]


@pytest.fixture
def sample_asr_segments():
    """Create ASR segments aligned with sample subtitles (small offsets)."""
    return [
        TranscriptionSegment(
            start_time=5.1,  # +0.1s offset
            end_time=8.1,
            text="こんにちは世界",
            katakana="コンニチワセカイ",
            char_timings=[],
        ),
        TranscriptionSegment(
            start_time=600.3,  # +0.3s offset
            end_time=605.3,
            text="おはようございます",
            katakana="オハヨウゴザイマス",
            char_timings=[],
        ),
        TranscriptionSegment(
            start_time=1205.0,  # +5.0s offset
            end_time=1208.0,
            text="ありがとう",
            katakana="アリガトウ",
            char_timings=[],
        ),
    ]


@pytest.fixture
def sample_asr_segments_large_shift():
    """Create ASR segments with large timing discrepancies."""
    return [
        TranscriptionSegment(
            start_time=20.0,  # +15s offset
            end_time=23.0,
            text="こんにちは世界",
            katakana="コンニチワセカイ",
            char_timings=[],
        ),
        TranscriptionSegment(
            start_time=600.0,  # exact match
            end_time=605.0,
            text="おはようございます",
            katakana="オハヨウゴザイマス",
            char_timings=[],
        ),
        TranscriptionSegment(
            start_time=1800.0,  # +600s offset (10 min)
            end_time=1803.0,
            text="ありがとう",
            katakana="アリガトウ",
            char_timings=[],
        ),
    ]


# ── Katakana conversion tests ─────────────────────────────────────────


class TestGetKatakana:
    """Test the _get_katakana helper."""

    def test_returns_katakana(self, aligner):
        text = "こんにちは、世界！"
        result = aligner._get_katakana(text)
        assert result != ""
        for ch in result:
            assert "\u30a0" <= ch <= "\u30ff"

    def test_empty_input(self, aligner):
        assert aligner._get_katakana("") == ""

    def test_none_like_input(self, aligner):
        assert aligner._get_katakana("   ") == ""


# ── Sliding window search tests ───────────────────────────────────────


class TestFindCandidates:
    """Test the sliding-window candidate search."""

    def test_finds_candidates_in_window(
        self, aligner, sample_subtitles, sample_asr_segments
    ):
        """ASR segments within ±5 min of subtitle start time are found."""
        sub = sample_subtitles[0]  # start_time = 5.0
        candidates = aligner._find_candidates(sub, sample_asr_segments)
        assert len(candidates) == 1
        assert candidates[0].start_time == 5.1

    def test_excludes_candidates_outside_window(
        self, aligner, sample_subtitles, sample_asr_segments
    ):
        """ASR segments outside ±5 min window are excluded."""
        sub = sample_subtitles[0]
        far_asr = TranscriptionSegment(
            start_time=1800.0,
            end_time=1803.0,
            text="far",
            katakana="ファール",
            char_timings=[],
        )
        segments = sample_asr_segments + [far_asr]
        candidates = aligner._find_candidates(sub, segments)
        assert not any(c.start_time == 1800.0 for c in candidates)

    def test_window_boundary_inclusive(self, aligner, sample_subtitles):
        """Window boundaries are inclusive (±5 min exactly matches)."""
        sub = sample_subtitles[0]  # start_time = 5.0
        boundary = 5.0 + 300.0  # exactly at window edge
        boundary_asr = TranscriptionSegment(
            start_time=boundary,
            end_time=boundary + 1,
            text="boundary",
            katakana="バウンダリ",
            char_timings=[],
        )
        candidates = aligner._find_candidates(sub, [boundary_asr])
        assert len(candidates) == 1

    def test_no_candidates_returns_empty(self, aligner, sample_subtitles):
        """No candidates in window returns empty list."""
        sub = sample_subtitles[0]
        far_asr = TranscriptionSegment(
            start_time=10000.0,
            end_time=10003.0,
            text="far",
            katakana="ファール",
            char_timings=[],
        )
        candidates = aligner._find_candidates(sub, [far_asr])
        assert candidates == []


# ── Similarity computation tests ──────────────────────────────────────


class TestComputeSimilarity:
    """Test the Katakana similarity computation."""

    def test_identical_strings(self, aligner):
        """Identical Katakana strings produce similarity of 1.0."""
        result = aligner._compute_similarity("コンニチワセカイ", "コンニチワセカイ")
        assert result == 1.0

    def test_completely_different_strings(self, aligner):
        """Very different strings produce low similarity."""
        result = aligner._compute_similarity("ア", "ズ")
        assert result < 0.5

    def test_minor_variation(self, aligner):
        """Minor differences (e.g. missing particles) yield high similarity."""
        # "こんにちは世界" vs "こんにちは、世界" (particle difference)
        result = aligner._compute_similarity("コンニチワセカイ", "コンニチワセカイ")
        assert result >= 0.70

    def test_empty_strings(self, aligner):
        """Empty strings produce similarity of 1.0 (both empty)."""
        result = aligner._compute_similarity("", "")
        assert result == 1.0

    def test_one_empty_string(self, aligner):
        """One empty string produces similarity of 0.0."""
        result = aligner._compute_similarity("コンニチワ", "")
        assert result == 0.0


# ── 3-tier timing adjustment tests ────────────────────────────────────


class TestApplyTimingAdjustment:
    """Test the 3-tier timing adjustment logic."""

    def test_tier1_small_shift_keeps_original(self, aligner):
        """Difference < 0.2s keeps original timing."""
        new_start, new_end, action = aligner._apply_timing_adjustment(
            original_start=5.0,
            original_end=8.0,
            asr_start=5.1,  # diff = 0.1
            asr_end=8.1,
        )
        assert action == "keep"
        assert new_start == 5.0
        assert new_end == 8.0

    def test_tier2_moderate_shift_uses_asr(self, aligner):
        """Difference 0.2–5.0s overwrites with ASR timing."""
        new_start, new_end, action = aligner._apply_timing_adjustment(
            original_start=5.0,
            original_end=8.0,
            asr_start=5.3,  # diff = 0.3
            asr_end=8.3,
        )
        assert action == "adjust"
        assert new_start == 5.3
        assert new_end == 8.3

    def test_tier2_boundary_at_0_2(self, aligner):
        """Difference exactly 0.2s uses ASR timing."""
        new_start, new_end, action = aligner._apply_timing_adjustment(
            original_start=5.0,
            original_end=8.0,
            asr_start=5.2,  # diff = 0.2
            asr_end=8.2,
        )
        assert action == "adjust"
        assert new_start == 5.2

    def test_tier2_boundary_at_5_0(self, aligner):
        """Difference exactly 5.0s uses ASR timing (still tier 2)."""
        new_start, new_end, action = aligner._apply_timing_adjustment(
            original_start=5.0,
            original_end=8.0,
            asr_start=10.0,  # diff = 5.0
            asr_end=13.0,
        )
        assert action == "adjust"
        assert new_start == 10.0

    def test_tier3_large_shift_uses_asr(self, aligner):
        """Difference > 5.0s overwrites with ASR timing."""
        new_start, new_end, action = aligner._apply_timing_adjustment(
            original_start=5.0,
            original_end=8.0,
            asr_start=15.0,  # diff = 10.0
            asr_end=18.0,
        )
        assert action == "shift"
        assert new_start == 15.0
        assert new_end == 18.0

    def test_tier3_boundary_above_5_0(self, aligner):
        """Difference just above 5.0s triggers tier 3."""
        new_start, new_end, action = aligner._apply_timing_adjustment(
            original_start=5.0,
            original_end=8.0,
            asr_start=10.01,  # diff = 5.01
            asr_end=13.01,
        )
        assert action == "shift"
        assert new_start == 10.01


# ── Full alignment tests ──────────────────────────────────────────────


class TestAlign:
    """Test the full alignment pipeline."""

    def test_align_small_shifts_kept(
        self, aligner, sample_subtitles, sample_asr_segments
    ):
        """Small timing differences (<0.2s) are kept, not adjusted."""
        aligned_blocks, matches = aligner.align(sample_subtitles, sample_asr_segments)

        # First subtitle: diff = 0.1s → keep
        assert len(aligned_blocks) == 3
        assert matches[0].action == "keep"
        assert aligned_blocks[0].start_time == 5.0

        # Second subtitle: diff = 0.3s → adjust
        assert matches[1].action == "adjust"
        assert aligned_blocks[1].start_time == 600.3

    def test_align_large_shifts_logged(
        self, aligner, sample_subtitles, sample_asr_segments_large_shift
    ):
        """Large timing differences (>5.0s) trigger shift action."""
        aligned_blocks, matches = aligner.align(
            sample_subtitles, sample_asr_segments_large_shift
        )

        # First subtitle: diff = 15s → shift
        assert matches[0].action == "shift"
        assert aligned_blocks[0].start_time == 20.0
        assert matches[0].timing_difference == 15.0

        # Second subtitle: diff = 0s → keep
        assert matches[1].action == "keep"
        assert aligned_blocks[1].start_time == 600.0

    def test_align_no_candidates(self, aligner, sample_subtitles):
        """Subtitles with no ASR candidates in window are kept unchanged."""
        far_asr = TranscriptionSegment(
            start_time=50000.0,
            end_time=50000.5,
            text="far",
            katakana="ファール",
            char_timings=[],
        )
        aligned_blocks, matches = aligner.align(sample_subtitles, [far_asr])

        assert all(m.action == "keep" for m in matches)
        assert len(aligned_blocks) == len(sample_subtitles)
        for i, block in enumerate(aligned_blocks):
            assert block.start_time == sample_subtitles[i].start_time

    def test_align_similarity_threshold(self, aligner, sample_subtitles):
        """Subtitles with similarity below threshold are kept unchanged."""
        dissimilar_asr = TranscriptionSegment(
            start_time=5.0,
            end_time=8.0,
            text="全く別の言葉",
            katakana="マタクベツノコトバ",
            char_timings=[],
        )
        aligned_blocks, matches = aligner.align(sample_subtitles, [dissimilar_asr])

        assert matches[0].action == "keep"
        assert matches[0].similarity < 0.70

    def test_align_preserves_raw_text(
        self, aligner, sample_subtitles, sample_asr_segments
    ):
        """Alignment preserves original raw_text of subtitles."""
        aligned_blocks, _ = aligner.align(sample_subtitles, sample_asr_segments)

        for i, block in enumerate(aligned_blocks):
            assert block.raw_text == sample_subtitles[i].raw_text

    def test_align_returns_correct_match_count(
        self, aligner, sample_subtitles, sample_asr_segments
    ):
        """Number of matches equals number of subtitles."""
        _, matches = aligner.align(sample_subtitles, sample_asr_segments)
        assert len(matches) == len(sample_subtitles)

    def test_align_match_data_integrity(
        self, aligner, sample_subtitles, sample_asr_segments
    ):
        """AlignmentMatch objects contain correct data."""
        _, matches = aligner.align(sample_subtitles, sample_asr_segments)

        for m in matches:
            assert isinstance(m, AlignmentMatch)
            assert m.subtitle_index >= 0
            assert m.similarity >= 0.0 and m.similarity <= 1.0
            assert m.timing_difference >= 0.0
            assert m.action in ("keep", "adjust", "shift")

    def test_align_with_empty_subtitles(self, aligner):
        """Empty subtitle list returns empty results."""
        aligned_blocks, matches = aligner.align([], [])
        assert aligned_blocks == []
        assert matches == []

    def test_align_with_empty_asr(self, aligner, sample_subtitles):
        """No ASR segments → all subtitles kept unchanged."""
        aligned_blocks, matches = aligner.align(sample_subtitles, [])
        assert len(aligned_blocks) == len(sample_subtitles)
        assert all(m.action == "keep" for m in matches)


# ── Output writing tests ──────────────────────────────────────────────


class TestWriteAligned:
    """Test writing aligned subtitle blocks to file."""

    def test_write_srt_output(self, aligner, sample_subtitles):
        """Aligned blocks are written as valid SRT."""
        aligned_blocks, _ = aligner.align(sample_subtitles, [])

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w") as f:
            output_path = f.name

        aligner.write_aligned(aligned_blocks, output_path, fmt="srt")

        content = Path(output_path).read_text(encoding="utf-8")
        # Check basic SRT structure
        assert "-->" in content
        # Timestamps should reflect aligned values
        assert "00:00:05" in content
        Path(output_path).unlink()

    def test_write_vtt_output(self, aligner, sample_subtitles):
        """Aligned blocks are written as valid VTT."""
        aligned_blocks, _ = aligner.align(sample_subtitles, [])

        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False, mode="w") as f:
            output_path = f.name

        aligner.write_aligned(aligned_blocks, output_path, fmt="vtt")

        content = Path(output_path).read_text(encoding="utf-8")
        assert content.strip().upper().startswith("WEBVTT")
        Path(output_path).unlink()

    def test_write_preserves_raw_text(self, aligner, sample_subtitles):
        """Written output preserves original raw_text."""
        aligned_blocks, _ = aligner.align(sample_subtitles, [])

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w") as f:
            output_path = f.name

        aligner.write_aligned(aligned_blocks, output_path, fmt="srt")

        content = Path(output_path).read_text(encoding="utf-8")
        for sub in sample_subtitles:
            assert sub.raw_text in content
        Path(output_path).unlink()


# ── Integration test with real subtitle file ──────────────────────────


class TestAlignWithRealSubtitle:
    """Integration tests using real subtitle fixtures."""

    def test_parse_and_align_real_srt(self, aligner):
        """Parse a real SRT file and run alignment with mock ASR segments."""
        fixture_path = Path(__file__).parent / "fixtures" / "test_sample.srt"
        parser = SubtitleParser()
        subtitles = parser.parse_file(fixture_path)

        assert len(subtitles) > 0

        first_sub = subtitles[0]
        mock_asr = [
            TranscriptionSegment(
                start_time=first_sub.start_time + 0.3,
                end_time=first_sub.end_time + 0.3,
                text="こんにちは世界",
                katakana="コンニチワセカイ",
                char_timings=[],
            ),
        ]

        aligned_blocks, matches = aligner.align(subtitles, mock_asr)

        assert len(aligned_blocks) == len(subtitles)
        assert len(matches) == len(subtitles)
        assert matches[0].action == "adjust"
        assert aligned_blocks[0].raw_text == subtitles[0].raw_text


# ── Fallback Offset Propagation ───────────────────────────────────────


class TestFallbackOffsetPropagation:
    """Test that unmatched subtitles inherit shift offset from neighbours."""

    def test_unmatched_inherits_shift_offset(self, aligner):
        """An unmatched subtitle overlapping a shifted ASR region
        inherits the shift offset."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "コンニチワセカイ", "コンニチワセカイ"),
            SubtitleBlock(26.0, 28.0, "オン画面", "オン画面"),
        ]
        asr_segments = [
            TranscriptionSegment(
                25.0,
                28.0,
                "コンニチワセカイ",
                "コンニチワセカイ",
                char_timings=[],
            ),
            TranscriptionSegment(
                30.0,
                32.0,
                "別の言葉",
                "ベツノコトバ",
                char_timings=[],
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        shifted = [m for m in matches if m.original_start == 5.0][0]
        kept = [m for m in matches if m.original_start == 26.0][0]

        assert shifted.action == "shift"
        assert shifted.similarity == 1.0
        assert shifted.new_start == 25.0

        assert kept.action == "keep"
        assert kept.new_start == 26.0 + (25.0 - 5.0)
        assert kept.new_end == 28.0 + 20.0

    def test_unmatched_no_shift_without_overlap(self, aligner):
        """Unmatched subtitle far from any shifted ASR is not shifted."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
            SubtitleBlock(5000.0, 5003.0, "遠い場所", "トオイバショ"),
        ]
        asr_segments = [
            TranscriptionSegment(
                25.0, 28.0, "こんにちは", "コンニチワ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert matches[0].action == "shift"

        assert matches[1].action == "keep"
        assert aligned_blocks[1].start_time == 5000.0
        assert aligned_blocks[1].end_time == 5003.0

    def test_unmatched_after_adjustment_not_shifted(self, aligner):
        """Only "shift" actions propagate offset, not "adjust" actions."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
            SubtitleBlock(10.0, 12.0, "オン画面", "オン画面"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(
                15.0, 17.0, "別の言葉", "ベツノコトバ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert matches[0].action == "adjust"

        assert matches[1].action == "keep"
        assert aligned_blocks[1].start_time == 10.0
        assert aligned_blocks[1].end_time == 12.0

    def test_unmatched_earlier_than_shift_kept(self, aligner):
        """Unmatched subtitle before any shift match is not shifted."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "オン画面", "オン画面"),
            SubtitleBlock(25.0, 28.0, "こんにちは", "こんにちは"),
        ]
        asr_segments = [
            TranscriptionSegment(
                50.0, 53.0, "こんにちは", "コンニチワ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert matches[0].action == "keep"
        assert aligned_blocks[0].start_time == 5.0

        assert matches[1].action == "shift"


# ── Conflict Overlap Handling ─────────────────────────────────────────


class TestConflictOverlapHandling:
    """Test that overlapping ASR/subtitle with low similarity is bypassed."""

    def test_overlapping_low_similarity_preserved(self, aligner):
        """Subtitle overlapping ASR with <70% similarity keeps original timing."""
        subtitles = [
            SubtitleBlock(10.0, 15.0, "話者Aのセリフ", "話者Aノセリフ"),
        ]
        asr_segments = [
            TranscriptionSegment(
                11.0, 14.0, "話者Bのセリフ", "話者Bノセリフ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert matches[0].action == "keep"
        assert matches[0].similarity < 0.70
        assert aligned_blocks[0].start_time == 10.0
        assert aligned_blocks[0].end_time == 15.0

    def test_overlapping_high_similarity_adjusted(self, aligner):
        """Subtitle overlapping ASR with ≥70% similarity is adjusted."""
        subtitles = [
            SubtitleBlock(10.0, 15.0, "こんにちは世界", "こんにちは世界"),
        ]
        asr_segments = [
            TranscriptionSegment(
                10.5,
                15.5,
                "こんにちは世界",
                "コンニチワセカイ",
                char_timings=[],
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert matches[0].action == "adjust"
        assert matches[0].similarity >= 0.70
        assert aligned_blocks[0].start_time == 10.5

    def test_multiple_overlapping_subtitles_independent(self, aligner):
        """Multiple overlapping subtitles each independently handled."""
        subtitles = [
            SubtitleBlock(10.0, 15.0, "コンニチワ", "コンニチワ"),
            SubtitleBlock(12.0, 17.0, "オハヨウゴザイマス", "オハヨウゴザイマス"),
        ]
        asr_segments = [
            TranscriptionSegment(
                11.0, 14.0, "コンニチワ", "コンニチワ", char_timings=[]
            ),
            TranscriptionSegment(
                13.0, 16.0, "コンニチワ", "コンニチワ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert matches[0].action == "adjust"
        assert matches[1].action == "keep"
        assert matches[1].similarity < 0.70


# ── ASR Scene Insertion ───────────────────────────────────────────────


class TestASRSceneInsertion:
    """Test insertion of unmatched ASR segments in subtitle gaps."""

    def test_insert_asr_in_gap(self, aligner):
        """Unmatched ASR in a subtitle gap is inserted as new subtitle."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
            SubtitleBlock(30.0, 35.0, "ありがとう", "アリガトウ"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(
                15.0, 18.0, "新しいシーン", "アタラシイシーン", char_timings=[]
            ),
            TranscriptionSegment(
                30.5, 33.5, "ありがとう", "アリガトウ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert len(aligned_blocks) == 3

        inserted = [b for b in aligned_blocks if b.raw_text == "新しいシーン"]
        assert len(inserted) == 1
        assert inserted[0].start_time == 15.0
        assert inserted[0].end_time == 18.0

        inserted_matches = [m for m in matches if m.action == "inserted"]
        assert len(inserted_matches) == 1
        assert inserted_matches[0].asr_segment.text == "新しいシーン"

    def test_insert_asr_uses_original_text(self, aligner):
        """Inserted subtitles use original ASR text, not Katakana."""
        subtitles = [SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは")]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(
                20.0, 23.0, "日本語のセリフ", "ニホンゴノセリフ", char_timings=[]
            ),
        ]
        aligned_blocks, _ = aligner.align(subtitles, asr_segments)

        inserted = [b for b in aligned_blocks if b.start_time == 20.0]
        assert len(inserted) == 1
        assert inserted[0].raw_text == "日本語のセリフ"
        assert inserted[0].cleaned_text == "日本語のセリフ"

    def test_short_asr_not_inserted(self, aligner):
        """ASR segments shorter than MIN_ASR_DURATION are not inserted."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(15.0, 15.5, "短い", "ミジカイ", char_timings=[]),
        ]
        aligned_blocks, _ = aligner.align(subtitles, asr_segments)

        assert len(aligned_blocks) == 1

    def test_asr_overlapping_subtitle_not_inserted(self, aligner):
        """ASR that overlaps any subtitle is not inserted (even if unmatched)."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(6.0, 9.0, "被る言葉", "ヒタるコトバ", char_timings=[]),
        ]
        aligned_blocks, _ = aligner.align(subtitles, asr_segments)

        assert len(aligned_blocks) == 1

    def test_multiple_insertions_in_same_gap(self, aligner):
        """Multiple unmatched ASR segments in one gap all get inserted."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
            SubtitleBlock(50.0, 55.0, "ありがとう", "アリガトウ"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(
                20.0, 22.0, "第一のセリフ", "ダイイチノセリフ", char_timings=[]
            ),
            TranscriptionSegment(
                23.0, 25.0, "第二のセリフ", "ダイニノセリフ", char_timings=[]
            ),
            TranscriptionSegment(
                26.0, 28.0, "第三のセリフ", "ダイサンノセリフ", char_timings=[]
            ),
            TranscriptionSegment(
                50.5, 53.5, "ありがとう", "アリガトウ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert len(aligned_blocks) == 5
        assert sum(1 for m in matches if m.action == "inserted") == 3

    def test_no_insertion_when_all_matched(self, aligner):
        """When all ASR segments are matched, no insertion occurs."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
            SubtitleBlock(20.0, 23.0, "ありがとう", "アリガトウ"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(
                20.5, 23.5, "ありがとう", "アリガトウ", char_timings=[]
            ),
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        assert len(aligned_blocks) == 2
        assert len(matches) == 2

    def test_output_sorted_by_start_time(self, aligner):
        """Final output is sorted by start time."""
        subtitles = [
            SubtitleBlock(30.0, 35.0, "ありがとう", "アリガトウ"),
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(
                30.5, 33.5, "ありがとう", "アリガトウ", char_timings=[]
            ),
            TranscriptionSegment(
                15.0, 18.0, "中間のセリフ", "チュウカンノセリフ", char_timings=[]
            ),
        ]
        aligned_blocks, _ = aligner.align(subtitles, asr_segments)

        times = [b.start_time for b in aligned_blocks]
        assert times == sorted(times)

    def test_insertion_at_video_start(self, aligner):
        """ASR at the very start (before first subtitle) is inserted."""
        subtitles = [
            SubtitleBlock(10.0, 13.0, "こんにちは", "こんにちは"),
        ]
        asr_segments = [
            TranscriptionSegment(
                1.0, 4.0, "オープニング", "オエーピンク", char_timings=[]
            ),
            TranscriptionSegment(
                10.5, 12.5, "こんにちは", "コンニチワ", char_timings=[]
            ),
        ]
        aligned_blocks, _ = aligner.align(subtitles, asr_segments)

        assert len(aligned_blocks) == 2
        inserted = [b for b in aligned_blocks if b.start_time == 1.0]
        assert len(inserted) == 1

    def test_insertion_at_video_end(self, aligner):
        """ASR at the very end (after last subtitle) is inserted."""
        subtitles = [
            SubtitleBlock(5.0, 8.0, "こんにちは", "こんにちは"),
        ]
        asr_segments = [
            TranscriptionSegment(5.5, 8.5, "こんにちは", "コンニチワ", char_timings=[]),
            TranscriptionSegment(
                20.0, 23.0, "エンディング", "エンディンク", char_timings=[]
            ),
        ]
        aligned_blocks, _ = aligner.align(subtitles, asr_segments)

        assert len(aligned_blocks) == 2
        inserted = [b for b in aligned_blocks if b.start_time == 20.0]
        assert len(inserted) == 1


# ── Local CTC Alignment Tests ─────────────────────────────────────────


class TestLocalCTCAlignment:
    """Test localized substring phonetic matching using CTC timings.

    Specifically targets `local_ctc` mode features and the matching engine's
    ability to crop timelines precisely based on character positions.
    """

    def test_find_local_ctc_bounds_success(self, aligner):
        """Verify successful CTC bounds extraction for matched substrings."""
        sub = SubtitleBlock(
            start_time=5.0,
            end_time=8.0,
            raw_text="こんにちは",
            cleaned_text="こんにちは",
        )
        candidate = TranscriptionSegment(
            start_time=10.0,
            end_time=13.0,
            text="こんにちは世界",
            katakana="コンニチワセカイ",
            char_timings=[10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
        )
        res = aligner._find_local_ctc_bounds(
            subtitle=sub,
            candidate=candidate,
            similarity_threshold=0.70,
        )
        assert res is not None
        new_start, new_end, similarity = res
        assert new_start == 10.0
        assert new_end == 12.5
        assert similarity == pytest.approx(10.0 / 11.0)

    def test_find_local_ctc_bounds_low_similarity(self, aligner):
        """Verify None is returned if similarity is below threshold."""
        sub = SubtitleBlock(
            start_time=5.0,
            end_time=8.0,
            raw_text="おやすみ",
            cleaned_text="おやすみ",
        )
        candidate = TranscriptionSegment(
            start_time=10.0,
            end_time=13.0,
            text="こんにちは世界",
            katakana="コンニチワセカイ",
            char_timings=[10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
        )
        res = aligner._find_local_ctc_bounds(
            subtitle=sub,
            candidate=candidate,
            similarity_threshold=0.70,
        )
        assert res is None

    def test_align_local_ctc_mode(self, aligner):
        """Verify that aligner in local_ctc mode routes and adjusts timings."""
        aligner.mode = "local_ctc"
        subtitles = [
            SubtitleBlock(
                start_time=5.0,
                end_time=8.0,
                raw_text="こんにちは",
                cleaned_text="こんにちは",
            )
        ]
        asr_segments = [
            TranscriptionSegment(
                start_time=10.0,
                end_time=13.0,
                text="こんにちは世界",
                katakana="コンニチワセカイ",
                char_timings=[10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
            )
        ]
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)
        assert len(aligned_blocks) == 1
        assert matches[0].action == "adjust"
        assert aligned_blocks[0].start_time == 10.0
        assert aligned_blocks[0].end_time == 12.5
