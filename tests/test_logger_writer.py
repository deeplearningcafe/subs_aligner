"""Tests for the parseable Markdown-table logger writer."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

# Load logger_writer directly from file to avoid the __init__.py import chain
# which pulls in aligner → asr_transcriber → reazonspeech (not installed).
_test_root = Path(__file__).resolve().parent.parent
_src = _test_root / "src"
_logger_path = _src / "subtitle_aligner" / "logger_writer.py"

_spec = importlib.util.spec_from_file_location(
    "subtitle_aligner.logger_writer", _logger_path
)
_logger_module = importlib.util.module_from_spec(_spec)
sys.modules["subtitle_aligner.logger_writer"] = _logger_module
_spec.loader.exec_module(_logger_module)

LogEntry = _logger_module.LogEntry
LogWarning = _logger_module.LogWarning
LoggerWriter = _logger_module.LoggerWriter


# ── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_entries():
    """Create sample LogEntry objects for testing."""
    return [
        LogEntry(
            subtitle_index=1,
            action="keep",
            original_start=5.0,
            original_end=8.0,
            new_start=5.0,
            new_end=8.0,
            timing_difference=0.1,
            similarity=0.95,
            text="こんにちは、世界！",
        ),
        LogEntry(
            subtitle_index=2,
            action="adjust",
            original_start=600.0,
            original_end=605.0,
            new_start=600.3,
            new_end=605.3,
            timing_difference=0.3,
            similarity=0.88,
            text="おはようございます",
        ),
        LogEntry(
            subtitle_index=3,
            action="shift",
            original_start=1200.0,
            original_end=1203.0,
            new_start=1215.0,
            new_end=1218.0,
            timing_difference=15.0,
            similarity=0.82,
            text="ありがとう！",
        ),
        LogEntry(
            subtitle_index=-1,
            action="inserted",
            original_start=300.0,
            original_end=305.0,
            new_start=300.0,
            new_end=305.0,
            timing_difference=0.0,
            similarity=1.0,
            text="ASR inserted scene",
        ),
    ]


@pytest.fixture
def sample_warnings():
    """Create sample LogWarning objects for testing."""
    return [
        LogWarning(
            warning_type="shift",
            subtitle_index=3,
            description="Large timing shift detected: 15.00s",
        ),
        LogWarning(
            warning_type="inserted",
            subtitle_index=None,
            description="ASR scene inserted at 300.00s–305.00s",
        ),
    ]


@pytest.fixture
def logger_writer():
    """Create a LoggerWriter with a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = LoggerWriter(
            video_name="test_video",
            subtitle_name="test_subtitles",
            output_dir=tmpdir,
        )
        yield writer


# ── Filename sanitization tests ───────────────────────────────────────


class TestFilenameSanitization:
    """Test filename sanitization logic."""

    def test_sanitize_colons(self):
        """Colons are replaced with underscores."""
        result = LoggerWriter._sanitize_filename("video:2024")
        assert ":" not in result
        assert result == "video_2024"

    def test_sanitize_slashes(self):
        """Forward and back slashes are replaced."""
        result = LoggerWriter._sanitize_filename("path/to/file")
        assert "/" not in result

    def test_sanitize_pipe(self):
        """Pipe characters are replaced."""
        result = LoggerWriter._sanitize_filename("video|sub")
        assert "|" not in result
        assert result == "video_sub"

    def test_sanitize_japanese(self):
        """Japanese characters are preserved."""
        result = LoggerWriter._sanitize_filename("日本語_video")
        assert "日本語" in result

    def test_sanitize_multiple_underscores(self):
        """Multiple consecutive underscores are collapsed."""
        result = LoggerWriter._sanitize_filename("video___sub")
        assert result == "video_sub"

    def test_sanitize_empty(self):
        """Empty name falls back to 'unknown'."""
        result = LoggerWriter._sanitize_filename("")
        assert result == "unknown"

    def test_sanitize_whitespace(self):
        """Leading/trailing whitespace and underscores are stripped."""
        result = LoggerWriter._sanitize_filename("  _test_  ")
        assert result == "test"

    def test_sanitize_preserves_safe_chars(self):
        """Letters, numbers, dots, and hyphens are preserved."""
        result = LoggerWriter._sanitize_filename("my-video_2024.test")
        assert result == "my-video_2024.test"


# ── Log content generation tests ──────────────────────────────────────


class TestGenerateLog:
    """Test Markdown log content generation."""

    def test_generates_summary_section(self, logger_writer):
        """Summary section contains Key: Value lines."""
        entries = [
            LogEntry(1, "keep", 1.0, 2.0, 1.0, 2.0, 0.1, 0.95, "test"),
        ]
        content = logger_writer.generate_log(entries, [])
        assert "# 処理概要 (Summary)" in content
        assert "- Video File:" in content
        assert "- Subtitle File:" in content
        assert "- Total Subtitles: 1" in content
        assert "- Kept: 1" in content
        assert "- Adjusted: 0" in content
        assert "- Shifted: 0" in content
        assert "- Inserted ASR Scenes: 0" in content
        assert "- Total Matches: 1" in content

    def test_generates_details_table(self, logger_writer, sample_entries):
        """Details section contains a Markdown table."""
        content = logger_writer.generate_log(sample_entries, [])
        assert "# 変更詳細 (Details)" in content
        assert "| #" in content
        assert "| Action |" in content
        assert "| Original Start (s) |" in content

    def test_details_table_has_pipe_delimiters(self, logger_writer, sample_entries):
        """Each data row in the details table uses pipe delimiters."""
        content = logger_writer.generate_log(sample_entries, [])
        rows = LoggerWriter.parse_log_table(content, "# 変更詳細 (Details)")
        assert len(rows) == len(sample_entries)
        for row in rows:
            assert len(row) == len(LoggerWriter.TABLE_COLUMNS_DETAILS)

    def test_details_table_action_values(self, logger_writer, sample_entries):
        """Details table contains correct action values."""
        content = logger_writer.generate_log(sample_entries, [])
        assert "| keep |" in content
        assert "| adjust |" in content
        assert "| shift |" in content
        assert "| inserted |" in content

    def test_details_table_timing_values(self, logger_writer, sample_entries):
        """Details table contains correctly formatted timing values."""
        content = logger_writer.generate_log(sample_entries, [])
        assert "5.000" in content
        assert "15.000" in content
        assert "0.100" in content

    def test_details_table_similarity_values(self, logger_writer, sample_entries):
        """Details table contains correctly formatted similarity values."""
        content = logger_writer.generate_log(sample_entries, [])
        assert "0.950" in content
        assert "0.880" in content
        assert "0.820" in content

    def test_generates_warnings_table(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Warnings section contains a Markdown table."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        assert "# 警告 (Warnings)" in content

    def test_warnings_table_has_pipe_delimiters(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Warnings table uses pipe delimiters."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        rows = LoggerWriter.parse_log_table(content, "# 警告 (Warnings)")
        assert len(rows) == len(sample_warnings)
        for row in rows:
            assert len(row) == len(LoggerWriter.TABLE_COLUMNS_WARNINGS)

    def test_warnings_table_contains_warning_types(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Warnings table contains correct warning types."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        assert "shift" in content
        assert "inserted" in content

    def test_warnings_table_subtitle_index(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Warnings table contains correct subtitle indices."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        # First warning has subtitle_index=3
        assert "3" in content
        # Second warning has subtitle_index=None (shown as "-")
        assert "-" in content

    def test_empty_entries_no_details(self, logger_writer):
        """Empty entries produce a 'No changes recorded' message."""
        content = logger_writer.generate_log([], [])
        assert "No changes recorded." in content

    def test_empty_warnings(self, logger_writer):
        """No warnings produces a 'No warnings' message."""
        content = logger_writer.generate_log([], [])
        assert "No warnings." in content

    def test_text_truncation(self, logger_writer):
        """Long subtitle text is truncated in the table."""
        long_text = "HelloWorld-" + "x" * 200
        entries = [
            LogEntry(1, "keep", 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, long_text),
        ]
        content = logger_writer.generate_log(entries, [])
        # The original long text should NOT appear in full
        assert long_text not in content
        # But the prefix should appear with ellipsis
        assert long_text[: LoggerWriter.TEXT_TRUNCATE_LENGTH - 3] + "..." in content

    def test_inserted_action_in_summary(self, logger_writer, sample_entries):
        """Inserted ASR scenes are counted in the summary."""
        content = logger_writer.generate_log(sample_entries, [])
        assert "- Inserted ASR Scenes: 1" in content

    def test_summary_counts(self, logger_writer, sample_entries):
        """Summary counts match the entries."""
        content = logger_writer.generate_log(sample_entries, [])
        assert "- Total Subtitles: 3" in content
        assert "- Kept: 1" in content
        assert "- Adjusted: 1" in content
        assert "- Shifted: 1" in content
        assert "- Total Matches: 4" in content

    def test_log_filename(self, logger_writer):
        """Log filename is built correctly."""
        expected = "test_video_test_subtitles.log"
        assert logger_writer._build_log_filename() == expected

    def test_log_filename_with_special_chars(self):
        """Log filename sanitizes special characters in names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LoggerWriter(
                video_name="video:2024",
                subtitle_name="sub|test.srt",
                output_dir=tmpdir,
            )
            assert ":" not in writer._build_log_filename()
            assert "|" not in writer._build_log_filename()


# ── File writing tests ────────────────────────────────────────────────


class TestWriteLog:
    """Test log file writing."""

    def test_writes_log_file(self, logger_writer, sample_entries, sample_warnings):
        """Log file is written to the output directory."""
        log_path = logger_writer.write_log(sample_entries, sample_warnings)
        assert log_path.exists()
        assert log_path.is_file()

    def test_writes_correct_filename(self, logger_writer):
        """Log file uses the correct filename."""
        log_path = logger_writer.write_log([], [])
        expected = logger_writer.output_dir / "test_video_test_subtitles.log"
        assert log_path == expected

    def test_writes_valid_markdown(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Written log file contains valid Markdown structure."""
        log_path = logger_writer.write_log(sample_entries, sample_warnings)
        content = log_path.read_text(encoding="utf-8")
        # Check all three sections exist
        assert "# 処理概要 (Summary)" in content
        assert "# 変更詳細 (Details)" in content
        assert "# 警告 (Warnings)" in content

    def test_writes_utf8(self, logger_writer, sample_entries):
        """Log file is written in UTF-8 encoding."""
        entries = [
            LogEntry(1, "keep", 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, "こんにちは"),
        ]
        log_path = logger_writer.write_log(entries, [])
        content = log_path.read_text(encoding="utf-8")
        assert "こんにちは" in content


# ── Table parsing tests ───────────────────────────────────────────────


class TestParseLogTable:
    """Test Markdown table parsing for CSV conversion."""

    def test_parses_details_table(self, logger_writer, sample_entries):
        """Details table can be parsed into rows."""
        content = logger_writer.generate_log(sample_entries, [])
        rows = LoggerWriter.parse_log_table(content, "# 変更詳細 (Details)")
        assert len(rows) == len(sample_entries)

    def test_parses_details_columns(self, logger_writer, sample_entries):
        """Parsed rows have the correct number of columns."""
        content = logger_writer.generate_log(sample_entries, [])
        rows = LoggerWriter.parse_log_table(content, "# 変更詳細 (Details)")
        expected_cols = len(LoggerWriter.TABLE_COLUMNS_DETAILS)
        for row in rows:
            assert len(row) == expected_cols

    def test_parses_warnings_table(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Warnings table can be parsed into rows."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        rows = LoggerWriter.parse_log_table(content, "# 警告 (Warnings)")
        assert len(rows) == len(sample_warnings)

    def test_parses_warnings_columns(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Parsed warnings have the correct number of columns."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        rows = LoggerWriter.parse_log_table(content, "# 警告 (Warnings)")
        expected_cols = len(LoggerWriter.TABLE_COLUMNS_WARNINGS)
        for row in rows:
            assert len(row) == expected_cols

    def test_parses_no_table(self, logger_writer):
        """Parsing an empty table returns empty list (no data rows)."""
        content = logger_writer.generate_log([], [])
        rows = LoggerWriter.parse_log_table(content, "# 変更詳細 (Details)")
        # "No changes recorded." is not a pipe-delimited data row
        assert len(rows) == 0

    def test_parse_stops_at_section_boundary(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Parsing stops when it hits the next section heading."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        details_rows = LoggerWriter.parse_log_table(content, "# 変更詳細 (Details)")
        # Should only get details rows, not warning rows
        assert len(details_rows) == len(sample_entries)

    def test_parse_returns_empty_for_missing_section(self, logger_writer):
        """Parsing a non-existent section returns empty list."""
        content = logger_writer.generate_log([], [])
        rows = LoggerWriter.parse_log_table(content, "# Nonexistent Section")
        assert rows == []

    def test_parsed_values_are_stripped(self, logger_writer, sample_entries):
        """Parsed values have leading/trailing whitespace stripped."""
        content = logger_writer.generate_log(sample_entries, [])
        rows = LoggerWriter.parse_log_table(content, "# 変更詳細 (Details)")
        for row in rows:
            for cell in row:
                assert cell == cell.strip()


# ── CSV conversion tests ──────────────────────────────────────────────


class TestToCsv:
    """Test Markdown-to-CSV conversion."""

    def test_converts_details_to_csv(self, logger_writer, sample_entries):
        """Details table converts to valid CSV."""
        content = logger_writer.generate_log(sample_entries, [])
        csv_content = LoggerWriter.to_csv(content, "# 変更詳細 (Details)")
        csv_lines = csv_content.strip().splitlines()
        # Data rows only (no header)
        assert len(csv_lines) == len(sample_entries)
        # First row contains action value
        assert "keep" in csv_lines[0]

    def test_converts_warnings_to_csv(
        self, logger_writer, sample_entries, sample_warnings
    ):
        """Warnings table converts to valid CSV."""
        content = logger_writer.generate_log(sample_entries, sample_warnings)
        csv_content = LoggerWriter.to_csv(content, "# 警告 (Warnings)")
        csv_lines = csv_content.strip().splitlines()
        assert len(csv_lines) == len(sample_warnings)

    def test_csv_uses_comma_delimiter(self, logger_writer, sample_entries):
        """CSV output uses comma delimiters."""
        content = logger_writer.generate_log(sample_entries, [])
        csv_content = LoggerWriter.to_csv(content, "# 変更詳細 (Details)")
        csv_lines = csv_content.strip().splitlines()
        for line in csv_lines:
            assert "," in line

    def test_csv_no_pipe_characters(self, logger_writer, sample_entries):
        """CSV output does not contain pipe characters."""
        content = logger_writer.generate_log(sample_entries, [])
        csv_content = LoggerWriter.to_csv(content, "# 変更詳細 (Details)")
        assert "|" not in csv_content

    def test_csv_empty_table(self, logger_writer):
        """Empty table returns empty string in CSV."""
        content = logger_writer.generate_log([], [])
        csv_content = LoggerWriter.to_csv(content, "# 変更詳細 (Details)")
        assert csv_content == ""

    def test_csv_preserves_action_values(self, logger_writer, sample_entries):
        """CSV preserves action values from entries."""
        content = logger_writer.generate_log(sample_entries, [])
        csv_content = LoggerWriter.to_csv(content, "# 変更詳細 (Details)")
        assert "keep" in csv_content
        assert "adjust" in csv_content
        assert "shift" in csv_content
        assert "inserted" in csv_content


# ── Integration test ──────────────────────────────────────────────────


class TestLoggerWriterIntegration:
    """End-to-end integration tests."""

    def test_full_log_roundtrip(self, sample_entries, sample_warnings):
        """Generate a log, write it, re-read and parse it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LoggerWriter(
                video_name="integration_test",
                subtitle_name="integration_sub",
                output_dir=tmpdir,
            )

            # Write
            log_path = writer.write_log(sample_entries, sample_warnings)
            assert log_path.exists()

            # Read back
            content = log_path.read_text(encoding="utf-8")

            # Parse details
            details_rows = LoggerWriter.parse_log_table(content, "# 変更詳細 (Details)")
            assert len(details_rows) == len(sample_entries)

            # Parse warnings
            warnings_rows = LoggerWriter.parse_log_table(content, "# 警告 (Warnings)")
            assert len(warnings_rows) == len(sample_warnings)

            # CSV conversion
            csv_details = LoggerWriter.to_csv(content, "# 変更詳細 (Details)")
            csv_lines = csv_details.strip().splitlines()
            assert len(csv_lines) == len(sample_entries)

    def test_log_with_japanese_names(self):
        """Log handles Japanese filenames correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LoggerWriter(
                video_name="日本語の動画",
                subtitle_name="日本語字幕",
                output_dir=tmpdir,
            )
            entries = [
                LogEntry(1, "keep", 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, "テスト"),
            ]
            log_path = writer.write_log(entries, [])
            content = log_path.read_text(encoding="utf-8")
            assert "テスト" in content
            assert "日本語" in content
