"""Logger Writer — parseable Markdown-table log generator.

Records execution metrics, timing shift details, and alignment warnings
into a strictly structured Markdown table format.

Sections:
    - 処理概要 (Summary):  ``- Key: Value`` text layout
    - 変更詳細 (Details):  Markdown table with pipe delimiters
    - 警告 (Warnings):     Markdown table with pipe delimiters

Designed for downstream CSV conversion via ``line.split('|')``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from subtitle_aligner.aligner import AlignmentMatch


@dataclass
class LogWarning:
    """A single warning entry for the log.

    Attributes:
        warning_type: Category of the warning (e.g. ``"shift"``,
            ``"inserted"``, ``"low_similarity"``).
        subtitle_index: 1-based index of the affected subtitle, or ``None``
            for global warnings.
        description: Human-readable warning description.
    """

    warning_type: str
    subtitle_index: Optional[int]
    description: str


@dataclass
class LogEntry:
    """A single row in the Details table.

    Mirrors ``AlignmentMatch`` but is serialisable to Markdown.

    Attributes:
        subtitle_index: 1-based subtitle index (``-1`` for inserted scenes).
        action: One of ``"keep"``, ``"adjust"``, ``"shift"``, ``"inserted"``.
        original_start: Original start time in seconds.
        original_end: Original end time in seconds.
        new_start: Aligned start time in seconds.
        new_end: Aligned end time in seconds.
        timing_difference: Absolute time difference in seconds.
        similarity: Katakana similarity ratio (0.0–1.0).
        text: Subtitle text (truncated to 60 chars for table readability).
    """

    subtitle_index: int
    action: str
    original_start: float
    original_end: float
    new_start: float
    new_end: float
    timing_difference: float
    similarity: float
    text: str


class LoggerWriter:
    """Generates parseable Markdown-table logs for alignment runs.

    Args:
        video_name: Base name of the processed video file.
        subtitle_name: Base name of the processed subtitle file.
        output_dir: Directory to write log files into.
    """

    TABLE_COLUMNS_DETAILS: list[str] = [
        "#",
        "Action",
        "Original Start (s)",
        "Original End (s)",
        "New Start (s)",
        "New End (s)",
        "Timing Diff (s)",
        "Similarity",
        "Text",
    ]

    TABLE_COLUMNS_WARNINGS: list[str] = [
        "#",
        "Type",
        "Subtitle #",
        "Description",
    ]

    TEXT_TRUNCATE_LENGTH: int = 60
    LOW_SIMILARITY_THRESHOLD: float = 0.5

    def __init__(
        self,
        video_name: str,
        subtitle_name: str,
        output_dir: str | Path = "outputs/logs/",
    ) -> None:
        """Initialize the logger writer.

        Args:
            video_name: Name of the video file (used in log filename).
            subtitle_name: Name of the subtitle file (used in log filename).
            output_dir: Directory for log output.
        """
        self.video_name = self._sanitize_filename(video_name)
        self.subtitle_name = self._sanitize_filename(subtitle_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a filename to remove unsafe characters.

        Keeps the original name as-is when possible; only replaces
        characters that would be invalid in filesystem paths.

        Args:
            name: Raw filename string.

        Returns:
            Sanitized filename safe for use in file paths.
        """
        # Replace characters that are problematic in most filesystems
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        # Collapse multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Strip leading/trailing whitespace and underscores
        sanitized = sanitized.strip(" _")
        return sanitized if sanitized else "unknown"

    def _build_log_filename(self) -> str:
        """Build the log output filename.

        Returns:
            Filename in the form ``{video_name}_{subtitle_name}.log``.
        """
        # limit char len
        return f"{self.video_name[:20]}_{self.subtitle_name[:20]}.log"

    def _truncate_text(self, text: str) -> str:
        """Truncate text to a maximum length for table readability.

        Args:
            text: Original subtitle text.

        Returns:
            Text truncated to ``TEXT_TRUNCATE_LENGTH`` characters with
            an ellipsis appended if truncated.
        """
        if len(text) <= self.TEXT_TRUNCATE_LENGTH:
            return text
        return text[: self.TEXT_TRUNCATE_LENGTH - 3] + "..."

    def _format_float(self, value: float, precision: int = 3) -> str:
        """Format a float to a fixed number of decimal places.

        Args:
            value: Float value to format.
            precision: Number of decimal places.

        Returns:
            Formatted string.
        """
        return f"{value:.{precision}f}"

    def _build_summary_section(
        self,
        total_subtitles: int,
        kept: int,
        adjusted: int,
        shifted: int,
        inserted: int,
        total_matches: int,
    ) -> list[str]:
        """Build the Summary section lines.

        Args:
            total_subtitles: Total number of original subtitles.
            kept: Count of ``"keep"`` actions.
            adjusted: Count of ``"adjust"`` actions.
            shifted: Count of ``"shift"`` actions.
            inserted: Count of ``"inserted"`` ASR scene insertions.
            total_matches: Total number of alignment matches.

        Returns:
            List of ``"- Key: Value"`` lines.
        """
        lines = [
            "# 処理概要 (Summary)",
            f"- Video File: {self.video_name}",
            f"- Subtitle File: {self.subtitle_name}",
            f"- Total Subtitles: {total_subtitles}",
            f"- Kept: {kept}",
            f"- Adjusted: {adjusted}",
            f"- Shifted: {shifted}",
            f"- Inserted ASR Scenes: {inserted}",
            f"- Total Matches: {total_matches}",
        ]
        return lines

    def _build_details_table(self, entries: list[LogEntry]) -> list[str]:
        """Build the Details Markdown table.

        Args:
            entries: List of LogEntry dataclass instances.

        Returns:
            List of lines forming the Markdown table (header + body).
        """
        lines: list[str] = [""]
        lines.append("# 変更詳細 (Details)")
        lines.append("")

        if not entries:
            lines.append("| No changes recorded. |")
            return lines

        # Header
        header = "| " + " | ".join(self.TABLE_COLUMNS_DETAILS) + " |"
        separator = "| " + " | ".join(["---"] * len(self.TABLE_COLUMNS_DETAILS)) + " |"

        lines.append(header)
        lines.append(separator)

        # Body rows
        for idx, entry in enumerate(entries, start=1):
            row = [
                str(idx),
                entry.action,
                self._format_float(entry.original_start),
                self._format_float(entry.original_end),
                self._format_float(entry.new_start),
                self._format_float(entry.new_end),
                self._format_float(entry.timing_difference),
                self._format_float(entry.similarity),
                self._truncate_text(entry.text),
            ]
            lines.append("| " + " | ".join(row) + " |")

        return lines

    def _build_warnings_table(self, warnings: list[LogWarning]) -> list[str]:
        """Build the Warnings Markdown table.

        Args:
            warnings: List of LogWarning dataclass instances.

        Returns:
            List of lines forming the Markdown table (header + body).
        """
        lines: list[str] = [""]
        lines.append("# 警告 (Warnings)")
        lines.append("")

        if not warnings:
            lines.append("| No warnings. |")
            return lines

        # Header
        header = "| " + " ".join(self.TABLE_COLUMNS_WARNINGS) + " |"
        separator = "| " + " | ".join(["---"] * len(self.TABLE_COLUMNS_WARNINGS)) + " |"

        lines.append(header)
        lines.append(separator)

        # Body rows
        for idx, warning in enumerate(warnings, start=1):
            sub_num = (
                str(warning.subtitle_index)
                if warning.subtitle_index is not None
                else "-"
            )
            row = [
                str(idx),
                warning.warning_type,
                sub_num,
                warning.description,
            ]
            lines.append("| " + " | ".join(row) + " |")

        return lines

    def generate_log(
        self,
        entries: list[LogEntry],
        warnings: list[LogWarning],
    ) -> str:
        """Generate the full Markdown log content.

        Args:
            entries: Alignment detail entries.
            warnings: Warning entries.

        Returns:
            Complete Markdown log as a string.
        """
        # Compute summary counts
        kept = sum(1 for e in entries if e.action == "keep")
        adjusted = sum(1 for e in entries if e.action == "adjust")
        shifted = sum(1 for e in entries if e.action == "shift")
        inserted = sum(1 for e in entries if e.action == "inserted")
        total_matches = len(entries)
        total_subtitles = sum(
            1 for e in entries if e.action in ("keep", "adjust", "shift")
        )

        all_lines: list[str] = []
        all_lines.extend(
            self._build_summary_section(
                total_subtitles=total_subtitles,
                kept=kept,
                adjusted=adjusted,
                shifted=shifted,
                inserted=inserted,
                total_matches=total_matches,
            )
        )
        all_lines.extend(self._build_details_table(entries))
        all_lines.extend(self._build_warnings_table(warnings))

        return "\n".join(all_lines) + "\n"

    def write_log(
        self,
        entries: list[LogEntry],
        warnings: list[LogWarning],
    ) -> Path:
        """Generate and write the log file.

        Args:
            entries: Alignment detail entries.
            warnings: Warning entries.

        Returns:
            Path to the written log file.
        """
        content = self.generate_log(entries, warnings)
        log_path = self.output_dir / self._build_log_filename()
        log_path.write_text(content, encoding="utf-8")
        return log_path

    @staticmethod
    def parse_log_table(log_content: str, section: str) -> list[list[str]]:
        """Parse a Markdown table section from log content into CSV arrays.

        Splits each row by the pipe character and strips whitespace,
        producing a list of column-value lists (one per row, excluding
        the header and separator rows).

        Args:
            log_content: Full Markdown log string.
            section: Section heading to locate (e.g. ``"# 変更詳細 (Details)"``).

        Returns:
            List of lists, where each inner list contains the column values
            of one data row (no header, no separator).
        """
        lines = log_content.splitlines()
        in_section = False
        header_skipped = False
        rows: list[list[str]] = []

        def _parse_pipe_row(stripped_line: str) -> list[str] | None:
            """Parse a pipe-delimited line into columns, or None if not a table row."""
            if not stripped_line.startswith("| "):
                return None
            parts = [p.strip() for p in stripped_line.split("|")]
            # Remove empty first/last elements from leading/trailing pipes
            while parts and parts[0] == "":
                parts.pop(0)
            while parts and parts[-1] == "":
                parts.pop()
            return parts

        for line in lines:
            stripped = line.strip()

            if stripped == section:
                in_section = True
                header_skipped = False
                continue

            if in_section:
                # End of section if we hit another heading
                if stripped.startswith("# "):
                    break

                # Skip empty lines
                if not stripped:
                    continue

                # Parse pipe-delimited row
                parsed = _parse_pipe_row(stripped)
                if parsed is None:
                    continue

                # Skip separator row (all cells are '---')
                if all(p == "---" for p in parsed):
                    continue

                # Skip header row (first non-separator table row)
                if not header_skipped:
                    header_skipped = True
                    continue

                rows.append(parsed)

        return rows

    @staticmethod
    def prepare_log_data(
        matches: list[AlignmentMatch],
    ) -> tuple[list[LogEntry], list[LogWarning]]:
        """Convert alignment matches to log entries and structured warnings.

        Decoupled helper to clean up run_aligner's core execution flow.
        """
        entries: list[LogEntry] = []
        warnings: list[LogWarning] = []

        for m in matches:
            text = m.asr_segment.text if m.asr_segment is not None else ""
            entry = LogEntry(
                subtitle_index=m.subtitle_index,
                action=m.action,
                original_start=m.original_start,
                original_end=m.original_end,
                new_start=m.new_start,
                new_end=m.new_end,
                timing_difference=m.timing_difference,
                similarity=m.similarity,
                text=text,
            )
            entries.append(entry)

            sub_idx = m.subtitle_index if m.subtitle_index >= 0 else None

            if m.action == "shift":
                warnings.append(
                    LogWarning(
                        warning_type="shift",
                        subtitle_index=sub_idx,
                        description=(
                            f"Large timing shift detected: {m.timing_difference:.2f}s"
                        ),
                    )
                )
            elif m.action == "inserted":
                warnings.append(
                    LogWarning(
                        warning_type="inserted",
                        subtitle_index=None,
                        description=(
                            f"ASR scene inserted at {m.new_start:.2f}s–{m.new_end:.2f}s"
                        ),
                    )
                )
            elif m.action == "keep" and 0.0 < m.similarity < 0.5:
                warnings.append(
                    LogWarning(
                        warning_type="low_similarity",
                        subtitle_index=sub_idx,
                        description=(
                            f"Low similarity match ({m.similarity:.2f}) "
                            f"kept with original timing"
                        ),
                    )
                )

        return entries, warnings

    @staticmethod
    def to_csv(log_content: str, section: str) -> str:
        """Convert a Markdown table section to CSV format.

        Args:
            log_content: Full Markdown log string.
            section: Section heading to convert.

        Returns:
            CSV string with data rows (no header row).
        """
        table_rows = LoggerWriter.parse_log_table(log_content, section)
        if not table_rows:
            return ""

        return "\n".join(",".join(cell for cell in row) for row in table_rows)
