"""Subtitle writer — writes SubtitleBlock lists back to .srt / .vtt files."""

from __future__ import annotations

import math
from pathlib import Path

from .subtitle_parser import SubtitleBlock


class SubtitleWriter:
    """Writes subtitle blocks to SRT or VTT format, preserving original text."""

    @staticmethod
    def write_blocks(
        blocks: list[SubtitleBlock],
        output_path: str | Path,
        fmt: str = "srt",
    ) -> None:
        """
        Write a list of SubtitleBlock instances to a subtitle file.

        Detects BOM and line-ending style from the first block's attached
        metadata (set by SubtitleParser) so the output is byte-for-byte
        identical to the original.

        Args:
            blocks:  List of SubtitleBlock instances to write.
            output_path: Destination file path.
            fmt:     Output format — ``"srt"`` or ``"vtt"``.
        """
        path = Path(output_path)

        # Extract preserved file metadata from the first block
        bom = getattr(blocks[0], "_bom", False)
        line_ending = getattr(blocks[0], "_line_ending", "\n")

        if fmt.lower() == "vtt":
            content = SubtitleWriter._to_vtt(blocks, line_ending)
        else:
            content = SubtitleWriter._to_srt(blocks, line_ending)

        # Prepend BOM if the original had one
        if bom:
            content = "\ufeff" + content

        path.write_text(content, encoding="utf-8", newline="")

    # ── SRT output ──────────────────────────────────────────────────────

    @staticmethod
    def _to_srt(blocks: list[SubtitleBlock], line_ending: str) -> str:
        """Render blocks as an SRT-formatted string."""
        parts: list[str] = []
        trailing_blank = getattr(blocks[0], "_trailing_blank", False)
        for i, block in enumerate(blocks):
            prefix = f"{i + 1}{line_ending}"
            ts = (
                f"{SubtitleWriter._format_srt_ts(block.start_time)} --> "
                f"{SubtitleWriter._format_srt_ts(block.end_time)}{line_ending}"
            )
            # Normalize internal \\n to the original line ending style
            raw = block.raw_text.replace("\n", line_ending)

            if i < len(blocks) - 1:
                # Between blocks: blank line
                parts.append(prefix + ts + raw + line_ending + line_ending)
            else:
                # Last block: preserve trailing blank line if present
                parts.append(
                    prefix + ts + raw + line_ending * (2 if trailing_blank else 1)
                )
        return "".join(parts)

    @staticmethod
    def _format_srt_ts(seconds: float) -> str:
        """Format seconds as an SRT timestamp (HH:MM:SS,mmm)."""
        if math.isnan(seconds):
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ── VTT output ──────────────────────────────────────────────────────

    @staticmethod
    def _to_vtt(blocks: list[SubtitleBlock], line_ending: str) -> str:
        """Render blocks as a VTT-formatted string."""
        parts: list[str] = [f"WEBVTT{line_ending}{line_ending}"]
        for i, block in enumerate(blocks):
            ts = (
                f"{SubtitleWriter._format_vtt_ts(block.start_time)} --> "
                f"{SubtitleWriter._format_vtt_ts(block.end_time)}{line_ending}"
            )
            # Normalize internal \\n to the original line ending style
            raw = block.raw_text.replace("\n", line_ending)

            if i < len(blocks) - 1:
                # Between blocks: blank line
                parts.append(ts + raw + line_ending + line_ending)
            else:
                # Last block: single trailing newline
                parts.append(ts + raw + line_ending)
        return "".join(parts)

    @staticmethod
    def _format_vtt_ts(seconds: float) -> str:
        """Format seconds as a VTT timestamp (HH:MM:SS.mmmmmm)."""
        if math.isnan(seconds):
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        us = int(round((seconds - int(seconds)) * 1_000_000))
        return f"{h:02d}:{m:02d}:{s:02d}.{us:06d}"
