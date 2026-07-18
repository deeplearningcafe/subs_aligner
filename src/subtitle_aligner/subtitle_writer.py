"""Subtitle writer — writes SubtitleBlock lists back to .srt / .vtt files."""

from __future__ import annotations

import math
from pathlib import Path

from .subtitle_parser import SubtitleBlock


class SubtitleWriter:
    """Writes subtitle blocks to SRT, VTT, or ASS format."""

    @staticmethod
    def write_blocks(
        blocks: list[SubtitleBlock],
        output_path: str | Path,
        fmt: str = "srt",
    ) -> None:
        """Write a list of SubtitleBlock instances to a subtitle file.

        Args:
            blocks:  List of SubtitleBlock instances to write.
            output_path: Destination file path.
            fmt:     Output format — ``"srt"``, ``"vtt"`` or ``"ass"``.
        """
        path = Path(output_path)

        # Extract preserved file metadata from the first block safely
        bom = getattr(blocks[0], "bom", False)
        line_ending = getattr(blocks[0], "line_ending", "\n")

        if fmt.lower() == "ass":
            content = SubtitleWriter._to_ass(blocks, line_ending)
        elif fmt.lower() == "vtt":
            content = SubtitleWriter._to_vtt(blocks, line_ending)
        else:
            content = SubtitleWriter._to_srt(blocks, line_ending)

        # Prepend BOM if the original had one
        if bom:
            content = "\ufeff" + content

        path.write_text(content, encoding="utf-8", newline="")

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
            # Support both SubtitleBlock (raw_text) and TranscriptionSegment (text)
            raw_text = getattr(block, "raw_text", None)
            if raw_text is None:
                raw_text = getattr(block, "text", "")
            if raw_text is None:
                raw_text = ""
            raw = raw_text.replace("\n", line_ending)

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

    @staticmethod
    def _to_vtt(blocks: list[SubtitleBlock], line_ending: str) -> str:
        """Render blocks as a VTT-formatted string."""
        parts: list[str] = [f"WEBVTT{line_ending}{line_ending}"]
        for i, block in enumerate(blocks):
            ts = (
                f"{SubtitleWriter._format_vtt_ts(block.start_time)} --> "
                f"{SubtitleWriter._format_vtt_ts(block.end_time)}{line_ending}"
            )

            raw_text = getattr(block, "raw_text", None)
            if raw_text is None:
                raw_text = getattr(block, "text", "")
            if raw_text is None:
                raw_text = ""
            raw = raw_text.replace("\n", line_ending)

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

    @staticmethod
    def _to_ass(blocks: list[SubtitleBlock], line_ending: str) -> str:
        """Render blocks as an ASS-formatted string.

        Args:
            blocks: List of blocks to serialize.
            line_ending: Preferred line ending character.

        Returns:
            The formatted ASS string content.
        """
        header_lines = getattr(blocks[0], "ass_header", None)
        if not header_lines:
            # Fallback default headers if no metadata was preserved
            header_lines = [
                "[Script Info]",
                "Title: Aligned Subtitles",
                "ScriptType: v4.00+",
                "WrapStyle: 0",
                "ScaledBorderAndShadow: yes",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, "
                "SecondaryColour, OutlineColour, BackColour, Bold, "
                "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
                "Angle, BorderStyle, Outline, Shadow, Alignment, "
                "MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,"
                "&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, "
                "MarginR, MarginV, Effect, Text",
            ]

        parts: list[str] = []
        for line in header_lines:
            parts.append(line + line_ending)

        # Sort blocks to ensure chronological order of events
        sorted_blocks = sorted(blocks, key=lambda b: b.start_time)
        for block in sorted_blocks:
            line_str = SubtitleWriter._format_ass_line(block)
            parts.append(line_str + line_ending)

        return "".join(parts)

    @staticmethod
    def _format_ass_line(block: SubtitleBlock) -> str:
        """Format a SubtitleBlock as a standard ASS Dialogue line.

        Args:
            block: The SubtitleBlock to format.

        Returns:
            Formated ASS dialog string.
        """
        metadata = getattr(block, "ass_metadata", None)
        if metadata is None:
            metadata = {
                "prefix": "Dialogue",
                "layer": "0",
                "style": "Default",
                "name": "",
                "margin_l": "0",
                "margin_r": "0",
                "margin_v": "0",
                "effect": "",
            }
        prefix = metadata.get("prefix", "Dialogue")
        layer = metadata.get("layer", "0")
        style = metadata.get("style", "Default")
        name = metadata.get("name", "")
        margin_l = metadata.get("margin_l", "0")
        margin_r = metadata.get("margin_r", "0")
        margin_v = metadata.get("margin_v", "0")
        effect = metadata.get("effect", "")
        start_str = SubtitleWriter._format_ass_ts(block.start_time)
        end_str = SubtitleWriter._format_ass_ts(block.end_time)
        # Convert internal literal newlines back to ASS's \N tag
        raw_text = block.raw_text.replace("\n", r"\N")

        return (
            f"{prefix}: {layer},{start_str},{end_str},{style},{name},"
            f"{margin_l},{margin_r},{margin_v},{effect},{raw_text}"
        )

    @staticmethod
    def _format_ass_ts(seconds: float) -> str:
        """Format seconds as an ASS timestamp (H:MM:SS.cc).

        Args:
            seconds: Absolute time in seconds.

        Returns:
            ASS formatted timestamp string.
        """
        if math.isnan(seconds) or seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        if cs >= 100:
            s += cs // 100
            cs = cs % 100
            if s >= 60:
                m += s // 60
                s = s % 60
                if m >= 60:
                    h += m // 60
                    m = m % 60
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
