"""Subtitle parser — reads .srt and .vtt files preserving original formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .text_processing import TextProcessor


@dataclass
class SubtitleBlock:
    """A single subtitle entry with timing, raw text, and cleaned text."""

    start_time: float
    end_time: float
    raw_text: str
    cleaned_text: str
    bom: bool = False
    line_ending: str = "\n"
    trailing_blank: bool = False
    ass_header: list[str] | None = None
    ass_metadata: dict[str, str] | None = None


class SubtitleParser:
    """
    Parses .srt and .vtt subtitle files.

    Each subtitle block is returned as a SubtitleBlock containing:
      - start_time:  start timestamp in seconds
      - end_time:    end timestamp in seconds
      - raw_text:    exact unmodified text (tags, colours, brackets preserved)
      - cleaned_text: TextProcessor-normalised version for phonetic analysis

    File-level metadata (BOM, line endings) is attached to the first block
    so the writer can reproduce the original byte-for-byte.
    """

    def __init__(self) -> None:
        self._text_processor = TextProcessor()

    def parse_file(self, filepath: str | Path) -> list[SubtitleBlock]:
        """
        Parse a subtitle file (auto-detect .srt or .vtt).

        Returns:
            List of SubtitleBlock instances.
        """
        path = Path(filepath)
        raw = path.read_bytes()

        # Detect BOM
        bom = raw[:3] == b"\xef\xbb\xbf"
        if bom:
            raw = raw[3:]

        # Detect line ending style
        line_ending = "\r\n" if b"\r\n" in raw else "\n"

        # Decode, normalising line endings for parsing
        text = raw.decode("utf-8").replace("\r\n", "\n")

        if self._is_vtt(text):
            blocks = self._parse_vtt(text)
        elif self._is_ass(text):
            blocks = self._parse_ass(text)
        else:
            blocks = self._parse_srt(text)

        # Attach metadata to first block for round-trip writing
        if blocks:
            blocks[0].bom = bom
            blocks[0].line_ending = line_ending
            # Detect trailing blank line (file ends with double line_ending)
            le_bytes = line_ending.encode("utf-8")
            blocks[0].trailing_blank = raw.endswith(le_bytes + le_bytes)

        return blocks

    @staticmethod
    def _is_vtt(text: str) -> bool:
        return text.strip().upper().startswith("WEBVTT")

    @staticmethod
    def _is_ass(text: str) -> bool:
        return "[events]" in text.lower() or "[script info]" in text.lower()

    def _parse_srt(self, text: str) -> list[SubtitleBlock]:
        """Parse an SRT file into a list of SubtitleBlock instances."""
        blocks: list[SubtitleBlock] = []

        # Split on blank lines (handles \r\n and \n)
        raw_blocks = re.split(r"\n\s*\n", text.strip())

        for chunk in raw_blocks:
            lines = chunk.strip().splitlines()
            if len(lines) < 3:
                continue

            # First line should be the index number (skip it)
            # Second line is the timestamp
            ts_line = lines[1].strip()
            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
                ts_line,
            )
            if not ts_match:
                continue

            start = self._srt_ts_to_seconds(ts_match.group(1))
            end = self._srt_ts_to_seconds(ts_match.group(2))

            # Remaining lines are the raw subtitle text
            raw_text = "\n".join(lines[2:])

            cleaned = self._text_processor.extract_main_text(raw_text)

            blocks.append(SubtitleBlock(start, end, raw_text, cleaned))

        return blocks

    def _parse_vtt(self, text: str) -> list[SubtitleBlock]:
        """Parse a VTT file into a list of SubtitleBlock instances."""
        blocks: list[SubtitleBlock] = []
        lines = text.splitlines()

        # Skip WEBVTT header
        idx = 0
        if lines and lines[0].strip().upper().startswith("WEBVTT"):
            idx = 1

        while idx < len(lines):
            # Skip blank lines between cues
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if idx >= len(lines):
                break

            # Try to find a timestamp line
            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3,6})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3,6})",
                lines[idx].strip(),
            )

            if ts_match:
                start = self._vtt_ts_to_seconds(ts_match.group(1))
                end = self._vtt_ts_to_seconds(ts_match.group(2))
                idx += 1

                # Collect text lines until the next blank line or new cue
                text_lines: list[str] = []
                while idx < len(lines):
                    line = lines[idx]
                    # Stop if we hit a blank line or another cue/timestamp
                    if not line.strip():
                        break
                    if re.match(
                        r"\d{2}:\d{2}:\d{2}\.\d{3,6}\s*-->",
                        line.strip(),
                    ):
                        break
                    text_lines.append(line)
                    idx += 1

                raw_text = "\n".join(text_lines)
                cleaned = self._text_processor.extract_main_text(raw_text)

                blocks.append(SubtitleBlock(start, end, raw_text, cleaned))
            else:
                # Non-timestamp line (settings, IDs, comments) — skip
                idx += 1

        return blocks

    def _parse_ass(self, text: str) -> list[SubtitleBlock]:
        """Parse an ASS file into a list of SubtitleBlock instances.

        Args:
            text: Decoded text of the ASS subtitle.

        Returns:
            List of SubtitleBlock instances.
        """
        blocks: list[SubtitleBlock] = []
        lines = text.splitlines()

        header_lines: list[str] = []
        is_events = False

        for line in lines:
            stripped = line.strip()
            if stripped.upper() == "[EVENTS]":
                is_events = True
                header_lines.append(line)
                continue

            if not is_events:
                header_lines.append(line)
                continue

            if stripped.upper().startswith("FORMAT:"):
                header_lines.append(line)
                continue

            # Process Event lines (Dialogue or Comment)
            match = re.match(r"^(Dialogue|Comment):\s*(.*)$", line, re.IGNORECASE)
            if match:
                prefix = match.group(1)
                fields_str = match.group(2)
                # Standard ASS has 10 fields; the last field is the Text
                fields = fields_str.split(",", 9)
                if len(fields) >= 10:
                    start_str = fields[1].strip()
                    end_str = fields[2].strip()
                    raw_text = fields[9]

                    try:
                        start_time = self._ass_ts_to_seconds(start_str)
                        end_time = self._ass_ts_to_seconds(end_str)
                    except Exception:
                        header_lines.append(line)
                        continue

                    # Map ASS line breaks to standard literal \n
                    internal_raw = raw_text.replace(r"\N", "\n").replace(r"\n", "\n")
                    cleaned = self._text_processor.extract_main_text(internal_raw)

                    block = SubtitleBlock(start_time, end_time, internal_raw, cleaned)
                    block.ass_metadata = {
                        "prefix": prefix,
                        "layer": fields[0],
                        "style": fields[3],
                        "name": fields[4],
                        "margin_l": fields[5],
                        "margin_r": fields[6],
                        "margin_v": fields[7],
                        "effect": fields[8],
                    }
                    blocks.append(block)
                else:
                    header_lines.append(line)
            else:
                header_lines.append(line)

        if blocks:
            blocks[0].ass_header = header_lines

        return blocks

    @staticmethod
    def _ass_ts_to_seconds(ts: str) -> float:
        """Convert an ASS timestamp (H:MM:SS.cs) to seconds.

        Args:
            ts: Raw ASS timestamp string.

        Returns:
            Timestamp converted to absolute seconds (float).
        """
        parts = ts.split(":")
        h = int(parts[0])
        m = int(parts[1])
        sec_parts = parts[2].split(".")
        s = int(sec_parts[0])
        frac_str = sec_parts[1] if len(sec_parts) > 1 else "0"
        # Support arbitrary length centiseconds
        frac_str = frac_str.ljust(3, "0")[:3]
        ms = int(frac_str)
        return h * 3600 + m * 60 + s + ms / 1000.0

    @staticmethod
    def _srt_ts_to_seconds(ts: str) -> float:
        """Convert an SRT timestamp (HH:MM:SS,mmm) to seconds."""
        parts = ts.split(":")
        h = int(parts[0])
        m = int(parts[1])
        sec_parts = parts[2].split(",")
        s = int(sec_parts[0])
        ms = int(sec_parts[1]) if len(sec_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    @staticmethod
    def _vtt_ts_to_seconds(ts: str) -> float:
        """Convert a VTT timestamp (HH:MM:SS.mmm[xxx]) to seconds."""
        parts = ts.split(":")
        h = int(parts[0])
        m = int(parts[1])
        sec_parts = parts[2].split(".")
        s = int(sec_parts[0])
        frac = sec_parts[1] if len(sec_parts) > 1 else "0"
        # Pad or truncate to 6 digits for microseconds
        frac = frac.ljust(6, "0")[:6]
        us = int(frac)
        return h * 3600 + m * 60 + s + us / 1_000_000.0
