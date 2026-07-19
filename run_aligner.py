#!/usr/bin/env python3
"""CLI entry point for the subtitle aligner.

Usage:
    python run_aligner.py --video-dir <dir> --subtitle-dir <dir>
    python run_aligner.py --audio --video-file <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so the src package is importable.
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.subtitle_aligner.aligner import SubtitleAligner
from src.subtitle_aligner.asr_transcriber import ASRTranscriber
from src.subtitle_aligner.audio_segmenter import AudioSegmenter
from src.subtitle_aligner.logger_writer import LogEntry, LogWarning, LoggerWriter
from src.subtitle_aligner.subtitle_parser import SubtitleParser
from src.subtitle_aligner.subtitle_writer import SubtitleWriter


def detect_format(filepath: Path) -> str:
    """Detect whether a subtitle file is SRT or VTT by its first line."""
    text = filepath.read_text(encoding="utf-8-sig")
    if text.strip().upper().startswith("WEBVTT"):
        return "vtt"
    return "srt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subtitle Aligner — parse and verify subtitle preservation."
    )
    parser.add_argument(
        "--video-dir",
        help="Path to the directory containing video files.",
    )
    parser.add_argument(
        "--subtitle-dir",
        help="Path to the directory containing subtitle files.",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Run audio processing pipeline (vocal extraction + segmentation).",
    )
    parser.add_argument(
        "--video-file",
        help="Path to a single video file for audio processing.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=300.0,
        help="Target segment duration in seconds (default: 300 = 5 min).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for audio processing (default: cpu).",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Run full alignment pipeline (audio + transcription + align).",
    )
    args = parser.parse_args()

    # ── Audio processing mode ───────────────────────────────────────
    if args.audio:
        if not args.video_file:
            print("Error: --video-file is required with --audio")
            sys.exit(1)

        video_path = Path(args.video_file)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)

        # Ensure data output directory exists
        data_dir = _project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        segmenter = AudioSegmenter(
            device=args.device,
            target_duration=args.target_duration,
        )
        segments = segmenter.process_video(
            str(video_path),
            output_dir=str(data_dir),
            target_duration=args.target_duration,
        )
        print(f"\n[AudioSegmenter] Generated {len(segments)} segment(s)")
        for seg in segments:
            print(
                f"  {seg.filepath}: offset={seg.start_time:.2f}s, "
                f"duration={seg.duration:.2f}s"
            )
        return

    # ── Alignment mode ──────────────────────────────────────────────
    if args.align:
        if not args.video_dir or not args.subtitle_dir:
            print("Error: --video-dir and --subtitle-dir are required with --align")
            sys.exit(1)

        video_dir = Path(args.video_dir)
        subtitle_dir = Path(args.subtitle_dir)

        if not video_dir.exists():
            print(f"Error: Video directory not found: {video_dir}")
            sys.exit(1)
        if not subtitle_dir.exists():
            print(f"Error: Subtitle directory not found: {subtitle_dir}")
            sys.exit(1)

        output_subtitles = _project_root / "outputs" / "subtitles"
        output_subtitles.mkdir(parents=True, exist_ok=True)

        # Find video file
        video_files = list(video_dir.glob("*.mp4"))
        if not video_files:
            video_files = list(video_dir.glob("*.mkv"))
        if not video_files:
            print("Error: No video files (.mp4/.mkv) found in --video-dir")
            sys.exit(1)
        video_path = video_files[0]

        # Find subtitle file
        subtitle_files = []
        for ext in ("*.srt", "*.vtt"):
            subtitle_files.extend(subtitle_dir.glob(ext))
        if not subtitle_files:
            print(f"No .srt or .vtt files found in {subtitle_dir}")
            sys.exit(1)
        subtitle_path = subtitle_files[0]

        # Step 1: Parse subtitles
        print(f"[Align] Parsing subtitles: {subtitle_path.name}")
        subtitle_parser = SubtitleParser()
        subtitles = subtitle_parser.parse_file(subtitle_path)
        print(f"[Align]   -> Parsed {len(subtitles)} subtitle block(s)")

        # Step 2: Process audio (extract vocals + segment)
        print(f"[Align] Processing audio: {video_path.name}")
        segmenter = AudioSegmenter(
            device=args.device,
            target_duration=args.target_duration,
        )
        segments = segmenter.process_video(
            str(video_path),
            output_dir=str(_project_root / "data"),
            target_duration=args.target_duration,
            subtitles=subtitles,
        )
        print(f"[Align]   -> Generated {len(segments)} audio segment(s)")

        # Step 3: Transcribe audio segments
        print("[Align] Transcribing audio segments...")
        transcriber = ASRTranscriber(device=args.device)
        asr_segments = transcriber.transcribe_segments(segments)
        print(f"[Align]   -> {len(asr_segments)} ASR transcription segment(s)")

        # Step 4: Align subtitles with ASR
        print("[Align] Running alignment engine...")
        aligner = SubtitleAligner(device=args.device)
        aligned_blocks, matches = aligner.align(subtitles, asr_segments)

        # Summary
        keep_count = sum(1 for m in matches if m.action == "keep")
        adjust_count = sum(1 for m in matches if m.action == "adjust")
        shift_count = sum(1 for m in matches if m.action == "shift")
        print(
            f"[Align]   -> Kept: {keep_count}, Adjusted: {adjust_count}, "
            f"Shifted: {shift_count}"
        )

        # Step 5: Write aligned output
        fmt = detect_format(subtitle_path)
        output_path = output_subtitles / subtitle_path.name
        aligner.write_aligned(aligned_blocks, output_path, fmt=fmt)
        print(f"[Align]   -> Written aligned subtitles to {output_path}")

        # Step 6: Generate log file
        output_logs = _project_root / "outputs" / "logs"
        output_logs.mkdir(parents=True, exist_ok=True)

        video_base = video_path.stem
        subtitle_base = subtitle_path.stem

        logger_writer = LoggerWriter(
            video_name=video_base,
            subtitle_name=subtitle_base,
            output_dir=str(output_logs),
        )

        entries: list[LogEntry] = []
        warnings: list[LogWarning] = []

        for m in matches:
            entry = LogEntry(
                subtitle_index=m.subtitle_index,
                action=m.action,
                original_start=m.original_start,
                original_end=m.original_end,
                new_start=m.new_start,
                new_end=m.new_end,
                timing_difference=m.timing_difference,
                similarity=m.similarity,
                text=m.asr_segment.text if m.asr_segment is not None else "",
            )
            entries.append(entry)

            # Warning: large shift
            if m.action == "shift":
                warnings.append(
                    LogWarning(
                        warning_type="shift",
                        subtitle_index=m.subtitle_index
                        if m.subtitle_index >= 0
                        else None,
                        description=(
                            f"Large timing shift detected: {m.timing_difference:.2f}s"
                        ),
                    )
                )

            # Warning: inserted ASR scene
            if m.action == "inserted":
                warnings.append(
                    LogWarning(
                        warning_type="inserted",
                        subtitle_index=None,
                        description=(
                            f"ASR scene inserted at {m.new_start:.2f}s–{m.new_end:.2f}s"
                        ),
                    )
                )

            # Warning: low similarity but kept
            if m.action == "keep" and m.similarity < 0.5 and m.similarity > 0.0:
                warnings.append(
                    LogWarning(
                        warning_type="low_similarity",
                        subtitle_index=m.subtitle_index
                        if m.subtitle_index >= 0
                        else None,
                        description=(
                            f"Low similarity match ({m.similarity:.2f}) "
                            f"kept with original timing"
                        ),
                    )
                )

        log_path = logger_writer.write_log(entries, warnings)
        print(f"[Align]   -> Written log to {log_path}")
        return

    # ── Subtitle processing mode ────────────────────────────────────
    if not args.video_dir or not args.subtitle_dir:
        print("Error: --video-dir and --subtitle-dir are required")
        sys.exit(1)

    video_dir = Path(args.video_dir)
    subtitle_dir = Path(args.subtitle_dir)

    # Create output directories
    output_subtitles = _project_root / "outputs" / "subtitles"
    output_subtitles.mkdir(parents=True, exist_ok=True)

    # Find all subtitle files
    subtitle_files = []
    for ext in ("*.srt", "*.vtt"):
        subtitle_files.extend(subtitle_dir.glob(ext))

    if not subtitle_files:
        print(f"No .srt or .vtt files found in {subtitle_dir}")
        return

    subtitle_parser = SubtitleParser()
    subtitle_writer = SubtitleWriter()

    for subtitle_path in sorted(subtitle_files):
        fmt = detect_format(subtitle_path)
        output_path = output_subtitles / subtitle_path.name

        print(f"Processing: {subtitle_path.name} (format: {fmt})")

        # Parse
        blocks = subtitle_parser.parse_file(subtitle_path)
        print(f"  -> Parsed {len(blocks)} subtitle block(s)")

        # Write back
        subtitle_writer.write_blocks(blocks, output_path, fmt=fmt)
        print(f"  -> Written to {output_path}")

        # Verify text preservation (raw_text round-trip)
        original_text = subtitle_path.read_text(encoding="utf-8-sig")
        written_text = output_path.read_text(encoding="utf-8-sig")

        # Compare content, ignoring line-number re-indexing differences
        # For SRT: strip line numbers and compare text content
        if fmt == "srt":
            orig_lines = original_text.strip().splitlines()
            written_lines = written_text.strip().splitlines()

            def strip_line_numbers(lines: list[str]) -> list[str]:
                """Remove sequence number lines (standalone integers)."""
                result = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.isdigit():
                        continue
                    result.append(line)
                return result

            orig_content = "\n".join(strip_line_numbers(orig_lines))
            written_content = "\n".join(strip_line_numbers(written_lines))

            if orig_content == written_content:
                print("  ✓ Byte-for-byte content preserved (ignoring line numbers)")
            else:
                print(f"  ✗ Content mismatch detected!")
                sys.exit(1)
        else:
            # VTT: WEBVTT header may differ slightly; compare subtitle content
            orig_content = "\n".join(
                l
                for l in original_text.strip().splitlines()
                if not l.strip().upper().startswith("WEBVTT")
            )
            written_content = "\n".join(
                l
                for l in written_text.strip().splitlines()
                if not l.strip().upper().startswith("WEBVTT")
            )

            if orig_content == written_content:
                print(f"  ✓ Byte-for-byte content preserved")
            else:
                print(f"  ✗ Content mismatch detected!")
                sys.exit(1)


if __name__ == "__main__":
    main()
