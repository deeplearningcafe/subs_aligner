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
import torch
import numpy as np

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from subtitle_aligner.aligner import SubtitleAligner
from subtitle_aligner.asr_transcriber import ASRTranscriber
from subtitle_aligner.audio_segmenter import AudioSegmenter
from subtitle_aligner.logger_writer import LoggerWriter
from subtitle_aligner.subtitle_parser import SubtitleParser
from subtitle_aligner.subtitle_writer import SubtitleWriter

torch.manual_seed(46)
np.random.seed(46)


def detect_format(filepath: Path) -> str:
    """Detect whether a subtitle file is SRT, VTT, or ASS."""
    ext = filepath.suffix.lower()
    if ext == ".ass":
        return "ass"
    text = filepath.read_text(encoding="utf-8-sig")
    if text.strip().upper().startswith("WEBVTT"):
        return "vtt"
    if "[events]" in text.lower() or "[script info]" in text.lower():
        return "ass"
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
    parser.add_argument(
        "--align-mode",
        default="local_ctc",
        choices=["local_ctc", "global"],
        help="Device for audio processing (default: cpu).",
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

        # Identify the first matching video file using lazy evaluation
        video_extensions = {".mp4", ".mkv", ".webm"}
        video_path = next(
            (
                p
                for p in video_dir.iterdir()
                if p.is_file() and p.suffix.lower() in video_extensions
            ),
            None,
        )
        if not video_path:
            print("Error: No video files found in --video-dir")
            sys.exit(1)

        # Find subtitle file
        subtitle_files = []
        for ext in ("*.srt", "*.vtt", "*.ass"):
            subtitle_files.extend(subtitle_dir.glob(ext))
        if not subtitle_files:
            print(f"No .srt, .vtt, or .ass files found in {subtitle_dir}")
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
        aligner = SubtitleAligner(device=args.device, mode=args.align_mode)
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

        # Step 6: Generate structured log file inside a folder per transcription
        transcription_dir_name = f"{video_path.stem}_{subtitle_path.stem}"
        transcription_dir_name = LoggerWriter._sanitize_filename(transcription_dir_name)

        output_logs = _project_root / "outputs" / "logs" / transcription_dir_name
        output_logs.mkdir(parents=True, exist_ok=True)

        logger_writer = LoggerWriter(
            video_name=video_path.stem,
            subtitle_name=subtitle_path.stem,
            output_dir=str(output_logs),
        )
        logger_writer = LoggerWriter(
            video_name=video_path.stem,
            subtitle_name=subtitle_path.stem,
            output_dir=str(output_logs),
        )
        entries, warnings = logger_writer.prepare_log_data(matches)
        log_path = logger_writer.write_log(entries, warnings)
        print(f"[Align]   -> Written log to {log_path}")

        # Step 7: Store the ASR prediction subtitles for debugging
        asr_sub_path = output_logs / f"asr_prediction.{fmt}"
        transcriber.write_aligned(asr_segments, asr_sub_path, fmt)
        print(f"[Align]   -> Written ASR prediction subtitles to {asr_sub_path}")
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
    for ext in ("*.srt", "*.vtt", "*.ass"):
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
