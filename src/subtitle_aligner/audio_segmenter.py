"""Audio segmenter — vocal extraction and audio chunking.

Handles isolating vocals from anime audio via UVR and splitting the result
into manageable .wav segments at natural silence boundaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .audio_loader import AudioLoader
from .preprocessing import AudioPreprocessor
from .subtitle_parser import SubtitleBlock

load_dotenv()


@dataclass
class AudioSegment:
    """A single audio segment with timing metadata."""

    filepath: str
    start_time: float  # absolute offset in seconds relative to original
    duration: float
    source_sr: int = 0  # original source sample rate


class AudioSegmenter:
    """Handles vocal extraction and audio segmentation.

    1. Uses ``AudioPreprocessor`` to extract clean vocals via UVR.
    2. Uses Silero VAD on the vocal track to detect silence regions.
    3. Splits audio at silence points near target duration thresholds.
    """

    def __init__(
        self,
        device: str = "cpu",
        target_duration: float = 300.0,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        vad_threshold: float = 0.3,
        uvr_model_dir: Optional[str] = None,
        uvr_model_filename: Optional[str] = None,
    ) -> None:
        self.device = device
        self.target_duration = target_duration
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.vad_threshold = vad_threshold
        self.uvr_model_dir = uvr_model_dir or os.getenv("UVR_MODEL_DIR", "models/uvr")
        self.uvr_model_filename = uvr_model_filename or os.getenv(
            "UVR_MODEL_FILENAME", "6_HP-Karaoke-UVR.pth"
        )
        self._preprocessor: Optional[AudioPreprocessor] = None

    def _get_preprocessor(self) -> AudioPreprocessor:
        """Lazily initialize the AudioPreprocessor."""
        if self._preprocessor is None:
            self._preprocessor = AudioPreprocessor(
                uvr_model_dir=self.uvr_model_dir,
                uvr_model_filename=self.uvr_model_filename,
            )
        return self._preprocessor

    def extract_vocals(self, audio_path: str, output_dir: str = "data") -> str:
        """Extract clean vocal track from audio using UVR.

        Args:
            audio_path: Path to input audio.
            output_dir: Directory to write isolated vocal file.

        Returns:
            Absolute path to the isolated vocal .wav file.
        """
        preprocessor = self._get_preprocessor()
        preprocessor.set_output_dir(output_dir)

        vocal_path = preprocessor.preprocess(audio_path)
        return vocal_path

    def _get_speech_timestamps(
        self,
        audio_path: str,
        sampling_rate: int = 16000,
    ) -> list[dict]:
        """Get speech timestamp regions using AudioPreprocessor's VAD."""
        preprocessor = self._get_preprocessor()
        return preprocessor.get_speech_timestamps(
            audio_path=audio_path,
            sampling_rate=sampling_rate,
            threshold=self.vad_threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    @staticmethod
    def _compute_silent_intervals(
        speech_ts: list[dict], total_duration: float
    ) -> list[tuple[float, float]]:
        """
        Compute silent intervals from speech timestamps.

        Args:
            speech_ts: Speech timestamp dicts from Silero VAD.
            total_duration: Total audio duration.

        Returns:
            List of (start, end) tuples for silent regions.
        """
        if not speech_ts:
            # Entire audio is silent
            return [(0.0, total_duration)]

        intervals: list[tuple[float, float]] = []

        # Leading silence
        first_speech = speech_ts[0]["start"]
        if first_speech > 0.05:
            intervals.append((0.0, first_speech))

        # Gaps between speech regions
        for i in range(len(speech_ts) - 1):
            gap_start = speech_ts[i]["end"]
            gap_end = speech_ts[i + 1]["start"]
            if gap_end - gap_start > 0.05:
                intervals.append((gap_start, gap_end))

        # Trailing silence
        last_speech = speech_ts[-1]["end"]
        if last_speech < total_duration - 0.05:
            intervals.append((last_speech, total_duration))

        return intervals

    @staticmethod
    def _find_nearest_silent_boundary(
        target_time: float,
        silent_intervals: list[tuple[float, float]],
        window: float = 30.0,
    ) -> Optional[float]:
        """Find the silence midpoint closest to the target split time.

        Args:
            target_time: The ideal timestamp (seconds) to split at.
            silent_intervals: List of (start_time, end_time) silence zones.
            window: Maximum allowed distance (seconds) from the target time.

        Returns:
            The midpoint of the closest silence zone, or None if none found.
        """
        best_split: Optional[float] = None
        best_distance = float("inf")

        for interval_start, interval_end in silent_intervals:
            interval_center = (interval_start + interval_end) / 2.0
            distance = abs(interval_center - target_time)

            if distance <= window and distance < best_distance:
                best_distance = distance
                best_split = interval_center

        return best_split

    def _locate_subtitle_gap(
        self,
        subtitles: list[SubtitleBlock],
        target_time: float,
        max_distance: float = 120.0,
    ) -> tuple[float, float] | None:
        """Find the subtitle gap >= 2.0s closest to the target split time.

        Args:
            subtitles: List of parsed SubtitleBlock instances.
            target_time: Target split timestamp (seconds).
            max_distance: Maximum distance (seconds) from the target split mark.

        Returns:
            A tuple of (gap_start, gap_end) if found, otherwise None.
        """
        best_gap = None
        min_dist = float("inf")

        for i in range(len(subtitles) - 1):
            gap_start = subtitles[i].end_time
            gap_end = subtitles[i + 1].start_time
            gap_duration = gap_end - gap_start

            if gap_duration >= 2.0:
                gap_midpoint = (gap_start + gap_end) / 2.0
                dist = abs(gap_midpoint - target_time)

                # Keep the closest gap that sits within our logical search range
                if dist < min_dist and dist <= max_distance:
                    min_dist = dist
                    best_gap = (gap_start, gap_end)

        return best_gap

    def _cross_verify_silence(
        self,
        silent_intervals: list[tuple[float, float]],
        gap_midpoint: float,
        window: float = 30.0,
    ) -> float | None:
        """Find a verified VAD silence >= 1.0s inside a subtitle gap window.

        Args:
            silent_intervals: List of (start_time, end_time) VAD silence zones.
            gap_midpoint: Midpoint of the located subtitle gap (seconds).
            window: Search window (+/- seconds) around the gap midpoint.

        Returns:
            Midpoint of the closest verified VAD silence, or None if none found.
        """
        best_split = None
        min_dist = float("inf")

        for s_start, s_end in silent_intervals:
            s_duration = s_end - s_start
            if s_duration >= 1.0:
                s_midpoint = (s_start + s_end) / 2.0
                dist = abs(s_midpoint - gap_midpoint)

                # Select the closest acoustically quiet point within the window
                if dist <= window and dist < min_dist:
                    min_dist = dist
                    best_split = s_midpoint

        return best_split

    def _find_split_points(
        self,
        audio_path: str,
        total_duration: float,
        subtitles: list[SubtitleBlock] | None = None,
        sampling_rate: int = 16000,
    ) -> list[float]:
        """Find optimal split points using VAD and subtitle cross-verification.

        Args:
            audio_path: Path to the vocal WAV file.
            total_duration: Total duration of the audio in seconds.
            subtitles: Optional list of parsed subtitle blocks.
            sampling_rate: Target VAD sampling rate.

        Returns:
            Sorted list of physical split timestamps.
        """
        speech_ts = self._get_speech_timestamps(audio_path, sampling_rate)
        silent_intervals = self._compute_silent_intervals(speech_ts, total_duration)

        split_points: list[float] = []
        t = self.target_duration

        while t < total_duration - 1.0:
            split_found = False

            # Core Hybrid Path: Use subtitle gaps cross-verified with audio VAD
            if subtitles:
                gap = self._locate_subtitle_gap(subtitles, t)
                if gap is not None:
                    gap_start, gap_end = gap
                    gap_midpoint = (gap_start + gap_end) / 2.0

                    # Look for VAD silences in +/- 30s window around gap midpoint
                    vad_midpoint = self._cross_verify_silence(
                        silent_intervals,
                        gap_midpoint,
                        window=30.0,
                    )

                    if vad_midpoint is not None:
                        # Success: split at verified VAD silence midpoint
                        split_points.append(vad_midpoint)
                    else:
                        # Fallback 2: No VAD silence, split at subtitle boundary
                        split_points.append(gap_midpoint)
                    split_found = True

            # Fallback 1: No subtitle gaps, or subtitles not supplied
            if not split_found:
                # Find the longest VAD silence region in a +/- 30s target window
                best_split = self._find_nearest_silent_boundary(
                    t,
                    silent_intervals,
                    window=30.0,
                )
                if best_split is not None:
                    split_points.append(best_split)
                else:
                    # Final fallback: split exactly at the target time
                    split_points.append(t)

            t += self.target_duration

        # Deduplicate splitting boundaries (min 1 second gap)
        deduped: list[float] = []
        for sp in sorted(split_points):
            if not deduped or (sp - deduped[-1]) >= 1.0:
                deduped.append(sp)

        return deduped

    def segment_audio(
        self,
        audio_path: str,
        output_dir: str,
        target_duration: Optional[float] = None,
        sampling_rate: int = 16000,
        subtitles: Optional[list[SubtitleBlock]] = None,
    ) -> list[AudioSegment]:
        """Segment vocal audio into physical .wav chunks at silence boundaries.

        The output segments preserve the original sample rate of the source
        audio file.

        Args:
            audio_path: Path to the vocal audio file.
            output_dir: Directory to write segmented .wav files.
            target_duration: Target segment duration in seconds (default 300).
            sampling_rate: Target sample rate for VAD analysis.
            subtitles: Optional list of parsed SubtitleBlock instances for
                       hybrid cross-verification splitting.

        Returns:
            List of AudioSegment instances.
        """
        target = target_duration or self.target_duration
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        loader = AudioLoader(audio_path)
        waveform, sr = loader.load_torchaudio()
        total_duration = waveform.shape[1] / sr

        print(
            f"[AudioSegmenter] Segmenting audio: {audio_path} "
            f"({total_duration:.1f}s, {sr} Hz, "
            f"{waveform.shape[0]} ch)"
        )

        split_points = self._find_split_points(
            audio_path, total_duration, subtitles, sampling_rate
        )
        print(f"[AudioSegmenter] Found {len(split_points)} split point(s)")

        segments: list[AudioSegment] = []
        prev_end_sample = 0

        for i, split_time in enumerate(split_points):
            split_sample = int(split_time * sr)
            segment_waveform = waveform[:, prev_end_sample:split_sample]

            filename = f"segment_{i:04d}.wav"
            filepath = os.path.join(output_dir, filename)
            AudioLoader.save_wav(filepath, segment_waveform, sr)

            segments.append(
                AudioSegment(
                    filepath=filepath,
                    start_time=prev_end_sample / sr,
                    duration=segment_waveform.shape[1] / sr,
                    source_sr=sr,
                )
            )
            prev_end_sample = split_sample

        last_waveform = waveform[:, prev_end_sample:]
        filename = f"segment_{len(split_points):04d}.wav"
        filepath = os.path.join(output_dir, filename)
        AudioLoader.save_wav(filepath, last_waveform, sr)

        segments.append(
            AudioSegment(
                filepath=filepath,
                start_time=prev_end_sample / sr,
                duration=last_waveform.shape[1] / sr,
                source_sr=sr,
            )
        )
        return segments

    def process_video(
        self,
        video_path: str,
        output_dir: str = "data",
        target_duration: Optional[float] = None,
        subtitles: Optional[list[SubtitleBlock]] = None,
    ) -> list[AudioSegment]:
        """Full pipeline: extract audio, isolate vocals, then segment.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory for output files.
            target_duration: Target segment duration in seconds.
            subtitles: Optional parsed subtitle blocks for hybrid splitting.

        Returns:
            List of AudioSegment instances.
        """
        video_path = os.path.abspath(video_path)
        # the output dir uses the video name such that the segments
        # are saved on their video folder
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.abspath(output_dir), video_filename)
        os.makedirs(output_dir, exist_ok=True)
        print(
            f"[AudioSegmenter] Processing video: {video_path} With output path at {output_dir}"
        )
        temp_audio = os.path.join(output_dir, f"{video_filename}_extracted.wav")

        AudioLoader.extract_audio_from_video(
            video_path=video_path,
            output_path=temp_audio,
        )

        vocal_path = self.extract_vocals(temp_audio, output_dir)

        segments = self.segment_audio(
            vocal_path,
            output_dir,
            target_duration=target_duration,
            subtitles=subtitles,
        )

        print(f"[AudioSegmenter] Complete: {len(segments)} segment(s) generated")
        return segments
