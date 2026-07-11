import os
import subprocess
import tempfile
import torch
import torchaudio


class AudioLoader:
    """Handles all audio and video ingestion for the project.

    Converts media formats to standard WAV using ffmpeg, provides clean loading
    via torchaudio, and supports direct audio extraction from video files.
    """

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a source file path.

        Args:
            file_path: Path to the input file (audio or video).
        """
        self.original_path = file_path
        self.wav_path = self._convert_to_wav(file_path)

    def _convert_to_wav(self, file_path: str) -> str:
        """Converts any audio/video file to a temporary WAV file using ffmpeg.

        Args:
            file_path: Path to the source file.

        Returns:
            Path to the temporary wav file.
        """
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".wav":
            return file_path

        print(f"Converting {ext} to .wav using ffmpeg...")
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()

        # Run ffmpeg to convert to 16-bit PCM WAV
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(file_path),
            "-acodec",
            "pcm_s16le",
            temp_wav_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Failed to convert {file_path} to WAV. "
                "Ensure ffmpeg is installed and accessible in your PATH. "
                f"Error: {e}"
            ) from e

        return temp_wav_path

    def load_parselmouth(self):
        """Loads the audio into a parselmouth.Sound object."""
        import parselmouth

        return parselmouth.Sound(self.wav_path)

    def load_torchaudio(
        self,
        sampling_rate: int | None = None,
        mono: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Loads the audio using standard torchaudio.

        Args:
            sampling_rate: Optional sample rate to resample to.
            mono: Whether to downmix multi-channel audio to mono.

        Returns:
            A tuple of (waveform, sample_rate).
        """
        waveform, orig_sr = torchaudio.load(self.wav_path)
        target_sr = sampling_rate if sampling_rate is not None else orig_sr

        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)

        return waveform, target_sr

    @staticmethod
    def save_wav(filepath: str, waveform: torch.Tensor, sample_rate: int) -> None:
        """Saves a PyTorch tensor waveform as a WAV file.

        Args:
            filepath: Destination file path.
            waveform: PyTorch audio tensor.
            sample_rate: Sampling rate of the audio in Hz.
        """
        torchaudio.save(filepath, waveform, sample_rate)

    @staticmethod
    def extract_audio_from_video(
        video_path: str,
        output_path: str,
        target_sr: int | None = None,
        mono: bool = False,
    ) -> None:
        """Extracts audio from a video file directly using ffmpeg.

        Args:
            video_path: Path to the source video file.
            output_path: Destination WAV file path.
            target_sr: Optional target sample rate.
            mono: If True, downmixes extracted audio to a single channel.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
        ]
        if target_sr is not None:
            cmd.extend(["-ar", str(target_sr)])
        if mono:
            cmd.extend(["-ac", "1"])
        cmd.append(str(output_path))

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg failed to extract audio from video. Error:\n{err_msg}"
            ) from e
