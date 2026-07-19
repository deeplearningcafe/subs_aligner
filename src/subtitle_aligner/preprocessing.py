import os
import shutil

from silero_vad import (
    get_speech_timestamps,
    load_silero_vad,
)
from audio_separator.separator import Separator
from .audio_loader import AudioLoader


class AudioPreprocessor:
    """
    Handles audio preprocessing by applying Voice Activity Detection (VAD)
    to remove long pauses, and Ultimate Vocal Remover (UVR) to extract
    clean vocals from noisy anime audio.
    """

    def __init__(
        self,
        uvr_model_dir: str,
        uvr_model_filename: str,
        output_dir: str = "data",
    ):
        """
        Initializes VAD and UVR models to avoid reloading on every call.

        Args:
            uvr_model_dir (str): Directory where the UVR model is stored.
            uvr_model_filename (str): The specific UVR model filename.
            output_dir (str): Default output directory for UVR results.
        """
        print("Loading Silero VAD model...")
        self.vad_model = load_silero_vad(onnx=True)

        print(f"Loading UVR model: {uvr_model_filename}...")
        self.separator = Separator(
            log_level=10,
            model_file_dir=uvr_model_dir,
            output_single_stem="Vocals",
            output_dir=output_dir,
        )
        self.separator.load_model(model_filename=uvr_model_filename)
        # Track both the active target output_dir and the immutable base dir
        self.output_dir: str = output_dir
        self._base_separator_dir: str = output_dir

    def set_output_dir(self, output_dir: str) -> None:
        """Update the target output directory for the next separation.

        Args:
            output_dir: New target output directory path.
        """
        self.output_dir = output_dir

    def get_speech_timestamps(
        self,
        audio_path: str,
        sampling_rate: int = 16000,
        threshold: float = 0.3,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
    ) -> list[dict[str, float]]:
        """Run Silero VAD on the audio and return speech timestamps.

        Does not physically trim or modify the source audio file.
        """
        loader = AudioLoader(audio_path)
        waveform, sr = loader.load_torchaudio(
            sampling_rate=sampling_rate,
            mono=True,
        )
        waveform = waveform.squeeze(0)

        timestamps = get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=sampling_rate,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        )
        return timestamps

    def _extract_vocals(self, audio_path: str) -> str:
        """Extracts the vocal stem from the audio file using UVR.

        Saves to the base separator folder and moves the file to the target.
        """
        output_files = self.separator.separate(audio_path)

        if not output_files:
            raise RuntimeError("UVR separation failed to return files.")

        vocal_filename = output_files[0]

        generated_path = os.path.join(self._base_separator_dir, vocal_filename)
        final_path = os.path.join(self.output_dir, vocal_filename)

        # Move the file if the active directory has been redirected
        if generated_path != final_path:
            os.makedirs(self.output_dir, exist_ok=True)
            shutil.move(generated_path, final_path)

        return final_path

    def preprocess(self, audio_path: str) -> str:
        """Run UVR vocal extraction pipeline and return the vocal file path."""
        print(f"[Preprocess] Extracting vocals from: {audio_path}")
        loader = AudioLoader(audio_path)
        vocal_path = self._extract_vocals(loader.wav_path)
        print(f"[Preprocess] Done. Clean vocals saved at: {vocal_path}")
        return vocal_path
