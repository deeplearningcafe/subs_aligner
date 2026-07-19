# Japanese Subtitle Aligner

**Japanese Subtitle Aligner** is a specialized tool designed to automatically
correct and synchronize Japanese subtitles (`.srt`, `.vtt`, `.ass`) with spoken
dialogue in video and audio files.

Using Voice Activity Detection (VAD), Japanese automatic speech recognition
(ASR), and a high-resolution phonetic forced-alignment engine, it improves
precise character and syllable boundaries. The objetive is to create a
tool for cleaning subtitle alignments, generating cropped speech
datasets for Text-to-Speech (TTS) and ASR training.

---

## 🌟 Key Features

### 1. Robust Preprocessing & Verification
*   **Vocal Isolation (UVR):** Integrates Ultimate Vocal Remover models via
    `audio-separator` to isolate vocals from music and sound effects.
*   **ASR Hallucination Filtering:** Calculates the Voice Activity Ratio (VAD
    overlap) of ASR segments to automatically suppress false transcriptions.
*   **Speech Hypothesis Snapping:** By combining VAD detection and the
    pydomino phoneme aligner, subtitles are snapped and padded to accurate
    boundaries.

### 2. Validation-Driven Dynamic Alignment Loop
*   **High-Resolution Forced Alignment:** Converts Kanji and Hiragana subtitle
    texts to 39 standard JEIDA phonetic tokens using `pyopenjtalk` G2P, and
    performs Viterbi search on the audio using `pydomino`.

### 3. Clean Final Outputs
*   **Linear Interpolation Fallback:** Falls back to safe, non-overlapping,
    monotonic timeline interpolation if forced alignment fails validation.
*   **Precise Trimming:** Adjusts the start end boundaries of subtitle cards
    precisely to the start and ending frames of the last spoken phoneme + 100ms,
    clamped to the next card's start.

---

## 📦 Installation

You can install the dependencies in your environment as follows:

### Prerequisites
* Python 3.12+
* PyTorch 2.8+
* [pydomino](https://github.com/DwangoMediaVillage/pydomino)
* [ReazonSpeech](https://github.com/reazon-research/ReazonSpeech)
* `torchaudio`, `pykakasi`, `pyopenjtalk`, `safetensors`

```bash
# Clone the repository
git clone https://github.com/deeplearningcafe/subs_aligner
cd subtitle_aligner

# Install packages using pip
pip install -e .
```

### Required Checkpoints
To run the alignment pipeline successfully, download the following model
checkpoints and configure them in your `.env` file:

1.  **VAD Engine:** Silero VAD (Loaded on the fly via `silero_vad`).
2.  **ASR Model:** `reazon-research/japanese-wav2vec2-large-rs35kh`.
3.  **UVR Model:** `6_HP-Karaoke-UVR.pth`.
4.  **Pydomino(optional):** `phoneme_transition_model.onnx`.

Place them in your checkpoints path and update your configuration:

```env
ALIGNER_MODEL_PATH="checkpoints/pydomino.onnx"
UVR_MODEL_DIR="checkpoints/uvr"
UVR_MODEL_FILENAME="6_HP-Karaoke-UVR.pth"
```

---

## 🚀 Usage

You can run the application directly using the CLI entry point `run_aligner.py`:

### 1. Run the Full Alignment Pipeline
Extracts vocals, chunks audio, transcribes with ASR, performs local CTC
 forced alignment, and saves the output subtitles alongside a markdown log:

```bash
python run_aligner.py \
    --video-dir "data/video/" \
    --subtitle-dir "data/subtitles/" \
    --align \
    --align-mode "local_ctc" \
    --device "cuda"
```

### 2. Standalone Audio Processing Mode
If you only need to extract isolated vocals and segment the target media into
5-minute contiguous physical chunks at clean silent boundaries:

```bash
python run_aligner.py \
    --audio \
    --video-file "data/video/episode_01.mp4" \
    --target-duration 300.0 \
    --device "cpu"
```

---

## TODO
A complete list of future steps can be found at [TODO](/docs/TODO.md)

*   **Mora Length Verification:** Flags and dynamically retries alignment if
    any syllable (mora) duration registers as anomalous (<30ms or >350ms).
*   **Dynamic Prompt Corrections:** Resolves speech pauses via dynamic `pau`
    token insertion, and handles phonetic contractions by auto-pruning
    unpronounced phonemes.
*   **Kanji Polyphony Solver:** Safely swaps reading candidates for names,
    variations (e.g., "私" pronounced as *watashi* vs. *watakushi*), and
    re-aligns.


## References

* **Ultimate Vocal Remover**: [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
* **Pydomino**: [pydomino](https://github.com/DwangoMediaVillage/pydomino)
* **ReazonSpeech**: [ReazonSpeech](https://github.com/reazon-research/ReazonSpeech)

## Author

[aipracticecafe](https://github.com/deeplearningcafe)
[aipracticecafe-codeberg](https://codeberg.org/aipracticecafe)

## 🔒 Privacy & Scope

This project runs completely locally. No audio, video, or transcription
data is ever transmitted to remote APIs or third-party servers. It is designed
exclusively for personal educational research and text-to-speech dataset
generation.

## License

This project is licensed under the MIT License [LICENSE](LICENSE.txt).
