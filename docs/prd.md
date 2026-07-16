## Problem Statement

Anime and video subtitles (in formats such as `.srt`, `.vtt`, and `.ass`) are frequently mistimed, drifted, or poorly aligned with spoken audio. This poses a major barrier for downstream natural language applications, such as extracting paired speech and text datasets for Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) model training, or performing pitch-accent analysis.

Automated alignment is historically difficult due to:
*   **ASR Hallucination**: Generative speech-to-text models (such as Whisper or ReazonSpeech) fabricate words during silent gaps, sound effects (SFX), or scenes with heavy background music (BGM).
*   **One-to-Many Mappings**: Human-tailored subtitles split single spoken sentences into multiple short cards for readability, whereas ASR transcriptions output a single continuous block of text, leading to overlapping or collapsed boundaries.
*   **Kanji Polyphony**: Standard G2P (Grapheme-to-Phoneme) conversion struggles with context-dependent Japanese readings (e.g., "私" pronounced as *watashi*, *watakushi*, or *atashi* depending on the character's personality/archetype).
*   **Boundary Overestimation**: ASR segments often overshoot, leaving trailing silence at the end of subtitles. This degrades downstream TTS/ASR model training datasets where audio segments must be tightly cropped around active speech.

---

## Solution

The solution is a multi-tiered, validation-driven alignment pipeline. Instead of relying on raw generative ASR timings, the pipeline uses ASR solely as a "rough container anchor" and applies a Japanese phonetic forced-alignment engine (`pydomino`) directly on the original subtitle text.

The system leverages Voice Activity Detection (VAD) to filter hallucinations and establish speech boundaries. It implements a **Validation-Driven Dynamic Prompt Alignment Loop** that detects abnormal mora lengths (indicating alignment drift) and automatically attempts to resolve them. It corrects text-audio mismatches using localized phonetic searches against the ASR stream and a lightweight polyphony lookup dictionary. Finally, the tool exports synchronized subtitles with tight tail-trimmed endings (発音終了 + 100ms) and outputs highly detailed, parseable Markdown logs to support automated dataset cleaning.

---

## User Stories

1. As a machine learning engineer, I want to extract tightly trimmed speech-text pairs from anime, so that I can train high-quality TTS and ASR models without silent buffers or trailing noise.
2. As a language student, I want my aligned subtitles to drop off screen exactly 100 milliseconds after speech ends, so that the timing matches the physical rhythm of the dialogue.
3. As an operator, I want unmatched ASR segments occurring during heavy music or action scenes to be classified as hallucinations and discarded, so that they do not generate false subtitles.
4. As a developer, I want a single 5-minute physical audio segment to remain contiguous in memory, so that I do not suffer from complex file-handle management or timeline drift.
5. As a developer, I want an ASR segment to be verified against logical VAD speech intervals before running phonetic alignment, so that silent lead-ins are pre-trimmed and consonant cutoffs are avoided via safeland buffers.
6. As a viewer, when a single spoken line is split into three separate subtitle blocks, I want them to appear sequentially in perfect sync with the character's natural speech rate, rather than overlapping or appearing all at once.
7. As an algorithm developer, if a dramatic pause stretches a mora beyond realistic speaking rates, I want the aligner to dynamically test inserting silence tokens (`pau`) before and after that mora to find the correct alignment path.
8. As a developer, if a character contracts their speech (e.g., saying "けど" instead of "だけど" written in the subtitle), I want the system to detect this collapse and prune the unpronounced phonemes from the alignment prompt.
9. As a developer, if a Kanji reading is converted incorrectly by G2P (e.g., a formal character pronouncing "私" as *watakushi* but G2P outputting *watashi*), I want the system to swap the G2P reading using a lightweight polyphony map and retry the alignment.
10. As an operator, if a subtitle fails all phonetic alignment retries, I want the system to fall back to a linear timeline interpolation to keep the block safe from massive drifts, while logging the fallback event.
11. As a user processing raw YouTube auto-generated subtitles, I want a dedicated flag (`--youtube-subs`) to ignore the original timestamps and rebuild human-readable, beautifully segmented subtitles from scratch using VAD boundaries and maximum line-length constraints.
12. As a data analyst, I want the Markdown log file to include a "Method" column specifying how each segment was resolved, so that I can easily filter out fallback entries from my machine learning training pool.

---

## Implementation Decisions

### 1. Unified Pipeline Architecture
The system will process input media in a linear, logical flow divided into several clean stages.

```
[Video/Audio] ─> [UVR Vocals] ─> [5-Min Contiguous Chunking]
                                           │
                                           ▼
[VAD Active Intervals] <── [VAD Post-Verification] ──> [ASR Decodes]
                                           │
                                           ▼
                                [Rough Container Snap]
                                           │
                                           ▼
                                [G2P Subtitle Phonemes]
                                           │
                                           ▼
                           [pydomino Iterative Retries]
                                           │
                             (Success) ├──> [Tail Trim +100ms]
                             (Failure) └──> [Linear Interpolation]
                                           │
                                           ▼
                                 [Markdown Table Logs]
```

### 2. Module Specifications

#### ASR Container Snapping & VAD Verification
*   **VAD Pre-filtering (Chunking)**: The system continues to divide long media into 5-minute contiguous physical chunks at clean silent boundaries. No smaller physical files are generated.
*   **VAD Post-verification**: For each segment returned by ASR, the system calculates the Voice Activity Ratio:
    $$\text{VAD\_Ratio} = \frac{\text{Speech Duration within Segment}}{\text{Total Segment Duration}}$$
    If this ratio is below a threshold (e.g., 25%), the ASR segment is flagged as a hallucination and discarded.
*   **Speech Hypothesis Snapping**: For validated ASR segments, the system crops the boundaries to match the edges of overlapping VAD active intervals, adding a safe leading and trailing padding buffer of `150ms` to prevent cutting off weak starting/ending consonants (e.g., voiceless fricatives like *s*, *sh*, *h*).

#### Validation-Driven Dynamic Prompt Alignment Loop
*   **Validation Thresholds**: The system monitors the output mora duration $D$ returned by `pydomino`.
    *   Minimum duration $T_{\text{min}} = 30\text{ms}$ (approximately 3 audio frames).
    *   Maximum duration $T_{\text{max}} = 350\text{ms}$ (excluding actual deliberate pauses).
*   **Anomaly Correction Loop**:
    1.  **Too Long ($D > T_{\text{max}}$)**: The system suspects an unannotated breath or dramatic pause. It generates two modified phoneme sequences: one with `pau` placed before the anomalous mora, and one with `pau` placed after. It runs `pydomino` on both and selects the sequence that minimizes the anomaly.
    2.  **Too Short ($D < T_{\text{min}}$)**: The system suspects a contraction or deletion. It extracts the collapsed mora $M_k$ and its neighbors $M_{k-1}$ and $M_{k+1}$.
        *   It searches the ASR phonetic stream. If $M_{k-1}$ and $M_{k+1}$ appear contiguously without $M_k$, $M_k$ is deleted from the sequence, and `pydomino` is re-run.
        *   If there is a different phoneme $X$ between them, $M_k$ is replaced with $X$ and re-run.
        *   If multiple matches occur, the search window expands to 4 neighboring moras to anchor the context.
    3.  **Kanji Mismatch**: If the subtitle text matches the ASR text but the readings collapse, the system extracts the word and looks up alternative pronunciations using a static, domain-specific anime polyphony dictionary (e.g., `{"私": ["わたし", "あたし", "わたくし"]}`). It replaces the G2P output and re-runs `pydomino`.
    4.  **Fallback**: If validation fails after 3 attempts, the system falls back to **Linear Timeline Interpolation** within the snapped container, mapping original subtitle timings to the adjusted container boundaries monotonically to guarantee zero overlaps.

#### YouTube Subtitle Rebuilding Mode (`--youtube-subs`)
*   If active, the system completely ignores the original subtitle timing cards.
*   It merges the verified ASR stream and slices it into new subtitle segments based on VAD-detected silent intervals.
*   It formats and splits the segments into readable cards respecting a maximum limit of 30 characters and a minimum display time of 1.0 second.

#### Tail-Trimming
*   The system sets the final subtitle card end time to:
    $$\text{New\_End} = \min\left(T_{\text{speech\_end}} + 100\text{ms},\ S_{\text{next}}.\text{start\_time}\right)$$
    where $T_{\text{speech\_end}}$ is the ending timestamp of the last non-`pau` phoneme spoken in that card as returned by `pydomino`.

### 3. Log Metadata Extension
The `LoggerWriter` is updated to include an extra column, `Method`, in the `# 変更詳細 (Details)` Markdown table.

| Column | Type | Description |
|---|---|---|
| `#` | Integer | 1-based subtitle card index |
| `Action` | String | `keep`, `adjust`, `shift`, `inserted` |
| `Method` | String | `pydomino-v1`, `pydomino-v2(pau)`, `pydomino-v2(pruned)`, `linear-fallback`, `youtube-rebuild` |
| `Original Start (s)`| Float | Original starting time |
| `Original End (s)` | Float | Original ending time |
| `New Start (s)` | Float | Aligned starting time |
| `New End (s)` | Float | Aligned ending time |
| `Timing Diff (s)` | Float | Absolute starting offset |
| `Similarity` | Float | Phonetic match confidence |
| `Text` | String | Truncated subtitle text |

---

## Testing Decisions

### Test Strategy & Seams
Tests must verify the external behavior of the alignment and correction loop rather than internal state transitions, ensuring refactoring can happen cleanly.

The primary test seam is the **`SubtitleAligner.align()`** method. By passing controlled inputs (synthetic subtitle blocks, ASR segments with simulated drift, mock VAD active intervals, and mocked `pydomino` outputs), we can assert exact boundary snap calculations, prompt retry conditions, and fallback results.

### Core Testing Scenarios
1.  **VAD Hallucination Suppression Test**:
    *   *Input*: An ASR segment placed in a region with 0% VAD overlap.
    *   *Assertion*: Assert that the segment is discarded and no incorrect insertions or alignment shifts occur.
2.  **One-to-Many Multi-Block Alignment Test**:
    *   *Input*: 3 consecutive subtitle cards matched to 1 long ASR segment.
    *   *Assertion*: Verify that `pydomino` correctly resolves individual boundaries, or that the system falls back to linear interpolation without any overlapping start/end times.
3.  **Prompt-Tuning Loop Tests**:
    *   *Too Long*: Input an abnormally long mora. Verify that `pydomino` runs again with `pau` inserted and recovers a normal mora length.
    *   *Too Short/Contraction*: Input a subtitle with "だけど" where ASR contains "けど". Verify that the system identifies the collapse of "だ" via localized anchor search, prunes it, and successfully aligns.
    *   *Polyphony Mismatch*: Input a subtitle with "私" (*watakushi*) where G2P defaulted to *watashi*. Verify that the alternative dictionary lookup replaces the reading and retries successfully.
4.  **Tail Trim Precision Test**:
    *   *Input*: A subtitle segment ending at 15.0s, but speech ends at 13.5s.
    *   *Assertion*: Assert that the new end time is exactly 13.6s (13.5s + 100ms padding) and does not overlap the next card's start time.
5.  **Log Parseability Test**:
    *   *Action*: Execute a mock pipeline, parse the output Markdown table via pipe-splitting, and assert that the `Method` column is correctly populated and parseable.

---

## Out of Scope

*   **Real-time streaming synchronization**: The alignment tool is strictly a batch-processing, local command-line utility.
*   **Foreign Language Translation Alignment**: Subtitle parsing, normalisation, and G2P processing are tailored strictly for the Japanese phonology domain.
*   **Acoustic Feature Extraction Tuning**: We do not retrain the underlying `pydomino` ONNX model weights; we only optimize the input phonetic sequence (G2P prompt tuning) and validate its output.

---

## Further Notes

*   **UVR Integration**: Running the UVR preprocessor is essential to obtain clean vocal stems. Heavy background music in action scenes degrades `pydomino`'s transition probability peaks; vocal isolation acts as a crucial accuracy safeguard.
*   **High-Fidelity Dataset Filtering**: The inclusion of the `Method` column directly supports downstream ML operations. Developers can filter their output metadata to use only `pydomino-v1` and `pydomino-v2` segments for building extremely high-quality, noise-free datasets for text-to-speech model training.
