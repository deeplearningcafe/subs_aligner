# Product Requirements Document (PRD): Automated Japanese Subtitle Alignment Tool

## Problem Statement

As a student of the Japanese language utilizing a pitch-accent analysis application, the user suffers from frequently mistimed or shifted anime subtitles (`.srt` and `.vtt` formats). When subtitle timestamps do not accurately align with the actual spoken dialogue, downstream systems—such as pitch-accent extractors—fail to match audio to text correctly.

Manually correcting subtitle timing is extremely time-consuming. Additionally, automated alignment is historically difficult due to:
- Minor variations and errors in speech recognition (ASR) systems.
- Differences between video releases (e.g., television broadcasts containing sponsor blocks vs. clean Blu-ray releases with unrecorded or deleted scenes), leading to sudden, massive timing discrepancies (e.g., 10-second offsets).
- Complex on-screen texts (e.g., signboard translations, opening/closing lyrics) where no spoken audio exists, which often get accidentally deleted or misaligned by traditional automatic aligners.

## Solution

The solution is a command-line utility structured under a modern `src` packaging layout that automatically aligns existing Japanese subtitle timestamps with video/audio speech, while guaranteeing the verbatim preservation of the original subtitle text, styles, and metadata.

The tool combines Ultimate Vocal Remover (UVR) vocal isolation, Silero Voice Activity Detection (VAD), and ReazonSpeech (espnet-asr) transcription. It resolves timeline shifts through a localized, segment-by-segment window search matching Katakana representations of original subtitles against ASR outputs. It splits long-form media at safe, cross-verified silent transitions to optimize memory, automatically inserts generated subtitles for unrecorded scenes, shifts unmatched text blocks based on neighboring offsets, and exports comprehensive, machine-parseable Markdown-table logs.

---

## User Stories

1. As a Japanese pitch-accent learner, I want my anime subtitles to be aligned within a fraction of a second to the actual spoken audio, so that my pitch-accent analysis tools work reliably.
2. As a language student, I want to keep the exact verbatim text, formatting, font colors, and bracketed notes of my original subtitle file, so that I do not lose reading context or translation aids.
3. As an automation developer, I want to pass my input video and subtitle directories dynamically via a command-line interface, so that I can integrate this system into batch scripts or a future graphical user interface.
4. As a system operator, I want the output subtitles and logs to be saved predictably in designated output directories inside the repository, so that I can easily locate and review aligned files.
5. As a developer, I want long video files to be partitioned into manageable chunks at natural pause points before transcribing them, so that I avoid out-of-memory errors on my hardware.
6. As a developer, I want these partition points to be verified by both subtitle gaps and audio silence, so that the audio is never chopped in the middle of a spoken sentence.
7. As an algorithm developer, I want minor timing errors (under 0.2s) to be left untouched, so that human-tailored subtitle reading buffers and lead times are not overwritten by noisy ASR boundaries.
8. As a viewer watching a Blu-ray version with an extra scene, I want the aligner to detect this gap, insert new ASR-transcribed subtitles during the scene, and automatically shift the remaining original subtitles forward by the correct offset.
9. As a viewer, I want signboard translations, quiet whispers, or theme lyrics without active speech detection to remain in the file and shift together with the local dialogue timeline, so that they do not get lost or misaligned.
10. As a developer, I want the comparison engine to convert both subtitles and ASR text to standardized phonetic Katakana before matching, so that orthographic differences (Kanji vs. Kana, alternate spellings) do not break the alignment.
11. As a data analyst, I want the execution logs to be output as strictly formatted Markdown tables, so that they are readable in my text editor while remaining trivial to parse into a CSV or spreadsheet format later.
12. As a package maintainer, I want the entire application to reside within a standard `src/` layout, so that imports are structured reliably and do not conflict with the parent pitch-accent application.

---

## Implementation Decisions

### 1. Repository & Package Architecture
- **Packaging Standard**: The repository is structured using the standard Python `src/` layout. All active modules reside under `src/subtitle_aligner/` to support clean relative imports and packaging.
- **Repository Outputs**: Output files and logs are saved to structured paths within the repository (`outputs/subtitles/` and `outputs/logs/`), while inputs are supplied dynamically as CLI parameters to decouple processing from static folders.

### 2. Module Boundaries & Interfaces

*   **Subtitle Parser** (`subtitle_parser.py`)
    *   *Modification*: Reuses and extends the existing custom parser.
    *   *Behavior*: Parses `.srt` and `.vtt` formats. Instead of returning a 3-tuple that strips text formatting, it outputs a list of 4-element structures containing: `(start_time, end_time, raw_text, cleaned_text)`.
    *   *Data Preservation*: `raw_text` holds the exact, untouched string (preserving tags, font colors, brackets). `cleaned_text` stores the output of the text normalizer.

*   **Audio Segmenter** (`audio_segmenter.py`)
    *   *Behavior*: Extracts a clean vocal track via UVR and applies Silero VAD to identify spoken segments.
    *   *Verification Split Logic*: Partitions the vocal audio into roughly 5-to-10 minute blocks at logical breaks.
    *   *Matching Gap Evaluation*: To find a split boundary near a target time (e.g., 5 minutes):
        1. It identifies a subtitle gap of $\ge 2.0$ seconds.
        2. It searches a $\pm 30$-second audio search window around that gap.
        3. If Silero VAD detects an actual silent gap of $\ge 1.0$ second in that window, it splits the audio at the midpoint.
    *   *Edge Cases*:
        *   *No Subtitle Gap*: If no subtitle gaps exist, it falls back to trusting the VAD silence detection entirely.
        *   *No VAD Pause*: If noise/music obscures the audio silence, it falls back to splitting at the subtitle transition time.

*   **ASR Transcriber** (`asr_transcriber.py`)
    *   *Behavior*: Wraps ReazonSpeech (espnet-asr) execution. Transcribes the segmented audio blocks, tracks relative timings, and maps them back to absolute timelines by adding the segment start-time offset.

*   **Text Normalizer** (`text_normalizer.py`)
    *   *Behavior*: Cleans input text via the existing custom `TextProcessor` (removing brackets, emojis, URLs, and applying `neologdn`). It then uses `pykakasi` to convert the normalized text to uniform Katakana, isolating phonetic moras for distance comparison.

*   **Aligner Core** (`aligner.py`)
    *   *Behavior*: Matches original subtitles with ASR outputs row-by-row.
    *   *Search Window*: Limits search candidate segments to a sliding temporal window of $\pm 5$ minutes around the subtitle's original timestamp.
    *   *Similarity Evaluation*: Measures Katakana text similarity using a localized character ratio via `difflib.SequenceMatcher`.
    *   *Decision Matrix*:
        *   *Best Match $\ge 70\%$ Similarity*:
            *   *Difference $< 0.2$ seconds*: Keep original timestamp (no change).
            *   *Difference $0.2$ to $5.0$ seconds*: Adjust to the ASR segment timing.
            *   *Difference $> 5.0$ seconds*: Apply the offset shift and record a log entry.
        *   *Mismatch / No Match ($< 70\%$ Similarity)*:
            *   If overlapping with another active ASR transcription of conflicting text, skip modification (keep original) to avoid corruption from multi-speaker overlaps.
            *   If quiet (no overlapping speech), preserve the line and shift its timestamp by the last-known successful alignment offset to prevent it from drifting from the timeline.
        *   *ASR Insertion (Unrecorded Scenes)*:
            *   If an ASR segment remains unmatched after aligning all subtitles, and it sits in a silent subtitle gap, insert the transcribed text as a new subtitle line.

*   **Logger Writer** (`logger_writer.py`)
    *   *Behavior*: Records summary stats and granular row modifications.
    *   *Structure*: Follows a strict Markdown format. The modification sections are formatted as Markdown Tables using pipe delimiters (`|`), ensuring uniform column alignment so they can be easily parsed or converted to CSV via regex.

---

## Testing Decisions

### Test Strategy
Tests will prioritize **integration and behavior-driven assertions** over checking internal class implementations, ensuring refactoring can happen without breaking the test suite.

### Key Testing Scenarios & Modules
*   **Subtitle Parser Tests**:
    *   Input a dirty `.srt` / `.vtt` file with rich text tags, emojis, and brackets.
    *   Assert that the parser yields exact `raw_text` strings matching the file, and that formatting is preserved when writing back.
*   **Hybrid Segmenter Tests**:
    *   Verify that audio segment boundaries correctly fall inside silence regions when given synthetic audio with specific spoken segments and background music.
*   **Aligner Core Tests**:
    *   Test alignment using mock subtitle tracks shifted by small offsets ($0.1$s), medium offsets ($2$s), and large offsets containing a 10-second sponsor block.
    *   Assert that small shifts remain unchanged, medium/large shifts are successfully aligned, and unmatched lines (like lyrics) preserve their relative timing via offset propagation.
    *   Assert that unrecorded scenes successfully generate new subtitle blocks in the output.
*   **Log Output Parser Validation**:
    *   Run a mock execution, read the generated `.txt` log file, parse it with a quick python `split('|')` script, and assert that the columns are consistent and contain no malformed rows.

---

## Out of Scope

- **Graphical User Interface (GUI)**: This PRD covers only the command-line interface execution and its underlying core engine.
- **Multilingual LLM Translation**: The translation capabilities inside the repository (e.g., Gemini-based translation in `translate.py`) are decoupled from this alignment feature.
- **Real-Time / Streaming Subtitle Sync**: The alignment is designed purely for local file batch-processing.

---

## Further Notes

- **Morphological Dependencies**: The system relies on existing linguistic configurations within the repository (e.g., `pyopenjtalk` directories and dictionaries utilized by the pitch-accent app). Consolidated module placement prevents directory mapping conflicts.
- **Hardware Profile**: ReazonSpeech processing is resource-intensive. The hybrid segmenter represents a critical performance safeguard, capping execution blocks to moderate durations (5-10 mins) to prevent CUDA or system memory depletion.
