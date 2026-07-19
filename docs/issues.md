---

### 1. **Title**: Package Scaffolding & Verbatim Subtitle Parsing
*   **Type**: AFK
*   **Blocked by**: None (Can start immediately)
*   **User Stories Covered**: 2, 3, 4, 12
*   **Description**: Create the modern `src/` directory layout. Move the existing files (`preprocessing.py`, `text_processing.py`, `predictor.py`) inside `src/subtitle_aligner/` and refactor their imports. Build the dynamic CLI shell `run_aligner.py` and implement the extended `subtitle_parser.py` which extracts 4-element tuples `(start_time, end_time, raw_text, cleaned_text)` to preserve original formatting.
*   **Verification**: Running the CLI reads any SRT/VTT file, parses it, and writes it back to `outputs/subtitles/` with 100% identical formatting, styles, and text (no timing adjustments yet).

### 2. **Title**: Basic Audio Vocal Extraction & Segmentation
*   **Type**: AFK
*   **Blocked by**: Slice 1
*   **User Stories Covered**: 5, 12
*   **Description**: Implement the basic audio slicing interface in `audio_segmenter.py`. When a video file is supplied, the pipeline runs UVR to isolate vocals, applies Silero VAD, and performs physical chunk splits at raw silent boundaries closest to target intervals (e.g., every 5-10 minutes) to save runtime memory.
*   **Verification**: Running the CLI with a video output successfully populates the `/data` folder with isolated and physically segmented `.wav` vocal chunks.

### 3. **Title**: Hybrid Splitting & Cross-Verification Logic
*   **Type**: AFK
*   **Blocked by**: Slice 2
*   **User Stories Covered**: 6
*   **Description**: Implement the advanced hybrid splitting verification inside `audio_segmenter.py`. Find subtitle gaps near target boundaries, open a local $\pm 30$-second search window in the audio, cross-verify against VAD speech pauses, and execute the physical split. Implement both edge-case fallback rules (no subtitle gaps or no VAD silences found).
*   **Verification**: Unit tests feed complex timelines (with and without silent gaps) to ensure splitting points occur strictly at cross-verified silent transitions.

### 4. **Title**: Phonetic Normalization & ASR Transcription
*   **Type**: AFK
*   **Blocked by**: Slice 1
*   **User Stories Covered**: 10
*   **Description**: Implement `text_normalizer.py` (which runs the clean `TextProcessor` then converts the output to Katakana using `pykakasi`) and `asr_transcriber.py` (which runs ReazonSpeech on the segmented audio chunks).
*   **Verification**: Running a pipeline run returns matching-ready, normalized Katakana strings for both the original subtitles and the reconstructed timeline of ASR text.

### 5. **Title**: Local Sliding-Window Timeline Alignment
*   **Type**: AFK
*   **Blocked by**: Slice 3, Slice 4
*   **User Stories Covered**: 1, 7
*   **Description**: Implement the core sliding-window search engine in `aligner.py`. Search candidate ASR segments in a local $\pm 5$-minute window around each subtitle's original timestamp. Match using Katakana character similarity ratios (`SequenceMatcher.ratio()`). Apply the 3-tier timing adjustment logic (ignore if $<0.2$s, update if $0.2$- $5.0$s, shift if $>5.0$s) and output the synchronized subtitle file.
*   **Verification**: Inputting a shifted subtitle file results in a newly synchronized SRT/VTT file with timing offsets corrected to match actual spoken dialogue.

### 6. **Title**: Alignment Robustness (Fallback Shifting & Scene Insertion)
*   **Type**: AFK
*   **Blocked by**: Slice 5
*   **User Stories Covered**: 8, 9
*   **Description**: Extend `aligner.py` with fallback and insertion handlers. If a subtitle line is unmatched (e.g. theme song or signboards), automatically shift it based on the last-known matched offset. If an ASR segment remains completely unmatched and falls within a blank gap, insert it as a new ASR subtitle line.
*   **Verification**: A modified video with added scenes/silences is aligned successfully; on-screen text remains synchronized, and the missing scenes are populated with generated captions.

### 7. **Title**: Parseable Markdown-Table Logger
*   **Type**: AFK
*   **Blocked by**: Slice 6
*   **User Stories Covered**: 11
*   **Description**: Implement `logger_writer.py` to write execution logs to `outputs/logs/` in a strict Markdown format. The modification metrics use pipe (`|`) delimiters.
*   **Verification**: Running the pipeline creates logs that render beautifully as grid tables in markdown editors, while verifying that a parsing script can successfully split the tables with `line.split('|')` to export clean CSV sheets.

---

