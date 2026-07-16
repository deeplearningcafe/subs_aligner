---

# 01 — Core `pydomino` Integration & Japanese Phonetic Wrapper

**What to build:**
A Japanese phonetic forced-alignment interface that takes mono WAV audio and a subtitle's text card, performs G2P (Grapheme-to-Phoneme) tokenization to extract the standard 39 JEIDA phonetic tokens, runs the `pydomino` transition-event model, and outputs sub-segment phoneme-level start and end timings.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Implement a wrapper that takes raw Japanese subtitle text, normalizes it, and converts it to a clean 39 JEIDA phoneme token sequence wrapped by leading and trailing `pau` (silence) tokens.
- [ ] Implement the `_align_pydomino` logic to pass the 16kHz mono audio waveform and phoneme sequence to the `pydomino` Viterbi engine.
- [ ] Ensure that a minimum frame allocation of 3 frames (30ms) is configured for Viterbi stability.
- [ ] Write integration tests verifying that aligning a short, clean vocal WAV with its corresponding text returns a chronological list of phoneme segments with precise millisecond-level start and end times.

---

# 02 — VAD-Based Post-Verification & Speech Hypothesis Snapping

**What to build:**
A robust pre-alignment boundary optimization and suppression layer that filters out generative ASR hallucinations using Voice Activity Detection (VAD) and snaps rough ASR container timelines to the true boundary of spoken audio.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Implement VAD post-verification that calculates the `VAD_Ratio` (ratio of active VAD speech duration to the total segment duration) on each raw ASR output segment.
- [ ] Automatically discard any ASR segments with a `VAD_Ratio` below 25% to eliminate background music (BGM) and noise-induced hallucinations.
- [ ] Implement the Speech Hypothesis Snapping logic: Crop the starting and ending times of valid ASR segments to snap exactly to the outer boundaries of overlapping VAD speech intervals.
- [ ] Apply a safe leading and trailing padding buffer of exactly 150 milliseconds to the snapped boundaries to protect weak starting and ending consonants (e.g., voiceless fricatives *s*, *sh*, *h*) from cutoff.
- [ ] Write integration tests verifying that hallucinated segments in silence are suppressed, and that valid speech boundaries are padded and snapped correctly.

---

# 03 — Validation-Driven Anomaly Detection & Phonetic Retrying System

**What to build:**
A validation engine that inspects the output mora durations of `pydomino` alignments and automatically attempts to correct timing drifts by dynamically modifying the phoneme prompts and retrying alignment.

**Blocked by:** 01 — Core `pydomino` Integration & Japanese Phonetic Wrapper.

**Status:** ready-for-agent

- [ ] Implement a mora validation scanner that computes the duration $D$ of each aligned mora.
- [ ] Flag any mora as anomalous if its duration is $D < 30\text{ms}$ (too short / collapsed) or $D > 350\text{ms}$ (too long / stretched).
- [ ] Implement the "Too Long" correction rule: Dynamically test inserting a `pau` (silence) token both before and after the long mora, re-run `pydomino` on both sequences, and select the layout that resolves the duration anomaly.
- [ ] Implement the "Too Short / Contraction" correction rule: Extract the short mora $M_k$ and its immediate neighbors $M_{k-1}$ and $M_{k+1}$.
- [ ] Perform a search in the ASR phonetic stream. If $M_{k-1}$ and $M_{k+1}$ are contiguous in the ASR stream (meaning $M_k$ was unpronounced), delete $M_k$ from the alignment prompt and re-run.
- [ ] If there is a different phoneme $X$ between $M_{k-1}$ and $M_{k+1}$ in the ASR stream, replace $M_k$ with $X$ in the prompt and re-run.
- [ ] Expand the check to 4 neighboring moras ($M_{k-2}, M_{k-1}, M_{k+1}, M_{k+2}$) if multiple matches are found in the ASR stream.
- [ ] Limit the retry loop to a maximum of 3 attempts per subtitle card.

---

# 04 — Kanji Polyphony Alternative Dictionary Resolver

**What to build:**
A local dictionary-based polyphony solver that resolves G2P read-errors on Japanese characters (e.g., character voices and formal variations) when a mismatch causes a mora to collapse during alignment.

**Blocked by:** 03 — Validation-Driven Anomaly Detection & Phonetic Retrying System.

**Status:** ready-for-agent

- [ ] Create a lightweight static dictionary mapping common Japanese anime polyphonic words to their alternative G2P readings (e.g., `{"私": ["わたし", "あたし", "わたくし"], "俺": ["おれ", "おいら"]}`).
- [ ] When the ASR text and subtitle text share the same Kanji but the aligned mora collapses due to a phonetic reading difference, extract the affected word.
- [ ] Retrieve the alternative G2P reading list from the custom dictionary, swap the reading in the alignment prompt, and re-run `pydomino`.
- [ ] Verify that if the alternative reading matches the spoken pronunciation, the alignment is successfully accepted by the validation scanner.

---

# 05 — Linear Interpolation Fallback & Tight Tail-Trimming

**What to build:**
A safe fallback layout engine and high-fidelity tail-trimmer that prevents overlapping boundaries when alignments fail, and crops subtitles tightly for machine learning datasets.

**Blocked by:**
- 03 — Validation-Driven Anomaly Detection & Phonetic Retrying System
- 04 — Kanji Polyphony Alternative Dictionary Resolver

**Status:** ready-for-agent

- [ ] Implement the Linear Interpolation Fallback strategy: If a subtitle card fails all validation retry attempts, discard the `pydomino` output and map original subtitle timings to the adjusted container boundaries monotonically, guaranteeing zero overlaps.
- [ ] Implement the Tail-Trimming logic: Set the final subtitle card end time to exactly $T_{\text{speech\_end}} + 100\text{ms}$ (using the end time of the last non-`pau` spoken phoneme returned by `pydomino`).
- [ ] Add a safety clamping rule to the tail-trimmer: The adjusted end time must never exceed the start time of the subsequent subtitle card: $\min(T_{\text{speech\_end}} + 100\text{ms},\ S_{\text{next}}.\text{start\_time})$.
- [ ] Write integration tests asserting that fallback is triggered under bad conditions, that overlap constraints are strictly kept, and that tail-trimming crops trailing silence tightly.

---

# 06 — YouTube Subtitles Readability Rebuilder

**What to build:**
A specialized alignment mode (`--youtube-subs`) that ignores the original low-quality subtitle card boundaries, merges the continuous speech stream, and rebuilds readable, elegantly formatted Japanese subtitle blocks from scratch.

**Blocked by:** 02 — VAD-Based Post-Verification & Speech Hypothesis Snapping.

**Status:** ready-for-agent

- [ ] Implement a command-line flag `--youtube-subs` to toggle this reconstruction behavior.
- [ ] If enabled, bypass original timestamps. Merge the validated ASR speech stream and slice it into clean segments using VAD silent intervals as physical boundaries.
- [ ] Format and split the merged text into clean, sequential subtitle blocks respecting a maximum limit of 30 characters per card and a minimum on-screen duration of 1.0 second.
- [ ] Write tests verifying that auto-generated YouTube templates are successfully rebuilt into standard, human-readable SRT segments.

---

# 07 — Markdown Table Log Extension with Method Profiling

**What to build:**
An extended logger module that profiles how each subtitle segment was resolved in the pipeline, outputting clear metadata directly into the Markdown-table files for automated dataset filtering.

**Blocked by:**
- 05 — Linear Interpolation Fallback & Tight Tail-Trimming
- 06 — YouTube Subtitles Readability Rebuilder

**Status:** ready-for-agent

- [ ] Extend the `# 変更詳細 (Details)` Markdown table layout in `LoggerWriter` to include a new `Method` column.
- [ ] Populate this `Method` column with exactly one of the following profiling labels depending on the alignment history:
    *   `pydomino-v1`: Succeeded on the first forced-alignment run.
    *   `pydomino-v2(pau)`: Succeeded after resolving a long mora via pau insertion.
    *   `pydomino-v2(pruned)`: Succeeded after trimming a collapsed/short mora.
    *   `linear-fallback`: Failed validation and fell back to Linear Interpolation.
    *   `youtube-rebuild`: Built from scratch via YouTube reconstruction mode.
- [ ] Write tests to verify that the generated Markdown log tables can be parsed cleanly by splitting strings on the pipe separator (`|`), and assert that all rows have a consistent column count.
