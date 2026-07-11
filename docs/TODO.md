# TODO & Known Issues

This document tracks planned features, known bugs, and technical debt for the Time Tracker application.

## 🐛 Bugs
- **ASR hallucination**: The asr models can hallucinate transcribing text where there is only silence. So we must use the vad to check if there is some human speech there or not to filter this hallucinated transcriptions.
- **Subtitles length**: The original subtitles might divide long sentences on several subtitles despite having no pause between them, the asr can not do this so it transcribes a long sentence, making the adjustment incorrect.

## ✨ Features

- **Youtube sub flag**: Youtube subtitles are low quality so here the asr should be considered as the correct subtitles.
- **Subtitles tail trim**: The subtitles adjustment must target the end time as well, in some cases the subtitles continue until the next start time.
- **Webm and ass support**: Add support for these types of formats.
- **ASR validation**: Add validation of the asr transcription results with vads, using 2 asrs, etc.

## 🛠 Refactoring & Tech Debt

## 🧪 Testing


