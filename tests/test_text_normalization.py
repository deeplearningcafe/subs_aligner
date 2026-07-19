"""Tests for text normalization (Katakana conversion) in text_processing."""

from __future__ import annotations

import pytest

from src.subtitle_aligner.text_processing import TextProcessor


class TestTextToKatakana:
    """Tests for TextProcessor.text_to_katakana."""

    @pytest.fixture
    def processor(self):
        return TextProcessor()

    def test_kanji_to_katakana(self, processor):
        """Hiragana characters are converted to their Katakana reading."""
        result = processor.text_to_katakana("こんにちは")
        # "は" in こんにちは is the particle wa -> ハ (ha)
        assert result == "コンニチハ"

    def test_mixed_kanji_kana(self, processor):
        """Mixed Kanji and Kana are all converted to Katakana."""
        result = processor.text_to_katakana("よくぞ集った精鋭たちよ")
        # Should be pure Katakana, no spaces or punctuation
        assert result != ""
        # All characters should be Katakana
        for ch in result:
            assert "\u30a0" <= ch <= "\u30ff"

    def test_strips_punctuation(self, processor):
        """Punctuation marks are removed from the output."""
        result = processor.text_to_katakana("こんにちは、世界！")
        # No commas, exclamation marks, or other punctuation
        for ch in result:
            assert ch not in "、！。！？"

    def test_strips_spaces(self, processor):
        """Spaces are removed from the output."""
        result = processor.text_to_katakana("よくぞ集った 精鋭たちよ")
        assert " " not in result
        # Should be a single continuous string
        assert all("\u30a0" <= ch <= "\u30ff" for ch in result)

    def test_strips_symbols(self, processor):
        """Japanese brackets and symbols are removed."""
        result = processor.text_to_katakana("（ローレック）よくぞ集った")
        # Parentheses should be stripped
        assert "（" not in result
        assert "）" not in result

    def test_empty_string(self, processor):
        """Empty string returns empty string."""
        result = processor.text_to_katakana("")
        assert result == ""

    def test_only_punctuation(self, processor):
        """String with only punctuation returns empty string."""
        result = processor.text_to_katakana("、。！")
        assert result == ""

    def test_already_katakana(self, processor):
        """Text that is already Katakana is returned unchanged."""
        result = processor.text_to_katakana("コンニチハ")
        assert result == "コンニチハ"

    def test_cleaned_text_roundtrip(self, processor):
        """Pipeline: extract_main_text -> text_to_katakana produces clean output."""
        raw = "（ローレック）よくぞ集った 精鋭たちよ。"
        cleaned = processor.extract_main_text(raw)
        katakana = processor.text_to_katakana(cleaned)
        assert katakana != ""
        # All characters must be Katakana
        assert all("\u30a0" <= ch <= "\u30ff" for ch in katakana)
        # No spaces or punctuation
        assert " " not in katakana
        assert "、" not in katakana
        assert "。" not in katakana

    def test_real_subtitle_text(self, processor):
        """Test with real subtitle text from wataoshi_06.srt."""
        raw = "学院騎士団は 諸君らを歓迎する！"
        cleaned = processor.extract_main_text(raw)
        katakana = processor.text_to_katakana(cleaned)
        assert katakana != ""
        assert all("\u30a0" <= ch <= "\u30ff" for ch in katakana)

    def test_special_kana_handling(self, processor):
        """Small kana (ゃゅょ etc.) are kept but not counted as separate moras
        in the pykakasi output — they should appear as part of the preceding
        character's Katakana representation."""
        result = processor.text_to_katakana("きゃ")
        # Should produce Katakana (e.g. キャ)
        assert result != ""
