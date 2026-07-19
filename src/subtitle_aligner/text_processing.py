import re
import demoji
import neologdn
import pykakasi


class TextProcessor:
    """
    Handles Natural Language Processing and text normalization tasks
    for Japanese subtitles.
    """

    @staticmethod
    def extract_main_text(
        line: str | None = None, remove_brackets: bool = False
    ) -> str:
        """
        Normalizes Japanese text by removing noise, URLs, emojis, and
        standardizing character widths.

        Args:
            line (str, optional): The raw text to normalize. Defaults to None.
            remove_brackets (bool, optional): If True, removes parentheses
                (both half and full width) and their contents. Defaults to
                False to preserve speaker context for LLM translation.

        Returns:
            str: The cleaned and normalized text.
        """
        if not line:
            return ""

        text = line.strip()

        translation_table = str.maketrans(
            {"\n": "", "\t": "", "\r": "", "\u3000": "", "《": "", "》": ""}
        )
        text = text.translate(translation_table)

        # Remove URLs
        text = re.sub(r"https?://[\w/:%#\$&\?\.\=\+\-]+", "", text)

        text = demoji.replace(string=text, repl="")

        text = neologdn.normalize(text)

        text = text.lower()

        if remove_brackets:
            # Matches standard (), full-width （）, and Japanese brackets 【】
            pattern = r"[\(（【][^()（）【】]*[\)）】]"
            text = re.sub(pattern, "", text)

        return text

    def text_to_katakana(self, text: str) -> str:
        """Convert text to pure Katakana using pykakasi.

        Strips spaces, punctuation, and non-phonetic characters,
        returning only Katakana moras for phonetic comparison.

        Args:
            text: Input text (preferably already cleaned by
                  ``extract_main_text``).

        Returns:
            Pure Katakana string with no spaces, punctuation, or
            non-phonetic characters.
        """

        kks = pykakasi.kakasi()
        converted = kks.convert(text)

        # Collect all Katakana characters from each token
        katakana_chars: list[str] = []
        for token in converted:
            kana = token.get("kana", "")
            # Keep only Katakana (range U+30A0 to U+30FF)
            for ch in kana:
                if "\u30a0" <= ch <= "\u30ff":
                    katakana_chars.append(ch)

        return "".join(katakana_chars)
