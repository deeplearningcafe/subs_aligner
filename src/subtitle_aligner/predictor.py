import re


class AccentPredictor:
    """
    Predicts theoretical pitch accent (High/Low) for Japanese text
    using pyopenjtalk-plus and MARINE.
    """

    def __init__(self):
        # Lazy imports for optional predict dependencies
        import pykakasi
        import pyopenjtalk

        self.kks = pykakasi.kakasi()
        self.run_marine = True

    def predict(self, text: str) -> list[bool]:
        """
        Predicts the pitch accent for each character in the text.

        Args:
            text (str): The Japanese transcript.

        Returns:
            list[bool]: Pitch for each character (True=High, False=Low).
        """
        import pyopenjtalk

        if not text.strip():
            return []

        try:
            labels = pyopenjtalk.extract_fullcontext(text, run_marine=self.run_marine)
        except TypeError:
            labels = pyopenjtalk.extract_fullcontext(text)

        # Parse labels to extract the pitch of each mora.
        mora_pitches = []
        last_p2 = None

        for label in labels:
            ph_match = re.search(r"\-(.*?)\+", label)
            if not ph_match:
                continue
            ph = ph_match.group(1)

            if ph in ["sil", "pau"]:
                continue

            # Extract accent phrase info: /A:p1+p2+p3
            # p1: distance to accent nucleus (can be negative)
            # p2: mora index in current phrase
            # p3: remaining moras in phrase
            a_match = re.search(r"/A:([0-9\-]+)\+([0-9]+)\+([0-9]+)", label)
            if not a_match:
                continue

            p1 = int(a_match.group(1))
            p2 = int(a_match.group(2))

            # When p2 changes, we've moved to a new mora
            if p2 != last_p2:
                if p1 > 0:
                    is_high = False
                elif p1 < 0 and p2 == 1:
                    # Before the nucleus, and it is the first mora
                    is_high = False
                else:
                    # At the nucleus, or mora 2+ before the nucleus
                    is_high = True

                mora_pitches.append(is_high)
                last_p2 = p2

        result = []
        mora_index = 0

        # Small kana don't count as independent moras
        small_kana = set(
            [
                "ャ",
                "ュ",
                "ョ",
                "ァ",
                "ィ",
                "ゥ",
                "ェ",
                "ォ",
                "ゃ",
                "ゅ",
                "ょ",
                "ぁ",
                "ぃ",
                "ぅ",
                "ぇ",
                "ぉ",
            ]
        )
        punct = set([" ", " ", "。", "、", "！", "？", "!", "?", ".", ","])

        for char in text:
            if char in punct:
                result.append(False)
                continue

            converted = self.kks.convert(char)
            if not converted:
                result.append(False)
                continue

            kana = converted[0]["kana"]
            mora_count = sum(1 for k in kana if k not in small_kana)

            # Assign the pitch of the first mora of this character
            if mora_index < len(mora_pitches):
                result.append(mora_pitches[mora_index])
            else:
                result.append(False)

            mora_index += mora_count

        return result
