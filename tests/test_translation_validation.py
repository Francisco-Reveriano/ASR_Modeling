"""Check untranslated source scripts without rejecting English names or IDs."""

import unittest

from src.translation_validation import contains_cjk


class TranslationValidationTests(unittest.TestCase):
    def test_mixed_source_scripts_are_detected_in_otherwise_english_text(self):
        for word in ("工序", "𠀀", "\U00031350", "\U000323b0", "神", "\U0002f800",
                     "ひらがな", "カタカナ", "ｶﾅ", "\U0001b000", "ㄅ", "ㆠ",
                     "한글", "ᄒ", "ㄱ", "\ua960", "\ud7b0", "〇"):
            with self.subTest(word=word):
                self.assertTrue(contains_cjk(f"Check the {word} process."))

    def test_english_names_numbers_and_identifiers_are_allowed(self):
        for text in ("", "Hello, world!", "José and Chloë meet at 8:30.",
                     "Check ETCH-07, LOT1234 and API_v2.", "Zhōngwén / Tâi-oân",
                     "Temperature: 30°C; pressure ≤ 5 Pa.", "English · with punctuation."):
            with self.subTest(text=text):
                self.assertFalse(contains_cjk(text))


if __name__ == "__main__":
    unittest.main()
