"""Check reference validation and real word alignment without any models."""

import unittest

from src.evaluation import (
    MAX_REFERENCE_BYTES, decode_reference, mixed_match_score, parse_reference, word_match_score,
)


class DecodeReferenceTests(unittest.TestCase):
    def test_utf8_bom_and_outer_whitespace_are_removed_without_rewriting_text(self):
        text = "Café １２３.\r\nWe're\tready."

        reference = decode_reference(("\ufeff \n" + text + "\n ").encode("utf-8"))

        self.assertEqual(reference, text)

    def test_letters_and_digits_from_unicode_are_valid(self):
        for text in ("123", "é", "你好"):
            with self.subTest(text=text):
                self.assertEqual(decode_reference(text.encode("utf-8")), text)

    def test_empty_or_non_word_reference_is_rejected(self):
        for text in ("", "\ufeff", " \t\r\n", "...—‘’！？", "🙂 + $"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "must contain words"):
                    decode_reference(text.encode("utf-8"))

    def test_invalid_encoding_is_rejected(self):
        for data in (b"caf\xe9", b"\xff\xfeh", b"\xfe\xff\xd8\x00", b"\x89PNG\r\n\x1a\n"):
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "UTF-8"):
                    decode_reference(data)

    def test_utf16_requires_bom_and_preserves_text_in_either_byte_order(self):
        text = "你好, Café!"
        for bom, encoding in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")):
            with self.subTest(encoding=encoding):
                self.assertEqual(decode_reference(bom + text.encode(encoding)), text)
        with self.assertRaises(ValueError):
            decode_reference("hello".encode("utf-16-le"))

    def test_binary_control_characters_are_rejected_even_with_words(self):
        for character in ("\x00", "\x01", "\x1b", "\x7f", "\u0085"):
            with self.subTest(character=character):
                with self.assertRaisesRegex(ValueError, "binary or unsupported control"):
                    decode_reference(("hello" + character + "world").encode("utf-8"))

    def test_size_limit_is_in_bytes_and_allows_exact_boundary(self):
        data = "é".encode("utf-8") * (MAX_REFERENCE_BYTES // 2)

        self.assertEqual(decode_reference(data), "é" * (MAX_REFERENCE_BYTES // 2))
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            decode_reference(data + b"a")


class ParseReferenceTests(unittest.TestCase):
    def parse(self, text, filename="reference.txt", **kwargs):
        return parse_reference(text.encode("utf-8"), filename, **kwargs)

    def test_plain_text_preserves_dialogue_colons_parentheses_numbers_and_unicode(self):
        text = "Answer: 42 (probably).\nMeet at 8:30; keep [these] words.\nCafé 你好"

        result = self.parse(text)

        self.assertEqual(result, {
            "text": text, "original_text": text, "format": "Plain text",
            "segment_count": 3, "removed_lines": 0, "has_cjk": True,
        })

    def test_timestamped_transcript_keeps_turn_order_and_spoken_annotations(self):
        text = (
            "# Synthetic meeting metadata\n\n"
            "[00:02.000 --> 00:03.000] SPK1 阿明 (Ming): (overlap) Hello Teams: keep (this) at 8:30.\n"
            "[00:01.500 --> 00:02.100] SPK2: (backchannel) 嗯，(overlap) 是一句話。\n"
            "[BG 00:04.000 --> 00:05.000] background speech\n"
            "[FX 00:05.000] chime\n"
            "# End metadata"
        )

        result = self.parse(text)

        self.assertEqual(result["text"], "Hello Teams: keep (this) at 8:30.\n嗯，(overlap) 是一句話。")
        self.assertEqual(result["original_text"], text)
        self.assertEqual(result["format"], "Timestamped transcript")
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["removed_lines"], 4)
        self.assertTrue(result["has_cjk"])

    def test_timestamped_named_speakers_and_multiline_turns(self):
        result = self.parse(
            "[00:01] Alice: First line\ncontinued (with detail).\n"
            "[00:02] Bob: Second line: the number is 123."
        )

        self.assertEqual(result["text"], "First line continued (with detail).\nSecond line: the number is 123.")
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["removed_lines"], 0)

    def test_explicit_speaker_labels_detected_and_manual_transcript_accepts_names(self):
        result = self.parse("SPK1: hello\nSpeaker 2: good morning")
        self.assertEqual(result["text"], "hello\ngood morning")
        self.assertEqual(result["format"], "Annotated transcript")
        self.assertEqual(result["segment_count"], 2)

        named = self.parse("Alice: hello\nBob: good morning", format="transcript")
        self.assertEqual(named["text"], "hello\ngood morning")
        self.assertEqual(named["segment_count"], 2)

    def test_plain_override_preserves_all_structural_content(self):
        text = "# Heading\n[00:01 --> 00:02] SPK1: (overlap) hello\n[BG 00:03] world"

        result = self.parse(text, "reference.vtt", format="plain")

        self.assertEqual(result["text"], text)
        self.assertEqual(result["format"], "Plain text")
        self.assertEqual(result["removed_lines"], 0)
        self.assertEqual(result["segment_count"], 3)

    def test_srt_cues_remove_indices_times_and_known_tags_but_preserve_dialogue(self):
        text = (
            "1\n00:00:01,000 --> 00:00:02,500\n"
            "<i>Hello</i>, <b>world</b> &amp; <unknown>literal</unknown>.\n"
            "Meet at 8:30 (tomorrow).\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n123"
        )

        result = self.parse(text, "captions.SRT")

        self.assertEqual(result["text"], "Hello, world & <unknown>literal</unknown>. Meet at 8:30 (tomorrow).\n123")
        self.assertEqual(result["format"], "SubRip captions")
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["removed_lines"], 4)
        self.assertFalse(result["has_cjk"])

    def test_caption_timing_is_detected_without_filename_extension(self):
        result = self.parse("00:01.000 --> 00:02.000\nHello there.")

        self.assertEqual(result["text"], "Hello there.")
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["removed_lines"], 1)

    def test_webvtt_blocks_identifiers_settings_voice_and_inline_timing(self):
        text = (
            "WEBVTT\nKind: captions\nLanguage: en\n\n"
            "NOTE translator note\nDo not score this.\n\n"
            "STYLE\n::cue { color: lime; }\n\n"
            "REGION\nid:fred\n\n"
            "opening-cue\n00:01.000 --> 00:02.500 line:90% align:start\n"
            "<v Alice><c.green>Hello</c> <00:01.300><lang en>world</lang></v>\n\n"
            "00:03.000 --> 00:04.000\n{\\an8}<font color=\"red\">你好</font> &lt;3"
        )

        result = self.parse(text)

        self.assertEqual(result["text"], "Hello world\n你好 <3")
        self.assertEqual(result["format"], "WebVTT captions")
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["removed_lines"], 12)

    def test_metadata_only_references_are_rejected_after_cleanup(self):
        for text, filename in (
            ("# Header\n[BG 00:01] speech\n[FX 00:02] chime", "ref.txt"),
            ("(overlap)", "ref.txt"),
            ("1\n00:00:01,000 --> 00:00:02,000", "ref.srt"),
            ("WEBVTT\n\nNOTE nothing spoken", "ref.vtt"),
        ):
            with self.subTest(filename=filename, text=text):
                with self.assertRaisesRegex(ValueError, "no spoken text"):
                    self.parse(text, filename)

    def test_metadata_words_inside_timed_cues_are_spoken_and_explicit_speaker_is_removed(self):
        result = self.parse(
            "WEBVTT\n\n00:01.000 --> 00:03.000\n"
            "NOTE this detail\nSTYLE\nREGION\nSPK1: (overlap) Keep this: detail."
        )

        self.assertEqual(result["text"], "NOTE this detail STYLE REGION Keep this: detail.")
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["removed_lines"], 2)

    def test_malformed_captions_and_unknown_format_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "outside a timed cue"):
            self.parse("Untimed content", "ref.srt")
        with self.assertRaisesRegex(ValueError, "format must be"):
            self.parse("hello", format="guess")

    def test_bom_marked_utf16_transcript_is_parsed(self):
        result = parse_reference("[00:01] SPK1: 你好".encode("utf-16"))

        self.assertEqual(result["text"], "你好")
        self.assertTrue(result["has_cjk"])

    def test_supplementary_han_is_detected(self):
        result = self.parse("A\U00020000B")

        self.assertTrue(result["has_cjk"])


class WordMatchScoreTests(unittest.TestCase):
    def assert_counts(self, result, *, hits, substitutions, deletions, insertions):
        self.assertEqual(
            [result[key] for key in ("hits", "substitutions", "deletions", "insertions")],
            [hits, substitutions, deletions, insertions],
        )
        self.assertEqual(result["reference_words"], hits + substitutions + deletions)
        self.assertEqual(result["hypothesis_words"], hits + substitutions + insertions)
        total = hits + substitutions + deletions + insertions
        expected = (substitutions + deletions + insertions) / total if total else 0.0
        self.assertAlmostEqual(result["wmer"], expected)
        self.assertAlmostEqual(result["score"], 1.0 - expected)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_normalization_handles_case_unicode_punctuation_and_whitespace(self):
        result = word_match_score("  ＣＡＦÉ—Straße, １２３!\nReady? ", "cafe\u0301 strasse\t123 ready")

        self.assert_counts(result, hits=4, substitutions=0, deletions=0, insertions=0)

    def test_straight_and_curly_apostrophes_and_hyphens_split_consistently(self):
        result = word_match_score("Don't re-enter.", "DON’T re enter")

        self.assert_counts(result, hits=4, substitutions=0, deletions=0, insertions=0)

    def test_accents_and_digits_are_not_discarded(self):
        result = word_match_score("café 123", "cafe 124")

        self.assert_counts(result, hits=0, substitutions=2, deletions=0, insertions=0)

    def test_substitution(self):
        result = word_match_score("the red boat", "the blue boat")

        self.assert_counts(result, hits=2, substitutions=1, deletions=0, insertions=0)

    def test_deletion(self):
        result = word_match_score("the red boat", "the boat")

        self.assert_counts(result, hits=2, substitutions=0, deletions=1, insertions=0)

    def test_insertions_use_alignment_length_so_score_never_becomes_negative(self):
        result = word_match_score("boat", "the very big blue boat today")

        self.assert_counts(result, hits=1, substitutions=0, deletions=0, insertions=5)
        self.assertAlmostEqual(result["score"], 1 / 6)

    def test_empty_texts_have_defined_scores_and_counts(self):
        for reference, hypothesis, counts in (
            ("", "", (0, 0, 0, 0)),
            ("...", "!?", (0, 0, 0, 0)),
            ("one two", "", (0, 0, 2, 0)),
            ("", "one two", (0, 0, 0, 2)),
        ):
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = word_match_score(reference, hypothesis)

                self.assert_counts(result, **dict(zip(
                    ("hits", "substitutions", "deletions", "insertions"), counts,
                )))

    def test_no_implicit_chinese_character_segmentation(self):
        result = word_match_score("你好 世界", "你好世界")

        self.assertEqual(result["reference_words"], 2)
        self.assertEqual(result["hypothesis_words"], 1)
        self.assertEqual(result["score"], 0.0)

    def test_missing_hypothesis_is_not_silently_scored_as_empty(self):
        with self.assertRaises(TypeError):
            word_match_score("hello world", None)


class MixedMatchScoreTests(unittest.TestCase):
    def test_han_is_split_independently_of_spaces_and_adjacent_english(self):
        result = mixed_match_score("你好Teams今天N3很好", "你 好 teams 今 天 N3 很 好")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["hits"], 8)
        self.assertEqual(result["reference_words"], 8)
        self.assertEqual(result["hypothesis_words"], 8)

    def test_english_stays_in_words_and_normalization_preserves_accents_digits(self):
        result = mixed_match_score("臺灣ＣＡＦÉ１２３，ready-to-go！", "臺 灣 cafe\u0301123 ready to go")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["hits"], 6)

    def test_supplementary_and_compatibility_han_are_individual_tokens(self):
        result = mixed_match_score("A\U00020000\U0002a700〇\ufa11B", "a \U00020000 \U0002a700 〇 \ufa11 b")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["hits"], 6)

    def test_match_denominator_includes_insertions(self):
        result = mixed_match_score("你", "你 好 Teams is ready")

        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["insertions"], 4)
        self.assertEqual(result["reference_words"], 1)
        self.assertAlmostEqual(result["wmer"], 4 / 5)
        self.assertAlmostEqual(result["score"], 1 / 5)

    def test_mixed_substitution_and_deletion_counts(self):
        result = mixed_match_score("你 好 Teams", "你 是")

        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["deletions"], 1)
        self.assertAlmostEqual(result["score"], 1 / 3)

    def test_empty_and_missing_inputs_follow_word_score_contract(self):
        self.assertEqual(mixed_match_score("", "")["score"], 1.0)
        self.assertEqual(mixed_match_score("你好", "")["score"], 0.0)
        self.assertEqual(mixed_match_score("", "你好")["insertions"], 2)
        with self.assertRaises(TypeError):
            mixed_match_score("你好", None)


if __name__ == "__main__":
    unittest.main()
