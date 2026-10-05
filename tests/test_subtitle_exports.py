"""Completed English captions and truthful selected-model export metadata."""

import csv
from io import StringIO
import json
import unittest

from src.subtitle_exports import conversation_records, export_bilingual_csv, export_captions


class SubtitleExportTests(unittest.TestCase):
    def inputs(self):
        return {"texts": ["第一句", "待定", "失敗", "背景", "混合", "最後"],
                "translations": ["First English.", None, None, "marker", "Mixed中文", "Last English."],
                "errors": [None, None, "PRIVATE provider details"], "filtered": [False, False, False, True],
                "timings": [{"start_s": index * 2 + 0.25, "end_s": index * 2 + 1.5} for index in range(6)],
                "profile_label": "Qwen fast", "model": "Qwen/example"}

    def test_csv_keeps_all_sources_and_truthful_selected_model_states(self):
        values = self.inputs()
        rows = list(csv.DictReader(StringIO(export_bilingual_csv(**values))))
        self.assertEqual([row["source_text"] for row in rows], values["texts"])
        self.assertEqual([row["status"] for row in rows], ["translated", "pending", "failed", "filtered", "failed", "translated"])
        self.assertEqual([row["translation"] for row in rows], ["First English.", "", "", "", "", "Last English."])
        self.assertEqual([row["segment_id"] for row in rows], [str(index) for index in range(1, 7)])
        self.assertTrue(all(row["model"] == "Qwen/example" and row["profile_label"] == "Qwen fast" for row in rows))
        self.assertNotIn("PRIVATE", export_bilingual_csv(**values))
        self.assertNotIn("speaker", rows[0])
        self.assertNotIn("seal_reason", rows[0])

    def test_json_records_contain_originals_and_only_completed_english(self):
        values = self.inputs()
        records = json.loads(json.dumps(conversation_records(**values), ensure_ascii=False))
        self.assertEqual([row["source_text"] for row in records], values["texts"])
        self.assertEqual([row["translation"] for row in records], ["First English.", None, None, None, None, "Last English."])
        self.assertEqual([row["filtered"] for row in records], [False, False, False, True, False, False])
        self.assertEqual(records[0]["start_s"], 0.25)
        self.assertNotIn("PRIVATE", json.dumps(records))

    def test_explicit_reasoning_markup_never_enters_translation_exports(self):
        for target in ("<think>PRIVATE</think>English.", "<think>PRIVATE", "PRIVATE</think>", "</analysis>PRIVATE",
                       "<reasoning>PRIVATE</reasoning>", "<|analysis|>PRIVATE",
                       "<|begin_of_thought|>PRIVATE<|end_of_thought|>"):
            with self.subTest(target=target):
                args = (["原文"], [target], [])
                kwargs = {"timings": [{"start_s": 0, "end_s": 1}]}
                record = conversation_records(*args, **kwargs)[0]
                self.assertEqual(record["status"], "failed")
                self.assertIsNone(record["translation"])
                self.assertEqual(record["source_text"], "原文")
                csv_row = list(csv.DictReader(StringIO(export_bilingual_csv(*args, **kwargs))))[0]
                self.assertEqual(csv_row["translation"], "")
                self.assertEqual(csv_row["status"], "failed")
                self.assertNotIn("PRIVATE", json.dumps(record) + str(csv_row))
                for format in ("srt", "vtt"):
                    self.assertEqual(export_captions(*args, **kwargs, format=format), "")

    def test_captions_exclude_pending_failed_filtered_and_invalid_english(self):
        for format, separator in (("srt", ","), ("vtt", ".")):
            with self.subTest(format=format):
                exported = export_captions(**self.inputs(), format=format)
                self.assertEqual(exported.count(" --> "), 2)
                self.assertIn(f"1\n00:00:00{separator}250 --> 00:00:01{separator}500\nFirst English.", exported)
                self.assertIn(f"2\n00:00:10{separator}250 --> 00:00:11{separator}500\nLast English.", exported)
                self.assertEqual(exported.startswith("WEBVTT"), format == "vtt")
                for absent in ("待定", "失敗", "背景", "中文", "marker", "unavailable", "pending", "PRIVATE"):
                    self.assertNotIn(absent, exported)

    def test_later_finished_rows_export_before_earlier_results(self):
        values = {"texts": ["first", "second"], "translations": [None, "Second English."],
                  "errors": [], "timings": [{"start_s": 1, "end_s": 2}, {"start_s": 3, "end_s": 4}]}
        self.assertIn("00:00:03,000 --> 00:00:04,000\nSecond English.", export_captions(**values))
        values["translations"][0] = "First English."
        complete = export_captions(**values)
        self.assertLess(complete.index("First English."), complete.index("Second English."))

    def test_missing_invalid_or_submillisecond_times_do_not_make_bad_cues(self):
        invalid = [None, {}, {"start_s": 0}, {"start_s": -1, "end_s": 1}, {"start_s": 2, "end_s": 1},
                   {"start_s": True, "end_s": 2}, {"start_s": float("nan"), "end_s": 2},
                   {"start_s": 0, "end_s": float("inf")}, {"start_s": 0.0001, "end_s": 0.0002}]
        for timing in invalid:
            with self.subTest(timing=timing):
                self.assertEqual(export_captions(["source"], ["English."], [], timings=[timing]), "")
        self.assertEqual(export_captions([], [], [], format="vtt"), "")

    def test_error_suppresses_stale_text_and_filtering_requires_explicit_flag(self):
        records = conversation_records(["first", "second", "third"],
                                       ["Old English", "[Background speech filtered]", "not relevant"],
                                       ["failed"], filtered=[False, False, True])
        self.assertEqual([row["status"] for row in records], ["failed", "translated", "filtered"])
        self.assertEqual([row["translation"] for row in records], [None, "[Background speech filtered]", None])

    def test_csv_formula_values_and_caption_markup_cannot_inject_structure(self):
        rows = list(csv.DictReader(StringIO(export_bilingual_csv(["=source"], ["+English"], [],
                                                                profile_label="@profile", model="-model"))))
        self.assertEqual(rows[0]["source_text"], "'=source")
        self.assertEqual(rows[0]["translation"], "'+English")
        self.assertEqual(rows[0]["profile_label"], "'@profile")
        self.assertEqual(rows[0]["model"], "'-model")
        captions = export_captions(["source"], ['<b>Hi</b>\n\n2\n00:00 --> 00:01'], [],
                                   timings=[{"start_s": 0, "end_s": 1}])
        self.assertEqual(captions.count(" --> "), 1)
        self.assertNotIn("<b>", captions)
        self.assertIn("‹b›Hi‹/b›", captions)

    def test_unknown_caption_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "srt or vtt"):
            export_captions([], [], [], format="html")


if __name__ == "__main__":
    unittest.main()
