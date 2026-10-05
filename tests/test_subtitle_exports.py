"""Verify only final records and actual audio timings enter QA exports."""

import csv
from io import StringIO
import unittest

from src.subtitle_exports import export_bilingual_csv, export_captions


class SubtitleExportTests(unittest.TestCase):
    def snapshot(self):
        return {"authoritative": ["A corrected line.", None, "=private formula", "No timing."], "segments": [
            {"segment_id": "s:1", "source_text": "原文", "timing": {"start_s": 1.25, "end_s": 3.5}, "seal_reason": "correction"},
            {"segment_id": "s:2", "source_text": "Pending", "timing": {"start_s": 4, "end_s": 5}},
            {"segment_id": "s:3", "source_text": "+source", "timing": {"start_s": 6, "end_s": 7}, "source_fallback": True},
            {"segment_id": "s:4", "source_text": "No timing", "timing": None},
        ]}

    def test_captions_use_audio_times_and_omit_unsealed_or_untimed_rows(self):
        snapshot = self.snapshot()
        srt = export_captions(snapshot, speakers=["Speaker 1"])
        self.assertIn("00:00:01,250 --> 00:00:03,500\nSpeaker 1: A corrected line.", srt)
        self.assertIn("[Source fallback]", srt)
        self.assertNotIn("Pending", srt)
        self.assertNotIn("No timing", srt)
        vtt = export_captions(snapshot, format="vtt")
        self.assertTrue(vtt.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:01.250 --> 00:00:03.500", vtt)

    def test_csv_keeps_identifiers_and_protects_formula_like_transcripts(self):
        rows = list(csv.DictReader(StringIO(export_bilingual_csv(self.snapshot()))))
        self.assertEqual([row["segment_id"] for row in rows], ["s:1", "s:3", "s:4"])
        self.assertEqual(rows[0]["final_text"], "A corrected line.")
        self.assertEqual(rows[1]["source_text"], "'+source")
        self.assertEqual(rows[1]["final_text"], "'=private formula")

    def test_missing_timings_never_receive_fabricated_caption_offsets(self):
        self.assertEqual(export_captions({"authoritative": ["Final"], "segments": [
            {"segment_id": "s:1", "source_text": "source", "timing": {}},
        ]}), "")


if __name__ == "__main__":
    unittest.main()
