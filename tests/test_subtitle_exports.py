"""Verify only final records and actual audio timings enter QA exports."""

import csv
from io import StringIO
from threading import Event
import time
import unittest

from src.slow_lane import SlowLaneSession
from src.conversation_review import ConversationReviewSession
from src.subtitle_exports import export_bilingual_csv, export_captions
from src.ui import export_conversation, render_transcript


class SubtitleExportTests(unittest.TestCase):
    def test_second_review_confirmation_keeps_first_correction_visible_and_audited(self):
        second_started, release_second = Event(), Event()
        calls = []
        def correct(request):
            calls.append(request)
            row = request["segments"][0]
            identity = {"segment_id": row["segment_id"], "base_version": row["base_version"]}
            if request.get("review_stage") == "conversation":
                second_started.set()
                release_second.wait()
                return {"corrections": [], "no_change": [identity]}
            return {"corrections": [{**identity, "target_text": "First corrected English.",
                                     "change_type": ["asr_fix"], "confidence": 0.95,
                                     "rationale": "Source-supported repair.", "term_pairs": []}],
                    "no_change": []}
        session = ConversationReviewSession(correct)
        self.addCleanup(session.close)
        self.addCleanup(release_second.set)
        session.submit(["Original source"], ["Fast draft"], timings=[{"start_s": 0, "end_s": 1}])
        self.assertTrue(second_started.wait(2))
        pending = session.snapshot()
        first_audit = pending["first_pass"]["segments"][0]
        pending["view"] = "authoritative"
        markup = render_transcript(["Original source"], ["Fast draft"], [], slow_lane=pending,
                                   providers=("openai", "astra"))
        self.assertIn("First corrected English.", markup)
        self.assertIn("astra-cell astra-corrected", markup)
        self.assertIn("Reviewing in context", markup)
        self.assertNotIn("Corrected · Final", markup)
        row = list(csv.DictReader(StringIO(export_bilingual_csv(pending))))[0]
        self.assertEqual((row["first_review_status"], row["conversation_review_status"]), ("corrected", "reviewing"))
        self.assertIn("First corrected English.", export_captions(pending))
        release_second.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            final = session.snapshot()
            if final["conversation_review"]["reviewed"] == 1:
                break
            time.sleep(0.01)
        self.assertEqual(final["conversation_review"]["reviewed"], 1)
        final["view"] = "authoritative"
        markup = render_transcript(["Original source"], ["Fast draft"], [], slow_lane=final)
        self.assertIn("astra-cell astra-corrected", markup)
        self.assertIn("Corrected · Final", markup)
        self.assertIn("Reviewed in context", markup)
        self.assertEqual(final["first_pass"]["segments"][0], first_audit)
        self.assertEqual(final["statuses"], ["corrected"])
        self.assertEqual(len(calls), 2)
        row = list(csv.DictReader(StringIO(export_bilingual_csv(final))))[0]
        self.assertEqual((row["first_review_status"], row["conversation_review_status"]), ("corrected", "confirmed"))
        downloaded = export_conversation(["Original source"], ["Fast draft"], [], slow_lane=final)
        self.assertIn("Astra first review: Corrected", downloaded)
        self.assertIn("Astra conversation review: Confirmed", downloaded)

    def test_conversation_review_exports_keep_accepted_text_and_both_stage_states(self):
        stages = ["waiting", "reviewing", "failed", "blocked", "corrected", "confirmed", "disabled"]
        snapshot = {"authoritative": [f"Accepted English {index}." for index in range(len(stages))],
                    "segments": [{"segment_id": f"s:{index + 1}", "source_text": f"Original {index}",
                                  "timing": {"start_s": index, "end_s": index + 1},
                                  "seal_reason": "correction", "first_pass_status": "corrected",
                                  "conversation_review_status": stage,
                                  "conversation_review_error": "PRIVATE provider error"}
                                 for index, stage in enumerate(stages)]}
        rows = list(csv.DictReader(StringIO(export_bilingual_csv(snapshot))))
        self.assertEqual([row["first_review_status"] for row in rows], ["corrected"] * len(stages))
        self.assertEqual([row["conversation_review_status"] for row in rows], stages)
        self.assertEqual([row["final_text"] for row in rows], snapshot["authoritative"])
        self.assertNotIn("PRIVATE", export_bilingual_csv(snapshot))
        for format in ("srt", "vtt"):
            captions = export_captions(snapshot, format=format)
            for text in snapshot["authoritative"]:
                self.assertIn(text, captions)
            self.assertNotIn("PRIVATE", captions)

    def test_out_of_order_reviews_publish_immediately_but_exports_keep_audio_order(self):
        first_started, release_first = Event(), Event()

        def correct(request):
            self.assertEqual(len(request["segments"]), 1)
            row = request["segments"][0]
            first = row["source_text"] == "First source."
            if first:
                first_started.set()
                release_first.wait()
            return {"corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": "First corrected line." if first else "Second corrected line.",
                "change_type": ["style"], "confidence": 0.95,
                "rationale": "Clearer English.", "term_pairs": [],
            }], "no_change": []}

        session = SlowLaneSession(correct)
        self.addCleanup(session.close)
        self.addCleanup(release_first.set)
        sources, drafts = ["First source.", "Second source."], ["First draft.", "Second draft."]
        timings = [{"start_s": 1, "end_s": 2}, {"start_s": 4, "end_s": 5}]
        session.submit(sources[:1], drafts[:1], timings=timings[:1])
        self.assertTrue(first_started.wait(2))
        session.submit(sources, drafts, timings=timings)

        def wait_for(predicate):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                snapshot = session.snapshot()
                if predicate(snapshot):
                    return snapshot
                time.sleep(0.01)
            self.fail("An independent correction did not publish while the earlier request was blocked.")

        partial = wait_for(lambda snapshot: snapshot["authoritative"][1] is not None)
        self.assertEqual(partial["authoritative"], [None, "Second corrected line."])
        self.assertEqual(partial["translations"], ["First draft.", "Second corrected line."])
        self.assertIn("00:00:04,000 --> 00:00:05,000\nSecond corrected line.", export_captions(partial))
        self.assertNotIn("First draft.", export_captions(partial))

        release_first.set()
        final = wait_for(lambda snapshot: snapshot["pending"] == 0)
        segment_ids = [row["segment_id"] for row in final["segments"]]
        self.assertEqual([event["segment"]["segment_id"] for event in final["events"] if event["type"] == "Seal"],
                         list(reversed(segment_ids)))
        exported = list(csv.DictReader(StringIO(export_bilingual_csv(final))))
        self.assertEqual([row["segment_id"] for row in exported], segment_ids)
        self.assertEqual([row["final_text"] for row in exported], ["First corrected line.", "Second corrected line."])
        for format in ("srt", "vtt"):
            captions = export_captions(final, format=format)
            self.assertLess(captions.index("First corrected line."), captions.index("Second corrected line."))

    def test_filtered_background_stays_in_audit_csv_but_is_omitted_from_captions(self):
        snapshot = {
            "authoritative": ["[Background speech filtered]", "Main conversation."],
            "segments": [
                {"segment_id": "s:1", "source_text": "Background speech", "seal_reason": "filtered",
                 "filtered": True, "timing": {"start_s": 0, "end_s": 1}},
                {"segment_id": "s:2", "source_text": "Main speech", "seal_reason": "no_change",
                 "timing": {"start_s": 2, "end_s": 3}},
            ],
        }
        for format in ("srt", "vtt"):
            captions = export_captions(snapshot, format=format)
            self.assertNotIn("Background", captions)
            self.assertIn("Main conversation.", captions)
            self.assertEqual(captions.count(" --> "), 1)
        rows = list(csv.DictReader(StringIO(export_bilingual_csv(snapshot))))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_text"], "Background speech")
        self.assertEqual(rows[0]["seal_reason"], "filtered")

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
        self.assertIn("[Translation unavailable]", srt)
        self.assertNotIn("=private formula", srt)
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
        self.assertEqual(rows[1]["final_text"], "[Translation unavailable]")

        snapshot = self.snapshot()
        snapshot["authoritative"][0] = "=English formula-like text"
        rows = list(csv.DictReader(StringIO(export_bilingual_csv(snapshot))))
        self.assertEqual(rows[0]["final_text"], "'=English formula-like text")

    def test_english_exports_keep_timed_rows_but_never_reuse_source_or_mixed_cjk(self):
        snapshot = {
            "authoritative": ["原始語音文字", "That batch of布拉吉.", "An English correction."],
            "segments": [
                {"segment_id": "s:1", "source_text": "原始語音文字", "source_fallback": True,
                 "seal_reason": "failed", "timing": {"start_s": 0, "end_s": 1}},
                {"segment_id": "s:2", "source_text": "上禮拜三那批布拉吉", "source_fallback": False,
                 "seal_reason": "correction", "timing": {"start_s": 1, "end_s": 2}},
                {"segment_id": "s:3", "source_text": "第三句", "source_fallback": False,
                 "seal_reason": "correction", "timing": {"start_s": 2, "end_s": 3}},
            ],
        }
        rows = list(csv.DictReader(StringIO(export_bilingual_csv(snapshot))))
        self.assertEqual([row["segment_id"] for row in rows], ["s:1", "s:2", "s:3"])
        self.assertEqual([row["source_text"] for row in rows], ["原始語音文字", "上禮拜三那批布拉吉", "第三句"])
        self.assertEqual([row["final_text"] for row in rows], [
            "[Translation unavailable]", "[Translation unavailable]", "An English correction.",
        ])
        self.assertEqual([row["seal_reason"] for row in rows], ["failed", "correction", "correction"])
        self.assertEqual([row["source_fallback"] for row in rows], ["True", "False", "False"])
        for format in ("srt", "vtt"):
            with self.subTest(format=format):
                exported = export_captions(snapshot, format=format)
                self.assertEqual(exported.count(" --> "), 3)
                self.assertEqual(exported.count("[Translation unavailable]"), 2)
                self.assertNotIn("原始語音文字", exported)
                self.assertNotIn("布拉吉", exported)
                self.assertIn("An English correction.", exported)

    def test_missing_timings_never_receive_fabricated_caption_offsets(self):
        self.assertEqual(export_captions({"authoritative": ["Final"], "segments": [
            {"segment_id": "s:1", "source_text": "source", "timing": {}},
        ]}), "")


if __name__ == "__main__":
    unittest.main()
