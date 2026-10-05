"""Check aligned conversation rows, safe model text, and readable downloads."""

from html.parser import HTMLParser
import unittest

from src.ui import export_conversation, render_transcript


class TranscriptParser(HTMLParser):
    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.numbers = []
        self.segments = []
        self.english = []
        self.tencent_english = []
        self.astra_english = []
        self.astra_statuses = []
        self.segment_ids = []
        self.speakers = []
        self.reference_numbers = []
        self.references = []
        self.reference_headings = []
        self.reference_inside_segment = False
        self.tags = []
        self._in_segment = False
        self._capture = None
        self._provider = None
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        classes = dict(attrs).get("class", "").split()
        if "transcript-row" in classes:
            self._in_segment = True
            self.segment_ids.append(dict(attrs).get("data-segment-id"))
        if "reference-number" in classes:
            self.reference_numbers.append("")
            self._capture = self.reference_numbers
        elif "reference-text" in classes:
            self.references.append("")
            self._capture = self.references
            self.reference_inside_segment |= self._in_segment
        elif "reference-heading" in classes:
            self.reference_headings.append("")
            self._capture = self.reference_headings
        elif "openai-cell" in classes:
            self._provider = self.english
        elif "tencent-cell" in classes:
            self._provider = self.tencent_english
        elif "astra-cell" in classes:
            self._provider = self.astra_english
        elif "slow-badge" in classes:
            self.astra_statuses.append("")
            self._capture = self.astra_statuses
        elif "speaker-label" in classes:
            self.speakers.append("")
            self._capture = self.speakers
        elif "line-number" in classes:
            self.numbers.append("")
            self._capture = self.numbers
        elif "segment-text" in classes:
            self.segments.append("")
            self._capture = self.segments
        elif any(name in classes for name in (
            "translation-text", "translation-pending", "translation-error"
        )):
            self._provider.append("")
            self._capture = self._provider

    def handle_data(self, data):
        if self._capture is not None:
            self._capture[-1] += data

    def handle_endtag(self, tag):
        self._capture = None
        if tag == "div":
            self._provider = None
        elif tag == "li":
            self._in_segment = False


class TranscriptRenderingTests(unittest.TestCase):
    def test_each_segment_gets_a_stable_number_as_new_segments_arrive(self):
        texts = ["第一段話。", "A second spoken segment."]
        initial = TranscriptParser(render_transcript(texts))
        updated = TranscriptParser(render_transcript(texts + ["Third segment."]))

        self.assertEqual(initial.numbers, ["01", "02"])
        self.assertEqual(initial.segments, texts)
        self.assertEqual(updated.numbers[:2], initial.numbers)
        self.assertEqual(updated.segments[:2], initial.segments)
        self.assertEqual(updated.numbers[2], "03")
        self.assertEqual(updated.segments[2], "Third segment.")

        longer = TranscriptParser(render_transcript([f"Segment {i}" for i in range(101)]))
        self.assertEqual(longer.numbers[98:], ["99", "100", "101"])
        self.assertEqual(len(longer.segments), 101)

    def test_providers_keep_rows_aligned_when_results_arrive_independently(self):
        texts = ["第一段。", "第二段。", "第三段。", "第四段。"]
        markup = render_transcript(
            texts,
            ["First.", None, "Third."],
            [None, None, None, "Private API error details"],
            ["One.", "Two."],
            [None, None, "Private model error details"],
        )
        rendered = TranscriptParser(markup)

        self.assertEqual(list(zip(
            rendered.numbers, rendered.segments, rendered.english, rendered.tencent_english,
        )), [
            ("01", texts[0], "First.", "One."),
            ("02", texts[1], "Translating…", "Two."),
            ("03", texts[2], "Third.", "Translation unavailable"),
            ("04", texts[3], "Translation unavailable", "Translating…"),
        ])
        self.assertNotIn("Private API error details", markup)
        self.assertNotIn("Private model error details", markup)

    def test_model_markup_in_all_columns_is_displayed_literally(self):
        text = '<script>alert("speech")</script> & <img src=x onerror="bad()">\nNext line.'
        translation = '<a href="javascript:bad()">Hello</a> & <iframe>test</iframe>'
        local_translation = '<input onfocus="bad()"> & <style>body {display:none}</style>'
        rendered = TranscriptParser(render_transcript(
            [text], [translation], tencent_translations=[local_translation],
        ))

        self.assertEqual(rendered.numbers, ["01"])
        self.assertEqual(rendered.segments, [text])
        self.assertEqual(rendered.english, [translation])
        self.assertEqual(rendered.tencent_english, [local_translation])
        self.assertNotIn("script", rendered.tags)
        self.assertNotIn("img", rendered.tags)
        self.assertNotIn("a", rendered.tags)
        self.assertNotIn("iframe", rendered.tags)
        self.assertNotIn("input", rendered.tags)
        self.assertNotIn("style", rendered.tags)

    def test_empty_transcript_has_no_numbered_segments(self):
        rendered = TranscriptParser(render_transcript([]))

        self.assertEqual(rendered.numbers, [])
        self.assertEqual(rendered.segments, [])
        self.assertEqual(rendered.english, [])
        self.assertEqual(rendered.tencent_english, [])

    def test_reference_has_its_own_order_and_numbers_for_different_line_counts(self):
        references = [f"Reference line {index}" for index in range(1, 44)]
        texts = [f"Speech segment {index}" for index in range(1, 19)]
        rendered = TranscriptParser(render_transcript(
            texts, reference_text="\n".join(references), reference_title="English reference",
        ))

        self.assertEqual(rendered.segments, texts)
        self.assertEqual(rendered.numbers, [f"{index:02d}" for index in range(1, 19)])
        self.assertEqual(rendered.references, references)
        self.assertEqual(rendered.reference_numbers, [f"R{index:02d}" for index in range(1, 44)])
        self.assertEqual(rendered.reference_headings, ["English reference"])
        self.assertFalse(rendered.reference_inside_segment)

        fewer_references = TranscriptParser(render_transcript(texts, reference_text="Only one line"))
        self.assertEqual(fewer_references.segments, texts)
        self.assertEqual(fewer_references.references, ["Only one line"])

    def test_reference_is_literal_and_visible_before_any_speech_results(self):
        title = 'Reference <script>bad()</script> "source"'
        lines = ['<img src=x onerror="bad()"> & hello', "", "Last line."]
        rendered = TranscriptParser(render_transcript(
            [], reference_text="\n".join(lines), reference_title=title,
        ))

        self.assertEqual(rendered.segments, [])
        self.assertEqual(rendered.references, lines)
        self.assertEqual(rendered.reference_numbers, ["R01", "R02", "R03"])
        self.assertEqual(rendered.reference_headings, [title])
        self.assertNotIn("script", rendered.tags)
        self.assertNotIn("img", rendered.tags)

    def test_reference_pane_is_absent_by_default(self):
        for texts in ([], ["A speech segment."]):
            with self.subTest(texts=texts):
                markup = render_transcript(texts)
                rendered = TranscriptParser(markup)

                self.assertEqual(rendered.references, [])
                self.assertEqual(rendered.reference_headings, [])
                self.assertNotIn("reference-pane", markup)
                self.assertNotIn("Reference lines follow the file", markup)

    def test_download_preserves_multiline_text_and_translation_states(self):
        exported = export_conversation(
            ["第一行\n第二行", "等候中", "失敗", "新的一段"],
            ["First line\nSecond line", None, None],
            [None, None, "The service timed out."],
            ["Line one\nLine two", "Waiting", None],
            [None, None, None, "The model failed."],
        )

        self.assertEqual(exported, (
            "01\nOriginal: 第一行\n第二行\nEnglish (OpenAI): First line\nSecond line\n"
            "English (Tencent local): Line one\nLine two\n\n"
            "02\nOriginal: 等候中\nEnglish (OpenAI): [Translation pending]\n"
            "English (Tencent local): Waiting\n\n"
            "03\nOriginal: 失敗\nEnglish (OpenAI): [Translation unavailable]\n"
            "English (Tencent local): [Translation pending]\n\n"
            "04\nOriginal: 新的一段\nEnglish (OpenAI): [Translation pending]\n"
            "English (Tencent local): [Translation unavailable]"
        ))
        self.assertEqual(export_conversation([], [], []), "")

    def test_astra_drafts_are_replaced_by_corrected_or_confirmed_text(self):
        drafts = ["Fast first.", "Fast second."]
        waiting = TranscriptParser(render_transcript(
            ["第一句", "第二句"], drafts, slow_lane={"statuses": ["waiting", "draft"]},
        ))
        reviewed = TranscriptParser(render_transcript(
            ["第一句", "第二句"], drafts,
            slow_lane={
                "translations": ["Corrected first.", "Fast second."],
                "statuses": ["corrected", "confirmed"],
            },
        ))

        self.assertEqual(waiting.astra_english, drafts)
        self.assertEqual(waiting.astra_statuses, ["Draft", "Draft"])
        self.assertEqual(reviewed.astra_english, ["Corrected first.", "Fast second."])
        self.assertEqual(reviewed.astra_statuses, ["Corrected", "Confirmed"])
        self.assertEqual(reviewed.english, drafts)

    def test_different_visible_and_stored_versions_are_explicit_in_display_and_export(self):
        snapshot = {
            "translations": ["A later correction."], "statuses": ["corrected"],
            "authoritative": ["The sealed version."],
            "segments": [{"segment_id": "segment-17"}],
        }
        live = TranscriptParser(render_transcript(["原文"], slow_lane=snapshot))
        final_snapshot = dict(snapshot, view="authoritative")
        final = TranscriptParser(render_transcript(["原文"], slow_lane=final_snapshot))

        self.assertEqual(live.astra_english, ["A later correction."])
        self.assertEqual(live.astra_statuses, ["Corrected"])
        self.assertEqual(final.astra_english, ["The sealed version."])
        self.assertEqual(final.astra_statuses, ["Final · Stored version"])
        self.assertEqual(final.segment_ids, ["segment-17"])
        for state in (snapshot, final_snapshot):
            exported = export_conversation(["原文"], [], [], slow_lane=state)
            self.assertIn("A later correction.", exported)
            self.assertIn("The sealed version.", exported)
            self.assertIn("Segment ID: segment-17", exported)
            self.assertIn("Astra status:", exported)

    def test_late_accepted_correction_is_final_without_relabelling_unchanged_screen_text(self):
        snapshot = {
            "translations": ["Fast draft."], "statuses": ["corrected"],
            "authoritative": ["A later accepted correction."],
            "segments": [{"segment_id": "segment-18", "version": 2, "screen_version": 1}],
        }
        live = TranscriptParser(render_transcript(["原文"], slow_lane=snapshot))
        final = TranscriptParser(render_transcript(
            ["原文"], slow_lane=dict(snapshot, view="authoritative"),
        ))

        self.assertEqual(live.astra_english, ["Fast draft."])
        self.assertEqual(live.astra_statuses, ["Draft · Final correction available"])
        self.assertEqual(final.astra_english, ["A later accepted correction."])
        self.assertEqual(final.astra_statuses, ["Corrected · Final"])
        exported = export_conversation(["原文"], [], [], slow_lane=snapshot)
        self.assertIn("English (Astra correction): Fast draft.", exported)
        self.assertIn("Astra authoritative: A later accepted correction.", exported)

    def test_authoritative_view_keeps_unsealed_rows_pending(self):
        rendered = TranscriptParser(render_transcript(
            ["原文"], ["Fast draft."],
            slow_lane={
                "translations": ["An unsealed correction."], "statuses": ["corrected"],
                "authoritative": [None], "view": "authoritative",
            },
        ))

        self.assertEqual(rendered.astra_english, ["Awaiting final text…"])
        self.assertEqual(rendered.astra_statuses, ["Pending final"])

    def test_degraded_reviews_keep_fallbacks_honest_and_missing_rows_visible(self):
        snapshot = {
            "translations": ["One.", "Two.", "Three."],
            "statuses": ["timeout", "paused", "failed", "waiting"],
            "authoritative": ["One.", "Two.", "Three.", None],
        }
        texts = ["一", "二", "三", "四", "五"]
        rendered = TranscriptParser(render_transcript(texts, slow_lane=snapshot))
        final = TranscriptParser(render_transcript(texts, slow_lane=dict(snapshot, view="authoritative")))

        self.assertEqual(rendered.segments, texts)
        self.assertEqual(rendered.astra_english, [
            "One.", "Two.", "Three.", "Waiting for fast translation…", "Waiting for fast translation…",
        ])
        self.assertEqual(rendered.astra_statuses, [
            "Draft · Timed out", "Draft · Paused", "Draft · Review failed", "Waiting", "Waiting",
        ])
        self.assertEqual(final.astra_statuses[:3], [
            "Final fallback · Timed out", "Final fallback · Paused", "Final fallback · Review failed",
        ])

    def test_astra_and_speaker_content_is_escaped_and_references_remain_independent(self):
        correction = '<script>bad()</script> & corrected'
        segment_id = 'id" onmouseover="bad()'
        speaker = '<img src=x> Speaker 1 + Speaker 2'
        rendered = TranscriptParser(render_transcript(
            ["原文", "另一句"], slow_lane={
                "translations": [correction], "statuses": ["corrected"],
                "segments": [{"segment_id": segment_id}],
            },
            speakers=[speaker, None], reference_text="First\nSecond\nThird",
        ))

        self.assertEqual(rendered.astra_english, [correction, "Waiting for fast translation…"])
        self.assertEqual(rendered.segment_ids, [segment_id, None])
        self.assertEqual(rendered.speakers, [speaker, "Speaker unknown"])
        self.assertEqual(rendered.references, ["First", "Second", "Third"])
        self.assertFalse(rendered.reference_inside_segment)
        self.assertNotIn("script", rendered.tags)
        self.assertNotIn("img", rendered.tags)
        exported = export_conversation(["原文", "另一句"], [], [], speakers=["Speaker 1 + Speaker 2"])
        self.assertIn("Speaker: Speaker 1 + Speaker 2", exported)
        self.assertIn("Speaker: Unknown", exported)

    def test_optional_astra_column_handles_empty_state_and_is_absent_by_default(self):
        ordinary = render_transcript(["Original."])
        self.assertNotIn("Astra", ordinary)
        self.assertNotIn("Astra", export_conversation(["Original."], [], []))
        empty = render_transcript([], slow_lane={}, reference_text="A reference remains available.")
        rendered = TranscriptParser(empty)
        self.assertIn("Astra · Correction", empty)
        self.assertEqual(rendered.astra_english, [])
        self.assertEqual(rendered.references, ["A reference remains available."])

    def test_source_fallback_is_not_labelled_an_english_draft(self):
        snapshot = {
            "translations": ["原始語音文字"], "authoritative": ["原始語音文字"],
            "statuses": ["failed"],
            "segments": [{"segment_id": "seg-1", "source_text": "原始語音文字", "source_fallback": True}],
        }
        markup = render_transcript(["原始語音文字"], slow_lane=snapshot)
        rendered = TranscriptParser(markup)
        final = TranscriptParser(render_transcript(
            ["原始語音文字"], slow_lane=dict(snapshot, view="authoritative"),
        ))

        self.assertEqual(rendered.astra_english, ["原始語音文字"])
        self.assertEqual(rendered.astra_statuses, ["Source fallback"])
        self.assertEqual(final.astra_statuses, ["Final · Source fallback"])
        self.assertNotIn('class="translation-text astra-text" lang="en"', markup)
        self.assertIn("Astra status: Source fallback", export_conversation(
            ["原始語音文字"], [], [], slow_lane=snapshot,
        ))

    def test_late_english_correction_preserves_source_fallback_label_on_stable_screen(self):
        snapshot = {
            "translations": ["原始語音文字"], "authoritative": ["An English correction."],
            "statuses": ["corrected"],
            "segments": [{
                "source_text": "原始語音文字", "source_fallback": False,
                "screen_source_fallback": True, "version": 2, "screen_version": 1,
            }],
        }
        markup = render_transcript(["原始語音文字"], slow_lane=snapshot)
        live = TranscriptParser(markup)
        final = TranscriptParser(render_transcript(
            ["原始語音文字"], slow_lane=dict(snapshot, view="authoritative"),
        ))

        self.assertEqual(live.astra_statuses, ["Source fallback"])
        self.assertNotIn('class="translation-text astra-text" lang="en"', markup)
        self.assertEqual(final.astra_english, ["An English correction."])
        self.assertEqual(final.astra_statuses, ["Corrected · Final"])


if __name__ == "__main__":
    unittest.main()
