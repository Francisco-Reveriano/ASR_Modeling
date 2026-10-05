"""Check aligned conversation rows, safe model text, and readable downloads."""

from html.parser import HTMLParser
from copy import deepcopy
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
        self.astra_cell_classes = []
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
            self.astra_cell_classes.append(classes)
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
    def test_selected_providers_hide_other_columns_and_download_fields(self):
        snapshot = {"translations": ["Reviewed English"], "authoritative": ["Reviewed English"],
                    "statuses": ["corrected"], "segments": [{"segment_id": "s:1"}]}
        for selected, expected in ((('openai',), (['Fast English'], [], [])),
                                   (('openai', 'astra'), (['Fast English'], [], ['Reviewed English'])),
                                   (('tencent',), ([], ['Local English'], [])),
                                   ((), ([], [], []))):
            with self.subTest(providers=selected):
                markup = render_transcript(["Original source"], ["Fast English"], [None],
                                           ["Local English"], [None], slow_lane=snapshot,
                                           providers=selected, reference_text="Reference line")
                parsed = TranscriptParser(markup)
                self.assertEqual((parsed.english, parsed.tencent_english, parsed.astra_english), expected)
                self.assertEqual(parsed.segments, ["Original source"])
                self.assertEqual(parsed.references, ["Reference line"])
                self.assertIn(f"provider-columns-{len(selected)}", markup)
                self.assertEqual(markup.count('class="translation-cell '), len(selected))
                downloaded = export_conversation(["Original source"], ["Fast English"], [None],
                                                 ["Local English"], [None], slow_lane=snapshot,
                                                 providers=selected)
                for provider, label in (("openai", "English (OpenAI)"), ("tencent", "English (Tencent local)"),
                                        ("astra", "English (Astra correction)")):
                    self.assertEqual(label in downloaded, provider in selected)
                self.assertIn("Original: Original source", downloaded)

    def test_provider_selection_rejects_unknown_or_duplicate_values_safely(self):
        for selected in (("openai", "PRIVATE"), ("openai", "openai"), "openai", ([],)):
            with self.subTest(providers=selected):
                for function in (render_transcript, export_conversation):
                    with self.assertRaisesRegex(ValueError, "providers") as raised:
                        function([], [], [], providers=selected)
                    self.assertNotIn("PRIVATE", str(raised.exception))

    def test_first_review_stays_visible_without_final_claim_while_context_review_is_unfinished(self):
        notes = {"waiting": "Context review pending", "reviewing": "Reviewing in context",
                 "failed": "Context review failed", "blocked": "Waiting for earlier review"}
        for stage, note in notes.items():
            for view in ("speculative", "authoritative"):
                with self.subTest(stage=stage, view=view):
                    snapshot = {"translations": ["First reviewed English"], "authoritative": ["First reviewed English"],
                                "statuses": ["corrected"], "view": view, "segments": [{
                                    "segment_id": "s:1", "first_pass_status": "corrected",
                                    "conversation_review_status": stage, "conversation_review_error": "PRIVATE error",
                                }]}
                    before = deepcopy(snapshot)
                    markup = render_transcript(["Original source"], ["Fast draft"], [None], slow_lane=snapshot)
                    parsed = TranscriptParser(markup)
                    self.assertEqual(parsed.astra_english, ["First reviewed English"])
                    self.assertEqual(parsed.astra_statuses, ["Corrected"])
                    self.assertIn("astra-corrected", parsed.astra_cell_classes[0])
                    self.assertIn(note, markup)
                    self.assertNotIn("PRIVATE error", markup)
                    downloaded = export_conversation(["Original source"], ["Fast draft"], [None], slow_lane=snapshot)
                    self.assertIn("Astra first review: Corrected", downloaded)
                    self.assertIn(f"Astra conversation review: {stage.capitalize()}", downloaded)
                    self.assertNotIn("Final", downloaded)
                    self.assertEqual(snapshot, before)

    def test_context_review_acceptance_marks_latest_english_without_losing_corrected_hint(self):
        for stage in ("corrected", "confirmed"):
            snapshot = {"translations": ["Context-reviewed English"], "authoritative": ["Context-reviewed English"],
                        "statuses": ["corrected"], "view": "authoritative", "segments": [{
                            "segment_id": "s:1", "first_pass_status": "corrected",
                            "conversation_review_status": stage,
                        }]}
            markup = render_transcript(["Original source"], ["Fast draft"], [None], slow_lane=snapshot)
            parsed = TranscriptParser(markup)
            self.assertEqual(parsed.astra_english, ["Context-reviewed English"])
            self.assertEqual(parsed.astra_statuses, ["Corrected · Final"])
            self.assertIn("astra-corrected", parsed.astra_cell_classes[0])
            self.assertIn("Reviewed in context", markup)
            self.assertIn('class="context-review-note complete"', markup)
            snapshot["translations"] = snapshot["authoritative"] = ["仍然中文"]
            invalid = render_transcript(["Original source"], ["Fast draft"], [None], slow_lane=snapshot)
            self.assertNotIn("Reviewed in context", invalid)
            self.assertNotIn('astra-cell astra-corrected', invalid)

    def test_filtered_background_is_visible_as_filtered_without_a_correction_badge(self):
        marker = "[Background speech filtered]"
        snapshot = {
            "translations": [marker], "authoritative": [marker], "statuses": ["filtered"],
            "segments": [{"segment_id": "s:1", "source_text": "Background speech", "filtered": True}],
        }
        for view in ("screen", "authoritative"):
            with self.subTest(view=view):
                state = dict(snapshot, view=view)
                rendered = TranscriptParser(render_transcript(
                    ["Background speech"], [marker], tencent_translations=["Local translation."], slow_lane=state,
                ))
                self.assertEqual(rendered.english, [marker])
                self.assertEqual(rendered.tencent_english, ["Local translation."])
                self.assertEqual(rendered.astra_english, [marker])
                self.assertEqual(rendered.astra_statuses, [
                    "Background filtered" + (" · Final" if view == "authoritative" else ""),
                ])
                self.assertNotIn("astra-corrected", rendered.astra_cell_classes[0])
                self.assertIn("Original: Background speech", export_conversation(
                    ["Background speech"], [marker], [], slow_lane=state,
                ))

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

        self.assertEqual(live.astra_english, ["The sealed version."])
        self.assertEqual(live.astra_statuses, ["Corrected"])
        self.assertEqual(final.astra_english, ["The sealed version."])
        self.assertEqual(final.astra_statuses, ["Corrected · Final"])
        self.assertEqual(final.segment_ids, ["segment-17"])
        for state in (snapshot, final_snapshot):
            exported = export_conversation(["原文"], [], [], slow_lane=state)
            self.assertIn("A later correction.", exported)
            self.assertIn("The sealed version.", exported)
            self.assertIn("Segment ID: segment-17", exported)
            self.assertIn("Astra status:", exported)

    def test_late_accepted_correction_replaces_the_live_draft_and_is_labelled_corrected(self):
        snapshot = {
            "translations": ["Fast draft."], "statuses": ["corrected"],
            "authoritative": ["A later accepted correction."],
            "segments": [{"segment_id": "segment-18", "version": 2, "screen_version": 1}],
        }
        live = TranscriptParser(render_transcript(["原文"], slow_lane=snapshot))
        final = TranscriptParser(render_transcript(
            ["原文"], slow_lane=dict(snapshot, view="authoritative"),
        ))

        self.assertEqual(live.astra_english, ["A later accepted correction."])
        self.assertEqual(live.astra_statuses, ["Corrected"])
        self.assertEqual(final.astra_english, ["A later accepted correction."])
        self.assertEqual(final.astra_statuses, ["Corrected · Final"])
        exported = export_conversation(["原文"], [], [], slow_lane=snapshot)
        self.assertIn("English (Astra correction): A later accepted correction.", exported)
        self.assertIn("Astra latest visible version: Fast draft.", exported)

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

    def test_only_accepted_english_corrections_tint_the_astra_cell_in_both_views(self):
        snapshot = {
            "translations": [
                "An older draft.", "Confirmed English.", "A failed draft.", None,
                "Mixed 中文.", "Literal source", "Unsealed English.",
            ],
            "authoritative": [
                "Accepted English correction.", "Confirmed English.", "A failed draft.", None,
                "Mixed 中文.", "Literal source", None,
            ],
            "statuses": ["corrected", "confirmed", "failed", "waiting", "corrected", "corrected", "corrected"],
            "segments": [{"source_fallback": index == 5} for index in range(7)],
        }
        for view in ("screen", "authoritative"):
            with self.subTest(view=view):
                rendered = TranscriptParser(render_transcript(
                    [f"Original {index}" for index in range(7)], slow_lane=dict(snapshot, view=view),
                ))

                self.assertEqual(
                    ["astra-corrected" in classes for classes in rendered.astra_cell_classes],
                    [True, False, False, False, False, False, False],
                )
                self.assertEqual(rendered.astra_english[0], "Accepted English correction.")
                self.assertEqual(
                    rendered.astra_statuses[0], "Corrected · Final" if view == "authoritative" else "Corrected",
                )

    def test_failed_final_review_requests_retry_without_displaying_a_draft_or_source(self):
        texts = ["第一句", "原文失敗", "等候翻譯", "審閱中"]
        snapshot = {
            "translations": ["An English draft.", "原文失敗", "等候翻譯", "A pending draft."],
            "authoritative": [None] * 4,
            "statuses": ["failed", "failed", "failed", "draft"],
            "segments": [
                {"error": "Correction failed."},
                {"error": "Correction failed.", "source_fallback": True},
                {"error": None, "source_fallback": True},
                {"error": None},
            ],
            "view": "authoritative",
        }
        rendered = TranscriptParser(render_transcript(texts, slow_lane=snapshot))

        self.assertEqual(rendered.astra_statuses, ["Review failed", "Review failed", "Pending final", "Pending final"])
        self.assertEqual(rendered.astra_english, [
            "Retry correction to finish this translation.", "Retry correction to finish this translation.",
            "Awaiting final text…", "Awaiting final text…",
        ])
        self.assertEqual(rendered.segments, texts)
        self.assertFalse(any("astra-corrected" in classes for classes in rendered.astra_cell_classes))
        exported = export_conversation(texts, [], [], slow_lane=snapshot)
        self.assertIn("English (Astra correction): [Retry correction to finish this translation.]", exported)
        self.assertEqual(exported.count("原文失敗"), 1)

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

    def test_source_fallback_stays_only_in_original_column_and_download_field(self):
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

        self.assertEqual(rendered.astra_english, ["Translation unavailable"])
        self.assertEqual(final.astra_english, ["Translation unavailable"])
        self.assertEqual(rendered.astra_statuses, ["Translation unavailable"])
        self.assertEqual(final.astra_statuses, ["Translation unavailable · Final"])
        self.assertEqual(rendered.segments, ["原始語音文字"])
        self.assertEqual(final.segments, ["原始語音文字"])
        self.assertNotIn('class="translation-text astra-text" lang="en"', markup)
        for state in (snapshot, dict(snapshot, view="authoritative")):
            exported = export_conversation(["原始語音文字"], [], [], slow_lane=state)
            self.assertEqual(exported.count("原始語音文字"), 1)
            self.assertIn("Original: 原始語音文字", exported)
            self.assertIn("English (Astra correction): [Translation unavailable]", exported)
            self.assertIn("Astra status: Translation unavailable", exported)

    def test_legacy_mixed_script_provider_outputs_are_not_presented_as_english(self):
        texts = ["上禮拜三那批布拉吉", "第二句"]
        mixed = "That batch of布拉吉 from last Wednesday."
        rendered = TranscriptParser(render_transcript(
            texts, [mixed, "Second sentence."],
            tencent_translations=[mixed, "The next sentence."],
        ))
        self.assertEqual(rendered.english, ["Translation unavailable", "Second sentence."])
        self.assertEqual(rendered.tencent_english, ["Translation unavailable", "The next sentence."])
        self.assertEqual(rendered.segments, texts)
        exported = export_conversation(texts, [mixed, "Second sentence."], [], [mixed, "The next sentence."])
        self.assertNotIn(mixed, exported)
        self.assertEqual(exported.count("[Translation unavailable]"), 2)
        self.assertIn(f"Original: {texts[0]}", exported)

    def test_invalid_astra_targets_are_unavailable_in_both_views_without_mutating_snapshot(self):
        texts = ["第一句", "第二句", "第三句", "Literal original"]
        snapshot = {
            "translations": ["That batch of布拉吉.", "Confirmed 中文.", "Corrected 中文.", "Literal original"],
            "authoritative": ["That batch of布拉吉.", "Confirmed 中文.", "Corrected 中文.", "Literal original"],
            "statuses": ["timeout", "confirmed", "corrected", "failed"],
            "segments": [
                {"segment_id": f"segment-{index}", "source_text": text, "source_fallback": index == 3}
                for index, text in enumerate(texts)
            ],
        }
        original = deepcopy(snapshot)
        for view in ("screen", "authoritative"):
            with self.subTest(view=view):
                state = dict(snapshot, view=view)
                rendered = TranscriptParser(render_transcript(texts, slow_lane=state))
                self.assertEqual(rendered.astra_english, ["Translation unavailable"] * 4)
                label = "Translation unavailable" + (" · Final" if view == "authoritative" else "")
                self.assertEqual(rendered.astra_statuses, [label] * 4)
                self.assertEqual(rendered.segments, texts)
                self.assertEqual(rendered.segment_ids, [f"segment-{index}" for index in range(4)])
                exported = export_conversation(texts, [], [], slow_lane=state)
                for target in snapshot["authoritative"][:3]:
                    self.assertNotIn(target, exported)
                self.assertEqual(exported.count("Literal original"), 1)
        self.assertEqual(snapshot, original)

    def test_download_supplementary_versions_never_leak_untranslated_source(self):
        snapshot = {
            "translations": ["原始語音文字", "An English draft."],
            "authoritative": ["An English correction.", "Stored 中文."],
            "statuses": ["corrected", "timeout"],
            "segments": [
                {"source_text": "原始語音文字", "source_fallback": False, "screen_source_fallback": True},
                {"source_text": "第二句", "source_fallback": False},
            ],
        }
        for view in ("screen", "authoritative"):
            with self.subTest(view=view):
                exported = export_conversation(
                    ["原始語音文字", "第二句"], [], [], slow_lane=dict(snapshot, view=view),
                )
                self.assertEqual(exported.count("原始語音文字"), 1)
                self.assertNotIn("Stored 中文.", exported)
                self.assertIn("English (Astra correction): An English correction.", exported)
                self.assertIn("Astra latest visible version: [Translation unavailable]", exported)
                self.assertIn("Astra authoritative: [Translation unavailable]", exported)

    def test_late_english_correction_replaces_source_fallback_on_screen(self):
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

        self.assertEqual(live.astra_english, ["An English correction."])
        self.assertEqual(live.astra_statuses, ["Corrected"])
        self.assertIn('class="translation-text astra-text" lang="en"', markup)
        self.assertEqual(final.astra_english, ["An English correction."])
        self.assertEqual(final.astra_statuses, ["Corrected · Final"])


if __name__ == "__main__":
    unittest.main()
