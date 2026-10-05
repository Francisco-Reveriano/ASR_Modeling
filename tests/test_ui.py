"""Selected English rendering and safe, independent local references."""

from copy import deepcopy
from html.parser import HTMLParser
import unittest

from src.ui import export_conversation, render_transcript


class TranscriptParser(HTMLParser):
    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.tags, self.statuses = [], []
        self.values = {}
        self._capture = None
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if "transcript-row" in classes:
            self.statuses.append(attrs.get("data-status"))
        for name in ("line-number", "segment-text", "translation-text", "translation-pending",
                     "translation-error", "translation-filtered", "reference-number",
                     "reference-text", "reference-heading"):
            if name in classes:
                self.values.setdefault(name, []).append("")
                self._capture = name

    def handle_data(self, data):
        if self._capture:
            self.values[self._capture][-1] += data

    def handle_endtag(self, tag):
        self._capture = None


class TranscriptRenderingTests(unittest.TestCase):
    def test_only_source_and_selected_english_are_rendered(self):
        markup = render_transcript(["原文"], ["English result."], [None],
                                   profile_label="Qwen fast", model="Qwen/example")
        parsed = TranscriptParser(markup)
        self.assertEqual(parsed.values["segment-text"], ["原文"])
        self.assertEqual(parsed.values["translation-text"], ["English result."])
        self.assertEqual(parsed.statuses, ["translated"])
        self.assertEqual(markup.count('class="translation-cell"'), 1)
        self.assertIn("Qwen fast", markup)
        for removed in ("OpenAI", "Tencent", "Astra", "Corrected", "Confirmed", "Speaker", "Review"):
            self.assertNotIn(removed, markup)

    def test_partial_results_keep_original_order_and_row_numbers(self):
        sources = ["第一句", "第二句", "第三句", "第四句"]
        parsed = TranscriptParser(render_transcript(sources, ["First.", None, "Third."],
                                                    [None, None, None, "PRIVATE failure"]))
        self.assertEqual(parsed.values["line-number"], ["01", "02", "03", "04"])
        self.assertEqual(parsed.values["segment-text"], sources)
        self.assertEqual(parsed.statuses, ["translated", "pending", "translated", "failed"])
        self.assertEqual(parsed.values["translation-pending"], ["Translating…"])
        self.assertEqual(parsed.values["translation-error"], ["Translation unavailable"])

    def test_pending_failed_filtered_and_invalid_targets_remain_distinct(self):
        markup = render_transcript(["原文"] * 6,
                                   [None, None, "marker", "That batch of布拉吉", " ", "stale"],
                                   [None, "PRIVATE key", None, None, None, "PRIVATE error"],
                                   filtered=[False, False, True])
        parsed = TranscriptParser(markup)
        self.assertEqual(parsed.statuses, ["pending", "failed", "filtered", "failed", "failed", "failed"])
        self.assertEqual(parsed.values["translation-filtered"], ["Background speech filtered"])
        for absent in ("布拉吉", "stale", "PRIVATE", "translation-text", "marker"):
            self.assertNotIn(absent, markup)

    def test_plain_filtered_marker_does_not_infer_filter_state(self):
        parsed = TranscriptParser(render_transcript(["source"], ["[Background speech filtered]"], []))
        self.assertEqual(parsed.statuses, ["translated"])
        self.assertEqual(parsed.values["translation-text"], ["[Background speech filtered]"])

    def test_reasoning_markup_is_unavailable_in_the_screen_and_txt(self):
        for target in ("<think>PRIVATE thought</think>English.", "<think>PRIVATE", "PRIVATE</think>",
                       "<ANALYSIS>PRIVATE</ANALYSIS>",
                       "<reasoning mode='brief'>PRIVATE</reasoning>",
                       "<|analysis|>PRIVATE", "<|begin_of_thought|>PRIVATE<|end_of_thought|>"):
            with self.subTest(target=target):
                markup = render_transcript(["原文"], [target], [])
                parsed = TranscriptParser(markup)
                self.assertEqual(parsed.statuses, ["failed"])
                self.assertEqual(parsed.values["translation-error"], ["Translation unavailable"])
                self.assertNotIn("translation-text", parsed.values)
                exported = export_conversation(["原文"], [target], [])
                self.assertIn("[Translation unavailable]", exported)
                self.assertIn("Original: 原文", exported)
                self.assertNotIn("PRIVATE", markup + exported)

    def test_normal_english_names_and_symbols_are_not_reasoning_markup(self):
        target = "I think José’s analysis of R2 > 0.7 is reasonable — 25 °C."
        parsed = TranscriptParser(render_transcript(["原文"], [target]))
        self.assertEqual(parsed.statuses, ["translated"])
        self.assertEqual(parsed.values["translation-text"], [target])

    def test_source_target_label_and_reference_are_escaped(self):
        source = '<script>alert("source")</script> & text'
        target = '<a href="javascript:bad()">Hello</a> & <iframe>text</iframe>'
        title = '<style>bad</style> "reference"'
        markup = render_transcript([source], [target], profile_label='<img src=x onerror="bad()">',
                                   reference_text='<input onfocus="bad()"> & hello', reference_title=title)
        parsed = TranscriptParser(markup)
        self.assertEqual(parsed.values["segment-text"], [source])
        self.assertEqual(parsed.values["translation-text"], [target])
        self.assertEqual(parsed.values["reference-heading"], [title])
        self.assertEqual(parsed.values["reference-text"], ['<input onfocus="bad()"> & hello'])
        for tag in ("script", "a", "iframe", "img", "style", "input"):
            self.assertNotIn(tag, parsed.tags)
        self.assertIn("&lt;img", markup)

    def test_references_have_independent_rows_even_before_results(self):
        parsed = TranscriptParser(render_transcript([], reference_text="One\n\nThree"))
        self.assertEqual(parsed.values["reference-text"], ["One", "", "Three"])
        self.assertEqual(parsed.values["reference-number"], ["R01", "R02", "R03"])
        self.assertNotIn("line-number", parsed.values)
        markup = render_transcript(["one", "two"], reference_segments=["Only one reference"])
        parsed = TranscriptParser(markup)
        self.assertEqual(parsed.values["reference-text"], ["Only one reference"])
        self.assertEqual(parsed.values["segment-text"], ["one", "two"])
        self.assertGreater(markup.index('class="reference-pane"'), markup.index("</ol>"))

    def test_empty_state_and_numbers_support_any_segment_count(self):
        markup = render_transcript([], profile_label="Selected model")
        self.assertIn("Selected model", markup)
        self.assertIn("Your conversation starts here", markup)
        self.assertNotIn("reference-pane", markup)
        parsed = TranscriptParser(render_transcript([str(number) for number in range(101)]))
        self.assertEqual(parsed.values["line-number"][-3:], ["99", "100", "101"])

    def test_render_and_export_do_not_mutate_snapshots(self):
        values = {"texts": ["原文"], "translations": ["English."], "errors": [None],
                  "filtered": [False], "timings": [{"start_s": 1, "end_s": 2}]}
        before = deepcopy(values)
        render_transcript(**values)
        export_conversation(**values)
        self.assertEqual(values, before)

    def test_txt_has_truthful_statuses_model_timing_and_originals(self):
        exported = export_conversation(["第一行\n第二行", "待定", "失敗", "背景", "錯誤"],
                                       ["First line\nSecond line", None, None, "marker", "Chinese中文"],
                                       [None, None, "PRIVATE failure"], filtered=[False, False, False, True],
                                       timings=[{"start_s": 1.25, "end_s": 3.5}],
                                       profile_label="Qwen fast", model="Qwen/example")
        self.assertIn("Original: 第一行\n第二行", exported)
        self.assertIn("English (Qwen fast): First line\nSecond line", exported)
        self.assertIn("Model: Qwen/example", exported)
        self.assertIn("Time: 1.250–3.500 s", exported)
        for status in ("Translated", "Pending", "Failed", "Filtered"):
            self.assertIn(f"Status: {status}", exported)
        for placeholder in ("[Translation pending]", "[Translation unavailable]", "[Background speech filtered]"):
            self.assertIn(placeholder, exported)
        for absent in ("PRIVATE", "Chinese中文", "marker", "Astra", "Tencent", "OpenAI", "Corrected"):
            self.assertNotIn(absent, exported)
        self.assertEqual(export_conversation([], [], []), "")


if __name__ == "__main__":
    unittest.main()
