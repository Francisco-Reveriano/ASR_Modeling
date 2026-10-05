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
        self.tags = []
        self._capture = None
        self._provider = None
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        classes = dict(attrs).get("class", "").split()
        if "openai-cell" in classes:
            self._provider = self.english
        elif "tencent-cell" in classes:
            self._provider = self.tencent_english
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


if __name__ == "__main__":
    unittest.main()
