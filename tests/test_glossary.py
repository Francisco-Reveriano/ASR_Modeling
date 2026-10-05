"""Check local terminology, DNT preservation, and learning with synthetic data."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openpyxl import Workbook

from src.glossary import Glossary, MAX_LEARNED, compare_dnt, load_glossary


class GlossaryTests(unittest.TestCase):
    def test_empty_glossary_has_only_recognizable_identifier_protection(self):
        glossary = load_glossary()

        self.assertEqual(glossary.entries, ())
        self.assertEqual(glossary.retrieve("Any ordinary words"), [])
        self.assertEqual(glossary.learned_terms(), [])
        self.assertEqual(glossary.dnt_hits("機台EXP-07/ETCH-07, LOT1234; ordinary pre-run words"),
                         ["EXP-07", "ETCH-07", "LOT1234"])

    def test_retrieval_matches_aliases_and_whole_terms_with_priority(self):
        glossary = Glossary([
            {"term_src": "蝕刻", "term_tgt": "etching", "aliases_src": "乾式蝕刻|dry etch", "priority": 2},
            {"term_src": "CMP", "term_tgt": "chemical mechanical polishing", "priority": 5},
            {"term_src": "etch", "term_tgt": "etch", "priority": 0},
        ])

        matches = glossary.retrieve("先 CMP，再 dry etch；不要 sketch")

        self.assertEqual([item["term_src"] for item in matches], ["CMP", "蝕刻", "etch"])
        self.assertEqual(glossary.retrieve("sketch"), [])
        self.assertEqual(glossary.retrieve("乾式蝕刻")[0]["term_tgt"], "etching")
        self.assertEqual(len(glossary.retrieve("CMP dry etch", limit=1)), 1)
        self.assertEqual(glossary.retrieve("CMP", limit=0), [])

    def test_retrieval_is_bounded_and_returned_data_cannot_mutate_snapshot(self):
        entries = [{"term_src": f"term{index}", "term_tgt": f"meaning{index}", "aliases_src": [f"alias{index}"]}
                   for index in range(50)]
        glossary = Glossary(entries)
        entries[0]["term_src"] = "changed outside"
        entries[0]["aliases_src"].append("outside alias")
        matches = glossary.retrieve(" ".join(f"term{index}" for index in range(50)), limit=1000)

        self.assertEqual(len(matches), 40)
        matches[0]["term_tgt"] = "changed result"
        matches[0]["aliases_src"].append("result alias")
        self.assertEqual(glossary.retrieve("term0")[0]["term_tgt"], "meaning0")
        self.assertEqual(glossary.retrieve("outside alias result alias"), [])
        with self.assertRaises(TypeError):
            glossary.entries[0]["term_tgt"] = "not allowed"
        self.assertIsInstance(glossary.entries[0]["aliases_src"], tuple)

    def test_dnt_checks_preserve_spelling_case_counts_and_detect_added_identifiers(self):
        glossary = Glossary([{"term_src": "Teams", "dnt": True, "aliases_src": "MS Teams"}])
        source = "Use MS Teams with EXP-07, then EXP-07 and ETCH-07."
        valid = "EXP-07、EXP-07 和 ETCH-07 使用 MS Teams。"
        self.assertTrue(glossary.compare_dnt(source, valid)["ok"])

        changed = compare_dnt(source, "Use ms teams with EXP-07 and ETCH-08.", glossary)

        self.assertEqual(changed["source"], ["MS Teams", "EXP-07", "EXP-07", "ETCH-07"])
        self.assertEqual(changed["missing"], ["MS Teams", "EXP-07", "ETCH-07"])
        self.assertEqual(changed["added"], ["ms teams", "ETCH-08"])
        self.assertFalse(changed["ok"])
        self.assertFalse(compare_dnt("EXP-07", "exp-07")["ok"])

    def test_learning_requires_two_distinct_segments_and_keeps_evidence(self):
        glossary = Glossary()
        self.assertFalse(glossary.observe("新製程", "new process", "segment-1"))
        self.assertFalse(glossary.observe("新製程", "new process", "segment-1"))
        self.assertEqual(glossary.learned_terms(), [])
        self.assertTrue(glossary.observe(" 新製程 ", "New Process", "segment-2"))
        self.assertFalse(glossary.observe("新製程", "new process", "segment-3"))

        learned = glossary.learned_terms()
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0]["term_src"], "新製程")
        self.assertEqual(learned[0]["term_tgt"], "new process")
        self.assertEqual(learned[0]["source"], "learned")
        self.assertEqual(learned[0]["evidence_segment_ids"], ["segment-1", "segment-2", "segment-3"])
        learned[0]["evidence_segment_ids"].clear()
        self.assertEqual(len(glossary.learned_terms()[0]["evidence_segment_ids"]), 3)
        self.assertEqual(glossary.retrieve("請測試新製程")[0]["term_tgt"], "new process")

    def test_learning_rejects_master_alias_dnt_and_promoted_term_conflicts(self):
        glossary = Glossary([
            {"term_src": "蝕刻", "term_tgt": "etching", "aliases_src": "乾蝕刻"},
            {"term_src": "Teams", "dnt": True},
        ])
        for source, target in (("蝕刻", "wrong"), ("乾蝕刻", "wrong"), ("Teams", "meetings"),
                               ("EXP-07 保養", "EXP-08 maintenance"), ("", "empty")):
            with self.subTest(source=source):
                self.assertFalse(glossary.observe(source, target, "segment-1"))
                self.assertFalse(glossary.observe(source, target, "segment-2"))
        self.assertEqual(glossary.learned_terms(), [])
        self.assertFalse(glossary.observe("新術語", "first meaning", "segment-1"))
        self.assertTrue(glossary.observe("新術語", "first meaning", "segment-2"))
        self.assertFalse(glossary.observe("新術語", "different meaning", "segment-3"))
        self.assertFalse(glossary.observe("新術語", "different meaning", "segment-4"))
        self.assertEqual(glossary.retrieve("新術語")[0]["term_tgt"], "first meaning")

    def test_different_candidate_pairs_do_not_combine_evidence(self):
        glossary = Glossary()
        self.assertFalse(glossary.observe("新詞", "first meaning", "one"))
        self.assertFalse(glossary.observe("新詞", "second meaning", "two"))
        self.assertEqual(glossary.learned_terms(), [])
        self.assertTrue(glossary.observe("新詞", "first meaning", "three"))
        self.assertEqual(glossary.learned_terms()[0]["evidence_segment_ids"], ["one", "three"])

    def test_learning_cap_does_not_replace_existing_terms(self):
        glossary = Glossary()
        for index in range(MAX_LEARNED + 1):
            glossary.observe(f"術語{index}", f"meaning {index}", "first")
            promoted = glossary.observe(f"術語{index}", f"meaning {index}", "second")
            self.assertEqual(promoted, index < MAX_LEARNED)
        self.assertEqual(len(glossary.learned_terms()), MAX_LEARNED)
        self.assertEqual(glossary.learned_terms()[0]["term_src"], "術語0")

    def test_concurrent_duplicate_observations_count_once(self):
        glossary = Glossary()
        with ThreadPoolExecutor(max_workers=4) as pool:
            first = list(pool.map(lambda _: glossary.observe("術語", "term", "same-segment"), range(20)))
            second = list(pool.map(lambda _: glossary.observe("術語", "term", "next-segment"), range(20)))
        self.assertFalse(any(first))
        self.assertEqual(sum(second), 1)
        self.assertEqual(glossary.learned_terms()[0]["evidence_segment_ids"], ["next-segment", "same-segment"])

    def test_master_reload_keeps_compatible_learning_and_removes_conflicts_atomically(self):
        glossary = Glossary([{"term_src": "舊詞", "term_tgt": "old term"}])
        for source, target in (("可保留", "retained term"), ("衝突詞", "old meaning"), ("NewBrand", "brand translation")):
            glossary.observe(source, target, "one")
            glossary.observe(source, target, "two")
        replacement = Glossary([
            {"term_src": "新詞", "term_tgt": "new term"},
            {"term_src": "衝突詞", "term_tgt": "authoritative meaning"},
            {"term_src": "NewBrand", "dnt": True},
        ])

        glossary.replace_master(replacement)

        self.assertEqual(glossary.retrieve("舊詞"), [])
        self.assertEqual(glossary.retrieve("新詞")[0]["term_tgt"], "new term")
        self.assertEqual(glossary.retrieve("衝突詞")[0]["term_tgt"], "authoritative meaning")
        self.assertEqual(glossary.dnt_hits("NewBrand"), ["NewBrand"])
        self.assertEqual([entry["term_src"] for entry in glossary.learned_terms()], ["可保留"])
        self.assertEqual(glossary.learned_terms()[0]["evidence_segment_ids"], ["one", "two"])
        replacement.replace_master(Glossary())
        self.assertEqual(len(glossary.entries), 3)

    def test_reload_resets_pending_evidence_and_deduplicates_promoted_master_pair(self):
        glossary = Glossary()
        glossary.observe("學習詞", "learned term", "one")
        glossary.observe("學習詞", "learned term", "two")
        glossary.observe("待確認", "pending term", "one")
        glossary.replace_master(Glossary([{"term_src": "學習詞", "term_tgt": "learned term"}]))

        self.assertEqual(len(glossary.retrieve("學習詞")), 1)
        self.assertEqual(len(glossary.learned_terms()), 1)
        self.assertFalse(glossary.observe("待確認", "pending term", "two"))
        self.assertTrue(glossary.observe("待確認", "pending term", "three"))

    def test_conflicting_master_aliases_and_invalid_entries_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            Glossary([{"term_src": "one", "term_tgt": "first", "aliases_src": "shared"},
                      {"term_src": "two", "term_tgt": "second", "aliases_src": "shared"}])
        for entry, message in (({"term_src": "x"}, "term_tgt"),
                               ({"term_src": "x", "dnt": True, "term_tgt": "y"}, "preserve"),
                               ({"term_src": "x", "term_tgt": "y", "priority": 1.5}, "integer")):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, message):
                Glossary([entry])


class GlossaryLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))

    def write(self, name, content):
        path = self.directory / name
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def test_csv_and_utf16_tsv_load_canonical_fields_and_pipe_aliases(self):
        for suffix, separator, encoding in (("csv", ",", "utf-8-sig"), ("tsv", "\t", "utf-16")):
            with self.subTest(suffix=suffix):
                rows = [separator.join(["entry_id", "term_src", "term_tgt", "aliases_src", "domain", "priority"]),
                        separator.join(["etch", "蝕刻", "etching", "乾式蝕刻|dry etch", "semiconductor", "9"])]
                glossary = load_glossary(self.write(f"terms.{suffix}", "\n".join(rows).encode(encoding)))
                entry = glossary.retrieve("dry etch")[0]
                self.assertEqual(entry["entry_id"], "etch")
                self.assertEqual(entry["term_tgt"], "etching")
                self.assertEqual(entry["aliases_src"], ["乾式蝕刻", "dry etch"])
                self.assertEqual(entry["priority"], 9)
                self.assertEqual(entry["source"], f"terms.{suffix}")

    def test_json_and_separate_dnt_list_load_without_writing_files(self):
        path = self.write("terms.json", json.dumps({"entries": [
            {"entry_id": "process", "term_src": "製程", "term_tgt": "process", "aliases_src": ["工藝"]},
        ]}))
        dnt_path = self.write("dnt.txt", "# Keep product names\nTeams\nEXP-07\n")
        original = path.read_bytes()
        glossary = load_glossary(path, dnt_path)

        self.assertEqual(glossary.retrieve("工藝")[0]["term_tgt"], "process")
        self.assertEqual(glossary.dnt_hits("Teams EXP-07"), ["Teams", "EXP-07"])
        self.assertFalse(glossary.observe("新詞", "new term", 1))
        self.assertTrue(glossary.observe("新詞", "new term", 2))
        self.assertEqual(path.read_bytes(), original)

    def test_json_dnt_strings_and_csv_dnt_rows_are_supported(self):
        for name, content in (("dnt.json", '["Teams", "ETCH-07"]'),
                              ("dnt.csv", "term_src\nTeams\nETCH-07\n")):
            with self.subTest(name=name):
                glossary = load_glossary(dnt_path=self.write(name, content))
                self.assertEqual(glossary.dnt_hits("Teams ETCH-07"), ["Teams", "ETCH-07"])
                self.assertTrue(all(entry["dnt"] for entry in glossary.entries))

    def test_xlsx_loads_values_and_rejects_formula_cells(self):
        workbook = Workbook()
        workbook.active.append(["term_src", "term_tgt"])
        workbook.active.append(["蝕刻", "etching"])
        path = self.directory / "terms.xlsx"
        workbook.save(path)
        self.assertEqual(load_glossary(path).retrieve("蝕刻")[0]["term_tgt"], "etching")
        workbook.active["B2"] = '=CONCAT("etch", "ing")'
        workbook.save(path)
        workbook.close()
        with self.assertRaisesRegex(ValueError, "formula or error"):
            load_glossary(path)

    def test_missing_or_malformed_configured_files_raise_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_glossary(self.directory / "missing.json")
        for name, content, message in (("bad.json", "not JSON", "JSON"),
                                       ("bad.csv", "source,target\nx,y", "term_src"),
                                       ("encoding.csv", b"term_src,term_tgt\nx,\xff", "UTF-8")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                load_glossary(self.write(name, content))


if __name__ == "__main__":
    unittest.main()
