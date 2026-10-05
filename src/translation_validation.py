"""Reject untranslated East Asian scripts in English translation output."""

import re


# Script ranges rather than a Latin-only rule keep accented names, symbols,
# numbers, and identifiers valid. Include supplementary Han planes so newer
# ideographs remain covered even when Python's Unicode database is older.
_CJK = re.compile(
    "[\u3005-\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002ffff\U00030000-\U0003ffff"
    "\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff"
    "\uff66-\uff9f\U0001aff0-\U0001afff\U0001b000-\U0001b16f"
    "\u3105-\u312f\u31a0-\u31bf"
    "\u1100-\u11ff\u3131-\u318e\ua960-\ua97f\uac00-\ud7ff\uffa0-\uffdc]"
)


def contains_cjk(text: str) -> bool:
    """Detect Han, kana, Bopomofo, or Hangul, including mixed English output.

    This is a source-script guard, not a general language detector: Latin
    transliterations and accented English names are intentionally accepted.
    """
    return _CJK.search(text) is not None
