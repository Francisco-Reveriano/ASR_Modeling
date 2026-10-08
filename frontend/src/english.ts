// Keep this source-script guard aligned with src/translation_validation.py.
// This does not identify every language: accented names and Latin identifiers remain valid.
const untranslatedScript = /[\u3005-\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u{20000}-\u{2ffff}\u{30000}-\u{3ffff}\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff\uff66-\uff9f\u{1aff0}-\u{1afff}\u{1b000}-\u{1b16f}\u3105-\u312f\u31a0-\u31bf\u1100-\u11ff\u3131-\u318e\ua960-\ua97f\uac00-\ud7ff\uffa0-\uffdc]/u;

export function validatedEnglish(value: unknown, sourceFallback = false): string | null {
  if (sourceFallback || typeof value !== 'string' || !value.trim() || untranslatedScript.test(value)) return null;
  return value.trim();
}
