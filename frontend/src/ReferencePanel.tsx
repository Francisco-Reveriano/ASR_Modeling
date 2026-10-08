import { useEffect, useRef, useState } from 'react';
import { api, errorMessage } from './api';
import type { ReferenceOptions, ReferencePreview } from './types';

export interface ReferenceSelection { file: File | null; options: ReferenceOptions; ready: boolean }
export default function ReferencePanel({ disabled, onChange }: { disabled: boolean; onChange: (value: ReferenceSelection) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] = useState<ReferenceOptions>({ format: 'auto' });
  const [preview, setPreview] = useState<ReferencePreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [stale, setStale] = useState(false);
  const generation = useRef(0);
  useEffect(() => () => { generation.current += 1; }, []);
  function choose(next: File | null) {
    generation.current += 1; setFile(next); setPreview(null); setError(''); setBusy(false);
    setOptions({ format: 'auto' }); onChange({ file: next, options: { format: 'auto' }, ready: false });
  }
  function configure(next: ReferenceOptions) {
    generation.current += 1; setOptions(next); setStale(true); setBusy(false);
    onChange({ file, options: next, ready: false });
  }
  async function load() {
    if (!file) return;
    const request = ++generation.current;
    setBusy(true); setError('');
    try {
      const result = await api.preview(file, options);
      if (request !== generation.current) return;
      const resolved = { ...options, sheet: result.sheet, header_row: result.header_row, source_column: result.source_column, english_column: result.english_column };
      setPreview(result); setOptions(resolved); setStale(false);
      onChange({ file, options: resolved, ready: !result.needs_source_selection && !!result.reference });
    } catch (failure) { if (request === generation.current) setError(errorMessage(failure)); }
    finally { if (request === generation.current) setBusy(false); }
  }
  const table = preview?.tables?.find(item => item.name === options.sheet) || preview?.tables?.[0];
  return <section className="reference-panel" aria-label="Evaluation reference">
    <label className="field">Reference file<input type="file" accept=".txt,.srt,.vtt,.xlsx,.csv,.tsv" disabled={disabled} onChange={event => choose(event.target.files?.[0] || null)} /></label>
    <p className="hint">TXT, SRT, VTT, XLSX, CSV or TSV · up to 1 MiB. References remain on this computer and never enter model prompts.</p>
    {file && <>
      <label className="field">Reference format<select value={options.format} disabled={disabled} onChange={event => configure({ ...options, format: event.target.value as 'auto' | 'plain' })}><option value="auto">Detect and remove transcript metadata</option><option value="plain">Plain text, keep all content</option></select></label>
      {!!preview?.tables?.length && <div className="reference-mapping">
        <label className="field">Sheet<select value={options.sheet || ''} disabled={disabled} onChange={event => configure({ format: options.format, sheet: event.target.value })}>{preview?.tables?.map(item => <option key={item.name}>{item.name}</option>)}</select></label>
        <label className="field">Header row <span className="hint">0 = no header</span><input type="number" min="0" max="10000" placeholder="Auto" value={options.header_row ?? ''} disabled={disabled} onChange={event => configure({ ...options, header_row: event.target.value === '' ? undefined : Number(event.target.value), source_column: undefined, english_column: undefined })} /></label>
        <label className="field">Source transcript column<select value={options.source_column ?? ''} disabled={disabled} onChange={event => configure({ ...options, source_column: event.target.value === '' ? null : Number(event.target.value) })}><option value="">Choose source column</option>{table?.columns.map((name, index) => <option key={index} value={index}>{name}</option>)}</select></label>
        <label className="field">English reference column<select value={options.english_column ?? ''} disabled={disabled} onChange={event => configure({ ...options, english_column: event.target.value === '' ? null : Number(event.target.value) })}><option value="">None</option>{table?.columns.map((name, index) => <option key={index} value={index}>{name}</option>)}</select></label>
      </div>}
      <button className="button secondary full" disabled={disabled || busy} onClick={() => void load()}>{busy ? 'Reading reference…' : preview ? 'Update reference preview' : 'Preview reference'}</button>
      {stale && <p className="hint">Update the preview to apply these column settings.</p>}
      {preview?.needs_source_selection && <p className="notice small">Select a source transcript column, then update the preview.</p>}
      {preview?.reference && <details className="reference-preview" open><summary>Source reference preview</summary><pre>{String(preview.reference.text || '')}</pre><p className="hint">{String(preview.reference.format || '')} · {Number(preview.reference.segment_count || 0)} segments</p></details>}
      {preview?.reference_view && preview.reference_view.text !== preview.reference?.text && <details className="reference-preview"><summary>{String(preview.reference_view.title || 'English reference')} preview</summary><pre>{String(preview.reference_view.text || '')}</pre></details>}
    </>}
    {error && <p className="notice error small" role="alert">{error}</p>}
  </section>;
}
