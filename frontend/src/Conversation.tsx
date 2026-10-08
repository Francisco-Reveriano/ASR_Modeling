import { useState } from 'react';
import { validatedEnglish } from './english';
import type { Json, Metadata, Segment, SessionSnapshot, Translation } from './types';

export function timecode(seconds?: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const minutes = Math.floor(seconds / 60);
  return `${minutes.toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
}
const providerLabel: Record<string, string> = { openai: 'OpenAI', tencent: 'Tencent · local', astra: 'Astra correction' };
function Status({ value }: { value: string }) {
  const accepted = ['corrected', 'confirmed', 'complete', 'ready', 'final'].includes(value.toLowerCase());
  const failed = ['failed', 'unavailable', 'error'].includes(value.toLowerCase());
  return <span className={`status-label ${accepted ? 'accepted' : failed ? 'failed' : ''}`}>{value.replaceAll('_', ' ')}</span>;
}
function Source({ segment }: { segment: Segment }) { return <p className="source-text" lang="zh-Hant">{segment.source}</p>; }
export function Subtitles({ session, original, visible, size }: { session: SessionSnapshot | null; original: boolean; visible: boolean; size: string }) {
  const turns = session?.segments.slice(-3) || [];
  return <section className={`subtitle-stage size-${size} ${visible ? '' : 'subtitles-hidden'}`} aria-label="Live subtitles">
    <div className="stage-label"><span className="eyebrow">Live English</span><span>{session?.realtime ? 'Continuous captions' : 'Latest three turns'}</span></div>
    {!visible ? <div className="stage-empty"><span className="empty-symbol">Aa</span><h2>Subtitles are hidden</h2><p>Translation continues. Show subtitles to follow along again.</p></div> : session?.realtime ? <div className="realtime-subtitle" aria-live="polite" aria-atomic="true"><p>{session.realtime.english || 'Listening for English translation…'}</p>{session.realtime.incomplete && <Status value="Incomplete" />}</div> : turns.length ? <div className="subtitle-turns" aria-live="polite" aria-atomic="false">{turns.map(segment => <article className="subtitle-turn" key={segment.id}>
      <div className="turn-meta"><span>{segment.speaker || 'Speaker pending'}</span><Status value={segment.status} /></div>
      <p className={segment.english ? '' : 'pending-text'}>{segment.english || (segment.status === 'filtered' ? 'Background speech filtered' : ['failed', 'unavailable', 'error'].includes(segment.status) ? 'Translation unavailable' : 'Translating…')}</p>
      {original && <Source segment={segment} />}
    </article>)}</div> : <div className="stage-empty"><div className="sound-mark" aria-hidden="true"><i /><i /><i /><i /><i /></div><h2>{session ? session.status === 'preparing' ? 'Preparing your session' : 'Listening, whenever you’re ready' : 'Every voice. In English.'}</h2><p>{session ? 'Subtitles appear as speech is recognized and translated.' : 'Start subtitles to follow a conversation, or translate a recording.'}</p>{!session && <span className="stage-language">中文 + EN <span aria-hidden="true">→</span> English</span>}</div>}
  </section>;
}
function TranslationCell({ value }: { value?: Translation }) {
  return <div className="translation-cell"><p className={value?.text ? '' : 'pending-text'}>{value?.text || (value?.status === 'filtered' ? 'Background speech filtered' : value?.error || ['failed', 'unavailable', 'error'].includes(value?.status || '') ? 'Translation unavailable' : 'Translating…')}</p>{value && <Status value={value.status} />}{value?.error && <p className="cell-error">{value.error}</p>}</div>;
}
function object(value: Json | undefined): Metadata | null { return value && typeof value === 'object' && !Array.isArray(value) ? value : null; }
export function Evaluation({ evaluation }: { evaluation: Metadata }) {
  const scoreMap = object(evaluation.results) || object(evaluation.scores) || { breeze: { label: 'Breeze', metric: evaluation.metric || 'Match', metrics: evaluation.metrics || null, status: evaluation.status || 'Waiting for transcription' } };
  const scores = Object.entries(scoreMap).map(([key, value]) => ({ key, item: object(value) })).filter(entry => entry.item);
  const reference = object(evaluation.reference);
  const referenceView = object(evaluation.reference_view);
  const englishReference = referenceView?.title === 'English reference' ? referenceView : null;
  return <section className="evaluation-results"><div className="section-title"><h2>Transcription evaluation</h2><span className="quiet">Full reference · literal match</span></div>
    <div className="score-grid">{scores.map(({ key, item }) => {
      const metrics = object(item?.metrics) || item;
      const score = metrics?.score;
      return <article className="score-card" key={key}><span className="eyebrow">{String(item?.label || key)} · {String(item?.metric || 'Match')}</span><strong>{typeof score === 'number' ? `${(score * 100).toFixed(1)}%` : '—'}</strong><span>{String(item?.status || 'Waiting for complete output')}</span>{typeof metrics?.hits === 'number' && <p className="hint">{metrics.hits} matches · {String(metrics.substitutions)} substitutions · {String(metrics.deletions)} deletions · {String(metrics.insertions)} insertions</p>}</article>;
    })}</div>
    {!scores.length && <p className="hint">A final score appears when transcription completes. Failed or pending output is not scored.</p>}
    {englishReference && <section className="evaluation-reference" aria-label="English reference"><h3>English reference</h3><p className="hint">Compare this reference with the English results. Translation is not scored.</p><pre className="reference-text">{String(englishReference.text || '')}</pre>{typeof englishReference.column === 'string' && <p className="hint">{String(englishReference.name || '')} · {englishReference.column}</p>}</section>}
    <details><summary>Reference and scoring details</summary><p className="hint">Scores compare the complete source transcript in order. They measure literal text matching, not meaning. References stay local and are never used in prompts.</p>{reference && <pre className="reference-text">{String(reference.text || '')}</pre>}<pre className="audit-json">{JSON.stringify(evaluation, null, 2)}</pre></details>
  </section>;
}
export default function Conversation({ session, original, onRetry, commandPending }: { session: SessionSnapshot | null; original: boolean; onRetry: (provider: string) => void; commandPending: boolean }) {
  const [view, setView] = useState<'transcript' | 'compare'>('transcript');
  const [finalOnly, setFinalOnly] = useState(false);
  const authoritative = session?.correction.authoritative;
  return <section className="transcript-card" aria-label="Conversation transcript">
    <div className="section-title"><div><span className="eyebrow">Conversation</span><h2>Full transcript <span className="count">{session?.segments.length || 0}</span></h2></div>{session && !session.realtime && <div className="segmented-control" aria-label="Transcript view"><button aria-pressed={view === 'transcript'} onClick={() => setView('transcript')}>Transcript</button><button aria-pressed={view === 'compare'} onClick={() => setView('compare')}>Compare</button></div>}</div>
    {session && <div className="transcript-metadata"><span>{session.transcription_label}</span>{session.providers.includes('astra') && <label className="inline-check"><input type="checkbox" checked={finalOnly} onChange={event => setFinalOnly(event.target.checked)} /> Final record</label>}</div>}
    {session?.realtime ? <div className="realtime-panes"><article><h3>English captions</h3><p>{session.realtime.english || 'Waiting for English…'}</p>{session.realtime.incomplete && <Status value="Incomplete" />}</article>{original && <article><h3>Original captions</h3><p lang="zh-Hant">{session.realtime.source || 'Waiting for source captions…'}</p></article>}<p className="hint">Source and English are continuous texts. Their caption boundaries do not necessarily align.</p></div> : session?.segments.length ? <>
      {view === 'compare' ? <div className="comparison-scroll" tabIndex={0} role="region" aria-label="Provider comparison"><table className="comparison"><thead><tr><th scope="col">Turn</th><th scope="col">Original transcript</th>{session.providers.map(provider => <th scope="col" key={provider}>English <span className={`provider ${provider}`}>{providerLabel[provider] || provider}</span></th>)}</tr></thead><tbody>{session.segments.map(segment => <tr key={segment.id}><th scope="row">{String(segment.index + 1).padStart(2, '0')}</th><td><span className="speaker-label">{segment.speaker || 'Speaker pending'}</span><Source segment={segment} /><span className="timecode">{timecode(segment.start_s)} – {timecode(segment.end_s)}</span></td>{session.providers.map(provider => <td key={provider}><TranslationCell value={segment.translations[provider]} /></td>)}</tr>)}</tbody></table></div> : <div className="transcript-scroll" tabIndex={0} role="region" aria-label="Transcript turns">{session.segments.map((segment, position) => {
        const accepted = Array.isArray(authoritative) ? authoritative[position] : null;
        const record = Array.isArray(session.correction.segments) ? object(session.correction.segments[position]) : null;
        const finalView = finalOnly && session.providers.includes('astra');
        const english = validatedEnglish(accepted, !!record?.source_fallback);
        const filtered = segment.status === 'filtered' || !!record?.filtered;
        const unavailable = accepted != null || !!record?.sealed || !!record?.source_fallback || !!record?.error;
        const finalPlaceholder = filtered ? 'Background speech filtered' : unavailable ? 'Translation unavailable' : 'Awaiting final review…';
        const text = finalView ? filtered ? finalPlaceholder : english || finalPlaceholder : segment.english;
        const status = finalView && (filtered || !english) ? filtered ? 'filtered' : unavailable ? 'unavailable' : 'waiting' : segment.status;
        return <article className="transcript-row" key={segment.id}><span className="turn-number">{String(segment.index + 1).padStart(2, '0')}</span><div className="turn-content"><div className="turn-meta"><span className="speaker-label">{segment.speaker || 'Speaker pending'}</span><time>{timecode(segment.start_s)} – {timecode(segment.end_s)}</time><Status value={status} /></div><p>{text || (segment.status === 'filtered' ? 'Background speech filtered' : ['failed', 'unavailable', 'error'].includes(segment.status) ? 'Translation unavailable' : 'Translating…')}</p>{original && <Source segment={segment} />}</div></article>;
      })}</div>}
      {session.providers.some(provider => session.segments.some(segment => segment.translations[provider]?.error)) && <div className="retry-row"><span className="hint">Some translations need attention.</span>{session.providers.filter(provider => session.segments.some(segment => segment.translations[provider]?.error)).map(provider => <button className="button small secondary" key={provider} disabled={commandPending} onClick={() => onRetry(provider)}>Retry {providerLabel[provider] || provider}</button>)}</div>}
    </> : <div className="transcript-empty"><span>01</span><p>Your translated conversation will appear here.<br /><small>Original text and provider comparison are available as you go.</small></p></div>}
    {session?.providers.includes('astra') && <details className="audit-details"><summary>Correction history & processing details</summary><p className="hint">Versioned corrections preserve earlier drafts. Speaker labels are anonymous estimates, not verified identities.</p><div className="detail-grid"><div><h3>Astra correction</h3><pre className="audit-json">{JSON.stringify(session.correction, null, 2)}</pre></div><div><h3>Speaker detection</h3><pre className="audit-json">{JSON.stringify(session.diarization, null, 2)}</pre></div></div></details>}
  </section>;
}
