import type { AppConfig, Segment, SessionSettings, SessionSnapshot } from './types';
export const settings: SessionSettings = {
  model_pair: 'gpt-live-transcribe + gpt-6-luna', translation_type: 'Compare all translations',
  speech_enabled: false, speech_model: 'gpt-4o-mini-tts', astra_speech_mode: 'Live corrections',
  speaker_voices: true, diarization: true, corrections_enabled: true, correction_model: 'gpt-6-astra',
  reasoning_effort: 'medium', max_output_tokens: 16384, confidence_threshold: 0.6, glossary_path: '', dnt_path: '',
};
export const config: AppConfig = {
  defaults: settings, model_pairs: [settings.model_pair, 'Breeze + OpenAI', 'gpt-realtime-translate'],
  translation_types: ['Compare all translations', 'Fast English', 'Corrected English', 'Fully reviewed English'],
  speech_models: ['gpt-4o-mini-tts', 'tts-1-hd'], speech_modes: ['Live corrections', 'Full review'],
  capabilities: { microphone: true, upload: true, evaluation: true }, models: { nemotron: 'local' },
  source_language: 'zh-TW+en', target_language: 'en', processing_disclosure: 'Audio is sent to OpenAI. Speaker detection is local.',
};
export function segment(index = 0, overrides: Partial<Segment> = {}): Segment {
  return { id: `segment-${index}`, index, source: `原文 ${index}`, english: `English turn ${index}`, start_s: index * 2, end_s: index * 2 + 1,
    speaker: 'Speaker 1', status: 'draft', translations: { openai: { text: `English turn ${index}`, status: 'complete' }, astra: { text: null, status: 'waiting' } }, ...overrides };
}
export function snapshot(overrides: Partial<SessionSnapshot> = {}): SessionSnapshot {
  return { id: 'session-1', revision: 1, kind: 'microphone', name: 'Microphone', status: 'recording', accepting: true,
    input_finished: false, finished: false, error: null, settings, providers: ['openai', 'astra'], transcription_label: 'OpenAI ASR · gpt-live-transcribe',
    processing_disclosure: config.processing_disclosure, segments: [], realtime: null,
    progress: { completed: 0, total: 0, received_seconds: 0, processed_seconds: 0 }, correction: {}, diarization: {}, evaluation: null, speech: null,
    ...overrides };
}
export class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = FakeWebSocket.OPEN;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  constructor(public url: string) { FakeWebSocket.instances.push(this); }
  send(message: string) { this.sent.push(message); }
  close() { if (this.readyState !== 3) { this.readyState = 3; this.onclose?.(); } }
  receive(message: unknown) { this.onmessage?.({ data: JSON.stringify(message) }); }
}
