export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
export type Metadata = { [key: string]: Json };
export interface SessionSettings {
  model_pair: string;
  translation_type: string;
  speech_enabled: boolean;
  speech_model: string;
  astra_speech_mode: string;
  speaker_voices: boolean;
  diarization: boolean;
  corrections_enabled: boolean;
  correction_model: string;
  reasoning_effort: string;
  max_output_tokens: number;
  confidence_threshold: number;
  glossary_path: string;
  dnt_path: string;
}
export interface AppConfig {
  defaults: SessionSettings;
  model_pairs: string[];
  translation_types: string[];
  speech_models: string[];
  speech_modes: string[];
  capabilities: Record<string, boolean>;
  models: Metadata;
  source_language: string;
  target_language: string;
  processing_disclosure: string;
}
export interface Translation { text: string | null; status: string; error?: string | null }
export interface Segment {
  id: string;
  index: number;
  source: string;
  start_s?: number | null;
  end_s?: number | null;
  speaker: string | null;
  english: string | null;
  status: string;
  translations: Record<string, Translation>;
}
export interface SessionSnapshot {
  id: string;
  revision: number;
  kind: 'microphone' | 'upload' | 'evaluation';
  name: string;
  status: 'preparing' | 'ready' | 'recording' | 'processing' | 'complete' | 'cancelled' | 'failed';
  accepting: boolean;
  input_finished: boolean;
  stop_requested?: boolean;
  finished: boolean;
  error: string | null;
  settings: SessionSettings;
  providers: string[];
  transcription_label: string;
  processing_disclosure: string;
  segments: Segment[];
  realtime: { source: string; english: string; incomplete: boolean } | null;
  progress: { completed: number; total: number; received_seconds: number; processed_seconds: number };
  correction: Metadata;
  diarization: Metadata;
  evaluation: Metadata | null;
  speech: unknown;
}
export type SessionUpdate = Partial<Omit<SessionSnapshot, 'id' | 'revision' | 'segments'>> &
  Pick<SessionSnapshot, 'id' | 'revision'> & { type: 'update'; segments: Segment[] };
export type SessionCommand = {
  type: 'stop' | 'cancel' | 'retry' | 'corrections' | 'speech' | 'terminology';
  provider?: string;
  enabled?: boolean;
  model?: string;
  mode?: string;
  speaker_voices?: boolean;
  glossary_path?: string;
  dnt_path?: string;
};
export interface ReferenceOptions {
  format: 'auto' | 'plain';
  sheet?: string;
  header_row?: number | null;
  source_column?: number | null;
  english_column?: number | null;
}
export interface ReferencePreview {
  reference: Metadata | null;
  reference_view: Metadata | null;
  tables?: { name: string; columns: string[]; suggested_source: number | null; suggested_english: number | null }[];
  sheet?: string;
  header_row?: number | null;
  source_column?: number | null;
  english_column?: number | null;
  needs_source_selection: boolean;
}
