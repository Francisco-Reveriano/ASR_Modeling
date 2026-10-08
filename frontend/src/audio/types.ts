export interface SpeechChunk {
  id: number;
  pcm: string;
}

/** PCM and deadlines are supplied by the Python SpeechSession, never browser TTS. */
export interface SpeechSnapshot {
  session_id: string;
  armed?: boolean;
  model?: string;
  sample_rate: number;
  /** Omitted in conversation metadata; speech events carry newly available chunks. */
  chunks?: SpeechChunk[];
  acked: number;
  closed: boolean;
  complete: boolean;
  generation_complete?: boolean;
  playback_delay_ms?: number | null;
  minimum_buffer_seconds?: number;
  pending?: number;
  error?: string | null;
  voices?: Record<string, string>;
  speaker_voices?: boolean;
  skipped?: number;
  buffered_seconds?: number;
}

export interface SpeechAcknowledgement {
  speech_session_id: string;
  played: number;
}

export interface SpeechStatus {
  phase: 'idle' | 'waiting' | 'lead-in' | 'buffering' | 'playing' | 'paused' |
    'blocked' | 'stopped' | 'complete' | 'error';
  message: string;
  enabled: boolean;
  paused: boolean;
  blocked: boolean;
  countdownSeconds: number | null;
  model: string | null;
  voices: Record<string, string>;
}

export interface CaptureStatus {
  phase: 'idle' | 'requesting' | 'connecting' | 'recording' | 'stopping' | 'stopped' | 'error';
  message: string;
  sampleRate: number | null;
}
