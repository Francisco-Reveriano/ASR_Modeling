import type { SpeechAcknowledgement, SpeechChunk, SpeechSnapshot, SpeechStatus } from './types';

interface SpeechCallbacks {
  onStatus?: (status: SpeechStatus) => void;
  onAcknowledgement?: (acknowledgement: SpeechAcknowledgement) => void;
}

/** Injectable clocks/audio factories keep tests independent of browser autoplay policy. */
export interface SpeechEnvironment {
  createContext: () => AudioContext;
  now: () => number;
  setInterval: (callback: () => void, milliseconds: number) => ReturnType<typeof setInterval>;
  clearInterval: (timer: ReturnType<typeof setInterval>) => void;
  decodeBase64: (pcm: string) => string;
}

const browserEnvironment: SpeechEnvironment = {
  createContext: () => new AudioContext({ sampleRate: 24000, latencyHint: 'interactive' }),
  now: () => performance.now(),
  setInterval: (callback, delay) => setInterval(callback, delay),
  clearInterval: timer => clearInterval(timer),
  decodeBase64: pcm => atob(pcm),
};

/** A session-scoped, bounded PCM scheduler. Capture uses a separate AudioContext. */
export class SpeechController {
  private state: SpeechSnapshot | null = null;
  private context: AudioContext | null = null;
  private sessionId: string | null = null;
  private retired = new Set<string>();
  private chunks = new Map<number, SpeechChunk>();
  private nodes = new Set<AudioBufferSourceNode>();
  private finished = new Set<number>();
  private scheduled = 0;
  private played = 0;
  private nextTime = 0;
  private enabled = false;
  private stopped = false;
  private paused = false;
  private blocked = false;
  private resuming = false;
  private disposed = false;
  private playbackAt: number | null = null;
  private buffering = true;
  private bufferingSince: number | null = null;
  private statusKey = '';
  private readonly timer: ReturnType<typeof setInterval>;

  constructor(private readonly callbacks: SpeechCallbacks = {},
    private readonly environment: SpeechEnvironment = browserEnvironment) {
    this.timer = environment.setInterval(() => { this.pump(); this.publishStatus(); }, 100);
  }

  update(snapshot: SpeechSnapshot | null): void {
    if (this.disposed || (snapshot && this.retired.has(snapshot.session_id))) return;
    if (!snapshot) {
      if (this.sessionId) this.retired.add(this.sessionId);
      this.sessionId = null;
      this.state = null;
      this.clearAudio();
      this.chunks.clear();
      this.enabled = false;
      this.stopped = this.paused = false;
      this.playbackAt = this.bufferingSince = null;
      this.publishStatus();
      return;
    }
    if (snapshot.session_id !== this.sessionId) {
      if (this.sessionId) this.retired.add(this.sessionId);
      this.clearAudio();
      this.chunks.clear();
      this.sessionId = snapshot.session_id;
      this.scheduled = this.played = snapshot.acked;
      this.stopped = this.paused = false;
      this.buffering = true;
      this.bufferingSince = this.playbackAt = null;
    }
    this.state = snapshot;
    // Events contain PCM deltas. Retain them through the lead-in, pauses and
    // reconnects until played; metadata snapshots may omit chunks altogether.
    this.played = Math.max(this.played, snapshot.acked);
    this.scheduled = Math.max(this.scheduled, this.played);
    for (const chunk of snapshot.chunks ?? []) {
      if (chunk.id > this.played && !this.chunks.has(chunk.id)) this.chunks.set(chunk.id, chunk);
    }
    for (const id of this.chunks.keys()) if (id <= this.played) this.chunks.delete(id);
    if (snapshot.acked < this.played) this.acknowledge();
    if (this.playbackAt === null && snapshot.playback_delay_ms != null) {
      this.playbackAt = this.environment.now() + Math.max(0, snapshot.playback_delay_ms);
    }
    if (snapshot.closed) this.stop();
    else if (snapshot.armed === false) {
      this.enabled = false;
      this.clearAudio();
      this.chunks.clear();
    } else if (!this.enabled) void this.activate(false);
    this.pump();
    this.publishStatus();
  }

  /** Call directly inside a trusted Start/Enable click, before any awaited work. */
  unlock(): Promise<void> {
    return this.activate(true);
  }

  async pause(): Promise<void> {
    if (this.disposed || this.stopped) return;
    this.paused = true;
    this.enabled = false;
    this.publishStatus();
    if (this.context) await this.context.suspend().catch(() => undefined);
  }

  async resume(): Promise<void> {
    if (this.disposed || this.stopped) return;
    this.paused = false;
    this.buffering = true;
    this.bufferingSince = null;
    await this.activate(true);
  }

  stop(): void {
    this.stopped = true;
    this.enabled = false;
    this.clearAudio();
    this.chunks.clear();
    this.publishStatus();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.environment.clearInterval(this.timer);
    this.clearAudio();
    this.chunks.clear();
    if (this.context) {
      this.context.onstatechange = null;
      void this.context.close().catch(() => undefined);
    }
  }

  private clearAudio(): void {
    for (const node of this.nodes) {
      node.onended = null;
      try { node.stop(); } catch { /* Already-ended Web Audio sources cannot restart. */ }
      node.disconnect();
    }
    this.nodes.clear();
    this.finished.clear();
    this.nextTime = 0;
  }

  private async activate(fromGesture: boolean): Promise<void> {
    // After Stop, no nodes remain: a new Start/Enable gesture can safely prime
    // the context while the replacement speech session is created. Merely
    // paused audio still needs the explicit Resume action.
    if (this.disposed || (this.stopped && !fromGesture) || (this.paused && !this.stopped) ||
      (!fromGesture && (!this.state || this.state.armed === false || this.resuming))) return;
    // A gesture may prime the playback context before a server session exists.
    try {
      if (!this.context) {
        this.context = this.environment.createContext();
        this.context.onstatechange = () => {
          if (this.disposed) return;
          this.enabled = this.context?.state === 'running' && !this.paused && !this.stopped &&
            !!this.state && this.state.armed !== false;
          if (this.enabled) this.blocked = false;
          this.pump();
          this.publishStatus();
        };
      }
      this.resuming = true;
      const resumed = this.context.resume();
      this.blocked = this.context.state !== 'running';
      this.publishStatus();
      await resumed;
      if (this.disposed || this.stopped || this.paused || !this.state || this.state.armed === false) return;
      this.enabled = this.context.state === 'running';
      this.blocked = !this.enabled;
      this.pump();
      this.publishStatus();
    } catch {
      this.blocked = true;
      this.enabled = false;
      this.publishStatus();
    } finally {
      this.resuming = false;
    }
  }

  private pump(): void {
    const state = this.state;
    const context = this.context;
    if (!this.enabled || this.stopped || this.paused || !context || context.state !== 'running' || !state ||
      this.playbackAt === null || this.environment.now() < this.playbackAt) return;
    // Resent WebSocket snapshots do not schedule duplicate PCM. Server IDs are ordered.
    const ready = [...this.chunks.values()].filter(chunk => chunk.id > this.scheduled).sort((a, b) => a.id - b.id);
    if (this.nextTime <= context.currentTime) this.buffering = true;
    if (!ready.length) { this.bufferingSince = null; return; }
    if (this.buffering) {
      this.bufferingSince ??= this.environment.now();
      const seconds = ready.reduce((sum, chunk) => sum + this.environment.decodeBase64(chunk.pcm).length /
        (state.sample_rate * 2), 0);
      if (seconds + 1e-6 < (state.minimum_buffer_seconds ?? 2) && !state.generation_complete &&
        this.environment.now() - this.bufferingSince < 2000) return;
      this.buffering = false;
      this.bufferingSince = null;
    }
    for (const chunk of ready) {
      if (chunk.id <= this.scheduled) continue;
      // Keep a four-second browser scheduling horizon; Python retains the rest.
      if (this.nextTime - context.currentTime > 4) break;
      const binary = this.environment.decodeBase64(chunk.pcm);
      if (binary.length % 2 !== 0 || state.sample_rate <= 0) {
        this.state = { ...state, error: 'The server returned invalid speech audio.' };
        this.stop();
        return;
      }
      const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
      const view = new DataView(bytes.buffer);
      const buffer = context.createBuffer(1, bytes.length / 2, state.sample_rate);
      const samples = buffer.getChannelData(0);
      for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
      const node = context.createBufferSource();
      node.buffer = buffer;
      node.connect(context.destination);
      const owner = this.sessionId;
      node.onended = () => {
        this.nodes.delete(node);
        node.disconnect();
        if (owner !== this.sessionId || this.stopped || this.disposed) return;
        this.finished.add(chunk.id);
        const previous = this.played;
        while (this.finished.delete(this.played + 1)) this.played += 1;
        for (const id of this.chunks.keys()) if (id <= this.played) this.chunks.delete(id);
        if (this.played > previous) this.acknowledge();
        this.pump();
        this.publishStatus();
      };
      this.nodes.add(node);
      if (this.nextTime <= context.currentTime) this.nextTime = context.currentTime + 0.08;
      node.start(this.nextTime);
      this.nextTime += buffer.duration;
      this.scheduled = chunk.id;
    }
  }

  private acknowledge(): void {
    if (this.sessionId) this.callbacks.onAcknowledgement?.({ speech_session_id: this.sessionId, played: this.played });
  }

  private publishStatus(): void {
    if (this.disposed) return;
    let phase: SpeechStatus['phase'] = 'waiting';
    let message = 'Waiting for English translation…';
    let countdownSeconds: number | null = null;
    const state = this.state;
    if (!state || state.armed === false) { phase = 'idle'; message = 'Spoken English is off.'; }
    else if (state.error) { phase = 'error'; message = state.error; }
    else if (this.stopped) { phase = 'stopped'; message = 'Voice stopped. Translation continues.'; }
    else if (this.paused) { phase = 'paused'; message = 'Voice paused.'; }
    else if (this.playbackAt === null) {
      if (state.complete) { phase = 'complete'; message = 'English playback complete.'; }
      else message = 'Waiting for the first English translation to start the 1-minute audio lead…';
    } else if (this.environment.now() < this.playbackAt) {
      phase = 'lead-in';
      countdownSeconds = Math.ceil((this.playbackAt - this.environment.now()) / 1000);
      message = `Building an audio lead · starts in ${countdownSeconds}s`;
    } else if (!this.enabled) { phase = 'blocked'; message = 'Your browser has paused sound. Enable sound to continue.'; }
    else if (this.nodes.size) { phase = 'playing'; message = 'Speaking English…'; }
    else if (state.complete) { phase = 'complete'; message = 'English playback complete.'; }
    else if (this.bufferingSince !== null) { phase = 'buffering'; message = 'Buffering English audio for smoother playback…'; }
    else if (state.pending) message = 'Generating English speech…';
    const status: SpeechStatus = { phase, message, enabled: this.enabled, paused: this.paused,
      blocked: this.blocked, countdownSeconds, model: state?.model ?? null, voices: state?.voices ?? {} };
    const key = JSON.stringify(status);
    if (key !== this.statusKey) { this.statusKey = key; this.callbacks.onStatus?.(status); }
  }
}
