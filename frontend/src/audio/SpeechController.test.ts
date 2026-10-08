import { describe, expect, it } from 'vitest';
import { SpeechController } from './SpeechController';
import type { SpeechEnvironment } from './SpeechController';
import type { SpeechAcknowledgement, SpeechSnapshot, SpeechStatus } from './types';

function player({ blocked = false } = {}) {
  let allowed = !blocked;
  let now = 0;
  let timer: (() => void) | null = null;
  const resumes: Array<() => void> = [];
  const sources: FakeSource[] = [];
  const contexts: FakeContext[] = [];
  const statuses: SpeechStatus[] = [];
  const acknowledgements: SpeechAcknowledgement[] = [];
  class FakeSource {
    buffer!: { duration: number; samples: Float32Array };
    time = 0;
    stopped = false;
    onended: (() => void) | null = null;
    connect() {}
    disconnect() {}
    start(time: number) { this.time = time; }
    stop() { this.stopped = true; }
    finish() { this.onended?.(); }
  }
  class FakeContext {
    currentTime = 0;
    state = 'suspended';
    destination = {};
    onstatechange: (() => void) | null = null;
    constructor() { contexts.push(this); }
    async resume() {
      if (!allowed) return new Promise<void>(resolve => resumes.push(resolve));
      this.state = 'running';
      this.onstatechange?.();
      resumes.splice(0).forEach(resolve => resolve());
    }
    async suspend() { this.state = 'suspended'; this.onstatechange?.(); }
    async close() { this.state = 'closed'; this.onstatechange?.(); }
    createBuffer(channels: number, length: number, rate: number) {
      expect(channels).toBe(1);
      const samples = new Float32Array(length);
      return { duration: length / rate, getChannelData: () => samples, samples };
    }
    createBufferSource() { const source = new FakeSource(); sources.push(source); return source; }
  }
  const environment: SpeechEnvironment = {
    createContext: () => new FakeContext() as unknown as AudioContext,
    now: () => now,
    setInterval: callback => { timer = callback; return 1 as unknown as ReturnType<typeof setInterval>; },
    clearInterval: () => { timer = null; },
    decodeBase64: pcm => atob(pcm),
  };
  const controller = new SpeechController({ onStatus: status => statuses.push(status),
    onAcknowledgement: acknowledgement => acknowledgements.push(acknowledgement) }, environment);
  const chunk = (id: number, seconds?: number) => ({ id, pcm: btoa(seconds === undefined ?
    String.fromCharCode(0, 128, 0, 0, 255, 127) : '\0'.repeat(Math.round(seconds * 48000))) });
  const render = async (state: Partial<SpeechSnapshot> = {}) => {
    controller.update({ session_id: 'one', armed: true, acked: 0, chunks: [], sample_rate: 24000,
      closed: false, complete: false, playback_delay_ms: 0, minimum_buffer_seconds: 0, ...state });
    await Promise.resolve();
  };
  const advance = (milliseconds: number) => {
    now += milliseconds;
    contexts.forEach(context => { if (context.state === 'running') context.currentTime += milliseconds / 1000; });
    timer?.();
  };
  const gesture = async () => { allowed = true; await controller.unlock(); };
  return { controller, render, chunk, advance, gesture, sources, contexts, statuses,
    acknowledgements, status: () => statuses.at(-1)!, hasTimer: () => timer !== null };
}

describe('continuous English speech', () => {
  it('plays PCM once across repeated snapshots and acknowledges completed chunks', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1)] });
    await p.render({ chunks: [p.chunk(1), p.chunk(2)] });
    await p.render({ chunks: [p.chunk(1), p.chunk(2)] });
    expect(p.sources).toHaveLength(2);
    expect(p.sources[1].time).toBe(p.sources[0].time + p.sources[0].buffer.duration);
    expect(p.sources[0].buffer.samples[0]).toBe(-1);
    expect(p.sources[0].buffer.samples[2]).toBe(32767 / 32768);
    p.sources[0].finish();
    expect(p.acknowledgements).toEqual([{ speech_session_id: 'one', played: 1 }]);
  });

  it('primes playback from a Start gesture before a session exists', async () => {
    const p = player({ blocked: true });
    await p.gesture();
    await p.render({ chunks: [p.chunk(1)] });
    expect(p.contexts).toHaveLength(1);
    expect(p.sources).toHaveLength(1);
    expect(p.status().phase).toBe('playing');
  });

  it('exposes an Enable sound fallback when autoplay is blocked', async () => {
    const p = player({ blocked: true });
    await p.render({ chunks: [p.chunk(1)] });
    expect(p.sources).toHaveLength(0);
    expect(p.status().blocked).toBe(true);
    expect(p.status().message).toMatch(/browser has paused sound/);
    await p.gesture();
    expect(p.sources).toHaveLength(1);
  });

  it('waits 60 seconds and repeated snapshots do not restart the countdown', async () => {
    const p = player();
    await p.render({ playback_delay_ms: 60000, chunks: [p.chunk(1)] });
    p.advance(59000);
    expect(p.sources).toHaveLength(0);
    expect(p.status().countdownSeconds).toBe(1);
    await p.render({ playback_delay_ms: 1000, chunks: [p.chunk(1)] });
    p.advance(1000);
    expect(p.sources).toHaveLength(1);
  });

  it('starts the minute with accepted text, before PCM and after session creation', async () => {
    const p = player();
    await p.render({ playback_delay_ms: null });
    p.advance(120000);
    expect(p.status().message).toMatch(/Waiting for the first English/);
    await p.render({ playback_delay_ms: 60000 });
    expect(p.status().countdownSeconds).toBe(60);
    p.advance(59000);
    await p.render({ playback_delay_ms: 1000, chunks: [p.chunk(1)] });
    expect(p.sources).toHaveLength(0);
    p.advance(1000);
    expect(p.sources).toHaveLength(1);
  });

  it('completes empty conversations without waiting for a deadline', async () => {
    const p = player();
    await p.render({ playback_delay_ms: null, complete: true, generation_complete: true });
    expect(p.status().phase).toBe('complete');
  });

  it('uses the remaining server deadline when reconnecting a short completed clip', async () => {
    const p = player();
    await p.render({ playback_delay_ms: 5000, chunks: [p.chunk(1)], complete: true });
    p.advance(4999);
    expect(p.sources).toHaveLength(0);
    p.advance(1);
    expect(p.sources).toHaveLength(1);
  });

  it('keeps pause across snapshots and unrelated gestures', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1)] });
    await p.controller.pause();
    await p.render({ chunks: [p.chunk(1), p.chunk(2)] });
    await p.gesture();
    expect(p.sources).toHaveLength(1);
    expect(p.contexts[0].state).toBe('suspended');
    await p.controller.resume();
    expect(p.sources).toHaveLength(2);
  });

  it('does not override pause or stop when the deadline expires', async () => {
    const p = player();
    await p.render({ playback_delay_ms: 60000, chunks: [p.chunk(1)] });
    await p.controller.pause();
    p.advance(61000);
    expect(p.sources).toHaveLength(0);
    await p.controller.resume();
    expect(p.sources).toHaveLength(1);
    p.controller.stop();
    expect(p.sources[0].stopped).toBe(true);
    await p.render({ chunks: [p.chunk(2)] });
    await p.gesture();
    expect(p.sources).toHaveLength(1);
  });

  it('cancels old session audio and fences late snapshots and ended callbacks', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1)] });
    const oldCallback = p.sources[0].onended!;
    await p.render({ session_id: 'two', playback_delay_ms: 60000, chunks: [p.chunk(1)] });
    expect(p.sources[0].stopped).toBe(true);
    oldCallback();
    expect(p.acknowledgements).toHaveLength(0);
    await p.render({ session_id: 'one', chunks: [p.chunk(2)] });
    p.advance(60000);
    expect(p.sources).toHaveLength(2);
  });

  it('skips acknowledged chunks when restoring and displays safe server errors', async () => {
    const p = player();
    await p.render({ acked: 3, chunks: [p.chunk(3), p.chunk(4)] });
    expect(p.sources).toHaveLength(1);
    await p.render({ closed: true, error: 'Speech unavailable.' });
    expect(p.sources[0].stopped).toBe(true);
    expect(p.status().message).toBe('Speech unavailable.');
  });

  it('bounds scheduling to approximately four seconds with contiguous starts', async () => {
    const p = player();
    await p.render({ chunks: Array.from({ length: 30 }, (_, i) => p.chunk(i + 1, 0.2)) });
    expect(p.sources.length).toBeGreaterThanOrEqual(19);
    expect(p.sources.length).toBeLessThanOrEqual(21);
    for (let i = 1; i < p.sources.length; i++) expect(p.sources[i].time).toBe(p.sources[i - 1].time + 0.2);
    p.contexts[0].currentTime = 1;
    p.sources[0].finish();
    expect(p.sources.length).toBeGreaterThan(21);
    expect(p.sources.length).toBeLessThan(30);
  });

  it('does not autoplay while disabled and disposes audio, timers, and contexts', async () => {
    const p = player();
    await p.render({ armed: false });
    expect(p.contexts).toHaveLength(0);
    await p.render({ chunks: [p.chunk(1)] });
    p.controller.dispose();
    expect(p.sources[0].stopped).toBe(true);
    expect(p.contexts[0].state).toBe('closed');
    expect(p.hasTimer()).toBe(false);
  });

  it('does not add a gap when PCM arrives just before the preceding chunk ends', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1, 0.2)] });
    const end = p.sources[0].time + p.sources[0].buffer.duration;
    p.contexts[0].currentTime = end - 0.03;
    await p.render({ chunks: [p.chunk(1, 0.2), p.chunk(2, 0.2)] });
    expect(p.sources[1].time).toBe(end);
  });

  it('builds a two-second buffer before first playback and after underruns', async () => {
    const p = player();
    const settings = { minimum_buffer_seconds: 2, generation_complete: false };
    await p.render({ ...settings, playback_delay_ms: 60000, chunks: [p.chunk(1, 0.2)] });
    p.advance(60000);
    expect(p.sources).toHaveLength(0);
    await p.render({ ...settings, chunks: Array.from({ length: 10 }, (_, i) => p.chunk(i + 1, 0.2)) });
    expect(p.sources).toHaveLength(10);
    p.advance(2200);
    p.sources.slice().forEach(source => source.finish());
    await p.render({ ...settings, chunks: [p.chunk(11, 0.2)] });
    expect(p.sources).toHaveLength(10);
    p.advance(500);
    await p.render({ ...settings, chunks: Array.from({ length: 10 }, (_, i) => p.chunk(i + 11, 0.2)) });
    expect(p.sources).toHaveLength(20);
  });

  it('caps a buffer wait at two seconds and flushes generation-complete clips immediately', async () => {
    const p = player();
    await p.render({ minimum_buffer_seconds: 2, chunks: [p.chunk(1, 0.2)] });
    p.advance(1999);
    expect(p.sources).toHaveLength(0);
    p.advance(1);
    expect(p.sources).toHaveLength(1);
    await p.render({ minimum_buffer_seconds: 2, session_id: 'two', generation_complete: true, chunks: [p.chunk(1, 0.2)] });
    expect(p.sources).toHaveLength(2);
  });

  it('reports active model and speaker voices and clears disclosures on disable', async () => {
    const p = player();
    await p.render({ model: 'tts-1-hd', voices: { 'Speaker 1': 'coral' } });
    expect(p.status().model).toBe('tts-1-hd');
    expect(p.status().voices).toEqual({ 'Speaker 1': 'coral' });
    p.controller.update(null);
    expect(p.status().model).toBeNull();
    expect(p.status().voices).toEqual({});
  });

  it('acknowledges only contiguous completion even if ended callbacks arrive out of order', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1), p.chunk(2)] });
    p.sources[1].finish();
    expect(p.acknowledgements).toEqual([]);
    p.sources[0].finish();
    expect(p.acknowledgements).toEqual([{ speech_session_id: 'one', played: 2 }]);
  });

  it('retains deduplicated PCM events across the minute-long lead-in', async () => {
    const p = player();
    await p.render({ playback_delay_ms: 60000, chunks: [p.chunk(1, 0.2)] });
    await p.render({ playback_delay_ms: 59900, chunks: [p.chunk(2, 0.2)] });
    await p.render({ playback_delay_ms: 59800, chunks: [] });
    p.advance(60000);
    expect(p.sources).toHaveLength(2);
    expect(p.sources[1].time).toBe(p.sources[0].time + 0.2);
  });

  it('retains unplayed deltas while paused and tolerates metadata without chunks', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1, 0.2)] });
    await p.controller.pause();
    await p.render({ chunks: [p.chunk(2, 0.2)] });
    await p.render({ chunks: undefined });
    await p.render({ chunks: [] });
    expect(p.sources).toHaveLength(1);
    await p.controller.resume();
    expect(p.sources).toHaveLength(2);
  });

  it('replays a cumulative acknowledgement lost during events-socket disconnection', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1, 0.2), p.chunk(2, 0.2)] });
    p.sources[0].finish();
    p.sources[1].finish();
    p.acknowledgements.length = 0; // Both callbacks were dropped by the disconnected transport.
    await p.render({ acked: 0, chunks: undefined }); // Reconnected conversation metadata.
    expect(p.acknowledgements).toEqual([{ speech_session_id: 'one', played: 2 }]);
    await p.render({ acked: 0, chunks: [p.chunk(1, 0.2), p.chunk(2, 0.2)] });
    expect(p.sources).toHaveLength(2);
    await p.render({ acked: 2, chunks: [p.chunk(3, 0.2)] });
    expect(p.sources).toHaveLength(3);
  });

  it('primes a stopped, previously paused context for the next speech session', async () => {
    const p = player();
    await p.render({ chunks: [p.chunk(1)] });
    await p.controller.pause();
    p.controller.stop();
    expect(p.contexts[0].state).toBe('suspended');
    await p.gesture();
    expect(p.contexts[0].state).toBe('running');
    expect(p.sources).toHaveLength(1);
    await p.render({ session_id: 'two', chunks: [p.chunk(1)] });
    expect(p.sources).toHaveLength(2);
  });

  it('clearing a session resets pause and stop without accepting stale deltas', async () => {
    const p = player();
    await p.render({ playback_delay_ms: 60000, chunks: [p.chunk(1)] });
    await p.controller.pause();
    p.controller.stop();
    p.controller.update(null);
    await p.gesture();
    await p.render({ session_id: 'one', chunks: [p.chunk(1)] });
    expect(p.sources).toHaveLength(0);
    await p.render({ session_id: 'two', chunks: [p.chunk(1)] });
    expect(p.sources).toHaveLength(1);
  });
});
