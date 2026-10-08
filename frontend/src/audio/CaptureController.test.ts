import { describe, expect, it } from 'vitest';
import { CaptureController } from './CaptureController';
import type { CaptureEnvironment } from './CaptureController';
import type { CaptureStatus } from './types';
import workletSource from './capture-worklet.js?raw';

const ticks = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

function capture({ permissionError, sampleRate = 48000 }: { permissionError?: Error; sampleRate?: number } = {}) {
  const sent: Array<string | ArrayBuffer> = [];
  const statuses: CaptureStatus[] = [];
  const errors: string[] = [];
  const timers = new Map<number, { callback: () => void; delay: number }>();
  let timerId = 0;
  let nodeCreated = false;
  let sourceConnected = false;
  const track = { stopped: false, onended: null as (() => void) | null, stop() { this.stopped = true; } };
  const stream = { getTracks: () => [track], getAudioTracks: () => [track] };
  const node = {
    port: { onmessage: null as ((event: { data: unknown }) => void) | null,
      commands: [] as unknown[], postMessage(data: unknown) { this.commands.push(data); } },
    connect() {}, disconnect() {},
  };
  const context = {
    sampleRate, closed: false, resumed: false, destination: {}, modules: [] as string[],
    audioWorklet: { async addModule(url: string) { context.modules.push(url); } },
    async resume() { this.resumed = true; },
    async close() { this.closed = true; },
    createMediaStreamSource() {
      return { connect() { sourceConnected = true; }, disconnect() { sourceConnected = false; } };
    },
  };
  const socket = {
    url: '', binaryType: '', readyState: 0, bufferedAmount: 0, closed: false,
    onopen: null as (() => void) | null, onclose: null as (() => void) | null,
    onerror: null as (() => void) | null, onmessage: null as ((event: { data: string }) => void) | null,
    send(data: string | ArrayBuffer) { sent.push(data); },
    close() { this.closed = true; this.readyState = 3; this.onclose?.(); },
    open() { this.readyState = 1; this.onopen?.(); },
    receive(data: unknown) { this.onmessage?.({ data: JSON.stringify(data) }); },
  };
  let constraints: MediaStreamConstraints | undefined;
  const environment: CaptureEnvironment = {
    getUserMedia: async value => {
      constraints = value;
      if (permissionError) throw permissionError;
      return stream as unknown as MediaStream;
    },
    createContext: () => context as unknown as AudioContext,
    createNode: () => { nodeCreated = true; return node as unknown as AudioWorkletNode; },
    createSocket: url => { socket.url = url; return socket as unknown as WebSocket; },
    socketUrl: sessionId => `ws://localhost:8000/api/sessions/${sessionId}/audio`,
    workletUrl: '/assets/capture-worklet.js',
    setTimeout: (callback, delay) => {
      timers.set(++timerId, { callback, delay });
      return timerId as unknown as ReturnType<typeof setTimeout>;
    },
    clearTimeout: id => timers.delete(id as unknown as number),
  };
  const controller = new CaptureController({ onStatus: status => statuses.push(status),
    onError: error => errors.push(error) }, environment);
  const start = () => controller.start({ sessionId: 'session-123', deviceId: 'mic-456' });
  const connected = async () => {
    const started = start();
    await ticks();
    socket.open();
    socket.receive({ type: 'capture.ready' });
    await started;
  };
  const pcm = (samples = 4800) => {
    const buffer = new ArrayBuffer(samples * 2);
    node.port.onmessage?.({ data: { type: 'pcm', pcm: buffer } });
    return buffer;
  };
  const flushed = () => node.port.onmessage?.({ data: { type: 'flushed' } });
  return { controller, start, connected, socket, node, context, sent, statuses, errors, track, timers, pcm,
    flushed, constraints: () => constraints, nodeCreated: () => nodeCreated, sourceConnected: () => sourceConnected };
}

describe('browser microphone transport', () => {
  it('declares actual sample rate and waits for server readiness before connecting capture', async () => {
    const c = capture({ sampleRate: 44100 });
    const started = c.start();
    await ticks();
    expect(c.context.resumed).toBe(true);
    expect(c.nodeCreated()).toBe(false);
    expect(c.constraints()).toMatchObject({ audio: { deviceId: { exact: 'mic-456' } }, video: false });
    c.socket.open();
    expect(JSON.parse(c.sent[0] as string)).toEqual({ type: 'capture.start', sample_rate: 44100, channels: 1, format: 'pcm_s16le' });
    expect(c.nodeCreated()).toBe(false);
    c.socket.receive({ type: 'capture.ready' });
    await started;
    expect(c.sourceConnected()).toBe(true);
    expect(c.statuses.at(-1)?.phase).toBe('recording');
    c.controller.dispose();
  });

  it('sends every PCM batch, flushes final partial bytes before finish, and stops tracks', async () => {
    const c = capture();
    await c.connected();
    const full = c.pcm();
    const stopped = c.controller.stop();
    expect(c.node.port.commands).toEqual([{ type: 'flush' }]);
    const partial = c.pcm(113);
    c.flushed();
    await ticks();
    expect(c.socket.closed).toBe(false);
    c.socket.receive({ type: 'capture.finished' });
    await stopped;
    expect(c.sent).toEqual([expect.any(String), full, partial, JSON.stringify({ type: 'capture.finish' })]);
    expect(c.track.stopped).toBe(true);
    expect(c.socket.closed).toBe(true);
    expect(c.context.closed).toBe(true);
    expect(c.errors).toEqual([]);
  });

  it('reports permission denial without opening the transport', async () => {
    const error = new Error('denied');
    error.name = 'NotAllowedError';
    const c = capture({ permissionError: error });
    await expect(c.start()).rejects.toThrow(/permission was denied/);
    expect(c.sent).toEqual([]);
    expect(c.context.closed).toBe(true);
    expect(c.errors).toHaveLength(1);
  });

  it('reports unavailable devices in user-facing language', async () => {
    const error = new Error('not found');
    error.name = 'NotFoundError';
    const c = capture({ permissionError: error });
    await expect(c.start()).rejects.toThrow(/selected microphone is unavailable/);
  });

  it('stops and reports overload instead of silently dropping or reconnecting', async () => {
    const c = capture();
    await c.connected();
    c.socket.bufferedAmount = 192000;
    c.pcm();
    expect(c.sent).toHaveLength(1);
    expect(c.errors.at(-1)).toMatch(/cannot keep up/);
    expect(c.track.stopped).toBe(true);
    expect(c.socket.closed).toBe(true);
    expect(c.statuses.at(-1)?.phase).toBe('error');
  });

  it('stops tracks and reports an abrupt socket disconnect without reconnecting', async () => {
    const c = capture();
    await c.connected();
    c.socket.close();
    expect(c.errors).toHaveLength(1);
    expect(c.errors[0]).toMatch(/connection lost/);
    expect(c.track.stopped).toBe(true);
    expect(c.context.closed).toBe(true);
  });

  it('preserves server rejection messages and cleans up', async () => {
    const c = capture();
    const started = c.start();
    const rejected = expect(started).rejects.toThrow('Microphone session is unavailable.');
    await ticks();
    c.socket.open();
    c.socket.receive({ type: 'error', message: 'Microphone session is unavailable.' });
    await rejected;
    expect(c.errors).toEqual(['Microphone session is unavailable.']);
    expect(c.track.stopped).toBe(true);
  });

  it('allows model initialization for at least 120 seconds but bounds handshake waiting', async () => {
    const c = capture();
    const started = c.start();
    const rejected = expect(started).rejects.toThrow(/did not accept/);
    await ticks();
    const timer = [...c.timers.values()][0];
    expect(timer.delay).toBeGreaterThanOrEqual(120000);
    timer.callback();
    await rejected;
    expect(c.track.stopped).toBe(true);
  });

  it('reports a worklet flush timeout instead of claiming a complete stop', async () => {
    const c = capture();
    await c.connected();
    const stopped = c.controller.stop();
    const rejected = expect(stopped).rejects.toThrow(/final partial batch/);
    [...c.timers.values()][0].callback();
    await rejected;
    expect(c.errors.at(-1)).toMatch(/final partial batch/);
    expect(c.sent.some(data => typeof data === 'string' && data.includes('capture.finish'))).toBe(false);
    expect(c.track.stopped).toBe(true);
  });

  it('waits for server confirmation so an HTTP Stop cannot overtake final PCM', async () => {
    const c = capture();
    await c.connected();
    let resolved = false;
    const stopped = c.controller.stop().then(() => { resolved = true; });
    c.pcm(123);
    c.flushed();
    await ticks();
    expect(c.sent.at(-1)).toBe(JSON.stringify({ type: 'capture.finish' }));
    expect(resolved).toBe(false);
    c.socket.receive({ type: 'capture.finished' });
    await stopped;
    expect(resolved).toBe(true);
    expect(c.track.stopped).toBe(true);
  });

  it('bounds waiting for server confirmation and stops tracks on acknowledgement timeout', async () => {
    const c = capture();
    await c.connected();
    const stopped = c.controller.stop();
    const rejected = expect(stopped).rejects.toThrow(/did not confirm/);
    c.flushed();
    await ticks();
    const timer = [...c.timers.values()][0];
    expect(timer.delay).toBe(10000);
    timer.callback();
    await rejected;
    expect(c.track.stopped).toBe(true);
    expect(c.statuses.at(-1)?.phase).toBe('error');
  });

  it('reports disconnect during finish instead of claiming the final audio was accepted', async () => {
    const c = capture();
    await c.connected();
    const stopped = c.controller.stop();
    const rejected = expect(stopped).rejects.toThrow(/before the server confirmed/);
    c.flushed();
    await ticks();
    c.socket.close();
    await rejected;
    expect(c.errors).toHaveLength(1);
    expect(c.track.stopped).toBe(true);
  });

  it('disposal cancels preparation and closes the microphone', async () => {
    const c = capture();
    const started = c.start();
    const rejected = expect(started).rejects.toThrow(/canceled/);
    await ticks();
    c.controller.dispose();
    await rejected;
    expect(c.track.stopped).toBe(true);
    expect(c.timers.size).toBe(0);
  });

  it('reports a device unplugged during recording', async () => {
    const c = capture();
    await c.connected();
    c.track.onended?.();
    expect(c.errors.at(-1)).toMatch(/microphone disconnected/);
    expect(c.socket.closed).toBe(true);
  });
});

describe('local PCM AudioWorklet', () => {
  interface WorkletMessage { type: string; pcm?: ArrayBuffer }
  function processor(rate = 16000) {
    const sent: WorkletMessage[] = [];
    class BaseProcessor {
      port = { onmessage: null as ((event: { data: { type: string } }) => void) | null,
        postMessage(data: WorkletMessage) { sent.push(data); } };
    }
    type Processor = BaseProcessor & { process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean };
    let Registered!: new () => Processor;
    new Function('AudioWorkletProcessor', 'sampleRate', 'registerProcessor', workletSource)(
      BaseProcessor, rate, (name: string, implementation: new () => Processor) => {
        expect(name).toBe('pcm-capture'); Registered = implementation;
      });
    const instance = new Registered();
    const flush = () => instance.port.onmessage?.({ data: { type: 'flush' } });
    return { instance, flush, sent };
  }

  it('averages channels, clamps PCM16LE, and never plays microphone samples', () => {
    const p = processor();
    const output = new Float32Array(4).fill(1);
    p.instance.process([[new Float32Array([-2, 1, 0.5, 0]), new Float32Array([-1, 1, -0.5, 0])]], [[output]]);
    p.flush();
    expect(Array.from(output)).toEqual([0, 0, 0, 0]);
    const view = new DataView(p.sent[0].pcm!);
    expect([0, 2, 4, 6].map(offset => view.getInt16(offset, true))).toEqual([-32768, 32767, 0, 0]);
    expect(p.sent[1]).toEqual({ type: 'flushed' });
  });

  it('preserves samples across variable native blocks, silence, and a final partial batch', () => {
    const p = processor(48000);
    const input = new Float32Array(10013);
    for (let i = 0; i < input.length; i++) input[i] = i % 4 === 0 ? 0 : (i % 61 - 30) / 30;
    let offset = 0;
    for (const length of [37, 128, 256, 3000, 1024, 5568]) {
      p.instance.process([[input.slice(offset, offset + length)]], [[new Float32Array(length)]]);
      offset += length;
    }
    expect(offset).toBe(input.length);
    p.flush();
    const batches = p.sent.filter(message => message.type === 'pcm').map(message => message.pcm!);
    expect(batches.map(buffer => buffer.byteLength)).toEqual([9600, 9600, 826]);
    const values = batches.flatMap(buffer => {
      const view = new DataView(buffer);
      return Array.from({ length: buffer.byteLength / 2 }, (_, i) => view.getInt16(i * 2, true));
    });
    expect(values).toEqual(Array.from(input, value => Math.round(value < 0 ? value * 32768 : value * 32767)));
    expect(p.sent.at(-1)?.type).toBe('flushed');
    const before = p.sent.length;
    p.instance.process([[new Float32Array(4800)]], [[new Float32Array(4800)]]);
    expect(p.sent).toHaveLength(before);
  });
});
