import type { CaptureStatus } from './types';

interface CaptureCallbacks {
  onStatus?: (status: CaptureStatus) => void;
  onError?: (message: string) => void;
}

interface CaptureOptions {
  sessionId: string;
  deviceId?: string;
}

export interface CaptureEnvironment {
  getUserMedia: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  createContext: () => AudioContext;
  createNode: (context: AudioContext) => AudioWorkletNode;
  createSocket: (url: string) => WebSocket;
  socketUrl: (sessionId: string) => string;
  workletUrl: string;
  setTimeout: (callback: () => void, milliseconds: number) => ReturnType<typeof setTimeout>;
  clearTimeout: (timer: ReturnType<typeof setTimeout>) => void;
}

const browserEnvironment: CaptureEnvironment = {
  getUserMedia: constraints => {
    if (!navigator.mediaDevices?.getUserMedia) {
      return Promise.reject(new Error('Microphone access requires localhost or HTTPS and a supported browser.'));
    }
    return navigator.mediaDevices.getUserMedia(constraints);
  },
  // Keep the browser's native sample rate; Python owns the persistent resampler.
  createContext: () => {
    if (typeof AudioContext === 'undefined') {
      throw new Error('This browser does not support microphone audio processing. Use a current browser on localhost or HTTPS.');
    }
    return new AudioContext({ latencyHint: 'interactive' });
  },
  createNode: context => new AudioWorkletNode(context, 'pcm-capture', {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
  }),
  createSocket: url => new WebSocket(url),
  socketUrl: sessionId => `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}` +
    `/api/sessions/${encodeURIComponent(sessionId)}/audio`,
  workletUrl: new URL('./capture-worklet.js', import.meta.url).href,
  setTimeout: (callback, delay) => setTimeout(callback, delay),
  clearTimeout: timer => clearTimeout(timer),
};

function microphoneError(error: unknown): string {
  if (error instanceof Error) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      return 'Microphone permission was denied. Allow microphone access in your browser and try again.';
    }
    if (error.name === 'NotFoundError' || error.name === 'OverconstrainedError') {
      return 'The selected microphone is unavailable. Connect a microphone or choose another device.';
    }
    if (error.name === 'NotReadableError') {
      return 'The microphone could not be opened. Check whether another application is using it.';
    }
    return error.message;
  }
  return 'Microphone capture could not start. Check your browser and microphone settings.';
}

/** Owns one microphone connection. It never reconnects or silently drops PCM. */
export class CaptureController {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private socket: WebSocket | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private flushResolve: (() => void) | null = null;
  private flushReject: ((error: Error) => void) | null = null;
  private finishResolve: (() => void) | null = null;
  private finishReject: ((error: Error) => void) | null = null;
  private operation = 0;
  private recording = false;
  private stopping = false;
  private disposed = false;
  private failureMessage: string | null = null;
  private stopPromise: Promise<void> | null = null;
  private timers = new Set<ReturnType<typeof setTimeout>>();

  constructor(private readonly callbacks: CaptureCallbacks = {},
    private readonly environment: CaptureEnvironment = browserEnvironment) {}

  async start({ sessionId, deviceId }: CaptureOptions): Promise<void> {
    if (this.disposed) throw new Error('This microphone controller has been closed.');
    if (this.context || this.stopping) throw new Error('Stop the current microphone recording first.');
    this.failureMessage = null;
    const operation = ++this.operation;
    this.publish('requesting', 'Requesting microphone access…');
    try {
      this.context = this.environment.createContext();
      const context = this.context;
      // Resume in the original gesture, before permission/network awaits.
      const resumed = context.resume().then(() => null, (error: unknown) => error);
      const stream = await this.environment.getUserMedia({
        audio: { channelCount: { ideal: 1 }, echoCancellation: true, noiseSuppression: true,
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}) },
        video: false,
      });
      if (operation !== this.operation) {
        stream.getTracks().forEach(track => track.stop());
        throw new Error('Microphone start was canceled.');
      }
      this.stream = stream;
      const resumeError = await resumed;
      if (resumeError) throw resumeError;
      if (operation !== this.operation) throw new Error('Microphone start was canceled.');
      if (!context.audioWorklet) throw new Error('This browser does not support AudioWorklet microphone capture. Use a current browser on localhost or HTTPS.');
      await context.audioWorklet.addModule(this.environment.workletUrl);
      if (operation !== this.operation) throw new Error('Microphone start was canceled.');
      this.publish('connecting', 'Connecting microphone to the local server…');
      const socket = this.environment.createSocket(this.environment.socketUrl(sessionId));
      this.socket = socket;
      socket.binaryType = 'arraybuffer';
      await new Promise<void>((resolve, reject) => {
        const timer = this.timeout(() => reject(new Error('The server did not accept microphone audio in time.')), 125000);
        this.readyReject = reject;
        socket.onopen = () => {
          if (operation !== this.operation) return;
          socket.send(JSON.stringify({ type: 'capture.start', sample_rate: context.sampleRate,
            channels: 1, format: 'pcm_s16le' }));
        };
        socket.onmessage = event => {
          if (operation !== this.operation) return;
          let data: { type?: string; message?: string };
          try { data = JSON.parse(String(event.data)); } catch { return; }
          if (data.type === 'capture.ready') {
            this.cancelTimer(timer);
            this.readyReject = null;
            resolve();
          } else if (data.type === 'error') {
            this.fail(data.message || 'The server rejected microphone audio.');
          } else if (data.type === 'capture.finished') {
            this.finishResolve?.();
            this.finishResolve = this.finishReject = null;
          }
        };
        socket.onerror = () => {
          if (operation === this.operation) this.fail('The microphone connection failed. Check that the local server is running.');
        };
        socket.onclose = () => {
          if (this.stopping && this.finishReject && operation === this.operation) {
            this.fail('The microphone connection closed before the server confirmed the final audio. Accepted audio will still finish.');
          } else if (!this.stopping && !this.disposed && operation === this.operation) {
            this.fail('Microphone connection lost. Accepted audio is finishing; start again to record more.');
          }
        };
      });
      if (operation !== this.operation) throw new Error('Microphone start was canceled.');
      const node = this.environment.createNode(context);
      this.node = node;
      node.port.onmessage = event => {
        if (operation !== this.operation) return;
        if (event.data?.type === 'pcm' && event.data.pcm instanceof ArrayBuffer) {
          this.sendPcm(event.data.pcm);
        } else if (event.data?.type === 'flushed') {
          this.flushResolve?.();
          this.flushResolve = null;
          this.flushReject = null;
        }
      };
      this.source = context.createMediaStreamSource(stream);
      this.source.connect(node);
      // A connected output keeps the worklet active; it emits zeros, never the microphone.
      node.connect(context.destination);
      this.recording = true;
      stream.getAudioTracks().forEach(track => {
        track.onended = () => {
          if (this.recording && !this.stopping) this.fail('The microphone disconnected. Choose a device and start again.');
        };
      });
      this.publish('recording', 'Microphone active. Audio is being translated.');
    } catch (error) {
      const message = microphoneError(error);
      if (operation === this.operation) this.fail(message);
      throw new Error(message);
    }
  }

  stop(): Promise<void> {
    if (this.stopPromise) return this.stopPromise;
    if (!this.context) return Promise.resolve();
    this.stopping = true;
    this.publish('stopping', 'Finishing microphone audio…');
    this.stopPromise = this.finish().finally(() => {
      this.stopping = false;
      this.stopPromise = null;
    });
    return this.stopPromise;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.cleanup();
  }

  private async finish(): Promise<void> {
    try {
      if (this.recording && this.node) {
        await new Promise<void>((resolve, reject) => {
          const timer = this.timeout(() => reject(new Error('Microphone audio could not finish cleanly. The final partial batch may be incomplete.')), 2000);
          this.flushReject = reject;
          this.flushResolve = () => { this.cancelTimer(timer); this.flushReject = null; resolve(); };
          this.node!.port.postMessage({ type: 'flush' });
        });
        if (this.socket?.readyState !== 1) throw new Error('The microphone connection closed before final audio could be confirmed.');
        // Do not let a following HTTP Stop overtake buffered PCM on this socket.
        // The server confirms only after consuming all preceding binary batches.
        await new Promise<void>((resolve, reject) => {
          const timer = this.timeout(() => reject(new Error('The server did not confirm the final microphone audio. Accepted audio will still finish.')), 10000);
          this.finishReject = reject;
          this.finishResolve = () => { this.cancelTimer(timer); this.finishReject = null; resolve(); };
          this.socket!.send(JSON.stringify({ type: 'capture.finish' }));
        });
      }
      this.cleanup();
      this.publish('stopped', 'Microphone stopped. Pending translations are finishing.');
    } catch (error) {
      const message = microphoneError(error);
      this.fail(message);
      throw new Error(message);
    }
  }

  private sendPcm(pcm: ArrayBuffer): void {
    if (!this.socket || !this.context || this.socket.readyState !== 1) {
      this.fail('The microphone connection closed before audio could be sent.');
      return;
    }
    // At most two seconds of PCM may wait in the browser's transport buffer.
    if (this.socket.bufferedAmount + pcm.byteLength > this.context.sampleRate * 2 * 2) {
      this.fail('Microphone upload cannot keep up. Recording stopped; check the server and connection before restarting.');
      return;
    }
    this.socket.send(pcm);
  }

  private fail(message: string): void {
    if (this.disposed || this.failureMessage !== null) return;
    this.failureMessage = message;
    this.cleanup(message);
    this.publish('error', message);
    this.callbacks.onError?.(message);
  }

  private cleanup(reason = 'Microphone start was canceled.'): void {
    this.operation += 1;
    this.recording = false;
    this.readyReject?.(new Error(reason));
    this.readyReject = null;
    this.flushReject?.(new Error(reason));
    this.flushResolve = null;
    this.flushReject = null;
    this.finishReject?.(new Error(reason));
    this.finishResolve = this.finishReject = null;
    for (const timer of this.timers) this.environment.clearTimeout(timer);
    this.timers.clear();
    if (this.node) { this.node.port.onmessage = null; this.node.disconnect(); }
    this.source?.disconnect();
    this.source = null;
    this.node = null;
    if (this.stream) this.stream.getTracks().forEach(track => { track.onended = null; track.stop(); });
    this.stream = null;
    if (this.socket) {
      this.socket.onopen = this.socket.onmessage = this.socket.onerror = this.socket.onclose = null;
      this.socket.close(1000, 'Capture finished');
    }
    this.socket = null;
    if (this.context) void this.context.close().catch(() => undefined);
    this.context = null;
  }

  private publish(phase: CaptureStatus['phase'], message: string): void {
    if (!this.disposed) this.callbacks.onStatus?.({ phase, message, sampleRate: this.context?.sampleRate ?? null });
  }

  private timeout(callback: () => void, delay: number): ReturnType<typeof setTimeout> {
    const timer = this.environment.setTimeout(() => { this.timers.delete(timer); callback(); }, delay);
    this.timers.add(timer);
    return timer;
  }

  private cancelTimer(timer: ReturnType<typeof setTimeout>): void {
    this.environment.clearTimeout(timer);
    this.timers.delete(timer);
  }
}
