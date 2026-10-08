import { act, renderHook } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SessionSnapshot } from './types';
import { FakeWebSocket, segment, snapshot } from './test-fixtures';
import { useSessionEvents } from './useSessionEvents';

describe('session event transport', () => {
  beforeEach(() => { vi.useFakeTimers(); FakeWebSocket.instances = []; vi.stubGlobal('WebSocket', FakeWebSocket); });
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
  it('restores the snapshot, applies corrections, and flags lost connections within five seconds', () => {
    const onSpeech = vi.fn();
    const hook = renderHook(() => {
      const [state, setState] = useState<SessionSnapshot | null>(snapshot());
      return { ...useSessionEvents(state?.id, setState, onSpeech), state };
    });
    const socket = FakeWebSocket.instances[0];
    act(() => socket.receive({ type: 'snapshot', snapshot: snapshot({ segments: [segment()] }) }));
    expect(hook.result.current.connection).toBe('connected');
    act(() => socket.receive({ ...snapshot(), type: 'update', revision: 2, segments: [segment(0, { english: 'Fixed in context' })] }));
    expect(hook.result.current.state?.segments[0].english).toBe('Fixed in context');
    act(() => vi.advanceTimersByTime(4500));
    expect(hook.result.current.connection).toBe('disconnected');
    act(() => vi.advanceTimersByTime(1000));
    expect(FakeWebSocket.instances.length).toBe(2);
    expect(FakeWebSocket.instances[1].url).toContain('/session-1/events');
    hook.unmount();
  });
  it('keeps an idle completed session connected while heartbeats arrive', () => {
    const hook = renderHook(() => {
      const [state, setState] = useState<SessionSnapshot | null>(snapshot({ status: 'complete', finished: true }));
      return useSessionEvents(state?.id, setState, vi.fn());
    });
    for (let index = 0; index < 8; index++) {
      act(() => { vi.advanceTimersByTime(1000); FakeWebSocket.instances[0].receive({ type: 'heartbeat' }); });
    }
    expect(hook.result.current.connection).toBe('connected');
    expect(FakeWebSocket.instances).toHaveLength(1);
    hook.unmount();
  });
  it('forwards speech acknowledgements and ends an externally cleared session', () => {
    const onSpeech = vi.fn();
    const hook = renderHook(() => {
      const [state, setState] = useState<SessionSnapshot | null>(snapshot());
      return { ...useSessionEvents(state?.id, setState, onSpeech), state };
    });
    act(() => hook.result.current.acknowledge({ type: 'speech.ack', speech_session_id: 'speech-1', played: 3 }));
    expect(JSON.parse(FakeWebSocket.instances[0].sent[0]).played).toBe(3);
    act(() => FakeWebSocket.instances[0].receive({ type: 'cleared' }));
    expect(hook.result.current.state).toBeNull();
    expect(onSpeech).toHaveBeenLastCalledWith(null);
    act(() => vi.advanceTimersByTime(10000));
    expect(FakeWebSocket.instances).toHaveLength(1);
    hook.unmount();
  });
});
