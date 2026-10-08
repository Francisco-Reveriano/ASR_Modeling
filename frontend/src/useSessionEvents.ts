import { useEffect, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { websocketUrl } from './api';
import { applyUpdate } from './session-state';
import type { SessionSnapshot, SessionUpdate } from './types';
export type ConnectionState = 'idle' | 'connecting' | 'connected' | 'disconnected';
export function useSessionEvents(id: string | undefined, setSession: Dispatch<SetStateAction<SessionSnapshot | null>>, onSpeech: (speech: unknown) => void) {
  const [connection, setConnection] = useState<ConnectionState>('idle');
  const socketRef = useRef<WebSocket | null>(null);
  const speechRef = useRef(onSpeech); speechRef.current = onSpeech;
  useEffect(() => {
    if (!id) { setConnection('idle'); return; }
    let disposed = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let lastMessage = Date.now(); let attempts = 0;
    const connect = () => {
      if (disposed) return;
      setConnection(attempts ? 'disconnected' : 'connecting');
      const socket = new WebSocket(websocketUrl(`/api/sessions/${encodeURIComponent(id)}/events`));
      socketRef.current = socket; lastMessage = Date.now();
      socket.onmessage = event => {
        if (disposed || socket !== socketRef.current) return;
        lastMessage = Date.now(); setConnection('connected'); attempts = 0;
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'snapshot' && message.snapshot?.id === id) {
            const snapshot = message.snapshot as SessionSnapshot;
            setSession(current => current?.id === id && snapshot.revision >= current.revision ? snapshot : current);
            speechRef.current(snapshot.speech);
          } else if (message.type === 'update') {
            setSession(current => applyUpdate(current, message as SessionUpdate));
          } else if (message.type === 'speech') { speechRef.current(message.speech); }
          else if (message.type === 'cleared') {
            disposed = true; setConnection('idle'); speechRef.current(null);
            setSession(current => current?.id === id ? null : current); socket.close();
          }
        } catch { /* Invalid events cannot corrupt the current transcript. */ }
      };
      socket.onclose = () => {
        if (disposed || socket !== socketRef.current) return;
        setConnection('disconnected'); attempts += 1;
        retry = setTimeout(connect, Math.min(1000 * attempts, 4000));
      };
      socket.onerror = () => { if (!disposed) setConnection('disconnected'); };
    };
    connect();
    const health = setInterval(() => {
      if (Date.now() - lastMessage >= 4000) { setConnection('disconnected'); socketRef.current?.close(); }
    }, 500);
    return () => {
      disposed = true; clearInterval(health); clearTimeout(retry);
      const socket = socketRef.current; socketRef.current = null; socket?.close();
    };
  }, [id, setSession]);
  function acknowledge(message: unknown) { const socket = socketRef.current; if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message)); }
  return { connection, acknowledge };
}
