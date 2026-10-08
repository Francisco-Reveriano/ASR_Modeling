import type { SessionSnapshot, SessionUpdate } from './types';
export function applyUpdate(current: SessionSnapshot | null, update: SessionUpdate): SessionSnapshot | null {
  if (!current || update.id !== current.id || update.revision <= current.revision) return current;
  const segments = new Map(current.segments.map(segment => [segment.id, segment]));
  for (const segment of update.segments) segments.set(segment.id, segment);
  const { type: _, ...metadata } = update;
  return { ...current, ...metadata, segments: [...segments.values()].sort((a, b) => a.index - b.index) };
}
export const isActive = (session: SessionSnapshot | null) => !!session && !session.finished && !['cancelled', 'failed', 'complete'].includes(session.status);
