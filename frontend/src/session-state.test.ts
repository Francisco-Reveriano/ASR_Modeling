import { describe, expect, it } from 'vitest';
import { applyUpdate } from './session-state';
import { segment, snapshot } from './test-fixtures';

describe('versioned conversation state', () => {
  it('replaces a corrected row in place and preserves source order', () => {
    const current = snapshot({ segments: [segment(0), segment(1)] });
    const update = { ...current, type: 'update' as const, revision: 2, segments: [segment(2), segment(0, { english: 'Corrected English', status: 'corrected' })] };
    const next = applyUpdate(current, update)!;
    expect(next.segments.map(row => row.id)).toEqual(['segment-0', 'segment-1', 'segment-2']);
    expect(next.segments[0].english).toBe('Corrected English');
    expect(current.segments[0].english).toBe('English turn 0');
  });
  it('discards stale updates and late results from replaced sessions', () => {
    const current = snapshot({ revision: 8 });
    expect(applyUpdate(current, { ...current, type: 'update', revision: 7 })).toBe(current);
    expect(applyUpdate(current, { ...current, type: 'update', id: 'closed-session', revision: 10 })).toBe(current);
  });
  it('preserves omitted metadata when only progress changes', () => {
    const current = snapshot({ segments: [segment(0)], correction: { events: ['retained history'] } });
    const next = applyUpdate(current, { type: 'update', id: current.id, revision: 2, segments: [], status: 'complete', finished: true })!;
    expect(next.status).toBe('complete');
    expect(next.finished).toBe(true);
    expect(next.correction).toBe(current.correction);
    expect(next.settings).toBe(current.settings);
    expect(next.segments).toEqual(current.segments);
  });
});
