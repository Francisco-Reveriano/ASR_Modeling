import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import Conversation, { Evaluation, Subtitles } from './Conversation';
import { segment, snapshot } from './test-fixtures';

describe('subtitle and transcript rendering', () => {
  it('shows only the last three turns and hides original text by default', () => {
    const session = snapshot({ segments: [segment(0), segment(1), segment(2), segment(3)] });
    const view = render(<Subtitles session={session} original={false} visible size="large" />);
    expect(screen.queryByText('English turn 0')).not.toBeInTheDocument();
    expect(screen.getByText('English turn 3')).toBeInTheDocument();
    expect(screen.queryByText('原文 3')).not.toBeInTheDocument();
    view.rerender(<Subtitles session={session} original visible size="large" />);
    expect(screen.getByText('原文 3')).toBeInTheDocument();
  });
  it('marks failed English as unavailable instead of a pending translation', () => {
    render(<Subtitles session={snapshot({ segments: [segment(0, { english: null, status: 'unavailable' })] })} original={false} visible size="regular" />);
    expect(screen.getByText('Translation unavailable')).toBeInTheDocument();
    expect(screen.queryByText('Translating…')).not.toBeInTheDocument();
  });
  it('keeps realtime captions separate without inventing turn alignment', () => {
    render(<Conversation session={snapshot({ realtime: { source: '原文', english: 'Continuous English', incomplete: true } })} original onRetry={vi.fn()} commandPending={false} />);
    expect(screen.getByRole('heading', { name: 'English captions' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Original captions' })).toBeInTheDocument();
    expect(screen.getByText('Incomplete')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });
  it('never labels source fallbacks, mixed scripts, or empty audit text as final English', async () => {
    const user = userEvent.setup();
    const values = ['Original fallback text', 'A mixed 原文 translation', 'A supplementary 𠮷 character', '', 'José confirmed the batch.'];
    const session = snapshot({ segments: values.map((_, index) => segment(index)), correction: {
      authoritative: values,
      segments: values.map((_, index) => ({ source_fallback: index === 0, sealed: true })),
    } });
    render(<Conversation session={session} original={false} onRetry={vi.fn()} commandPending={false} />);
    await user.click(screen.getByLabelText('Final record'));
    const transcript = screen.getByRole('region', { name: 'Transcript turns' });
    expect(within(transcript).getAllByText('Translation unavailable')).toHaveLength(4);
    expect(within(transcript).queryByText('Original fallback text')).not.toBeInTheDocument();
    expect(within(transcript).queryByText('A mixed 原文 translation')).not.toBeInTheDocument();
    expect(within(transcript).getByText('José confirmed the batch.')).toBeInTheDocument();
  });
  it('preserves the readable English reference for completed or restored evaluations', () => {
    render(<Evaluation evaluation={{ metric: 'Mixed match', status: 'Final score', reference: { text: '來源逐字稿' }, reference_view: { title: 'English reference', text: 'The English reference is available after processing.', name: 'meeting.xlsx', column: 'C: English' } }} />);
    const reference = screen.getByRole('region', { name: 'English reference' });
    expect(within(reference).getByRole('heading', { name: 'English reference' })).toBeInTheDocument();
    expect(within(reference).getByText('The English reference is available after processing.')).toBeInTheDocument();
    expect(within(reference).getByText('meeting.xlsx · C: English')).toBeInTheDocument();
  });
  it('displays direct evaluation metrics from the API', () => {
    render(<Evaluation evaluation={{ metric: 'Mixed match', status: 'Final score', metrics: { score: 0.85, hits: 85, substitutions: 10, deletions: 3, insertions: 2 } }} />);
    expect(screen.getByText('85.0%')).toBeInTheDocument();
    expect(screen.getByText('Final score')).toBeInTheDocument();
  });
});
