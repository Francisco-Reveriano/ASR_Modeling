import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { api } from './api';
import ReferencePanel from './ReferencePanel';
import type { ReferencePreview } from './types';

describe('reference mapping', () => {
  it('does not restore an uploaded reference after the panel has been cleared', async () => {
    const user = userEvent.setup(); const onChange = vi.fn();
    let finish!: (value: ReferencePreview) => void;
    vi.spyOn(api, 'preview').mockReturnValue(new Promise(resolve => { finish = resolve; }));
    const page = render(<ReferencePanel disabled={false} onChange={onChange} />);
    await user.upload(screen.getByLabelText('Reference file'), new File(['Hello'], 'reference.txt', { type: 'text/plain' }));
    await user.click(screen.getByRole('button', { name: 'Preview reference' }));
    const before = onChange.mock.calls.length;
    page.unmount();
    await act(async () => finish({ reference: { text: 'Hello' }, reference_view: null, needs_source_selection: false }));
    expect(onChange).toHaveBeenCalledTimes(before);
  });
  it('requires a refreshed preview after a column changes and submits one-based header rows and zero-based columns', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const preview = vi.spyOn(api, 'preview').mockResolvedValue({
      reference: null, reference_view: null, needs_source_selection: true, sheet: 'Transcript', header_row: 1, source_column: null, english_column: 1,
      tables: [{ name: 'Transcript', columns: ['A: Source', 'B: English'], suggested_source: null, suggested_english: 1 }],
    });
    render(<ReferencePanel disabled={false} onChange={onChange} />);
    const file = new File(['Source,English\n原文,Hello'], 'reference.csv', { type: 'text/csv' });
    await user.upload(screen.getByLabelText('Reference file'), file);
    await user.click(screen.getByRole('button', { name: 'Preview reference' }));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ ready: false }));
    await user.selectOptions(screen.getByLabelText('Source transcript column'), '0');
    expect(screen.getByText('Update the preview to apply these column settings.')).toBeInTheDocument();
    preview.mockResolvedValue({ reference: { text: '原文', format: 'Plain text', segment_count: 1 }, reference_view: { text: 'Hello' }, needs_source_selection: false, sheet: 'Transcript', header_row: 1, source_column: 0, english_column: 1, tables: [{ name: 'Transcript', columns: ['A: Source', 'B: English'], suggested_source: 0, suggested_english: 1 }] });
    await user.click(screen.getByRole('button', { name: 'Update reference preview' }));
    expect(preview).toHaveBeenLastCalledWith(file, expect.objectContaining({ source_column: 0, header_row: 1 }));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ ready: true }));
    expect(screen.getByText('原文')).toBeInTheDocument();
    const header = screen.getByRole('spinbutton');
    await user.clear(header); await user.type(header, '0');
    await user.click(screen.getByRole('button', { name: 'Update reference preview' }));
    expect(preview).toHaveBeenLastCalledWith(file, expect.objectContaining({ header_row: 0 }));
  });
});
