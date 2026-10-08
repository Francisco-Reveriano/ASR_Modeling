import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from './api';
import { config, FakeWebSocket, segment, snapshot } from './test-fixtures';
import App from './App';

const audio = vi.hoisted(() => ({ start: vi.fn(), stop: vi.fn(), dispose: vi.fn(), unlock: vi.fn(), update: vi.fn() }));
vi.mock('./audio', () => ({
  CaptureController: class {
    constructor(private callbacks: { onStatus: (state: unknown) => void }) {}
    async start(options: unknown) { await audio.start(options); this.callbacks.onStatus({ phase: 'recording', message: 'Live', sampleRate: 48000 }); }
    async stop() { await audio.stop(); }
    dispose() { audio.dispose(); }
  },
  SpeechController: class {
    update(value: unknown) { audio.update(value); }
    unlock() { audio.unlock(); }
    pause() {}
    resume() {}
    stop() {}
    dispose() {}
  },
}));

describe('voice workspace', () => {
  beforeEach(() => {
    history.replaceState(null, '', '/');
    const stored = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      get length() { return stored.size; }, getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => stored.set(key, value), clear: () => stored.clear(),
    });
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
    vi.stubGlobal('AudioWorkletNode', class {});
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(), enumerateDevices: vi.fn().mockResolvedValue([]), addEventListener: vi.fn(), removeEventListener: vi.fn() } });
    vi.spyOn(api, 'config').mockResolvedValue(config);
    vi.spyOn(api, 'create').mockResolvedValue(snapshot());
    vi.spyOn(api, 'get').mockResolvedValue(snapshot());
    vi.spyOn(api, 'upload').mockResolvedValue(snapshot({ kind: 'upload', status: 'processing', input_finished: true, stop_requested: false }));
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn().mockReturnValue('blob:test-recording') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    vi.spyOn(api, 'command').mockResolvedValue(snapshot({ revision: 2, status: 'processing', input_finished: true }));
    vi.spyOn(api, 'remove').mockResolvedValue();
    audio.start.mockResolvedValue(undefined); audio.start.mockClear(); audio.stop.mockClear();
  });
  it('starts with one click and text-only defaults, freezes model controls, then preserves stopped results', async () => {
    const user = userEvent.setup(); render(<App />);
    const start = await screen.findByRole('button', { name: 'Start subtitles' });
    await waitFor(() => expect(start).toBeEnabled());
    expect(screen.getByRole('switch', { name: /Spoken English/ })).not.toBeChecked();
    await user.click(start);
    expect(api.create).toHaveBeenCalledWith(expect.objectContaining({ model_pair: 'gpt-live-transcribe + gpt-6-luna', speech_enabled: false }));
    expect(audio.start).toHaveBeenCalledWith({ sessionId: 'session-1', deviceId: undefined });
    await user.click(screen.getByText('Models & translation settings'));
    expect(screen.getByLabelText('Transcription + translation model')).toBeDisabled();
    act(() => FakeWebSocket.instances[0].receive({ type: 'snapshot', snapshot: snapshot({ segments: [segment()] }) }));
    await user.click(screen.getByRole('button', { name: 'Stop recording' }));
    expect(audio.stop).toHaveBeenCalled();
    expect(api.command).toHaveBeenCalledWith('session-1', { type: 'stop' });
    expect(api.remove).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'End and clear' })).toBeInTheDocument();
  });
  it('keeps microphone and upload model choices independent while evaluation remains Breeze', async () => {
    const user = userEvent.setup(); render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    await user.click(screen.getByText('Models & translation settings'));
    const model = screen.getByLabelText('Transcription + translation model');
    await user.selectOptions(model, 'Breeze + OpenAI');
    await user.click(screen.getByRole('tab', { name: 'Upload WAV' }));
    expect(model).toHaveValue('gpt-live-transcribe + gpt-6-luna');
    await user.selectOptions(model, 'gpt-realtime-translate');
    await user.click(screen.getByRole('tab', { name: 'Microphone' }));
    expect(model).toHaveValue('Breeze + OpenAI');
    await user.click(screen.getByRole('tab', { name: 'Evaluate' }));
    expect(model).toHaveValue('Breeze + OpenAI'); expect(model).toBeDisabled();
    await user.click(screen.getByRole('tab', { name: 'Upload WAV' }));
    expect(model).toHaveValue('gpt-realtime-translate');
    await user.upload(screen.getByLabelText('WAV recording'), new File(['RIFF'], 'example.wav', { type: 'audio/wav' }));
    await user.click(screen.getByRole('button', { name: 'Translate recording' }));
    expect(api.upload).toHaveBeenCalledWith(expect.any(File), expect.objectContaining({ model_pair: 'gpt-realtime-translate' }), undefined, expect.any(Object));
  });
  it('offers only TXT and JSON for a realtime session, independent of the next model choice', async () => {
    const id = 'abcdef0123456789abcdef0123456789';
    history.replaceState(null, '', `/?session=${id}`);
    vi.mocked(api.get).mockResolvedValue(snapshot({ id, status: 'complete', finished: true, settings: { ...config.defaults, model_pair: 'gpt-realtime-translate' }, realtime: { source: '原文', english: 'English', incomplete: false } }));
    const user = userEvent.setup(); render(<App />);
    await screen.findByText('Your previous session is restored. Processing and available results are preserved.');
    await user.click(screen.getByText('Models & translation settings'));
    await user.selectOptions(screen.getByLabelText('Transcription + translation model'), 'Breeze + OpenAI');
    await user.click(screen.getByText('Export'));
    expect(screen.getByRole('link', { name: 'TXT' })).toHaveAttribute('href', `/api/sessions/${id}/exports/txt`);
    expect(screen.getByRole('link', { name: 'JSON' })).toBeInTheDocument();
    for (const name of ['CSV', 'SRT', 'VTT']) expect(screen.queryByRole('link', { name })).not.toBeInTheDocument();
  });
  it('marks completed provider failures clearly and returns to processing when retried', async () => {
    const id = 'abcdef0123456789abcdef0123456789';
    history.replaceState(null, '', `/?session=${id}`);
    const failed = snapshot({ id, status: 'complete', finished: true, segments: [segment(0, { english: null, status: 'unavailable', translations: { openai: { text: null, status: 'unavailable', error: 'Translation request failed.' } } })] });
    vi.mocked(api.get).mockResolvedValue(failed);
    vi.mocked(api.command).mockResolvedValue({ ...failed, revision: 2, status: 'processing', finished: false });
    const user = userEvent.setup(); render(<App />);
    expect(await screen.findByText('Completed with errors')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Retry OpenAI' }));
    expect(api.command).toHaveBeenCalledWith(id, { type: 'retry', provider: 'openai' });
    expect(await screen.findByText('Processing', { exact: true })).toBeInTheDocument();
    expect(screen.queryByText('Completed with errors')).not.toBeInTheDocument();
  });
  it('ignores a command failure from a session that has been cleared and replaced', async () => {
    let rejectOld!: (error: Error) => void;
    const user = userEvent.setup(); render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start subtitles' }));
    vi.mocked(api.command).mockReturnValueOnce(new Promise((_, reject) => { rejectOld = reject; }));
    await user.click(screen.getByRole('switch', { name: /Spoken English/ }));
    await user.click(screen.getByRole('button', { name: 'End and clear' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    vi.mocked(api.create).mockResolvedValue(snapshot({ id: 'replacement' }));
    await user.click(screen.getByRole('button', { name: 'Start subtitles' }));
    await act(async () => rejectOld(new Error('Old session request failed.')));
    expect(screen.queryByText('Old session request failed.')).not.toBeInTheDocument();
    expect(screen.getByRole('switch', { name: /Spoken English/ })).toBeEnabled();
  });
  it('updates the same transcript row when corrected and stores presentation preferences only', async () => {
    const user = userEvent.setup(); render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start subtitles' }));
    act(() => FakeWebSocket.instances[0].receive({ type: 'snapshot', snapshot: snapshot({ segments: [segment()] }) }));
    const transcript = screen.getByRole('region', { name: 'Transcript turns' });
    expect(within(transcript).queryByText('原文 0')).not.toBeInTheDocument();
    await user.click(screen.getByLabelText('Show original'));
    expect(within(transcript).getByText('原文 0')).toBeInTheDocument();
    act(() => FakeWebSocket.instances[0].receive({ ...snapshot(), type: 'update', revision: 2, segments: [segment(0, { english: 'A corrected translation', status: 'corrected' })] }));
    expect(within(transcript).getAllByRole('article')).toHaveLength(1);
    expect(within(transcript).getByText('A corrected translation')).toBeInTheDocument();
    expect(localStorage.length).toBe(1);
    const saved = localStorage.getItem('speech-voice-presentation-v1')!;
    expect(saved).toContain('"original":true');
    expect(saved).not.toContain('translation'); expect(saved).not.toContain('session-1');
  });
  it('restores previous results if replacement model preparation fails', async () => {
    const user = userEvent.setup(); render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start subtitles' }));
    const original = snapshot({ revision: 2, finished: true, status: 'complete', segments: [segment()] });
    act(() => FakeWebSocket.instances[0].receive({ type: 'snapshot', snapshot: original }));
    vi.mocked(api.create).mockResolvedValue(snapshot({ id: 'replacement', kind: 'microphone', status: 'preparing' }));
    audio.start.mockRejectedValueOnce(new Error('Local model could not load.'));
    await user.click(screen.getByRole('button', { name: 'Start new conversation' }));
    await screen.findByText('Local model could not load.');
    expect(screen.getAllByText('English turn 0')).toHaveLength(2);
    expect(api.remove).toHaveBeenCalledWith('replacement');
    expect(api.remove).not.toHaveBeenCalledWith('session-1');
  });
  it('surfaces microphone permission failure without claiming recording succeeded', async () => {
    audio.start.mockRejectedValueOnce(new Error('Microphone permission was denied.'));
    const user = userEvent.setup(); render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start subtitles' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Microphone permission was denied.');
    expect(api.command).toHaveBeenCalledWith('session-1', { type: 'stop' });
  });
  it('restores an opaque URL session without restarting microphone input', async () => {
    const id = 'abcdef0123456789abcdef0123456789';
    history.replaceState(null, '', `/?session=${id}`);
    vi.mocked(api.get).mockResolvedValue(snapshot({ id, segments: [segment()] }));
    render(<App />);
    await screen.findByText(/Microphone capture was interrupted by the page reload/);
    expect(api.get).toHaveBeenCalledWith(id);
    expect(audio.start).not.toHaveBeenCalled();
    expect(screen.getAllByText('English turn 0')).toHaveLength(2);
    expect(new URL(location.href).searchParams.get('session')).toBe(id);
  });
  it('clears an expired session URL and remains ready to start', async () => {
    history.replaceState(null, '', '/?session=abcdef0123456789abcdef0123456789');
    vi.mocked(api.get).mockRejectedValue(new ApiError('This session is unavailable.', 404));
    render(<App />);
    await screen.findByText('The previous session expired or was cleared. Start a new conversation.');
    expect(new URL(location.href).searchParams.has('session')).toBe(false);
    expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled();
  });
  it('allows stopping a fully uploaded file while processing and clears its preview on End and clear', async () => {
    const user = userEvent.setup(); render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    await user.click(screen.getByRole('tab', { name: 'Upload WAV' }));
    await user.upload(screen.getByLabelText('WAV recording'), new File(['RIFF'], 'example.wav', { type: 'audio/wav' }));
    await user.click(screen.getByRole('button', { name: 'Translate recording' }));
    const stop = screen.getByRole('button', { name: 'Stop file translation' });
    expect(stop).toBeEnabled();
    await user.click(stop);
    expect(api.command).toHaveBeenCalledWith('session-1', { type: 'stop' });
    expect(api.remove).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.getByRole('button', { name: 'End and clear' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'End and clear' }));
    await waitFor(() => expect(screen.queryByLabelText('Preview uploaded recording')).not.toBeInTheDocument());
    expect(screen.getByLabelText('WAV recording')).toHaveValue('');
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-recording');
    expect(new URL(location.href).searchParams.has('session')).toBe(false);
  });
  it('clears the server session and keeps the interface ready for a new conversation', async () => {
    const user = userEvent.setup(); render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start subtitles' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Start subtitles' }));
    await user.click(screen.getByRole('button', { name: 'End and clear' }));
    expect(api.remove).toHaveBeenCalledWith('session-1');
    expect(await screen.findByRole('button', { name: 'Start subtitles' })).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'End and clear' })).not.toBeInTheDocument();
  });
});
