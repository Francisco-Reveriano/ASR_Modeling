import type { AppConfig, ReferenceOptions, ReferencePreview, SessionCommand, SessionSettings, SessionSnapshot } from './types';
export class ApiError extends Error {
  constructor(message: string, public readonly status: number) { super(message); this.name = 'ApiError'; }
}
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(path, options); }
  catch { throw new Error('Cannot reach the local server. Check that FastAPI is running.'); }
  if (!response.ok) {
    let detail = '';
    try { const body = await response.json(); detail = typeof body.detail === 'string' ? body.detail : ''; } catch { /* HTTP fallback */ }
    throw new ApiError(detail || `The server could not complete this request (${response.status}).`, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
export const api = {
  config: () => request<AppConfig>('/api/config'),
  health: () => request<Record<string, unknown>>('/api/health'),
  get: (id: string) => request<SessionSnapshot>(`/api/sessions/${encodeURIComponent(id)}`),
  create: (settings: SessionSettings) => request<SessionSnapshot>('/api/sessions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings),
  }),
  upload: (file: File, settings: SessionSettings, reference?: File, options?: ReferenceOptions) => {
    const body = new FormData(); body.append('file', file); body.append('settings', JSON.stringify(settings));
    if (reference) { body.append('reference', reference); body.append('reference_options', JSON.stringify(options || { format: 'auto' })); }
    return request<SessionSnapshot>(`/api/sessions/${reference ? 'evaluate' : 'upload'}`, { method: 'POST', body });
  },
  preview: (file: File, options: ReferenceOptions) => {
    const body = new FormData(); body.append('file', file); body.append('options', JSON.stringify(options));
    return request<ReferencePreview>('/api/references/preview', { method: 'POST', body });
  },
  command: (id: string, command: SessionCommand) => request<SessionSnapshot>(`/api/sessions/${encodeURIComponent(id)}/commands`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(command),
  }),
  remove: (id: string) => request<void>(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
};
export function websocketUrl(path: string): string { return `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${path}`; }
export function errorMessage(error: unknown): string { return error instanceof Error ? error.message : 'Something went wrong. Please try again.'; }
