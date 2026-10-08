import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  build: { assetsInlineLimit: 0 },
  server: { proxy: {
    '/api': { target: 'http://127.0.0.1:8000', ws: true },
    '/guide': { target: 'http://127.0.0.1:8000' },
  } },
  test: { environment: 'jsdom', setupFiles: './src/test-setup.ts', restoreMocks: true },
});
