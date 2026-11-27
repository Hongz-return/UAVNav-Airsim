import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 纯离线静态产物：base 使用相对路径，dist/ 由 Python 后端直接托管（无需 CDN / 网络）。
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1800,
  },
  server: {
    port: 5173,
    proxy: {
      '/ws': { target: 'ws://127.0.0.1:8765', ws: true, rewriteWsOrigin: true },
      '/api': 'http://127.0.0.1:8765',
    },
  },
});
