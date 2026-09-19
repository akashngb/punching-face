import { defineConfig } from 'vite';
// Sentry browser profiling requires Document-Policy: js-profiling on the document response.
// Adding it here (dev + preview) is the one prerequisite; the SDK does the rest via browserProfilingIntegration.
const jsProfilingHeader = () => ({
  name: 'sentry-js-profiling-header',
  configureServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Document-Policy', 'js-profiling');
      next();
    });
  },
  configurePreviewServer(server) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader('Document-Policy', 'js-profiling');
      next();
    });
  },
});
export default defineConfig({
  plugins: [jsProfilingHeader()],
  server: {
    host: '127.0.0.1',
    port: Number(process.env.CONTACT_WEB_PORT || 5173),
    strictPort: true,
    fs: {
      deny: [
        '.env',
        '.env.*',
        '**/*.pem',
        '**/*.crt',
        '**/.git/**',
        '**/.local/**',
        '**/.venv/**',
      ],
    },
    watch: { ignored: ['**/.local/**', '**/.venv/**', '**/artifacts/**'] },
    proxy: {
      '/physics': `http://127.0.0.1:${process.env.CONTACT_PHYSICS_PORT || 5175}`,
      '/api': {
        target: `http://127.0.0.1:${process.env.CONTACT_API_PORT || 5174}`,
        timeout: 900000,
        proxyTimeout: 900000,
      },
    },
  },
  build: { chunkSizeWarningLimit: 2500 },
});
