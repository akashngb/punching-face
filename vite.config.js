import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Keep a live demo/camera session stable while other work changes files.
// Developers can opt into automatic updates with CONTACT_HMR=1 npm run dev.
const enableHmr = process.env.CONTACT_HMR === '1';

const quietGeneratedAssets = () => ({
  name: 'contact-quiet-generated-assets',
  configureServer(server) {
    if (!enableHmr) return;
    const publicDir = server.config.publicDir.replaceAll('\\', '/');
    const generated = (file = '') =>
      ['generated/', 'textures/', 'impact-preview/'].some((folder) =>
        [publicDir, '/public', ''].some((prefix) =>
          file.replaceAll('\\', '/').startsWith(`${prefix}/${folder}`),
        ),
      );
    // Keep public assets watched: Vite uses add/unlink events to discover new URLs.
    // Suppress their browser updates instead of breaking that serving cache.
    const hot = server.environments.client.hot;
    const send = hot.send.bind(hot);
    hot.send = (...args) => {
      const payload = args[0];
      if (payload?.type === 'full-reload') {
        if (generated(payload.path) || generated(payload.triggeredBy)) return;
      } else if (payload?.type === 'update') {
        const updates = payload.updates.filter(
          (update) => !generated(update.path) && !generated(update.acceptedPath),
        );
        if (!updates.length) return;
        return send({ ...payload, updates });
      }
      return send(...args);
    };
  },
});

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
  plugins: [react(), tailwindcss(), jsProfilingHeader(), quietGeneratedAssets()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '127.0.0.1',
    port: Number(process.env.CONTACT_WEB_PORT || 5173),
    strictPort: true,
    hmr: enableHmr,
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
    watch: {
      ignored: [
        '**/.local/**',
        '**/.venv/**',
        '**/artifacts/**',
        '**/dist-guest/**',
        '**/docs/**',
        '**/scripts/**',
        '**/tests/**',
        '**/*.md',
        '**/__pycache__/**',
      ],
    },
    proxy: {
      '/physics': `http://127.0.0.1:${process.env.CONTACT_PHYSICS_PORT || 5175}`,
      '/api': {
        target: `http://127.0.0.1:${process.env.CONTACT_API_PORT || 5174}`,
        timeout: 900000,
        proxyTimeout: 900000,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 2500,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        design: fileURLToPath(new URL('./design.html', import.meta.url)),
      },
    },
  },
});
