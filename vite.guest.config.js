// Builds only the guest page into dist-guest/, so it can be dropped on any https static host and
// opened on phones. Separate from vite.config.js on purpose: the main app, its dev server and its
// proxy settings are untouched. publicDir is off because the guest page loads MediaPipe from public
// CDNs when deployed, which keeps the upload to a few hundred kB instead of ~70 MB of lab assets.
import { defineConfig } from 'vite';
export default defineConfig({base:'./',publicDir:false,build:{outDir:'dist-guest',emptyOutDir:true,rollupOptions:{input:'guest.html'},chunkSizeWarningLimit:1500}});
