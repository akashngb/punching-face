import { defineConfig } from 'vite';
// Sentry browser profiling requires Document-Policy: js-profiling on the document response.
// Adding it here (dev + preview) is the one prerequisite; the SDK does the rest via browserProfilingIntegration.
// Block body on purpose: an implicit return hands Vite the connect app, which it then
// calls as a post-hook with no request and crashes the server on startup.
const configureServer = server => {
  server.middlewares.use((req, res, next) => {
    res.setHeader('Document-Policy', 'js-profiling');
    if (req.url === '/cv-debug') { res.statusCode = 302; res.setHeader('Location', '/cv-debug/'); res.end(); } else next();
  });
};
const jsProfilingHeader = () => ({
  name: 'sentry-js-profiling-header',
  configureServer,
  configurePreviewServer: configureServer,
});
export default defineConfig({plugins:[jsProfilingHeader()],server:{host:'127.0.0.1',port:5173,strictPort:true,fs:{deny:['.env','.env.*','**/*.pem','**/*.crt','**/.git/**','**/.local/**','**/.venv/**']},watch:{ignored:['**/.local/**','**/.venv/**','**/artifacts/**']},proxy:{'/physics':'http://127.0.0.1:5175','/api':{target:'http://127.0.0.1:5174',timeout:900000,proxyTimeout:900000}}},build:{rollupOptions:{input:['index.html','cv-debug/index.html']},chunkSizeWarningLimit:2500}});
