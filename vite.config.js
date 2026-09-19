import { defineConfig } from 'vite';
export default defineConfig({server:{host:'127.0.0.1',port:5173,strictPort:true,fs:{deny:['.env','.env.*','**/*.pem','**/*.crt','**/.git/**','**/.local/**','**/.venv/**']},watch:{ignored:['**/.local/**','**/.venv/**','**/artifacts/**']},proxy:{'/physics':'http://127.0.0.1:5175','/api':'http://127.0.0.1:5174'}},build:{chunkSizeWarningLimit:2500}});
