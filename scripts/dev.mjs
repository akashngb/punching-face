import { spawn } from 'node:child_process';
import { capturePython, newtonPython } from './venv.mjs';
const children = [
  spawn(newtonPython, ['physics_server.py'], {stdio:'inherit'}),
  spawn(capturePython, ['server.py'], {stdio:'inherit'}),
  spawn(capturePython, ['omni_relay.py'], {stdio:'inherit'}),
  spawn('node', ['node_modules/vite/bin/vite.js'], {stdio:'inherit'}),
];
let stopping=false;
function stop(code=0){if(stopping)return;stopping=true;for(const p of children)p.kill('SIGTERM');process.exitCode=code;}
for(const p of children){p.on('error',e=>{console.error(e.message);stop(1)});p.on('exit',code=>stop(code??0))}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop());
