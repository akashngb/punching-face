// Run a repo Python entry point with the capture venv, so package.json scripts stay platform-independent.
import { spawn } from 'node:child_process';
import { capturePython } from './venv.mjs';
const child = spawn(capturePython, process.argv.slice(2), {stdio:'inherit'});
child.on('error', e => {console.error(e.message);process.exit(1)});
child.on('exit', code => process.exit(code ?? 0));
