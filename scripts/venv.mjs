// Interpreter paths for the two environments. POSIX venvs expose bin/python, Windows venvs Scripts/python.exe.
const leaf = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
export const capturePython = `.venv/${leaf}`;
export const newtonPython = `.local/newton-env/${leaf}`;
