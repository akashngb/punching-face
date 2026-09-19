#!/usr/bin/env python3
"""Check that each Python environment matches the requirements file that describes it.

This project runs on more than one interpreter (see "Installation and checks" in the
README), so every requirements*.txt in the repo root starts with a header:

    # env: .venv                 environment folder(s), comma separated
    # python: >=3.9,<3.12        interpreter versions the pins can install on
    # optional: yes              only on files the app runs without

This script reads those headers, runs each environment's own interpreter, and reports
an interpreter outside the declared range or a pin the installed packages do not
satisfy. It installs nothing and changes nothing.

Usage (standard library only, any Python 3.9+):
    python3 scripts/check_python_envs.py
"""

from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = re.compile(r'^#\s*(env|python|optional):\s*(.+?)\s*$')

# Runs inside each environment, so it must work on that environment's Python and with the
# `packaging` copy bundled in its pip when the package itself is not installed.
PROBE = r'''
import json, sys
from importlib import metadata
try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet

job = json.loads(sys.argv[1])
version = '.'.join(str(part) for part in sys.version_info[:3])
result = {
    'version': version,
    'pythonOk': SpecifierSet(job['python']).contains(version, prereleases=True),
    'checked': 0,
    'missing': [],
    'wrong': [],
}
for line in job['requirements']:
    requirement = Requirement(line)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    result['checked'] += 1
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        result['missing'].append(line)
        continue
    if not requirement.specifier.contains(installed, prereleases=True):
        result['wrong'].append([line, installed])
print(json.dumps(result))
'''


def parse(path: Path) -> dict:
    """Header keys and requirement lines of one requirements file."""
    header, requirements = {}, []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line.startswith('#'):
            match = HEADER.match(line)
            if match:
                header.setdefault(match.group(1), match.group(2))
            continue
        # pip treats whitespace followed by '#' as the start of a comment.
        line = re.split(r'\s+#', line, maxsplit=1)[0].strip()
        if line and not line.startswith('-'):
            requirements.append(line)
    return {
        'envs': [e.strip() for e in header.get('env', '').split(',') if e.strip()],
        'python': header.get('python', ''),
        'optional': header.get('optional', 'no').lower() in ('yes', 'true'),
        'requirements': requirements,
    }


def interpreter(env: Path) -> Path | None:
    for candidate in (env / 'bin/python', env / 'Scripts/python.exe'):
        if candidate.exists():
            return candidate
    return None


def probe(python: Path, spec: dict) -> dict:
    job = json.dumps({'python': spec['python'], 'requirements': spec['requirements']})
    done = subprocess.run(
        [str(python), '-c', PROBE, job], capture_output=True, text=True, timeout=120
    )
    if done.returncode != 0:
        lines = done.stderr.strip().splitlines()
        return {'error': lines[-1] if lines else 'the probe failed'}
    return json.loads(done.stdout)


def check(name: str, spec: dict) -> tuple[list[str], list[str]]:
    """Report lines and failures for one requirements file."""
    report, failures = [], []
    label = ' (optional)' if spec['optional'] else ''
    if not spec['envs'] or not spec['python']:
        failures.append(f'{name}: add the "# env:" and "# python:" header lines')
        return [f'{name}: no env/python header'], failures
    for env in spec['envs']:
        report.append(f'{name} -> {env}{label}')
        python = interpreter(ROOT / env)
        if python is None:
            report.append('  not created')
            if not spec['optional']:
                failures.append(f'{env} does not exist: see the install line in {name}')
            continue
        result = probe(python, spec)
        if 'error' in result:
            report.append(f'  could not check: {result["error"]}')
            failures.append(f'{env}: could not check against {name}')
            continue
        verdict = 'ok' if result['pythonOk'] else 'OUTSIDE THE RANGE'
        report.append(
            f'  Python {result["version"]}, needs {spec["python"]}: {verdict}'
        )
        if not result['pythonOk']:
            failures.append(
                f'{env} is Python {result["version"]} but {name} needs {spec["python"]}'
            )
        for line, installed in result['wrong']:
            report.append(f'  {line}: installed {installed}')
            failures.append(f'{env} has a version outside the pin "{line}" in {name}')
        for line in result['missing']:
            report.append(f'  {line}: not installed')
            if not spec['optional']:
                failures.append(f'{env} is missing "{line}" from {name}')
        satisfied = result['checked'] - len(result['missing']) - len(result['wrong'])
        report.append(f'  {satisfied} of {result["checked"]} requirements satisfied')
    return report, failures


def main() -> int:
    files = sorted(ROOT.glob('requirements*.txt'))
    if not files:
        print(f'No requirements*.txt in {ROOT}', file=sys.stderr)
        return 2
    failures = []
    for path in files:
        report, failed = check(path.name, parse(path))
        print('\n'.join(report))
        failures.extend(failed)
    print()
    if failures:
        print('Python environments do NOT match the requirements files:')
        for failure in failures:
            print(f'  - {failure}')
        return 1
    print('Every Python environment matches its requirements file.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
