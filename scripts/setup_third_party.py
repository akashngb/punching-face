"""Fetch the pinned third-party sources listed in scripts/third_party_manifest.json.

Source code and ungated model data only. Nothing fetched here is imported or
executed, no binaries or model weights are downloaded, and submodules are never
followed. See OPEN_SOURCE_STACK.md for what each source is for.

Sources land in .local/third_party, beside .local/sources. The Vite watcher and
dependency scan already skip .local, so fetching never reloads a camera session.
"""

from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
import argparse, json, os, re, shutil, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
# Never prompt for credentials, and never let an LFS filter pull large files unasked.
ENV = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_LFS_SKIP_SMUDGE': '1'}
LFS = b'version https://git-lfs.github.com/spec'


def git(*args, cwd):
    return subprocess.run(
        ['git', *args], cwd=cwd, check=True, env=ENV, capture_output=True, text=True
    ).stdout.strip()


def target(base, name):
    # Manifest names become directories; keep them inside the destination.
    if not re.fullmatch(r'[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?', name) or '..' in name:
        raise ValueError(f'Unsafe entry name: {name}')
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'Unsafe entry name: {name}')
    return path


def fetch_repo(entry, work):
    if not re.fullmatch(r'https://github\.com/[\w.-]+/[\w.-]+', entry['url']):
        raise ValueError(f"Unexpected source: {entry['url']}")
    git('init', '-q', cwd=work)
    git('remote', 'add', 'origin', entry['url'], cwd=work)
    if entry.get('sparse'):
        git('sparse-checkout', 'set', *entry['sparse'], cwd=work)
    # One pinned commit without history, fetching only the blobs that get checked out.
    git(
        'fetch',
        '-q',
        '--depth',
        '1',
        '--filter=blob:none',
        'origin',
        entry['sha'],
        cwd=work,
    )
    git('-c', 'advice.detachedHead=false', 'checkout', '-q', 'FETCH_HEAD', cwd=work)
    head = git('rev-parse', 'HEAD', cwd=work)
    if head != entry['sha']:
        raise ValueError(f"{entry['name']}: fetched {head}, expected {entry['sha']}")
    # Keep a plain snapshot so the project never contains a nested repository.
    shutil.rmtree(work / '.git')


def fetch_files(entry, work):
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', entry['repo']):
        raise ValueError(f"Unexpected source: {entry['repo']}")
    base = f"https://raw.githubusercontent.com/{entry['repo']}/{entry['sha']}/"
    for name in entry['files']:
        if not re.fullmatch(r'[\w./-]+', name) or '..' in name:
            raise ValueError(f'Unsafe file path: {name}')
        with urlopen(base + name, timeout=60) as response:
            (work / Path(name).name).write_bytes(response.read())
    # Carry the licence text alongside copied files. Some sources state it per file instead.
    for name in (
        'LICENSE',
        'LICENSE.txt',
        'LICENSE.md',
        'License',
        'License.txt',
        'license',
        'COPYING',
    ):
        try:
            with urlopen(base + name, timeout=60) as response:
                (work / 'LICENSE.upstream').write_bytes(response.read())
            break
        except HTTPError:
            continue


def run(dest, only, force):
    manifest = json.loads((ROOT / 'scripts/third_party_manifest.json').read_text())
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    for entry in manifest['entries']:
        if only and entry['name'] not in only:
            continue
        if not re.fullmatch(r'[0-9a-f]{40}', entry['sha']):
            raise ValueError(f"{entry['name']}: pin a full commit hash.")
        path = target(dest, entry['name'])
        stamp = path / '.source.json'
        if (
            not force
            and stamp.exists()
            and json.loads(stamp.read_text()).get('sha') == entry['sha']
        ):
            report.append((entry['name'], 'current', path))
            continue
        work = path.with_name(path.name + '.partial')
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        try:
            (fetch_repo if entry['kind'] == 'repo' else fetch_files)(entry, work)
            source = entry.get('url') or f"https://github.com/{entry['repo']}"
            (work / '.source.json').write_text(
                json.dumps(
                    {
                        'name': entry['name'],
                        'source': source,
                        'sha': entry['sha'],
                        'license': entry['license'],
                        'sparse': entry.get('sparse'),
                        'fetched': time.strftime('%Y-%m-%d'),
                    },
                    indent=2,
                )
            )
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        if path.exists():
            shutil.rmtree(path)
        work.rename(path)
        report.append((entry['name'], 'fetched', path))
    for name, state, path in report:
        files = [p for p in path.rglob('*') if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        # An LFS pointer means the real asset was deliberately not downloaded.
        pointers = [
            p
            for p in files
            if p.stat().st_size < 400 and p.read_bytes().startswith(LFS)
        ]
        print(
            f"{state:8} {name:36} {size/1e6:8.1f} MB  {len(files):5} files"
            + (f"  ({len(pointers)} LFS pointers not downloaded)" if pointers else '')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--only', action='append', help='Entry name from the manifest; repeatable.'
    )
    parser.add_argument('--dest', type=Path, default=ROOT / '.local/third_party')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Refetch even when the pinned commit is already present.',
    )
    args = parser.parse_args()
    run(args.dest.resolve(), set(args.only or []), args.force)
