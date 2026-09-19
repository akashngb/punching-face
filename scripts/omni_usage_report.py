#!/usr/bin/env python3
"""Produce the two files the Huawei OMNI Live organizers want by 2026-09-20 11:59 EDT.

Runs the sponsor's canonical `summarize_usage.py` against `.local/usage/yibu_api_calls.jsonl`
and writes to `.local/usage/summary/`. Then prints a redaction reminder.

The ledger is per-machine: every laptop that made a call has its own
`.local/usage/yibu_api_calls.jsonl`. Pass the teammates' copies as extra
arguments and they are merged (de-duplicated by `call_id`) before summarizing,
otherwise the totals only cover this machine.

Usage:
    npm run omni:report                                  # this machine only
    npm run omni:report -- ../from-akash.jsonl ../from-jace.jsonl

The two files to attach to the submission email:
    .local/usage/summary/usage_summary.json
    .local/usage/summary/usage_by_model_key_purpose.csv
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / '.local/usage/yibu_api_calls.jsonl'
OUT_DIR = ROOT / '.local/usage/summary'
SUMMARIZE = ROOT / '.local/third_party/yibuapi-examples/yibuapi_examples_20260918_v01/summarize_usage.py'


def merge(extra: list[str]) -> Path:
    """Concatenate this machine's ledger with teammates' copies, de-duplicated by call_id."""
    seen, rows = set(), []
    for source in [LEDGER, *(Path(e) for e in extra)]:
        if not source.is_file():
            print(f'skipping {source}: not a file', file=sys.stderr)
            continue
        for line in source.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                call_id = json.loads(line).get('call_id')
            except ValueError:
                print(f'skipping unparseable row in {source}', file=sys.stderr)
                continue
            if call_id in seen:
                continue
            seen.add(call_id)
            rows.append(line)
    merged = OUT_DIR / 'merged_yibu_api_calls.jsonl'
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'merged {len(rows)} unique calls from {1 + len(extra)} ledger(s)')
    return merged


def main() -> int:
    if not SUMMARIZE.is_file():
        print(f'Sponsor package not extracted at {SUMMARIZE.parent}. Fetch and extract '
              'yibuapi_examples_20260918_v01.tar.gz into .local/third_party/yibuapi-examples/.',
              file=sys.stderr)
        return 2
    extra = sys.argv[1:]
    # Whoever compiles the report may not be the one who made the calls, so teammates'
    # ledgers alone are enough; only the no-ledger-anywhere case is an error.
    if not extra and not LEDGER.is_file():
        print(f'No ledger at {LEDGER}, and no teammate ledgers given. Run the app and trigger at '
              'least one OMNI call, or pass the ledgers from the other machines as arguments.',
              file=sys.stderr)
        return 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = merge(extra) if extra else LEDGER
    subprocess.check_call([
        sys.executable, str(SUMMARIZE),
        '--log', str(ledger),
        '--out-dir', str(OUT_DIR),
    ])
    summary_path = OUT_DIR / 'usage_summary.json'
    csv_path = OUT_DIR / 'usage_by_model_key_purpose.csv'
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        totals = payload.get('overall_totals') or payload.get('totals') or {}
        print('overall_totals:', json.dumps(totals, ensure_ascii=False))
    print()
    print('Attach these two files to your reply to the organizers:')
    print('  ', summary_path)
    print('  ', csv_path)
    print()
    print('Before sending: `source.path` in the JSON is an absolute local path — sanitize if needed.')
    print('Do NOT include the full API key, prompts, or the raw ledger in the reply.')
    print()
    print('The reply must also state, per docs/yibuapi-usage-reporting.md:')
    for field in ('team name', 'project link', 'application email', 'reporting period',
                  'key suffixes (last 4 only)', 'any gaps in logging'):
        print(f'  - {field}')
    print('If no API calls were made, say so explicitly instead of omitting the report.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
