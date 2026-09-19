#!/usr/bin/env python3
"""Produce the two files the Huawei OMNI Live organizers want by 2026-09-20 11:59 EDT.

Runs the sponsor's canonical `summarize_usage.py` against `.local/usage/yibu_api_calls.jsonl`
and writes to `.local/usage/summary/`. Then prints a redaction reminder.

Usage:
    .venv/bin/python scripts/omni_usage_report.py

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


def main() -> int:
    if not SUMMARIZE.is_file():
        print(f'Sponsor package not extracted at {SUMMARIZE.parent}. Fetch and extract '
              'yibuapi_examples_20260918_v01.tar.gz into .local/third_party/yibuapi-examples/.',
              file=sys.stderr)
        return 2
    if not LEDGER.is_file():
        print(f'No ledger at {LEDGER}. Run the app and trigger at least one OMNI call first.', file=sys.stderr)
        return 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        sys.executable, str(SUMMARIZE),
        '--log', str(LEDGER),
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
