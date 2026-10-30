#!/usr/bin/env python3
"""Check public source syntax, JSON, file sizes, and private path patterns."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(rb'/(?:mnt/data|file/share|data/dataset|data/data_[^/\s]+|home|root)/')
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', 'outputs', 'runs'}

def main():
    errors = []
    count = total = largest = 0
    for p in sorted(ROOT.rglob('*')):
        if SKIP.intersection(p.relative_to(ROOT).parts):
            continue
        if p.is_symlink():
            errors.append(f'Symlink: {p.relative_to(ROOT)}')
            continue
        if not p.is_file():
            continue
        data = p.read_bytes()
        count += 1
        total += len(data)
        largest = max(largest, len(data))
        if len(data) > 25 * 1024**2:
            errors.append(f'File exceeds conservative 25 MiB upload limit: {p.relative_to(ROOT)}')
        if PRIVATE.search(data) or SECRET.search(data):
            errors.append(f'Private path or credential pattern: {p.relative_to(ROOT)}')
        try:
            if p.suffix == '.py':
                ast.parse(data, filename=str(p.relative_to(ROOT)))
            elif p.suffix == '.json':
                json.loads(data)
        except (ValueError, SyntaxError) as exc:
            errors.append(f'{p.relative_to(ROOT)}: {exc}')
    print(json.dumps({'files': count, 'bytes': total, 'largest_file_bytes': largest, 'errors': errors}, indent=2))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
