"""Verify the committed release manifest without network access."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'data/release_manifest.json').read_text(encoding='utf-8'))
for row in manifest['files']:
    path = (ROOT / row['file']).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert path.is_file(), row['file']
    assert path.stat().st_size == row['bytes'], row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['file']
print(json.dumps({'status':'pass','files_checked':len(manifest['files']),'bytes':sum(r['bytes'] for r in manifest['files'])}))
