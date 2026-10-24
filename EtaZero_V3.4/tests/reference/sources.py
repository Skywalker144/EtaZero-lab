"""Pinned source locations shared by independent reference checks."""
import json
from pathlib import Path


def source_root(name='KataGo'):
    manifest = Path(__file__).resolve().parents[2] / 'reference_sources.json'
    return Path(json.loads(manifest.read_text())[name]['root'])
