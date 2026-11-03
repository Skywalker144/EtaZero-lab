"""EtaZero orchestration around the in-tree KataGo-derived engine and learner."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
__version__ = (ROOT / 'VERSION').read_text().strip()
