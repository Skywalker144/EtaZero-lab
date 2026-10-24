"""Version identity shared with the native build."""
from pathlib import Path

VERSION = (Path(__file__).resolve().parents[2] / "VERSION").read_text().strip()
