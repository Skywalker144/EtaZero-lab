"""Fixed configuration inputs for tests, independent of user run configurations."""
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent / 'fixtures'
CONFIGS = CONFIG_ROOT / 'configs'
