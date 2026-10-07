from pathlib import Path
import os
import sys

python_root=str(Path(__file__).resolve().parents[1]/"python")
sys.path.insert(0,python_root)
# Recovery tests spawn fresh interpreters, which do not inherit sys.path.
python_path=os.environ.get('PYTHONPATH')
os.environ['PYTHONPATH']=python_root+(os.pathsep+python_path if python_path else '')


import importlib
import pytest
from config_samples import CONFIG_ROOT


@pytest.fixture(scope="session", autouse=True)
def isolated_configuration_sources():
    """Parser fallback parents and umbrella defaults come from test samples."""
    with pytest.MonkeyPatch.context() as patch:
        for name in ('etazero.config', 'etazero.eval_config', 'etazero.autoelo'):
            module = importlib.import_module(name)
            patch.setattr(module, 'ROOT', CONFIG_ROOT)
        yield
