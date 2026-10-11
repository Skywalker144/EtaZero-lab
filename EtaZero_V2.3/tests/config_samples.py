"""Fixed configuration inputs for tests, independent of user run configurations."""
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent / 'fixtures'
CONFIGS = CONFIG_ROOT / 'configs'


def derive_config(directory, parent, run=''):
    """Construct a training profile using the same explicit includes as user profiles."""
    import os
    import shlex
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('run', 'env', 'net', 'selfplay', 'train'):
        source = shlex.quote(os.path.relpath(Path(parent)/f'{name}.cfg', directory))
        (directory/f'{name}.cfg').write_text(f'@include {source}\n' +
                                           (f'[run]\n{run}' if name == 'run' else ''))
    return directory
