"""Keep the exact native selfplay/gatekeeper inputs with the run."""
import re
from pathlib import Path


def native_inputs(config):
    files = {}
    for role, path, supplied in (
        ('selfplay', config.selfplay_config, config.native_selfplay),
        ('gatekeeper', config.gatekeeper_config, None),
    ):
        if path is None:
            continue
        seen = set()

        def capture(source, output, content):
            lines = []
            for line in content.splitlines(keepends=True):
                if not line.lstrip().startswith('@'):
                    lines.append(line)
                    continue
                # Only relocate includes. Values, comments and override boundaries
                # are still parsed by the native ConfigParser itself.
                directive = line.split('#', 1)[0].strip()
                match = re.fullmatch(r'@include[ \t\v\f=]+(.+)', directive)
                if not match:
                    raise ValueError(f'{source}: invalid native config directive {directive}')
                name = match[1].strip().strip("'").strip('"')
                child = source.parent / name
                if not name or name.startswith(('/', '\\')) or name in seen:
                    raise ValueError(f'{source}: invalid or repeated native include {name!r}')
                seen.add(name)
                slot = f'{role}-{len(seen)}.cfg'
                target = Path('native-includes') / slot
                ref = target if output.parent == Path('.') else Path(slot)
                lines.append(f'@include {ref}\n')
                capture(child, target, child.read_text())
            files[output] = ''.join(lines)

        source = Path(path)
        capture(source, Path(f'{role}.resolved.cfg'), source.read_text() if supplied is None else supplied)
    return files
