"""INI files with explicit relative includes and last-file-wins overrides."""
import configparser
from pathlib import Path
import shlex


def read_config_file(path, fields, *, local_only=(), section_aliases=None, ancestors=()):
    section_aliases = section_aliases or {}
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError('Configuration include cycle: ' + ' -> '.join(map(str, (*ancestors, path))))
    lines, includes = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith('@include'):
            if any(text.lstrip().startswith('[') for text in lines):
                raise ValueError(f'@include must precede sections: {path}:{number}')
            tokens = shlex.split(line, comments=True)
            if len(tokens) != 2 or tokens[0] != '@include':
                raise ValueError(f'Invalid include: {path}:{number}: {line}')
            includes.append(path.parent / tokens[1])
        else:
            lines.append(line)
    values = {}
    for included in includes:
        values.update(read_config_file(included, fields, local_only=local_only, section_aliases=section_aliases,
                                       ancestors=(*ancestors, path)))
    for key in local_only:
        values.pop(key, None)
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.read_string('\n'.join(lines), source=str(path))
    if parser.defaults():
        raise ValueError(f'DEFAULT fields are not supported: {path}')
    for section in parser.sections():
        if section not in fields:
            raise ValueError(f'Wrong section/file ownership: {path}: [{section}]')
        for key, value in parser.items(section):
            if key not in fields[section]:
                raise ValueError(f'Unknown key: {path}: {section}.{key}')
            values[section_aliases.get(section, section), key] = value
    return values
