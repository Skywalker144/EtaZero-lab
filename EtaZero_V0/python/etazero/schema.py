"""The single source of truth for the Python/C++ model and trajectory contract."""
import hashlib
import json

PLANES = ("own", "opponent", "black_to_move", "on_board", "standard", "renju")
RULES = ("freestyle", "standard", "renju")
REASONS = ("draw", "five", "forbidden")
RAW_DTYPES = {
    "observations": "uint8", "players": "int8", "actions": "int32",
    "policies": "float32", "visits": "int64", "simulations": "int32",
    "temperatures": "float32", "rewards": "float32", "game_offsets": "int64",
    "observation_offsets": "int64", "game_ids": "uint64", "seeds": "uint64",
    "sizes": "int16", "rules": "int8", "winners": "int8", "reasons": "int8",
    "metadata": "uint8",
}
CONTRACT = {"planes": PLANES, "rules": RULES, "reasons": REASONS,
            "raw_dtypes": RAW_DTYPES, "actions": "canvas-row-major",
            "value": "side-to-move", "model_outputs": ["logits[N,C*C]", "value[N]"]}
CONTRACT_ID = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()


def generate_header(path):
    from pathlib import Path
    names = ", ".join(json.dumps(x) for x in RULES)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        '#pragma once\nnamespace etazero {\n'
        f'inline constexpr int INPUT_PLANES = {len(PLANES)};\n'
        f'inline constexpr char CONTRACT_ID[] = "{CONTRACT_ID}";\n'
        'enum class Rule { FREESTYLE, STANDARD, RENJU };\n'
        f'inline constexpr const char* RULE_NAMES[] = {{{names}}};\n'
        '}\n')


if __name__ == "__main__":
    import sys
    generate_header(sys.argv[1])
