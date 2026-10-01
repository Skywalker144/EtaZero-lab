"""The single source of truth for the Python/C++ model and trajectory contract."""
import hashlib
import json

PLANES = ("on_board", "own", "opponent", "black_forbidden_black_to_move", "black_forbidden_white_to_move")
GLOBALS = ("standard", "renju", "renju_color", "forbidden_feature_enabled")
RULES = ("freestyle", "standard", "renju")
REASONS = ("draw", "five", "forbidden")
RAW_DTYPES = {
    "observations": "uint8", "globals": "float32", "players": "int8", "actions": "int32",
    "policies": "float32", "visits": "int64", "simulations": "int32",
    "temperatures": "float32", "rewards": "float32", "game_offsets": "int64",
    "observation_offsets": "int64", "game_ids": "uint64", "seeds": "uint64",
    "sizes": "int16", "rules": "int8", "winners": "int8", "reasons": "int8",
    "metadata": "uint8", "train_mask": "uint8", "row_repeats": "int32",
    "target_weights": "float32", "policy_surprises": "float32", "value_surprises": "float32",
    "network_wdl": "float32", "search_wdl": "float32", "cheap_search": "uint8", "opening_moves": "int16",
    "balanced_moves": "int16", "policy_moves": "int16", "opening_attempts": "int32",
    "opening_status": "int8", "start_values": "float32",
}
CONTRACT = {"planes": PLANES, "globals": GLOBALS,
            "global_layout": "float32[N,4]; standard,renju,Renju Black=-1 White=+1,forbidden enabled",
            "feature_dropout": "independent per training row, forbidden planes and enabled flag zero together", "rules": RULES, "reasons": REASONS,
            "observations": "uint8[N,planes,ceil(canvas*canvas/8)], MSB first, zero tail bits",
            "raw_dtypes": RAW_DTYPES, "rows": "sum(row_repeats); train_mask marks searched suffix; plies counts all actions",
            "opening": "leading train_mask=0 prefix, zero policy/visits/budget, full trajectory retained", "actions": "canvas-row-major",
            "value": "side-to-move W/D/L one-hot terminal target",
            "model_outputs": ["logits[N,C*C]", "wdl_logits[N,3]"],
            "sampling": "KataGo surprise weights stochastically rounded once per game; repeat rows in training view"}
CONTRACT_ID = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()


def pack_observations(observations):
    import numpy as np
    if observations.ndim != 4 or observations.shape[1] != len(PLANES) or not np.isin(observations, [0, 1]).all():
        raise ValueError("Expected binary NCHW observations")
    return np.packbits(observations.reshape(len(observations), len(PLANES), -1), axis=2, bitorder="big")


def unpack_observations(packed, canvas):
    import numpy as np
    if packed.dtype != np.uint8 or packed.shape[1:] != (len(PLANES), (canvas*canvas+7)//8):
        raise ValueError("Packed observation shape/dtype mismatch")
    return np.unpackbits(packed, axis=2, count=canvas*canvas, bitorder="big").reshape(-1, len(PLANES), canvas, canvas)


def generate_header(path):
    from pathlib import Path
    names = ", ".join(json.dumps(x) for x in RULES)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        '#pragma once\nnamespace etazero {\n'
        f'inline constexpr int INPUT_PLANES = {len(PLANES)};\n'
        f'inline constexpr int GLOBAL_FEATURES = {len(GLOBALS)};\n'
        f'inline constexpr char CONTRACT_ID[] = "{CONTRACT_ID}";\n'
        'enum class Rule { FREESTYLE, STANDARD, RENJU };\n'
        f'inline constexpr const char* RULE_NAMES[] = {{{names}}};\n'
        '}\n')


if __name__ == "__main__":
    import sys
    generate_header(sys.argv[1])
