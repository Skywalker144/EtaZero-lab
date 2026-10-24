"""Shared trajectory fields and legacy Gomoku/Hex features; game layouts live in games.py."""
import json
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from etazero.games import RULE_GAMES, SPECS
    from etazero.outcomes import REASONS, generate_cpp as outcome_cpp
except ImportError:
    from games import RULE_GAMES, SPECS
    from outcomes import REASONS, generate_cpp as outcome_cpp

PLANES = ("on_board", "own", "opponent", "black_forbidden_black_to_move", "black_forbidden_white_to_move")
GLOBALS = ("standard", "renju", "renju_color", "forbidden_feature_enabled", "pda_enabled", "pda_half_doublings", "hex", "hex_white_to_move")
HEX_GLOBAL = GLOBALS.index('hex')
HEX_WHITE_GLOBAL = GLOBALS.index('hex_white_to_move')
RULES = tuple(RULE_GAMES)
POLICY_HEADS = ("policy", "opponent_policy", "soft_policy", "soft_opponent_policy", "long_optimistic_policy", "short_optimistic_policy")
Q_POLICY_HEAD = "q_winloss_pretanh"
RAW_DTYPES = {
    "observations": "uint8", "globals": "float32", "players": "int8", "actions": "int32",
    "policies": "int16", "opponent_policies": "int16", "opponent_policy_weights": "float32",
    "visits": "int64", "simulations": "int64",
    "q_values": "int16", "q_visits": "int16",
    "temperatures": "float32", "rewards": "float32", "game_offsets": "int64",
    "observation_offsets": "int64", "game_ids": "uint64", "seeds": "uint64",
    "sizes": "int16", "rules": "int8", "winners": "int8", "reasons": "int8",
    "metadata": "uint8", "train_mask": "uint8", "row_repeats": "int32",
    "target_weights": "float32", "policy_surprises": "float32", "value_surprises": "float32",
    "network_wdl": "float32", "search_wdl": "float32", "cheap_search": "uint8", "opening_moves": "int16",
    "balanced_moves": "int16", "policy_moves": "int16", "opening_attempts": "int32",
    "initial_position_moves": "int16", "initial_position_kind": "int8", "hint_actions": "int32",
    "opening_status": "int8", "start_values": "float32",
    'training_weights': 'float32',
    "sample_indices": "int64", "forbidden_input": "uint8",
    "reanalyzed": "uint8", "reanalysis_used_outcome": "uint8", "reanalysis_original_visits": "int64",
    "reanalysis_policy_surprise": "float32", "reanalysis_value_surprise": "float32",
    "side_observations": "uint8", "side_globals": "float32", "side_players": "int8",
    "side_game_indices": "int32", "side_policies": "int16", "side_visits": "int64",
    "side_wdl": "float32", "side_row_repeats": "int32", "side_target_weights": "float32",
    "side_forbidden_input": "uint8",
    "side_q_values": "int16", "side_q_visits": "int16",
}
CONTRACT = {"planes": PLANES, "globals": GLOBALS,
            "global_layout": "float32[N,8]; standard,renju,Renju Black=-1 White=+1,forbidden enabled,PDA enabled,signed 0.5*log2(budget ratio),hex,Hex White to move",
            "feature_dropout": "full trajectory; forbidden_input uint8[sum(row_repeats)] sampled independently by writer; training view zeros both planes and enabled flag together", "rules": RULES, "reasons": REASONS,
            "observations": "uint8[N,planes,ceil(canvas*canvas/8)], MSB first, zero tail bits",
            "raw_dtypes": RAW_DTYPES, "side": "independent searched positions associated with parent game; value=all TD=own search WDL, full-game/error/optimistic/opponent gate zero",
            "rows": "metadata.row_begin:row_begin+rows selects final repeated rows in game order (main then side per game); full trajectories retained across shard boundaries; train_mask marks searched suffix; plies counts represented actions",
            "opening": "leading train_mask=0 prefix, zero budget, full packed trajectory retained; separate balanced/policy/initial prefix counts, initial kind ordinary/hint/earlyFork/gameFork/hintFork; hint actions canvas-indexed or -1", "actions": "Gomoku/Hex: canvas-row-major; Go: canvas-row-major + pass at canvas²; Xiangqi: games.XIANGQI_MOVES indices",
            "search_storage": "policy and visits stored once for sample_indices=flatnonzero(row_repeats); packed trajectory and lightweight WDL/weight diagnostics retain all steps",
            "value": "side-to-move W/D/L/no-result terminal; three finite-trajectory TD targets at actual board area",
            "policy_quantization": "unnormalized final selection weights; max at least 10, capped 30000; round to int16 before learner normalization",
            "model_outputs": ["logits[N,A]", "wdl_logits[N,4]", "short_optimistic_logits[N,A]", "shortterm_value_stdev[N]"],
            "training_policy_heads": POLICY_HEADS,
            "optional_training_policy_head": Q_POLICY_HEAD,
            "q_targets": "pure current-player W-L; float32 stochastic rounding times 32000 on each final repeated row; int16 visits=min(child NODE visits,32000), stored once per sampled main/side position; no Go score Q; reanalysis replaces both",
            "opponent_policy": "next actual turn search, independent of its row repeat; final turn uniform placeholder and weight zero",
            "reanalysis": "postgame cheap-only full searches replace policy/search/NN and target weights, preserving actual moves/outcome; reanalyzed rows without outcome targets suppress opponent/error/variance/default optimistic and future-position supervision via independent target weights; main value/TD and Q remain enabled; disabled optimistic branch retains fixed 0.5 weights",
            "pda": "nonzero doubling advantage activates flag; signed at each side-to-move, retained by main rows; side rows zero both PDA globals",
            "sampling": "KataGo surprise weights stochastically rounded once per game; repeat rows in training view"}


def pack_observations(observations):
    import numpy as np
    if observations.ndim != 4 or observations.shape[1] not in {spec.planes for spec in SPECS.values()} or not np.isin(observations, [0, 1]).all():
        raise ValueError("Expected binary NCHW observations")
    return np.packbits(observations.reshape(len(observations), observations.shape[1], -1), axis=2, bitorder="big")


def unpack_observations(packed, canvas):
    import numpy as np
    if packed.dtype != np.uint8 or packed.ndim != 3 or packed.shape[1] not in {spec.planes for spec in SPECS.values()} or packed.shape[2] != (canvas*canvas+7)//8:
        raise ValueError("Packed observation shape/dtype mismatch")
    return np.unpackbits(packed, axis=2, count=canvas*canvas, bitorder="big").reshape(-1, packed.shape[1], canvas, canvas)


def generate_header(path):
    from pathlib import Path
    try:
        from etazero.games import generate_cpp
    except ImportError:
        from games import generate_cpp
    names = ", ".join(json.dumps(x) for x in RULES)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        '#pragma once\nnamespace etazero {\n'
        f'inline constexpr int INPUT_PLANES = {len(PLANES)};\n'
        f'inline constexpr int GLOBAL_FEATURES = {len(GLOBALS)};\n'
        f'inline constexpr int HEX_GLOBAL = {HEX_GLOBAL};\n'
        f'inline constexpr int HEX_WHITE_GLOBAL = {HEX_WHITE_GLOBAL};\n'
        'enum class Rule { '+', '.join(rule.upper() for rule in RULES)+' };\n'
        f'inline constexpr const char* RULE_NAMES[] = {{{names}}};\n'
        + generate_cpp() + outcome_cpp() + '}\n')


if __name__ == "__main__":
    import sys
    generate_header(sys.argv[1])
