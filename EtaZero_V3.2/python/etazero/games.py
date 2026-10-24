"""Game layouts and the fixed Xiangqi action vocabulary shared with C++."""
from dataclasses import dataclass
import math


def validate_go_komi(komi):
    # KataGo Rules' user-input range and integer/half-integer scoring contract.
    if not math.isfinite(komi) or not -400 <= komi <= 400 or komi*2 != round(komi*2):
        raise ValueError('Go komi must be an integer or half-integer in [-400,400]')


def xiangqi_moves():
    moves = []
    for y in range(10):
        for x in range(9):
            for v in range(10):
                for u in range(9):
                    dx, dy = u-x, v-y
                    if (dx == 0) != (dy == 0) or sorted((abs(dx), abs(dy))) == [1, 2]:
                        moves.append((x, y, u, v))
    for flip in (False, True):
        for points, step in (([(3,0),(5,0),(4,1),(3,2),(5,2)], 1),
                             ([(2,0),(6,0),(0,2),(4,2),(8,2),(2,4),(6,4)], 2)):
            points = [(x, 9-y if flip else y) for x,y in points]
            moves.extend((x,y,u,v) for x,y in points for u,v in points
                         if abs(x-u) == step and abs(y-v) == step)
    return tuple(moves)


XIANGQI_MOVES = xiangqi_moves()
XIANGQI_INDEX = {move: i for i, move in enumerate(XIANGQI_MOVES)}
XIANGQI_DELTAS = (tuple((0,d) for d in range(-9,10) if d)
                  + tuple((d,0) for d in range(-8,9) if d)
                  + tuple((sx*d,sy*d) for d in (1,2) for sx in (-1,1) for sy in (-1,1))
                  + tuple((sx*a,sy*b) for a,b in ((1,2),(2,1)) for sx in (-1,1) for sy in (-1,1)))


@dataclass(frozen=True)
class GameSpec:
    name: str
    planes: int
    globals: int
    action_layout: str = 'points'
    fixed_actions: int = 0
    board_width: int = 0
    board_height: int = 0
    symmetry: str = 'd4'
    symmetries: int = 8
    subtree_value_bias: bool = True

    def actions(self, canvas):
        return self.fixed_actions or canvas*canvas + (self.action_layout == 'points_pass')

    def width(self, size):
        return self.board_width or size

    def height(self, size):
        return self.board_height or size

    def board_stride(self, size):
        # Public board snapshots use height-sized rows, including Xiangqi padding.
        return size

    def area(self, size):
        return self.width(size) * self.height(size)

    @property
    def layout(self):
        return self.planes, self.globals, self.action_layout, self.fixed_actions


SPECS = {'gomoku': GameSpec('gomoku', 5, 8),
         'hex': GameSpec('hex', 5, 8, symmetry='hex', symmetries=2),
         'go': GameSpec('go', 22, 19, action_layout='points_pass'),
         'xiangqi': GameSpec('xiangqi', 113, 8, action_layout='moves', fixed_actions=len(XIANGQI_MOVES),
                            board_width=9, board_height=10, symmetry='xiangqi', symmetries=2)}
RULE_GAMES = dict(freestyle='gomoku', standard='gomoku', renju='gomoku',
                  hex='hex', go='go', xiangqi='xiangqi')


def spec_for_rule(rule):
    return SPECS[rule if rule in SPECS else RULE_GAMES[rule]]


def spec_for_config(config):
    rules = [rule.strip() for rule in config['environment']['rules'].split(',') if rule.strip()]
    specs = [spec_for_rule(rule) for rule in rules]
    layouts = {s.layout for s in specs}
    if len(layouts) != 1:
        raise ValueError('One model requires one game input/action layout')
    return specs[0]


def policy_indices(canvas):
    delta = {d:i for i,d in enumerate(XIANGQI_DELTAS)}
    return [delta[u-x,v-y]*canvas*canvas + y*canvas+x for x,y,u,v in XIANGQI_MOVES]


def generate_cpp():
    deltas = ','.join('{'+','.join(map(str,d))+'}' for d in XIANGQI_DELTAS)
    moves = ',\n'.join('{'+','.join(map(str,m))+'}' for m in XIANGQI_MOVES)
    layouts = ',\n'.join(
        f'{{GameKind::{s.name.upper()}, "{s.name}", {s.planes}, {s.globals}, '
        f'ActionLayout::{s.action_layout.upper()}, {s.fixed_actions}, {s.board_width}, {s.board_height}, '
        f'SymmetryKind::{s.symmetry.upper()}, {s.symmetries}, {str(s.subtree_value_bias).lower()}}}'
        for s in SPECS.values())
    rules = ', '.join('GameKind::'+name.upper() for name in RULE_GAMES.values())
    return ('enum class GameKind { '+', '.join(name.upper() for name in SPECS)+' };\n'
        + '''enum class ActionLayout { POINTS, POINTS_PASS, MOVES };
enum class SymmetryKind { D4, HEX, XIANGQI };
struct GameLayout {
    GameKind kind;
    const char* name;
    int planes, globals;
    ActionLayout action_layout;
    int fixed_actions, board_width, board_height;
    SymmetryKind symmetry;
    int symmetries;
    bool subtree_value_bias;
};
''' + 'inline constexpr GameLayout GAME_LAYOUTS[] = {\n'+layouts+'\n};\n'
        + 'inline constexpr GameKind RULE_GAME_KINDS[] = {'+rules+'};\n'
        + f'inline constexpr int XIANGQI_DELTAS[{len(XIANGQI_DELTAS)}][2] = {{'+deltas+'};\n'
        + f'inline constexpr int XIANGQI_MOVES[{len(XIANGQI_MOVES)}][4] = {{\n'+moves+'\n};\n')


def layout_metadata(config):
    spec = spec_for_config(config)
    return dict(game=spec.name, input_planes=spec.planes, global_features=spec.globals,
                action_count=spec.actions(config['network']['canvas']))


def spec_for_metadata(metadata):
    return SPECS[metadata.get('game', 'gomoku')]


def static_action_mask(rule, size, canvas):
    import numpy as np
    spec = spec_for_rule(rule)
    if spec.action_layout == 'moves': return np.ones(spec.actions(canvas), np.int16)
    result = np.zeros(spec.actions(canvas), np.int16)
    result[:canvas*canvas].reshape(canvas,canvas)[:spec.height(size),:spec.width(size)] = 1
    if spec.action_layout == 'points_pass': result[-1] = 1
    return result
