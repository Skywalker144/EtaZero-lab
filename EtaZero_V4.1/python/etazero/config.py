"""Only the outer-round settings live here. Engine and learner own their options."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    selfplay_config: str
    canvas: int
    batch_size: int
    bootstrap_games: int
    train_steps: int
    native_selfplay: str | None = None
    model_kind: str | None = None
    shuffle_processes: int = 1
    shuffle_rows_per_file: int = 70000
    shuffle_keep_rows: int = 600000
    train_args: tuple[str, ...] = ()
    shuffle_args: tuple[str, ...] = ()
    game: str = 'go'
    algorithm: str = 'alphazero'
    muzero_network: dict | None = None
    muzero_unroll_steps: int | None = None
    publish_weights: str = 'ordinary'
    replay_ratio: float = 4.0
    min_rows: int = 100000
    bootstrap_game_multiplier: float = 1.05
    gatekeeper_config: str | None = None
    gatekeeper_required_win_prop: float = 0.5
    auto_plot: bool = True

    @property
    def samples_per_epoch(self):
        return self.train_steps * self.batch_size

    @property
    def is_muzero(self):
        return self.algorithm in {'muzero', 'gumbel_muzero'}

    @property
    def is_gumbel(self):
        return self.algorithm in {'gumbel_alphazero', 'gumbel_muzero'}

    def validate(self):
        if type(self.auto_plot) is not bool:
            raise ValueError('auto_plot must be boolean')
        if self.algorithm not in {'alphazero', 'gumbel_alphazero', 'muzero', 'gumbel_muzero'}:
            raise ValueError(f'Algorithm is not implemented: {self.algorithm}')
        if self.game not in {'go', 'freestyle', 'standard', 'renju', 'hex', 'xiangqi'}:
            raise ValueError(f'Game not implemented: {self.game}')
        if not Path(self.selfplay_config).is_file():
            raise ValueError(f'Missing selfplay config: {self.selfplay_config}')
        if self.gatekeeper_config is not None and not Path(self.gatekeeper_config).is_file():
            raise ValueError(f'Missing gatekeeper config: {self.gatekeeper_config}')
        if (type(self.gatekeeper_required_win_prop) not in (int, float) or
                not math.isfinite(self.gatekeeper_required_win_prop) or not 0 <= self.gatekeeper_required_win_prop <= 1):
            raise ValueError('gatekeeper_required_win_prop must be finite and in [0,1]')
        if self.gatekeeper_config is None and self.gatekeeper_required_win_prop != 0.5:
            raise ValueError('gatekeeper_required_win_prop requires gatekeeper_config')
        for key in ('canvas', 'batch_size', 'bootstrap_games', 'train_steps', 'min_rows',
                    'shuffle_processes', 'shuffle_rows_per_file', 'shuffle_keep_rows'):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if not 3 <= self.canvas <= 19:
            raise ValueError('The current source writer/learner supports canvas in [3,19]')
        if self.game == 'xiangqi' and self.canvas < 10:
            raise ValueError('Xiangqi requires canvas >= 10 for its 9x10 board')
        # A snapshot must be capable of supplying a complete batch.
        if self.shuffle_rows_per_file < self.batch_size or self.shuffle_keep_rows < self.batch_size:
            raise ValueError('Shuffle targets must provide at least one batch')
        if self.is_muzero:
            if self.model_kind is not None:
                raise ValueError('MuZero uses muzero_network, not model_kind')
            if not isinstance(self.muzero_network, dict):
                raise ValueError('MuZero requires muzero_network containing a source MuZeroConfig')
            from katago.train.muzero_model import MuZeroConfig
            try:
                network = MuZeroConfig(**self.muzero_network).validate()
            except (TypeError, KeyError) as error:
                raise ValueError(f'Invalid muzero_network: {error}') from error
            if network.game != self.game:
                raise ValueError('muzero_network game must match the outer game')
            if type(self.muzero_unroll_steps) is not int or self.muzero_unroll_steps < 1:
                raise ValueError('MuZero requires positive integer muzero_unroll_steps')
        else:
            if self.muzero_network is not None or self.muzero_unroll_steps is not None:
                raise ValueError('MuZero network/unroll settings require a MuZero algorithm')
            from katago.train.modelconfigs import config_of_name
            if self.model_kind not in config_of_name:
                raise ValueError(f'Unknown source model: {self.model_kind}')
        if self.publish_weights not in {'ordinary', 'swa'}:
            raise ValueError('publish_weights must be ordinary or swa')
        for key in ('replay_ratio', 'bootstrap_game_multiplier'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        # These settings determine stage ownership or the promised single-learner boundary.
        reserved = {
            'train_args': {'-traindir', '-datadir', '-latestdatadir', '-exportdir', '-exportprefix',
                           '-pos-len', '-batch-size', '-model-kind', '-input-features', '-max-epochs-this-instance',
                           '-max-training-samples', '-max-batches-this-instance', '-stop-at-samples', '-samples-per-epoch', '-multi-gpus',
                           '-no-export', '-gnorm-stats-debug', '-initial-checkpoint', '-initial-weights', '-always-initial-checkpoint',
                           '-initialize-only', '-muzero-config',
                           '-max-train-bucket-per-new-data', '-max-train-bucket-size', '-initial-train-bucket-level',
                           '-epochs-per-export', '-export-prob',
                           '-stop-when-train-bucket-limited', '-quit-if-no-data', '-no-repeat-files',
                           '-required-initial-checkpoint-train-steps'},
            'shuffle_args': {'-out-dir', '-out-tmp-dir', '-batch-size', '-num-processes',
                             '-approx-rows-per-out-file', '-keep-target-rows', '-dry-run-print-resource-cost',
                             '-muzero-unroll-steps', '-min-rows', '-input-files-stable'},
        }
        for key, forbidden in reserved.items():
            args = getattr(self, key)
            if not isinstance(args, (list, tuple)) or not all(isinstance(a, str) for a in args):
                raise ValueError(f'{key} must be a list of command-line strings')
            if any(a.startswith('-') and len(a.split('=', 1)[0]) > 1 and
                   any(flag.startswith(a.split('=', 1)[0]) for flag in forbidden) for a in args):
                raise ValueError(f'{key} overrides a setting owned by the outer controller')
        return self

    def serializable(self):
        return json.loads(json.dumps(asdict(self)))


def load_config(path):
    from .profiles import profile_path, load_run_profile
    path = profile_path(path, 'run.cfg')
    if path.suffix != '.json':
        return load_run_profile(path)[0]
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError('Run config must be a JSON object')
    unknown = set(values) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(f'Unknown outer settings: {sorted(unknown)}')
    for key in ('selfplay_config', 'gatekeeper_config'):
        if key in values and values[key] is not None:
            values[key] = str((path.parent / values[key]).resolve())
    return RunConfig(**values).validate()
