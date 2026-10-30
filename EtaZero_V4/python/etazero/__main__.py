"""Outer EtaZero entry point; native source protocols are forwarded explicitly."""
import argparse
import json
import os
import signal
import sys

from . import ROOT, __version__
from .config import load_config
from .profiles import default_output, load_run_profile, profile_path


def main():
    # Do not reinterpret source CLI arguments or its public GTP/JSON protocols.
    if len(sys.argv) > 1 and sys.argv[1] == 'native':
        binary = ROOT / 'build/katago'
        os.execv(str(binary), [str(binary), *sys.argv[2:]])
    parser = argparse.ArgumentParser(description='EtaZero V4 source-based integration')
    parser.add_argument('--version', action='version', version=f'EtaZero {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('native', help='Forward remaining arguments to the native source engine')
    check = commands.add_parser('check-config')
    check.add_argument('--config', '--config-dir', default=os.environ.get('CONFIG_DIR'),
                       help='Run profile directory, run.cfg, or resolved JSON')
    launch = commands.add_parser('run', help='Run synchronous training rounds for the configured game')
    launch.add_argument('--config', '--config-dir', default=os.environ.get('CONFIG_DIR'))
    launch.add_argument('--run-dir', help='Defaults to data/<profile path relative to configs/>')
    launch.add_argument('--weights', help='New run only: source checkpoint supplying initial model weights, without training state')
    launch.add_argument('--binary')
    launch.add_argument('--lease-fd', type=int, action='append', default=[], help=argparse.SUPPRESS)
    launch.add_argument('--iterations', type=int, help='Cumulative completed-round limit')
    launch.add_argument('--max-hours', type=float, help='Cumulative time budget, checked at completed-round boundaries')
    launch.add_argument('--dry-run', action='store_true', help='Print resolved configuration and launch settings without training')
    experiment = commands.add_parser('experiment', help='Queue independent training arms in GPU slots')
    experiment.add_argument('--config', '--config-dir', default=os.environ.get('CONFIG_DIR'),
                            help='Experiment directory with exp.cfg, or experiment JSON')
    experiment.add_argument('--work-dir', help='Defaults to data/<profile path relative to configs/>')
    experiment.add_argument('--binary')
    experiment.add_argument('--dry-run', action='store_true')
    plot = commands.add_parser('plot', help='Rebuild figures from committed V4 rounds')
    plot.add_argument('--run-dir', required=True)
    plot.add_argument('--output-dir', help='Optional separate destination for rebuilt figures')
    match = commands.add_parser('match', help='Run/resume source matches and fit Elo from completed games')
    match.add_argument('--config', help='Native match config; relative model paths use its directory')
    match.add_argument('--output-dir', required=True)
    match.add_argument('--binary')
    match.add_argument('--fit-only', action='store_true', help='Fit saved games without loading models or using a GPU')
    match.add_argument('--prior-games', type=float, default=2.)
    match.add_argument('--first-player-advantage', action='store_true')
    match.add_argument('--anchor', help='Participating bot name to place at Elo zero')
    match.add_argument('--max-loaded-models', type=int, help='Resident model service limit (at least 2)')
    arena = commands.add_parser('arena', help='Evaluate committed historical models by actual training time')
    arena.add_argument('--data', help='Training run or parent directory containing runs')
    arena.add_argument('--config', help='Independent native match profile')
    arena.add_argument('--output-dir', required=True)
    arena.add_argument('--binary')
    arena.add_argument('--dry-run', action='store_true')
    arena.add_argument('--fit-only', action='store_true')
    arena.add_argument('--prior-games', type=float, default=2.)
    arena.add_argument('--first-player-advantage', action='store_true')
    arena.add_argument('--max-loaded-models', type=int, help='Resident model service limit (at least 2)')
    # Suppressed defaults let fit-only distinguish supplied selection settings.
    for flag in ('stride', 'neighbors', 'games-per-pair'):
        arena.add_argument('--' + flag, type=int, default=argparse.SUPPRESS)
    for flag in ('sample-seconds', 'cross-seconds'):
        arena.add_argument('--' + flag, type=float, default=argparse.SUPPRESS)
    arena.add_argument('--anchor', default=argparse.SUPPRESS)
    arena.add_argument('--final-cross', action='store_true', default=argparse.SUPPRESS)
    arena.add_argument('--exclude-first', dest='include_first', action='store_false', default=argparse.SUPPRESS)
    autoelo = commands.add_parser('autoelo', help='Incremental historical Elo with reusable native pair games')
    autoelo.add_argument('--data')
    autoelo.add_argument('--config', help='Independent native match profile, indexed within each two-bot pair')
    autoelo.add_argument('--output-dir', required=True)
    autoelo.add_argument('--binary')
    autoelo.add_argument('--max-loaded-models', type=int, help='Maximum resident inference services (default: 2)')
    autoelo.add_argument('--dry-run', action='store_true')
    autoelo.add_argument('--fit-only', action='store_true', help='Refit a saved session without original inputs or GPU')
    autoelo.add_argument('--prior-games', type=float, default=2.)
    autoelo.add_argument('--first-player-advantage', action='store_true')
    for flag in ('stride', 'neighbors', 'games-per-pair'):
        autoelo.add_argument('--' + flag, type=int, default=argparse.SUPPRESS)
    for flag in ('sample-seconds', 'cross-seconds'):
        autoelo.add_argument('--' + flag, type=float, default=argparse.SUPPRESS)
    autoelo.add_argument('--anchor', default=argparse.SUPPRESS)
    autoelo.add_argument('--final-cross', action='store_true', default=argparse.SUPPRESS)
    autoelo.add_argument('--exclude-first', dest='include_first', action='store_false', default=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command in ('run', 'experiment', 'check-config') and not args.config:
        parser.error('Set CONFIG_DIR or --config-dir to a configuration directory (or use --config)')
    if args.command in ('match', 'arena', 'autoelo'):
        # A targeted SIGTERM must unwind the native child wait as SIGINT does,
        # so completed games survive and the child is stopped before releasing locks.
        signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        if args.command == 'experiment':
            from .experiment import experiment_plan, run_experiment
            plan = experiment_plan(args.config, args.work_dir)
            if args.dry_run:
                print(json.dumps(plan, indent=2, ensure_ascii=False))
            else:
                code = run_experiment(plan, binary=args.binary)
                if code:
                    sys.exit(code)
            return
        if args.command in ('arena', 'autoelo'):
            from .arena import run_arena
            from .autoelo import run_autoelo
            options = vars(args).copy()
            del options['command']
            output = options.pop('output_dir')
            result = (run_arena if args.command == 'arena' else run_autoelo)(output, **options)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.dry_run else
                  f'Fitted {result["games"]} games; anchor: {result["anchor"]}')
            return
        if args.command == 'match':
            from .evaluation import run_match
            result = run_match(args.output_dir, args.config, binary=args.binary, fit_only=args.fit_only,
                prior_games=args.prior_games, first_player_advantage=args.first_player_advantage, anchor=args.anchor,
                max_loaded_models=args.max_loaded_models)
            print(f'Fitted {result["games"]} games; anchor: {result["anchor"]}')
            return
        if args.command == 'plot':
            from .plotting import plot_run
            for path in plot_run(args.run_dir, args.output_dir):
                print(path)
            return
        path = profile_path(args.config, 'run.cfg')
        config, defaults = ((load_config(path), {}) if path.suffix == '.json' else load_run_profile(path))
        if args.command == 'check-config':
            print(json.dumps(config.serializable(), indent=2, ensure_ascii=False))
        else:
            from .run import run
            from .run import validate_budget
            options = {key: getattr(args, key) if getattr(args, key) is not None else defaults.get(key)
                       for key in ('run_dir', 'iterations', 'max_hours', 'weights')}
            if options['run_dir'] is None:
                options['run_dir'] = str(default_output(path, 'run.cfg'))
            seconds = None if options['max_hours'] is None else options['max_hours'] * 3600
            validate_budget(options['iterations'], seconds)
            if args.dry_run:
                print(json.dumps(dict(config=config.serializable(), **options), indent=2, ensure_ascii=False))
                return
            run(options['run_dir'], config, options['iterations'], binary=args.binary,
                max_seconds=seconds, weights=options['weights'], inherited_fds=tuple(args.lease_fd))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
