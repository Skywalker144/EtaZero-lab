"""Compare AlphaZero networks through the production LibTorch CUDA Graph backend.

Latency includes input packing, H2D, graph replay, blocking D2H and result
construction. Loading, graph capture, warmup, search, queues and caches are
excluded. Models use seeded initial weights; no training run is modified.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from etazero.config import load_config
from etazero.katago_configs import resolve
from etazero.network import make_network, inference_network
from etazero.runtime import provenance, source_snapshot
from etazero.storage import save_json
from etazero.games import SPECS, spec_for_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architectures', nargs='+', default=[])
    parser.add_argument('--config-dir', type=Path,
                        help='Include the AlphaZero network from an existing configuration directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--game', choices=tuple(SPECS), help='Game for named presets; otherwise use the config game or Gomoku')
    parser.add_argument('--canvas', type=int, default=15)
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=1.0,
                        help='Minimum measured seconds per model/batch/repeat')
    parser.add_argument('--warmup', type=int, default=30,
                        help='Minimum warmup calls, also lasting at least 0.2 seconds')
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    if (min(args.batches + [args.repeats, args.warmup]) < 1 or
            not math.isfinite(args.seconds) or args.seconds <= 0 or not 5 <= args.canvas <= 25):
        parser.error('Use positive batches/repeats/warmup/seconds and canvas 5..25')
    if not args.device.startswith('cuda:'):
        parser.error('An explicit CUDA device is required')
    if len(set(args.architectures)) != len(args.architectures) or len(set(args.batches)) != len(args.batches):
        parser.error('Architectures and batches must be unique')
    specs = []
    configured = load_config(args.config_dir.resolve()) if args.config_dir is not None else None
    if configured is not None:
        if configured['agent']['algorithm'] != 'alphazero':
            parser.error('--config-dir requires an AlphaZero configuration')
        network = configured['network']
        if network['canvas'] != args.canvas:
            parser.error('--canvas must match the configured network canvas')
        label = f"configured:{network['architecture']}:c{network['channels']}:b{network['blocks']}"
        specs.append((label, configured))
    game = args.game or (spec_for_config(configured).name if configured else 'gomoku')
    if configured and game != spec_for_config(configured).name:
        parser.error('--game must match --config-dir')
    if SPECS[game].board_height and args.canvas != SPECS[game].board_height:
        parser.error('Xiangqi requires canvas=10')
    for architecture in args.architectures:
        resolve(architecture)
        specs.append((architecture, dict(agent=dict(algorithm='alphazero'),
                                        environment=dict(rules='freestyle' if game=='gomoku' else game),
                                        network=dict(architecture=architecture, canvas=args.canvas))))
    if not specs:
        parser.error('Choose --architectures and/or --config-dir')
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run in a host environment with GPU access')
    torch.cuda.get_device_properties(device)
    binary = ROOT / 'build/inference_benchmark'
    if not binary.is_file():
        parser.error('Build first: cmake --build build --target inference_benchmark')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        timing_scope=__doc__, source_id=source_snapshot(output),
        binary=str(binary.resolve()), environment=provenance(), models=[],
        cpu=subprocess.check_output(['lscpu'], text=True),
        nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
        cmake_cache=(ROOT / 'build/CMakeCache.txt').read_text(),
        input_config=configured,
    )
    save_json(output / 'manifest.json', manifest)
    for index, (architecture, config) in enumerate(specs):
        torch.manual_seed(args.seed)
        model = make_network(config).eval()
        path = output / f'model_{index}.pt'
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', torch.jit.TracerWarning)
            torch.jit.script(inference_network(model)).save(str(path))
        manifest['models'].append(dict(
            architecture=architecture, parameters=sum(p.numel() for p in model.parameters()),
            config=getattr(model, 'source_config', config['network']),
            file=path.name,
            export_seconds=time.perf_counter()-started,
        ))
        save_json(output / 'manifest.json', manifest)
        print(f'Exported {architecture}: {manifest["models"][-1]["parameters"]:,} parameters', flush=True)
    command = [str(binary), args.device, args.precision, str(args.canvas),
               ','.join(map(str, args.batches)), str(args.repeats), str(args.seconds), str(args.warmup)]
    command += [str(output / model['file']) for model in manifest['models']]
    manifest['command'] = command
    save_json(output / 'manifest.json', manifest)
    records = []
    with (output / 'raw.jsonl').open('w') as raw, (output / 'stderr.log').open('w') as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True) as process:
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                record = json.loads(line)
                records.append(record)
                print(line.strip(), flush=True)
            if process.wait():
                raise RuntimeError(f'Native benchmark failed; see {output / "stderr.log"}')
    rows = []
    for index, (architecture, _) in enumerate(specs):
        for batch in args.batches:
            samples = [r for r in records if r['event'] == 'measure' and
                       r['model'] == index and r['batch'] == batch]
            if len(samples) != args.repeats:
                raise RuntimeError(f'Incomplete measurements: {architecture}, batch {batch}')
            rates = [r['positions_per_second'] for r in samples]
            rows.append(dict(
                architecture=architecture, parameters=manifest['models'][index]['parameters'],
                batch=batch, graph_batch=samples[0]['graph_batch'],
                median_batch_ms=statistics.median(r['batch_ms'] for r in samples),
                median_positions_per_second=statistics.median(rates),
                min_positions_per_second=min(rates), max_positions_per_second=max(rates),
            ))
    save_json(output / 'summary.json', rows, immutable=True)
    with (output / 'summary.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results: {output / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
