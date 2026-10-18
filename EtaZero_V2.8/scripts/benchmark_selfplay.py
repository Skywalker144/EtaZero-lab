"""Measure complete selfplay requests with a fixed model and production writer.

Each candidate gets its own persistent worker, an unmeasured warmup, and the
same measured game count/seeds. Outputs never enter an existing training run.
"""
import argparse
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from etazero.config import fingerprint, load_config, validate, write_native
from etazero.native import NativeWorker
from etazero.runtime import provenance, source_snapshot, verify_build
from etazero.storage import sha256


def candidate(text):
    values = tuple(map(int, text.split(':')))
    if len(values) != 5 or any(x < 1 for x in values[:4]) or values[4] < 0:
        raise argparse.ArgumentTypeError('Use game_threads:batch:servers:search_threads:wait_us')
    return values


def gpu_sample(device):
    result = subprocess.run([
        'nvidia-smi', '-i', device.split(':')[1],
        '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits',
    ], capture_output=True, text=True, check=True)
    utilization, memory, power = map(float, result.stdout.strip().split(','))
    return dict(gpu_percent=utilization, gpu_memory_mib=memory, gpu_power_w=power)


def process_sample(pid):
    # /proc stat fields 14/15 are CPU ticks; statm field 2 is resident pages.
    import os
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    ticks = int(fields[11]) + int(fields[12])
    pages = int(Path(f'/proc/{pid}/statm').read_text().split()[1])
    return ticks / os.sysconf('SC_CLK_TCK'), pages * os.sysconf('SC_PAGE_SIZE') / 2**20


def request(worker, directory, model, model_id, games, seed, device, timeout, evaluator):
    directory.mkdir()
    cpu_start, _ = process_sample(worker.process.pid)
    started = time.perf_counter()
    worker.submit([model, model_id, games, directory / 'selfplay', directory.name,
                   2, seed, evaluator], directory / 'events.jsonl')
    samples = []
    next_sample = started + 1
    while True:
        result = worker.take()
        if result is not None:
            if result['event'] != 'worker_complete' or result['code'] != 0:
                raise RuntimeError(f'Selfplay failed: {result}')
            break
        now = time.perf_counter()
        if now - started > timeout:
            worker.process.send_signal(signal.SIGINT)
            raise TimeoutError(f'Selfplay request exceeded {timeout}s: {directory}')
        if now >= next_sample:
            sample = gpu_sample(device)
            _, sample['rss_mib'] = process_sample(worker.process.pid)
            sample['seconds'] = now - started
            samples.append(sample)
            next_sample = now + 1
        time.sleep(.005)
    seconds = time.perf_counter() - started
    cpu_end, _ = process_sample(worker.process.pid)
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    inference = next(e for e in events if e['event'] == 'inference')
    complete = next(e for e in events if e['event'] == 'selfplay_complete')
    if complete['games'] != games:
        raise RuntimeError('Incomplete measured game quota')
    rows = sum(e['rows'] for e in events if e['event'] == 'shard')
    # Shards may repeat a game's trajectory. Count each game exactly once.
    import numpy as np
    seen = set()
    plies = simulations = 0
    for path in (directory / 'selfplay').glob('*.npz'):
        with np.load(path, allow_pickle=False) as data:
            offsets = data['game_offsets']
            for i, identity in enumerate(data['game_ids']):
                identity = int(identity)
                if identity in seen:
                    continue
                seen.add(identity)
                lo, hi = map(int, offsets[i:i+2])
                plies += hi - lo
                simulations += int(data['simulations'][lo:hi].sum())
    if len(seen) != games:
        raise RuntimeError('Serialized games disagree with completed quota')
    output = dict(seconds=seconds, games=games, rows=rows, plies=plies,
                  simulations=simulations, games_per_second=games/seconds,
                  rows_per_second=rows/seconds, simulations_per_second=simulations/seconds,
                  cpu_cores_used=(cpu_end-cpu_start)/seconds,
                  mean_batch=inference['requests']/max(1, inference['batches']),
                  mean_queue_us=inference['queue_wait_us']/max(1, inference['requests']),
                  cache_hit_rate=inference['cache_hits']/max(1, inference['submitted']),
                  inference=inference, resource_samples=samples)
    for key in ('gpu_percent', 'gpu_power_w'):
        output['mean_' + key] = sum(x[key] for x in samples)/len(samples) if samples else None
    for key in ('gpu_memory_mib', 'rss_mib'):
        output['peak_' + key] = max((x[key] for x in samples), default=None)
    (directory / 'metrics.json').write_text(json.dumps(output, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--evaluator', choices=('network', 'random'), default='network')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--games', type=int, default=512)
    parser.add_argument('--warmup-games', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--candidates', type=candidate, nargs='+', required=True,
                        help='Each value is game_threads:batch:servers:search_threads:wait_us')
    args = parser.parse_args()
    if min(args.games, args.warmup_games, args.repeats) < 1 or args.timeout <= 0:
        parser.error('Game counts, repeats and timeout must be positive')
    if not args.device.startswith('cuda:'):
        parser.error('This performance benchmark requires an explicit CUDA device')
    if args.evaluator == 'network' and args.model is None:
        parser.error('--model is required for network inference')
    config = load_config(args.config_dir.resolve())
    binary = ROOT / 'build/etazero'
    verify_build(binary)
    model = args.model.resolve() if args.model is not None else ''
    model_id = sha256(model) if args.evaluator == 'network' else f'random:{args.seed}'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_id = source_snapshot(output)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    model=str(model), model_sha256=model_id if args.evaluator == 'network' else None,
                    source_id=source_id,
                    binary_sha256=sha256(binary), environment=provenance(), base_config=config,
                    cpu=subprocess.check_output(['lscpu'], text=True),
                    memory=Path('/proc/meminfo').read_text())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    for index, values in enumerate(args.candidates):
        game_threads, batch, servers, search, wait = values
        current = copy.deepcopy(config)
        current['parallelism'].update(game_threads=game_threads, search_threads=search)
        current['inference'].update(max_batch=batch, server_threads=servers, batch_wait_us=wait)
        validate(current)
        directory = output / f'{index:02d}_g{game_threads}_b{batch}_s{servers}_t{search}_w{wait}'
        directory.mkdir()
        (directory / 'effective.json').write_text(json.dumps(current, indent=2) + '\n')
        write_native(current, directory / 'effective.cfg')
        command = [str(binary), 'worker', '--config', str(directory / 'effective.cfg'),
                   '--device', args.device, '--run-id', output.name, '--worker', '0',
                   '--config-id', fingerprint(current), '--source-id', source_id]
        worker = NativeWorker(command, directory / 'stderr.log')
        try:
            request(worker, directory / 'warmup', model, model_id,
                    args.warmup_games, args.seed - 1, args.device, args.timeout, args.evaluator)
            for repeat in range(args.repeats):
                metrics = request(worker, directory / f'measured_{repeat}', model, model_id,
                                  args.games, args.seed + repeat, args.device, args.timeout, args.evaluator)
                result = dict(candidate=values, repeat=repeat, metrics=metrics)
                results.append(result)
                (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(dict(candidate=values, repeat=repeat,
                                      **{k: v for k, v in metrics.items()
                                         if k not in ('inference', 'resource_samples')})), flush=True)
        finally:
            worker.close()


if __name__ == '__main__':
    main()
