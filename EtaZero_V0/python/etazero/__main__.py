import argparse
import json
import os
from pathlib import Path
import sys
from .config import ROOT, load_config, fingerprint


def main():
    if len(sys.argv)>1 and sys.argv[1]=='arena':
        from .arena import main as arena_main
        sys.argv.pop(1)
        return arena_main()
    parser=argparse.ArgumentParser(description="EtaZero V0 training and evaluation")
    subparsers=parser.add_subparsers(dest="command",required=True)
    subparsers.add_parser("arena",help="Resumable historical-model matches and Elo")
    for name in ("check-config","run","evaluate","match"):
        sub=subparsers.add_parser(name)
        sub.add_argument("--config-dir",default=os.environ.get("CONFIG_DIR") or str(ROOT/"configs"/"baseline"))
        if name!="check-config":
            sub.add_argument("--run-dir")
            sub.add_argument("--binary",default=str(ROOT/"build"/"etazero"))
        if name=="run":
            sub.add_argument("--resume",action="store_true",default=None,
                             help="Require an existing run; otherwise resume is detected automatically")
            sub.add_argument("--weights",type=Path)
            sub.add_argument("--iterations",type=int,
                             help="Total completed-iteration target; 0 means unlimited")
            sub.add_argument("--max-seconds",type=float, help="Cumulative committed-iteration wall-time budget; 0 means unlimited")
            sub.add_argument("--plot",action="store_true", help="Also rebuild figures after the run returns")
        if name in ("evaluate","match"):
            sub.add_argument("--model")
            sub.add_argument("--size",type=int)
            sub.add_argument("--rule",choices=("freestyle","standard","renju"))
            sub.add_argument("--moves",default="",help="Comma-separated board-row-major action indices (evaluate only)")
            sub.add_argument("--output")
        if name=="match":
            sub.add_argument("--model-b",required=True)
            sub.add_argument("--games",type=int)
    plot=subparsers.add_parser("plot");plot.add_argument("--run-dir",required=True)
    args=parser.parse_args()
    if args.command=="plot":
        from .plotting import plot_run
        print(plot_run(Path(args.run_dir).resolve()));return
    directory=Path(args.config_dir)
    if not directory.exists() and not directory.is_absolute():
        directory=ROOT/directory
    if args.command in ('evaluate','match'):
        from .eval_config import load_evaluation_config
        config=load_evaluation_config(directory, match=args.command=='match')
    else:
        config=load_config(directory)
    if args.command=="check-config":
        print(json.dumps({"id":fingerprint(config),"config":config},indent=2));return
    if args.command in ('evaluate','match') and not args.run_dir and not args.model:
        parser.error('Evaluation requires --model or --run-dir to identify the model')
    root=Path(args.run_dir).resolve() if args.run_dir else ROOT/(config["run"]["run_dir"] if "run" in config else "data")
    binary=Path(args.binary).resolve()
    if args.command=="run":
        if args.iterations is not None and args.iterations<0:
            parser.error("--iterations must be nonnegative; 0 means unlimited")
        import math
        if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds < 0):
            parser.error("--max-seconds must be nonnegative and finite")
        from .runtime import run_training
        run_training(root,config,binary,args.resume,args.weights,args.iterations,args.max_seconds)
        if args.plot:
            from .plotting import plot_run
            print(plot_run(root))
    else:
        if args.command=="match":
            from .process import install_signals
            install_signals()
        from .evaluation import evaluate
        path,result=evaluate(config,binary,root,args.model,getattr(args,"model_b",None),args.size,args.rule,
                             args.moves,args.output,getattr(args,"games",None))
        print(json.dumps(result["result"],indent=2));print(path)


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"EtaZero: {error}",file=sys.stderr)
        raise
