import argparse
import json
from pathlib import Path
import sys
from .config import ROOT, load_config, fingerprint


def main():
    parser=argparse.ArgumentParser(description="EtaZero V0 training and evaluation")
    subparsers=parser.add_subparsers(dest="command",required=True)
    for name in ("check-config","run","evaluate","match"):
        sub=subparsers.add_parser(name)
        sub.add_argument("--config-dir",default=str(ROOT/"configs"/"baseline"))
        if name!="check-config":
            sub.add_argument("--run-dir")
            sub.add_argument("--binary",default=str(ROOT/"build"/"etazero"))
        if name=="run":
            sub.add_argument("--resume",action="store_true")
            sub.add_argument("--weights",type=Path)
            sub.add_argument("--cycles",type=int)
            sub.add_argument("--plot",action="store_true")
        if name in ("evaluate","match"):
            sub.add_argument("--model")
            sub.add_argument("--size",type=int)
            sub.add_argument("--rule",choices=("freestyle","standard","renju"))
            sub.add_argument("--moves",default="",help="Comma-separated canvas-row-major action indices")
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
    config=load_config(directory)
    if args.command=="check-config":
        print(json.dumps({"id":fingerprint(config),"config":config},indent=2));return
    root=Path(args.run_dir).resolve() if args.run_dir else ROOT/config["run"]["run_dir"]
    binary=Path(args.binary).resolve()
    if args.command=="run":
        if args.cycles is not None and args.cycles<1:
            parser.error("--cycles must be a positive total completed-cycle target")
        from .runtime import run_training
        run_training(root,config,binary,args.resume,args.weights,args.cycles)
        if args.plot:
            from .plotting import plot_run
            print(plot_run(root))
    else:
        from .evaluation import evaluate
        path,result=evaluate(config,binary,root,args.model,getattr(args,"model_b",None),args.size,args.rule,
                             args.moves,args.output,getattr(args,"games",None))
        print(json.dumps(result["result"],indent=2));print(path)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        print(f"EtaZero: {error}",file=sys.stderr)
        raise
