"""Plots derive from raw journal values and committed checkpoint lineage."""
import json
import os
from pathlib import Path
from .storage import load_json


def plot_run(run_dir):
    root=Path(run_dir)
    os.environ.setdefault("MPLCONFIGDIR",str(root/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    state=load_json(root/"state.json")
    reference=state["checkpoint"]
    progress=root/"cycles"/f'{state["cycle"]:06d}'/"learner.json"
    if progress.exists():
        current=load_json(progress)["checkpoint"]
        if current["total_steps"]>reference["total_steps"]:
            reference=current
    committed=set()
    while reference:
        sidecar=load_json((root/reference["path"]).with_suffix(".json"))
        committed.update(sidecar["committed_updates"]);reference=sidecar["parent"]
    events=[]
    for line in (root/"events.jsonl").read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    updates=sorted((e for e in events if e["event"]=="update" and e["update_id"] in committed),key=lambda e:e["total_steps"])
    cycles={e["cycle"]:e for e in events if e["event"]=="cycle_complete"}
    cycles=[cycles[k] for k in sorted(cycles)]
    figure,axes=plt.subplots(1,3,figsize=(13,3.5))
    for key,label in (("loss","total"),("policy_loss","policy"),("value_loss","value")):
        axes[0].plot([e["total_steps"] for e in updates],[e[key] for e in updates],label=label)
    axes[0].set(xlabel="Committed updates",ylabel="Loss (raw, unsmoothed)");axes[0].legend()
    axes[1].plot([e["cycle"] for e in cycles],[e["unique_rows"] for e in cycles],label="unique selfplay rows")
    axes[1].plot([e["cycle"] for e in cycles],[e["total_samples"] for e in cycles],label="training samples consumed")
    axes[1].set(xlabel="Cycle",ylabel="Cumulative samples");axes[1].legend()
    for phase in ("selfplay","shuffle","train","export"):
        rows=[e for e in events if e["event"]=="phase_end" and e["phase"]==phase]
        axes[2].plot([e["cycle"] for e in rows],[e["seconds"] for e in rows],"o-",label=phase)
    axes[2].set(xlabel="Cycle",ylabel="Phase wall seconds (each attempt)");axes[2].legend()
    figure.tight_layout();destination=root/"plots"/"training.png";destination.parent.mkdir(exist_ok=True)
    figure.savefig(destination,dpi=160);plt.close(figure)
    return destination
