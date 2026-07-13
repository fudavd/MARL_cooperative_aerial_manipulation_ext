#!/usr/bin/env python3
"""Show how the decomposed reward distribution shifts over generations within each run.

Reads reward_components.csv; prints each term's share of total reward at milestone generations,
plus the task (pos+ori) vs regularizer (smoothness+body_rate+force+downwash) split, to reveal
whether training shifts reward toward actual tracking or stays regularizer-dominated.
"""
import csv
import glob
import json
import os
import sys

TASK = sys.argv[1] if len(sys.argv) > 1 else "Isaac-flycrane-payload-decentralized-hovering-v0"
TASK_TERMS = {"pos_reward", "ori_reward"}

for d in sorted(glob.glob(f"logs/hebbian/{TASK}/*/*seed*/")):
    cf = os.path.join(d, "reward_components.csv")
    sp = os.path.join(d, "settings.json")
    if not os.path.exists(cf):
        continue
    rows = list(csv.DictReader(open(cf)))
    if len(rows) < 4:
        continue
    s = json.load(open(sp)) if os.path.exists(sp) else {"cli_args": {}}
    a = s.get("cli_args", {})
    opt = a.get("optimizer", "?")
    lab = f"{opt} p{a.get('popsize')} clr{a.get('pgpe_center_lr')}" if opt == "pgpe" else f"{opt} p{a.get('popsize')}"
    lab += f" i{a.get('weight_init_scale')}{'+ew' if a.get('evolve_init_weights') else ''}"
    keys = [k for k in rows[0] if k not in ("generation", "env_steps", "total_mean")]
    g = len(rows)
    print(f"\n=== {lab} | gens={g} ===")
    print(f"  {'gen':>5} {'total':>7} {'task%':>6} {'reg%':>6}  " + "  ".join(f"{k.split('_')[0]:>6}" for k in keys))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = rows[min(g - 1, int(frac * (g - 1)))]
        comps = {k: float(r[k]) for k in keys}
        tot = sum(comps.values()) or 1e-9
        task = sum(v for k, v in comps.items() if k in TASK_TERMS)
        shares = "  ".join(f"{100 * comps[k] / tot:5.1f}" for k in keys)
        print(f"  {r['generation']:>5} {float(r['total_mean']):>7.1f} {100*task/tot:>5.1f} {100*(1-task/tot):>5.1f}  {shares}")
