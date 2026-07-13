#!/usr/bin/env python3
"""Summarize Hebbian EA runs: fitness trajectory (plateau check) + reward-component composition.

Run on the host that has the logs (stdlib only). Helps spot local optima: a stalled best_so_far
plus a composition where regularizer terms dominate while pos/ori tracking stays low.
"""
import csv
import glob
import json
import os
import sys

TASK = sys.argv[1] if len(sys.argv) > 1 else "Isaac-flycrane-payload-decentralized-hovering-v0"
root = f"logs/hebbian/{TASK}"

for d in sorted(glob.glob(f"{root}/*/*seed*/")):
    sp = os.path.join(d, "settings.json")
    rp = os.path.join(d, "results.csv")
    if not (os.path.exists(sp) and os.path.exists(rp)):
        continue
    s = json.load(open(sp))
    opt = s["cli_args"]["optimizer"]
    pop = s["cli_args"]["popsize"]
    rows = list(csv.DictReader(open(rp)))
    if not rows:
        print(f"{opt} pop{pop}: no generations yet")
        continue
    g = len(rows)
    bsf = [float(r["best_so_far"]) for r in rows]
    mean = [float(r["mean"]) for r in rows]
    q = max(1, g // 4)
    gain_q = bsf[-1] - bsf[-q]                 # best-so-far gain over last quarter
    mean_gain_q = mean[-1] - mean[-q]
    label = f"{opt}(F{s['cli_args'].get('de_F')}/CR{s['cli_args'].get('de_CR')})" if opt == "revde" else opt
    isc = s["cli_args"].get("weight_init_scale")
    ew = s["cli_args"].get("evolve_init_weights")
    tag = f"i{isc}" + ("+evoW" if ew else "")
    if opt == "pgpe":
        tag += f" clr{s['cli_args'].get('pgpe_center_lr')} slr{s['cli_args'].get('pgpe_stdev_lr')}"
        tag += f" sig{s['cli_args'].get('sigma')} bnd{s['cli_args'].get('bound')}"
    flag = "  <-- PLATEAU" if (g >= 8 and gain_q < 0.3) else ""
    print(f"=== {label} pop{pop} {tag} | gens={g} | best={bsf[-1]:.2f} mean={mean[-1]:.2f} | "
          f"gain(best/mean,last 1/4)={gain_q:+.2f}/{mean_gain_q:+.2f}{flag} ===")
    # best-so-far milestones
    marks = [int(g * f) - 1 for f in (0.25, 0.5, 0.75, 1.0)]
    print("    best_so_far @ 25/50/75/100%: " + " ".join(f"{bsf[m]:.2f}" for m in marks))
    # reward-component composition (last gen)
    cf = os.path.join(d, "reward_components.csv")
    if os.path.exists(cf):
        cr = list(csv.DictReader(open(cf)))
        if cr:
            last = cr[-1]
            comps = {k: float(v) for k, v in last.items()
                     if k not in ("generation", "env_steps", "total_mean")}
            tot = sum(comps.values()) or 1.0
            ordered = sorted(comps.items(), key=lambda x: -x[1])
            print("    reward composition: " + "  ".join(
                f"{k}={v:.4f} ({100 * v / tot:.0f}%)" for k, v in ordered))
