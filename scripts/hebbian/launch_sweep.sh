#!/usr/bin/env bash
# Initialize the PGPE W&B sweep and launch N agents (each runs --count configs sequentially).
# Usage (on remote): bash scripts/hebbian/launch_sweep.sh [n_agents] [count_per_agent]
set -u
cd /home/diggelen/AerialMan

N_AGENTS="${1:-4}"
COUNT="${2:-6}"
CFG="scripts/hebbian/sweeps/sweep_pgpe.yaml"

OUT=$(.venv/bin/wandb sweep --project hebbian-flycrane "$CFG" 2>&1)
echo "$OUT"
SID=$(echo "$OUT" | grep -oE "wandb agent [^ ]+" | tail -1 | awk '{print $3}')
if [ -z "$SID" ]; then echo "[sweep] FAILED to parse sweep id"; exit 1; fi
echo "[sweep] sweep id: $SID  (agents=$N_AGENTS count=$COUNT)"

for i in $(seq 1 "$N_AGENTS"); do
    setsid .venv/bin/wandb agent --count "$COUNT" "$SID" > "sweep_agent_${i}.log" 2>&1 < /dev/null &
    echo "[sweep] launched agent $i"
    sleep 30   # stagger Isaac Sim inits
done
echo "[sweep] $N_AGENTS agents launched for $SID"
