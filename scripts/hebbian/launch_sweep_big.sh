#!/usr/bin/env bash
# Initialize the EXPANDED Hebbian W&B sweep (sweep_big.yaml) and launch N agents.
# Each agent runs --count configs sequentially; keep concurrency low so each config gets
# real compute (popsize up to 1024 -> 8192 envs ~= 6.5 GB/proc; 2 agents ~= 13 GB on the 4090).
# Usage (on remote): bash scripts/hebbian/launch_sweep_big.sh [n_agents] [count_per_agent]
set -u
cd /home/diggelen/AerialMan

N_AGENTS="${1:-2}"          # low concurrency: each config trains up to 8 h (see sweep_big.yaml)
COUNT="${2:-8}"
CFG="scripts/hebbian/sweeps/sweep_big.yaml"

OUT=$(.venv/bin/wandb sweep --project hebbian-flycrane "$CFG" 2>&1)
echo "$OUT"
SID=$(echo "$OUT" | grep -oE "wandb agent [^ ]+" | tail -1 | awk '{print $3}')
if [ -z "$SID" ]; then echo "[sweep] FAILED to parse sweep id"; exit 1; fi
echo "[sweep] sweep id: $SID  (agents=$N_AGENTS count=$COUNT)"

for i in $(seq 1 "$N_AGENTS"); do
    setsid .venv/bin/wandb agent --count "$COUNT" "$SID" > "sweep_big_agent_${i}.log" 2>&1 < /dev/null &
    echo "[sweep] launched agent $i"
    sleep 40   # stagger Isaac Sim inits
done
echo "[sweep] $N_AGENTS agents launched for $SID"
