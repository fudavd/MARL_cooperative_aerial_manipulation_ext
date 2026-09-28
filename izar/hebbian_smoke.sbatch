#!/bin/bash
#SBATCH -A micro-515
#SBATCH --qos=debug
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J hebb_smoke
#SBATCH -o /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext/izar/hebbian_smoke.out
set -uo pipefail
source /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext/izar/env.sh
hostname
for task in Isaac-flycrane-payload-decentralized-hovering-v0 Isaac-flycrane-payload-decentralized-hovering-rand-v0; do
  echo "=== hebbian CMA-ES smoke: $task (3 gens, popsize 10, 2 repeats, 20 envs) ==="
  t0=$(date +%s)
  python scripts/hebbian/train.py --task=$task --headless --num_envs=20 --popsize=10 --n_repeats=2 --num_generations=3 --seed=1 2>&1 | grep -vE "omni.usd\] Warning" | tail -25 | cut -c1-200
  echo "exit=${PIPESTATUS[0]} wall=$(( $(date +%s) - t0 ))s"
done
echo "=== HEBB DONE ==="
