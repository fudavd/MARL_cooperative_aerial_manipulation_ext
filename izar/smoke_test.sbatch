#!/bin/bash
#SBATCH -A micro-515
#SBATCH --qos=debug
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J isaac_smoke
#SBATCH -o /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext/izar/smoke.out
set -uo pipefail
source /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext/izar/env.sh
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== isaacsim import ==="
python -c "import isaacsim; print('isaacsim import ok')"
echo "=== 3-iteration headless hook-grasp training (documented smoke test) ==="
python scripts/hook_grasp/train_rslrl.py --headless --num_envs 32 --max_iterations 3 --experiment_name izar_smoke
echo "exit=$?"
echo "=== SMOKE DONE ==="
echo "=== timing: 512 envs x 20 iterations ==="
date +%s > /tmp/$USER/t0; python scripts/hook_grasp/train_rslrl.py --headless --num_envs 512 --max_iterations 20 --experiment_name izar_timing > izar/timing.log 2>&1; echo "exit=$? wall=$(( $(date +%s) - $(cat /tmp/$USER/t0) ))s"
ls -t logs/rsl_rl/izar_timing/*/ | head -3
echo "=== TIMING DONE ==="
