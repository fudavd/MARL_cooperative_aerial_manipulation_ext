#!/bin/bash
# Generic 1-GPU training job. Usage (from repo root):
#   sbatch izar/train.sbatch scripts/hook_grasp/train_rslrl.py --headless --num_envs 512 --max_iterations 20000 --experiment_name v90_izar
#SBATCH -A micro-515
#SBATCH --qos=normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -J marl_train
#SBATCH -o /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext/izar/logs/%x_%j.out
set -uo pipefail
source /home/vandigge/projects/MARL_cooperative_aerial_manipulation_ext/izar/env.sh
mkdir -p izar/logs
echo "node=$(hostname) job=$SLURM_JOB_ID args=$*"
python "$@"
