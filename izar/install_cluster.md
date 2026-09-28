# Installing Isaac Sim 4.5 + IsaacLab 2.0.2 + this repo on SCITAS izar (V100)

Verified 2026-09-02 on `izar.hpc.epfl.ch` (RHEL 9.0, glibc 2.34, Tesla V100-32GB, driver 535.154).
Headless rsl_rl training and Hebbian CMA-ES training both run. Rendering/video is untested (V100 has no RT cores).

## Why it works (and why earlier attempts failed)

- **Nothing is compiled.** Isaac Sim 4.5, torch and warp are binary pip wheels. You never need `nvcc`, a `cuda/*`
  module, or a newer gcc. The only "wheels built" are pure-Python packages (isaaclab, skrl, the extension).
- **The hard requirement is glibc >= 2.34**, because the `isaacsim` wheel is tagged `manylinux_2_34`.
  izar's RHEL 9 has exactly 2.34. On RHEL 8 / CentOS 7 (glibc 2.28 / 2.17) pip refuses the wheel — that is the
  "old compiler/toolchain" wall people hit. No workaround on old OSes except a container.
- **V100 (compute 7.0) is fine for headless physics.** PhysX-GPU, torch cu124 and warp support it.
- **Do not upgrade IsaacLab past v2.1.1.** From v2.2.0 it needs torch 2.7 + Isaac Sim 5.1 + Python 3.11.

## 0. Prerequisites

```bash
ssh <user>@izar.hpc.epfl.ch
sacctmgr show assoc user=$USER format=account,qos -P     # note your account and QOS names
module load gcc/11.3.0 python/3.10.4                      # the only modules needed (gcc is just python's parent)
python --version                                          # 3.10.4
```
Slurm on izar needs an explicit QOS on every job: `build` (installs, no GPU, 10 days), `debug` (tests, 1 h),
`normal` (training, 3 days). Below, replace `micro-515` with your account.

The login node has **no GPU driver** — never run Isaac there. Compute nodes have internet, so pip works inside jobs.

git-lfs is not installed system-wide (needed for the USD assets):
```bash
mkdir -p ~/bin && cd ~/bin
curl -sL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \
  | tar xz --strip-components=1 git-lfs-3.5.1/git-lfs
export PATH=$HOME/bin:$PATH && git lfs install --skip-repo
```

## 1. Clone

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/fudavd/MARL_cooperative_aerial_manipulation_ext.git
cd MARL_cooperative_aerial_manipulation_ext
git lfs pull
git clone --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git thirdparty/IsaacLab
git clone https://github.com/jackzeng-robotics/skrl.git thirdparty/skrl   # 1.4.3 fork with CTDE/MAPPO models
git clone https://github.com/fudavd/revde.git thirdparty/revde             # only for the RevDE optimizer
```
(GitHub SSH keys are usually not on the cluster; https works for these public repos.)

## 2. Build the venv (as a job, ~2 h — the home FS is slow with Isaac Sim's ~100k small files)

Save as `izar/build_venv.sh` and run `sbatch izar/build_venv.sh`:

```bash
#!/bin/bash
#SBATCH -A micro-515
#SBATCH --qos=build
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -J venv_build
#SBATCH -o venv_build.out
set -euo pipefail
module purge
module load gcc/11.3.0 python/3.10.4
cd ~/projects/MARL_cooperative_aerial_manipulation_ext
export PIP_NO_CACHE_DIR=1
export TMPDIR=/tmp/$USER; mkdir -p "$TMPDIR"          # node-local /tmp is 2.9 TB

[ -d .venv ] || python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 1) torch first, pinned, from the cu124 index
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# 2) Isaac Sim (binary wheels, ~10 GB)
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
# 3) IsaacLab v2.0.2 source packages, editable
for p in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    pip install -e "thirdparty/IsaacLab/source/$p"
done
# 4) project deps (pinned to the versions that work on the laptop)
pip install cma==4.4.4 rsl-rl-lib==2.3.3 wandb==0.28.0 tensorboard==2.20.0 tensorboardX==2.6.5 \
    moviepy==2.2.1 imageio-ffmpeg==0.6.0 opencv-python==4.11.0.86 scipy==1.15.3 matplotlib==3.10.9 \
    evotorch==0.6.1 h5py==3.16.0 hydra-core==1.3.3 omegaconf==2.3.1 \
    gym==0.23.1 rl-games==1.6.1 stable_baselines3==2.8.0    # isaaclab_rl's __init__ imports rl_games + sb3
# 5) skrl fork WITHOUT deps (otherwise pip replaces it with stock skrl 2.x, which crashes on CTDE configs)
pip install -e thirdparty/skrl --no-deps
pip install -e exts/MARL_mav_carry_ext

python - <<'PY'
import torch, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.version.cuda)
for p in ["isaacsim","isaaclab","skrl","rsl-rl-lib","cma","numpy","gymnasium","MARL_mav_carry_ext"]:
    print(p, m.version(p))
PY
echo "=== BUILD DONE ==="
```

Expected result: torch 2.5.1+cu124, isaacsim 4.5.0.0, isaaclab 0.34.9, skrl 1.4.3 (path = thirdparty/skrl),
rsl-rl-lib 2.3.3, numpy 1.26.4, gymnasium 1.2.x, `.venv` ≈ 32 GB.

## 3. Environment file

Save as `izar/env.sh`; `source` it in every job/shell:

```bash
module purge
module load gcc/11.3.0 python/3.10.4
export REPO=$HOME/projects/MARL_cooperative_aerial_manipulation_ext
source "$REPO/.venv/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES                                        # headless EULA accept
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json   # never let Vulkan pick lvp/intel
export TMPDIR=/tmp/$USER; mkdir -p "$TMPDIR"
export PATH=$HOME/bin:$PATH
cd "$REPO"
```

## 4. Smoke test on a GPU node (qos=debug, ~2 min)

```bash
#!/bin/bash
#SBATCH -A micro-515 --qos=debug -p gpu --gres=gpu:1 -c 8 --mem=32G -t 00:30:00 -J isaac_smoke -o smoke.out
source ~/projects/MARL_cooperative_aerial_manipulation_ext/izar/env.sh
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import isaacsim; print('isaacsim ok')"
# rsl_rl hook-grasp, 3 iterations
python scripts/hook_grasp/train_rslrl.py --headless --num_envs 32 --max_iterations 3 --experiment_name smoke
# Hebbian CMA-ES, 3 generations
python scripts/hebbian/train.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 --headless \
    --num_envs=20 --popsize=10 --n_repeats=2 --num_generations=3
```
Success = `logs/rsl_rl/smoke/*/model_2.pt` and `logs/hebbian/.../results.csv` exist, and `smoke.out` has no
`[Error] ... Failed to import python module isaaclab_rl` lines. (The train scripts exit with `os._exit`, so
stdout can end abruptly; check the checkpoint dirs.) The first Isaac start on a node takes ~30–60 s.

## 5. Training job template

```bash
#!/bin/bash
#SBATCH -A micro-515 --qos=normal -p gpu --gres=gpu:1 -c 10 --mem=64G -t 2-00:00:00 -J train -o logs/%x_%j.out
source ~/projects/MARL_cooperative_aerial_manipulation_ext/izar/env.sh
python "$@"
```
`sbatch train.sbatch scripts/hook_grasp/train_rslrl.py --headless --num_envs 512 --max_iterations 20000 --experiment_name run1`

Measured: 512 envs of the hook-grasp task ≈ 9.0k env-steps/s on a V100 (≈ an RTX 4070 laptop, but 32 GB VRAM).

## Pitfalls checklist

- `srun: Invalid qos specification` → add `--qos=debug|normal|build` and `-A <account>`.
- `No module named 'gym' / 'rl_games'` at startup → step 4 deps missing (`gym`, `rl-games`, `stable_baselines3`).
- `pip show skrl` must say 1.4.3 at `thirdparty/skrl`. If it says 2.x, redo `pip install -e thirdparty/skrl --no-deps`.
- Never `pip install` anything that pulls a different torch (e.g. agilerl); torch must stay 2.5.1.
- `git status` shows USD/.pt assets as modified → phantom LFS diffs, ignore.
- wandb: run `wandb login` once on the login node (or export `WANDB_API_KEY` in the job) before `--wandb`.
- Isaac caches go to `~/.cache/ov`, `~/.local/share/ov`, `~/.nvidia-omniverse` — home, fine.
