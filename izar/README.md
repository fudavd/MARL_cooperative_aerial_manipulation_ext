# Running this repo on SCITAS izar (V100)

Repo: `~/projects/MARL_cooperative_aerial_manipulation_ext` on `vandigge@izar.hpc.epfl.ch`
(RHEL 9, glibc 2.34, Tesla V100-32GB, driver 535.154, Slurm account `micro-515`).

## Environment (every shell / job)
```bash
source ~/projects/MARL_cooperative_aerial_manipulation_ext/izar/env.sh
```
That loads `gcc/11.3.0 python/3.10.4`, activates `.venv`, sets `OMNI_KIT_ACCEPT_EULA=YES`,
`VK_ICD_FILENAMES` (NVIDIA Vulkan ICD) and `TMPDIR=/tmp/$USER`. The `cuda/*` modules are NOT needed
(torch cu124 wheels bundle the runtime). Login node has no GPU driver: run anything Isaac in a job.

## Stack in `.venv` (mirrors the laptop)
Python 3.10.4, torch 2.5.1+cu124, isaacsim 4.5.0.0, IsaacLab v2.0.2 (editable, `thirdparty/IsaacLab`),
numpy 1.26.4, gymnasium 1.3.0, gym 0.23.1, warp-lang 1.17.0, rsl-rl-lib 2.3.3, skrl 1.4.3 (fork, editable,
`thirdparty/skrl`), cma 4.4.4, wandb 0.28.0, MARL_mav_carry_ext (editable).
Rebuild: `sbatch izar/build_venv.sh` (qos=build, ~2 h — home FS is slow with Isaac Sim's many small files).

## Jobs
```bash
# quick GPU check (qos=debug, ≤1 h): isaacsim import + 3-iteration headless train_rslrl
sbatch izar/smoke_test.sbatch
# real training (qos=normal, ≤3 d), args = the python command
sbatch izar/train.sbatch scripts/hook_grasp/train_rslrl.py --headless --num_envs 512 --max_iterations 20000 --experiment_name v90_izar
# interactive GPU shell
srun -A micro-515 --qos=debug -p gpu --gres=gpu:1 -c 8 --mem=32G -t 01:00:00 --pty bash
```
Logs: `izar/logs/<jobname>_<jobid>.out`; rsl_rl checkpoints under `logs/rsl_rl/<experiment_name>/`.
Note the train scripts exit via `os._exit`, so stdout may stop early — check the checkpoint dir for success.

## Redeploy edits from the laptop
```bash
git push --no-verify izar <branch>          # laptop remote 'izar' = ssh://vandigge@izar.hpc.epfl.ch/~/projects/...
rsync -az --exclude=.git --exclude=.venv --exclude=wandb --exclude=logs --exclude=outputs --exclude=.idea \
      --exclude=__pycache__ --exclude=thirdparty/IsaacLab --exclude=thirdparty/hebbian_marl \
      ./ vandigge@izar.hpc.epfl.ch:~/projects/MARL_cooperative_aerial_manipulation_ext/
```
`git status` on izar shows the USD/.pt LFS assets as modified — phantom LFS diffs, harmless.
Rendering/video (`play_*_video.py`) is untested on V100 (no RT cores); train headless here, render locally.
