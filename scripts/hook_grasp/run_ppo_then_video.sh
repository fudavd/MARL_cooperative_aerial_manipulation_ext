#!/bin/bash
# Train PPO on the hook-grasp env for up to 11h, then auto-render a video of the trained policy.
# Runs unattended: when training ends (timeout or completion) it locates the newest checkpoint and
# renders a video, copying it to a predictable path.
set -u
cd /home/fuda/PycharmProjects/MARL_cooperative_aerial_manipulation_ext
export PYTHONUNBUFFERED=1

VENV=.venv/bin/python
TASK=Isaac-flycrane-hook-grasp-v0
LOGROOT=logs/skrl/ppo_flycrane_hook
RENDER_DIR=scripts/hook_grasp/_render
TRAIN_SECONDS="${TRAIN_SECONDS:-39600}"   # 11h training cap by default; video runs after
NUM_ENVS="${NUM_ENVS:-128}"
RESUME_CKPT="${RESUME_CKPT:-}"            # optional: warm-start from this checkpoint (--resume --checkpoint)

RESUME_ARGS=""
if [ -n "${RESUME_CKPT}" ]; then
  if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[wrap] ERROR: RESUME_CKPT set but not found: ${RESUME_CKPT}"; exit 3
  fi
  RESUME_ARGS="--resume --checkpoint ${RESUME_CKPT}"
  echo "[wrap] resuming (warm-start) from: ${RESUME_CKPT}"
fi

echo "[wrap] ==== PPO TRAIN START $(date) (cap ${TRAIN_SECONDS}s, num_envs=${NUM_ENVS}) ===="
# max_iterations set huge so the timeout is the limiter; checkpoints saved periodically (yaml: 2500).
# SIGKILL (-s KILL) at the deadline: kill training abruptly so its Isaac teardown does NOT run -- that
# teardown closes the X display (:1) and would break the subsequent video render. Final policy comes
# from the periodic checkpoints, so no clean-exit save is needed.
timeout -s KILL ${TRAIN_SECONDS} ${VENV} scripts/skrl/train.py \
  --task ${TASK} --algorithm PPO ${RESUME_ARGS} \
  --num_envs ${NUM_ENVS} --max_iterations 500000 --headless --seed 42
echo "[wrap] ==== PPO TRAIN ENDED $(date) (exit=$?) ===="
sleep 15   # let the GPU / display settle before the video render

# newest run dir for this task's PPO logs
RUN=$(ls -td ${LOGROOT}/*_ppo_torch* 2>/dev/null | head -1)
echo "[wrap] run dir: ${RUN}"
CKPT="${RUN}/checkpoints/best_agent.pt"
if [ ! -f "${CKPT}" ]; then
  CKPT=$(ls -t "${RUN}"/checkpoints/agent_*.pt 2>/dev/null | head -1)
fi
echo "[wrap] checkpoint: ${CKPT}"

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "[wrap] ERROR: no checkpoint found; cannot render video."
  exit 2
fi

echo "[wrap] ==== VIDEO RENDER START $(date) ===="
mkdir -p ${RENDER_DIR}
rm -f ${RENDER_DIR}/ppo_policy.mp4
# Manual-render script (env.render()+imageio+os._exit) -- reliable; gym RecordVideo hangs on this env in
# headless. Loads the skrl PPO agent and writes DIRECTLY to ${RENDER_DIR}/ppo_policy.mp4 (follow-cam).
timeout --kill-after=60 900 ${VENV} scripts/hook_grasp/play_ppo_video.py \
  --task ${TASK} --checkpoint "${CKPT}" \
  --num_envs 1 --steps 600 --frame_every 4 --fps 15 --headless --enable_cameras --seed 7
echo "[wrap] ==== VIDEO RENDER ENDED $(date) (exit=$?) ===="

if [ -f ${RENDER_DIR}/ppo_policy.mp4 ]; then
  echo "[wrap] video ready -> ${RENDER_DIR}/ppo_policy.mp4"
else
  echo "[wrap] WARNING: ppo_policy.mp4 not produced"
fi
echo "[wrap] ==== ALL DONE $(date) ===="
