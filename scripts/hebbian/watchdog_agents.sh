#!/usr/bin/env bash
# Self-healing watchdog for the Hebbian sweeps: keep a fixed number of wandb agents alive on
# each sweep (relaunching any that die/crash) so the GPU stays busy, WITHOUT OOMing the 4090.
# Maintains TARGET_PGPE agents on the pgpe-only sweep (prioritized) + TARGET_MAIN on the mixed
# sweep, under a shared MAX_TRAINERS cap. Launch detached on ALC:
#   cd /home/diggelen/AerialMan
#   SWEEP_PGPE=<id> setsid bash scripts/hebbian/watchdog_agents.sh >watchdog_agents.out 2>&1 </dev/null &
# Stop it with:  pkill -f watchdog_agents.sh
set -u
cd /home/diggelen/AerialMan

SWEEP_MAIN="${SWEEP_MAIN:-fudavd/hebbian-flycrane/ywv39wxa}"  # mixed pgpe+openai_es sweep
SWEEP_PGPE="${SWEEP_PGPE:-}"                                   # pgpe-only sweep id (empty = disabled)
TARGET_MAIN="${TARGET_MAIN:-2}"       # desired live agents on the mixed sweep
TARGET_PGPE="${TARGET_PGPE:-1}"       # desired live agents on the pgpe-only sweep
MAX_TRAINERS="${MAX_TRAINERS:-3}"     # hard cap on concurrent train.py (VRAM ~6.5 GB each @ p1024)
MIN_FREE_MB="${MIN_FREE_MB:-7500}"    # only launch when this much VRAM is free (avoid OOM)
COUNT="${COUNT:-50}"                  # configs an agent runs before exiting (long-lived)
INTERVAL="${INTERVAL:-180}"           # seconds between checks
STARTUP_GRACE="${STARTUP_GRACE:-150}" # pause after a launch so Isaac Sim finishes init before re-evaluating

log(){ echo "[$(date '+%F %T')] $*"; }
cnt(){ local n; n=$(pgrep -fc "$1" 2>/dev/null); [ -z "$n" ] && n=0; echo "$n"; }

launch_agent(){  # $1 = sweep id, $2 = label
  local sid="$1" lbl="$2" ts; ts=$(date '+%Y%m%d_%H%M%S')
  setsid .venv/bin/wandb agent --count "$COUNT" "$sid" > "watchdog_${lbl}_${ts}.log" 2>&1 < /dev/null &
  log "RELAUNCH $lbl agent on $sid -> watchdog_${lbl}_${ts}.log"
  sleep "$STARTUP_GRACE"
}

log "watchdog start: main=$SWEEP_MAIN (target $TARGET_MAIN) pgpe=${SWEEP_PGPE:-none} (target $TARGET_PGPE) max_trainers=$MAX_TRAINERS min_free=${MIN_FREE_MB}MB"
while true; do
  TRAINERS=$(cnt "scripts/hebbian/train.py")
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); FREE=${FREE:-0}
  # count agents by sweep id (the only process whose cmdline contains it); the --count <N>
  # value sits between "--count" and the id, so match the id alone, not "--count <id>".
  A_MAIN=$(cnt "$SWEEP_MAIN")
  A_PGPE=0; [ -n "$SWEEP_PGPE" ] && A_PGPE=$(cnt "$SWEEP_PGPE")
  if [ "$TRAINERS" -lt "$MAX_TRAINERS" ] && [ "$FREE" -ge "$MIN_FREE_MB" ]; then
    if [ -n "$SWEEP_PGPE" ] && [ "$A_PGPE" -lt "$TARGET_PGPE" ]; then
      launch_agent "$SWEEP_PGPE" "pgpe"
    elif [ "$A_MAIN" -lt "$TARGET_MAIN" ]; then
      launch_agent "$SWEEP_MAIN" "main"
    else
      log "ok (main=$A_MAIN/$TARGET_MAIN pgpe=$A_PGPE/$TARGET_PGPE trainers=$TRAINERS free=${FREE}MB)"
    fi
  else
    log "hold (main=$A_MAIN pgpe=$A_PGPE trainers=$TRAINERS free=${FREE}MB)"
  fi
  sleep "$INTERVAL"
done
