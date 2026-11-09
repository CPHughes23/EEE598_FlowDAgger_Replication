#!/usr/bin/env bash
# FlowDAgger pi0.5 on MetaWorld assembly (paper Table 1, Assembly row).
# Same training settings as run_assembly_conda.sh, but uses env.sh (project-local
# conda env, caches, outputs), writes one log per seed, and skips finished seeds.
#
# Usage:
#   bash run_assembly.sh                         # seeds 42 43 44 (~6 h per seed)
#   SEEDS="42" bash run_assembly.sh              # one seed
#   DRY_RUN=1 bash run_assembly.sh               # print what would run, run nothing
#   bash run_assembly.sh --eval_episodes 10      # extra args go to train_flowdagger.py
#   GROUP=base_eval bash run_assembly.sh --max_steps 0 --eval_episodes 50 --save_eval_video 0
#                                                # base-policy-only eval, no training
#
# Long runs: detach from the terminal so closing it doesn't kill the run, e.g.
#   nohup bash run_assembly.sh > ../runs/assembly_replication.nohup.log 2>&1 &
# A run needs ~12 GB host RAM while loading pi0.5 (add swap if you have <=16 GB RAM)
# and ~10 GB VRAM.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-42 43 44}"
DRY_RUN="${DRY_RUN:-0}"
GROUP="${GROUP:-assembly_replication}"

# EXP must be set before sourcing, since env.sh only defaults it when unset.
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$REPO_DIR")}"
export EXP="${EXP:-$PROJECT_ROOT/runs/$GROUP}"
source "$REPO_DIR/env.sh"
export PYTHONUNBUFFERED=1
LOG_DIR="$EXP/logs"
mkdir -p "$LOG_DIR"

python -W ignore -c "import jax; d = jax.devices(); print('[run] JAX devices:', d); assert 'cuda' in str(d[0]).lower(), 'JAX is on CPU'" 2>/dev/null

# Same settings as run_assembly_conda.sh.
ARGS=(--env metaworld --task_key metaworld_assembly
      --max_steps 5000
      --eval_episodes 25 --eval_interval 500
      --dual_buffer 1
      --launch_group_id "$GROUP")

# A seed counts as done if an eval_results.jsonl under EXP already has its final step.
seed_done() {
  local seed="$1" f
  for f in "$EXP"/*--s-"$seed"/eval_results.jsonl; do
    [ -f "$f" ] && grep -q '"step": 5000' "$f" && return 0
  done
  return 1
}

cd "$FLOWDAGGER_REPO/flowdagger_pi05"
for SEED in $SEEDS; do
  if seed_done "$SEED" && [ "$*" = "" ]; then
    echo "[run] seed $SEED already finished in $EXP, skipping"
    continue
  fi
  LOG="$LOG_DIR/seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"
  # train_utils.py starts a GLOBAL pynput keyboard listener during eval: an 's'
  # typed in ANY window aborts the current eval rollout and scores it as a
  # failure. Run without an X display so that listener can't start (MuJoCo
  # renders via EGL and doesn't need one). ALLOW_SKIP_KEY=1 keeps the skip key.
  NO_X=(env -u DISPLAY -u WAYLAND_DISPLAY)
  [ "${ALLOW_SKIP_KEY:-0}" = "1" ] && NO_X=()
  CMD=("${NO_X[@]}" python train_flowdagger.py "${ARGS[@]}" --seed "$SEED" "$@")
  echo "=== seed $SEED ($(date)) -> $LOG"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry run] ${CMD[*]}"
    continue
  fi
  rc=0
  "${CMD[@]}" > "$LOG" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[run] seed $SEED FAILED (exit $rc), see $LOG" >&2
    exit "$rc"
  fi
  echo "[run] seed $SEED done ($(date))"
done

echo "Done. Summarize with:"
echo "  python $REPO_DIR/summarize.py $EXP"
