#!/usr/bin/env bash
# FlowDAgger pi0.5 on MetaWorld assembly, conda version.
# Run with the conda env ALREADY ACTIVE (conda activate flowdagger).
#
# Usage:  bash run_assembly_conda.sh /path/to/FlowDAgger
# One seed first:  SEEDS="42" bash run_assembly_conda.sh /path/to/FlowDAgger
set -euo pipefail

REPO="${1:?Usage: bash run_assembly_conda.sh /path/to/FlowDAgger}"
SEEDS="${SEEDS:-42 43 44}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "flowdagger" ]]; then
  echo "Activate the env first: conda activate flowdagger" >&2
  exit 1
fi

# Old system CUDA libs shadow the pip ones and break JAX GPU support.
unset LD_LIBRARY_PATH
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Separate folder so smoke-test runs don't get mixed into the summary.
export EXP="${EXP:-$HOME/class-work/598_EEE/flowdagger_runs/assembly_replication}"
mkdir -p "$EXP"

python -c "import jax; d = jax.devices(); print(d); assert 'cuda' in str(d[0]).lower(), 'JAX is on CPU'"

cd "$REPO/flowdagger_pi05"
for SEED in $SEEDS; do
  echo "=== seed $SEED ($(date)) ==="
  # env -u DISPLAY: train_utils.py hooks the GLOBAL keyboard (pynput) during eval,
  # so an 's' typed in any window aborts the current eval rollout and scores it as
  # a failure. Without an X display the listener can't start; EGL rendering is fine.
  env -u DISPLAY -u WAYLAND_DISPLAY python train_flowdagger.py \
    --env metaworld --task_key metaworld_assembly --seed "$SEED" \
    --max_steps 5000 \
    --eval_episodes 25 --eval_interval 500 \
    --dual_buffer 1 \
    --launch_group_id assembly_replication
done
echo "Done. Summarize with: python summarize.py $EXP"