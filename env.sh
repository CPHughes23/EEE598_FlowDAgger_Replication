# Project environment for the FlowDAgger replication.
# Usage:  source env.sh   (from this repo, in every new terminal)
#
# Expected layout (PROJECT_ROOT defaults to the folder containing this repo):
#   PROJECT_ROOT/
#     EEE598_FlowDAgger_Replication/   <- this repo
#     FlowDAgger/                      <- upstream code (setup_env.sh clones it)
#     .conda-env/                      <- project-local conda env (setup_env.sh)
#     .cache/                          <- HF checkpoint, openpi assets, wandb
#     runs/                            <- training outputs
# Override any of these by exporting PROJECT_ROOT / EXP before sourcing.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR
export PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$REPO_DIR")}"
export FLOWDAGGER_REPO="$PROJECT_ROOT/FlowDAgger"

# conda: use the one on PATH if any, else Miniforge in the default location.
CONDA_BASE="${CONDA_BASE:-$( [ -n "${CONDA_EXE:-}" ] && dirname "$(dirname "$CONDA_EXE")" || echo "$HOME/miniforge3")}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$PROJECT_ROOT/.conda-env"

# System CUDA (e.g. /usr/local/cuda-12.8) on LD_LIBRARY_PATH shadows the pip-installed
# CUDA libs that jax[cuda12] / torch cu128 ship with, which breaks JAX GPU support.
unset LD_LIBRARY_PATH

# Headless MuJoCo rendering on the NVIDIA GPU; don't let JAX grab all VRAM up front.
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Keep downloads/caches/outputs in the project.
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"          # pi0.5 checkpoint (mmurray-ms/pi05-metaworld, 12.4 GB)
export OPENPI_DATA_HOME="$PROJECT_ROOT/.cache/openpi"      # openpi assets / tokenizer
# Hugging Face's Xet downloader stalled on the 12.4 GB checkpoint and can't resume
# partial files; plain HTTP downloads resume after interruptions.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export WANDB_DIR="$PROJECT_ROOT/.cache/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"                 # set WANDB_MODE=online to log to wandb.ai
export EXP="${EXP:-$PROJECT_ROOT/runs}"                    # train_flowdagger.py output root
mkdir -p "$HF_HOME" "$OPENPI_DATA_HOME" "$WANDB_DIR" "$EXP"

echo "[env] $(python --version) @ $CONDA_PREFIX | EXP=$EXP"
