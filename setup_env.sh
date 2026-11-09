#!/usr/bin/env bash
# Project-local environment setup for FlowDAgger pi0.5 + MetaWorld.
# Tested on an RTX 5070 (Blackwell, sm_120), driver 595, Ubuntu, Python 3.11.
# Safe to re-run. Usage:  bash setup_env.sh   (needs conda/Miniforge installed)
#
# Differences from setup.sh:
#   - conda env lives in PROJECT_ROOT/.conda-env instead of a global "flowdagger" env
#   - torch 2.7.1 is the cu128 build (the default PyPI cu126 build has no RTX 50-series kernels)
#   - openpi-client and lerobot are installed the way openpi's uv config pins them
#   - Google Cloud libs are pinned to work with protobuf 5 (step 5b)
# See README "Local setup (RTX 50-series)" for the reasoning behind each step.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$REPO_DIR")}"
UPSTREAM="$PROJECT_ROOT/FlowDAgger"
ENV="$PROJECT_ROOT/.conda-env"
CONDA="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"

unset LD_LIBRARY_PATH   # system CUDA libs would shadow the pip-provided ones

# 1. Upstream code + openpi submodule
[ -d "$UPSTREAM" ] || git clone https://github.com/microsoft/FlowDAgger.git "$UPSTREAM"
git -C "$UPSTREAM" submodule update --init flowdagger_pi05/openpi

# 2. Project-local conda env (Python 3.11)
[ -x "$ENV/bin/python" ] || "$CONDA" create -p "$ENV" python=3.11 pip -y
PIP="$ENV/bin/python -m pip"
$PIP install --upgrade pip

# 3. Blackwell-capable PyTorch (same version openpi pins)
$PIP install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 4. openpi's non-PyPI deps, as pinned in openpi/pyproject.toml [tool.uv.sources]
$PIP install -e "$UPSTREAM/flowdagger_pi05/openpi/packages/openpi-client"
$PIP install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# 5. openpi, then FlowDAgger pins LAST so they win (per upstream requirements.txt)
$PIP install -e "$UPSTREAM/flowdagger_pi05/openpi"
$PIP install -r "$UPSTREAM/flowdagger_pi05/requirements.txt"

# 5b. Keep Google Cloud libs (used by openpi to fetch the PaliGemma tokenizer from gs://)
#     compatible with protobuf 5.x, which TensorFlow 2.19 requires. Unconstrained, they
#     resolve to protobuf-6-only versions and the run fails with a protobuf VersionError.
mkdir -p "$PROJECT_ROOT/.cache"
cat > "$PROJECT_ROOT/.cache/protobuf5-constraints.txt" <<'EOF'
protobuf==5.29.6
tensorflow==2.19.0
jax==0.8.0
numpy==2.1.3
torch==2.7.1+cu128
fsspec<=2026.6.0
huggingface-hub==0.36.2
datasets==5.0.1
EOF
$PIP install --upgrade -c "$PROJECT_ROOT/.cache/protobuf5-constraints.txt" \
  gcsfs google-api-core googleapis-common-protos grpcio-status proto-plus \
  google-cloud-storage google-cloud-storage-control grpc-google-iam-v1 "fsspec[http,gcs]"

# 6. Sanity checks
source "$REPO_DIR/env.sh"
python - <<'PY'
import jax, torch
print("jax", jax.__version__, jax.devices())
assert "cuda" in str(jax.devices()[0]).lower(), "JAX is on CPU"
print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("torch arch list:", torch.cuda.get_arch_list())
major, minor = torch.cuda.get_device_capability(0)
assert f"sm_{major}{minor}" in torch.cuda.get_arch_list(), "torch build lacks kernels for this GPU"
import jax.numpy as jnp; print("jax matmul ok:", float((jnp.ones((512,512)) @ jnp.ones((512,512))).sum()))
x = torch.ones(512, 512, device="cuda"); print("torch matmul ok:", (x @ x).sum().item())
import mujoco, metaworld, gymnasium
env = gymnasium.make("Meta-World/MT1", env_name="assembly-v3", render_mode="rgb_array", camera_name="corner3")
env.reset(seed=0); print("metaworld render ok:", env.render().shape)
import gcsfs; from google.api_core import exceptions; print("gcsfs/protobuf ok")
PY
cd "$UPSTREAM/flowdagger_pi05" && python -c "import train_flowdagger; print('train_flowdagger imports ok')"
echo "Environment ready. Activate with: source $REPO_DIR/env.sh"
