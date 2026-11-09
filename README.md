This repository is to track our progress in replicating the results of the paper titles FlowDAgger.

We were assigned this task in our Reinforcement Learning in Robotics course.

The paper can be found here https://arxiv.org/pdf/2607.08877

## Local setup (RTX 50-series / project-local env)

`setup.sh` + `run_assembly_conda.sh` are the original conda workflow. `setup_env.sh`,
`env.sh` and `run_assembly.sh` are an alternative that keeps everything (conda env,
12.4 GB checkpoint, caches, outputs) inside one project folder next to this repo:

```
PROJECT_ROOT/                     # defaults to the folder containing this repo
  EEE598_FlowDAgger_Replication/  # this repo
  FlowDAgger/                     # cloned by setup_env.sh
  .conda-env/  .cache/  runs/
```

```bash
bash setup_env.sh                 # one-time; needs conda/Miniforge; ends with GPU sanity checks
source env.sh                     # every new terminal
SEEDS="42" bash run_assembly.sh   # one seed (~6 h); DRY_RUN=1 to preview
python summarize.py ../runs/assembly_replication
```

Issues hit on an RTX 5070 (Blackwell, sm_120, 15 GB RAM) and how the scripts handle them:

- **PyTorch has no RTX 50-series kernels by default.** openpi pins `torch==2.7.1`, whose PyPI
  build is CUDA 12.6 (no sm_120). `setup_env.sh` installs the same version's cu128 build.
  `jax[cuda12]==0.8.0` already supports Blackwell.
- **protobuf VersionError at startup** (`gencode 6.33.5 runtime 5.29.6`). TensorFlow 2.19 needs
  protobuf 5, but the Google Cloud libs that openpi uses to fetch the PaliGemma tokenizer resolve
  to protobuf-6-only versions. `setup_env.sh` step 5b pins them to protobuf-5-compatible versions.
- **Checkpoint download stalls.** Hugging Face's Xet downloader stalled and can't resume partial
  files. `env.sh` sets `HF_HUB_DISABLE_XET=1` (plain HTTP, resumable).
- **Killed (OOM) while loading pi0.5.** The checkpoint is float32 (12.4 GB) and is loaded in host
  RAM before casting to bf16 and moving to the GPU; peak was ~12.4 GB RSS + ~7 GB swap. With
  ~16 GB RAM, use a 16 GB swapfile and close heavy apps. Peak VRAM ~9.9 GB.
- `pip check` still lists conflicts (openpi wants jax 0.5.3 / numpy<2, lerobot wants gymnasium
  0.29, ...). These are expected: upstream installs its own pins over openpi's on purpose.

## Pitfall: typing 's' anywhere fails eval rollouts

Upstream `train_utils.py` starts a **global** `pynput` keyboard listener during evaluation: pressing
`s` in *any* window abandons the current eval rollout and scores it as a **failure**. Typing in a
terminal or browser while a run evaluates silently lowers success rates (one 50-rollout base eval
here lost 16 rollouts this way).

Both run scripts now launch training with `env -u DISPLAY`, so the listener can't start (MuJoCo
renders through EGL and doesn't need a display). `ALLOW_SKIP_KEY=1 bash run_assembly.sh` keeps the
skip key for supervised runs.

To check old results for skips without logs: unskipped failures always run the full 300 steps and
successes take ~92, so `(avg_episode_len - success_rate*92) / (1 - success_rate)` should be ~300.
Values well below 300 mean rollouts were skipped. In `flowdagger_runs/` only seed 42 step 1500
(~190) looks affected; the base (step 0) and final (step 5000) numbers are clean.

See [FINDINGS.md](FINDINGS.md) for why the trial runs were ~0.1 below the paper (short answer: base-policy sampling, not a bug).
