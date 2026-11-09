# Findings: why our assembly results were ~0.1 below the paper

Investigation from 2026-11-08, done before any full runs on the RTX 5070 machine.

## The question

The first trial runs (`flowdagger_runs/`, 3 seeds x 25 eval rollouts) were consistently about 0.1 below
the paper (Table 1, Assembly):

| | Ours (`flowdagger_runs/`) | Paper | Gap |
|---|---|---|---|
| Base pi0.5 (step 0) | 0.53 +/- 0.05 | 0.64 | -0.11 |
| FlowDAgger, 50 episodes (step 5000) | 0.75 +/- 0.04 | 0.89 (0.87 in Table 5) | -0.14 |
| Improvement over base | +0.22 | +0.25 | -0.03 |

The improvement itself reproduces; the whole curve is shifted down from step 0, where FlowDAgger is not
involved yet (step-0 eval feeds pi0.5 plain Gaussian noise, `train_utils.py` `perform_control_eval`).

## Short answer

**The base gap is sampling, not a bug.** A 50-rollout-per-seed base-only eval (150 rollouts) gives
**0.62 +/- 0.04**, matching the paper's 0.64. The 25 start states each seed evaluates on happen to be
harder than average, and because the *same* 25 start states are reused at every eval in a run, that
penalty shifts the whole curve, which is why the gap looked consistent.

The final-score gap (0.75 vs 0.89) is probably partly the same effect but is not fully explained; we
could not check it because the trial runs saved no checkpoints (see "Next steps").

## Evidence

### Ruled out

| Possible cause | Check | Result |
|---|---|---|
| Step 0 isn't really the base policy | Read eval code | It is: Gaussian noise, steering net not used at `i == 0` |
| Different protocol | Paper Sec. 5.1 | Same: N=50 episodes, 3 seeds x 25 rollouts |
| 300-step eval horizon too short | Dataset `mmurz/metaworld_mt50_v3` | Assembly demos are 83-97 steps, our successes ~92; failures get stuck until 300 |
| Image mismatch (camera / flip / resolution) | Dataset frame vs our `assembly-v3` render at the same init | Mean pixel diff 0.2/255 (flips: 45-70); initial state identical to 1e-4 |
| Typing 's' during eval (see README pitfall) | Implied failed-episode length | Trial runs clean except seed 42 step 1500 |

### Base-only eval, 50 rollouts per seed (`base_eval_runs/`)

Run with `run_assembly.sh --max_steps 0 --eval_episodes 50 --save_eval_video 0`, skip-key listener
disabled, 0 skipped rollouts. Eval start states depend only on the run seed, so rollouts 1-25 here are
the **same start states** as the trial runs' step-0 eval.

| Seed | Rollouts 1-25: this machine | Rollouts 1-25: trial run | Rollouts 26-50 (new starts) | All 50 |
|---|---|---|---|---|
| 42 | 13/25 | 12/25 | 14/25 | 0.54 |
| 43 | 17/25 | 15/25 | 17/25 | 0.68 |
| 44 | 13/25 | 13/25 | 19/25 | 0.64 |
| **Total** | **43/75 = 0.57** | **40/75 = 0.53** | **50/75 = 0.67** | **0.62** |

- On identical start states, two machines differ by 3/75 rollouts.
- The standard 25 start states are harder (0.57) than the next 25 (0.67).
- Same-machine run-to-run noise: seed 43's step-0 eval accidentally ran twice (quirk below); the repeat
  flipped 2 of 38 rollout outcomes. `per_rollout.csv` has both passes (`pass` column).

Statistically: 75 rollouts give a standard error of ~0.055, so a 0.11 base gap against the paper's own
75-rollout number is ~1.4 combined SE, which is unremarkable.

## Quirks found along the way

- **Global 's' key skips eval rollouts.** See README "Pitfall". Fixed in both run scripts.
- **Step-0 eval can repeat.** `dagger_loop.py` (~line 256) runs the step-0 eval after every collected
  episode while the step counter is 0. With `--dual_buffer 1` the counter stays at 0 until the
  autonomous buffer has data (a successful episode with autonomous chunks), so the full step-0 eval can
  repeat (seed 43 of the trial run logged it 3 times). Costs 10-20 min per repeat; `summarize.py` keeps
  one value per step so numbers are unaffected. With `--max_steps 0`, kill the process once its
  `eval_results.jsonl` line appears if it starts repeating.

## Next steps

1. Run full runs with `--checkpoint_interval 5000` so each seed saves its final steering policy:
   `SEEDS="42" bash run_assembly.sh --checkpoint_interval 5000`
   (extra args disable the skip-finished-seeds check; drop them to resume).
2. Re-evaluate each final policy on 50+ start states (like the base-only eval) to measure the
   final score against 0.89 without the fixed-start-state penalty. Not done yet: checkpoint
   saving/loading is untested, and this needs a small eval-only script that loads the saved
   steering policy.
3. If a real final gap remains, ask the authors (GitHub issue) what they get with the released code.

## Files

- `base_eval_runs/*/eval_results.jsonl`: per-seed results; `python summarize.py base_eval_runs`
- `base_eval_runs/per_rollout.csv`: `seed,pass,rollout,success,shared_with_teammate_step0`
  (`shared_with_teammate_step0=1` for rollouts 0-24, the trial runs' start states)
