"""Summarize FlowDAgger replication runs and compare with the paper.

Usage: python summarize.py [runs_dir]   (default: ./runs, or $EXP)
Reads every eval_results.jsonl under runs_dir.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

# Paper, Table 1, Assembly row (pi0.5 base, MetaWorld, mean over 3 seeds x 25 rollouts)
PAPER = {"Base": 0.64, "SFT": 0.85, "LoRA-DAgger": 0.81,
         "Residual-DAgger": 0.53, "DSRL": 0.64, "FlowDAgger": 0.89}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXP", "runs"))
    files = sorted(root.rglob("eval_results.jsonl"))
    if not files:
        sys.exit(f"No eval_results.jsonl found under {root}")

    by_seed = defaultdict(dict)  # seed -> {step: success_rate}
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_seed[r["seed"]][r["step"]] = r["success_rate"]

    steps = sorted({s for d in by_seed.values() for s in d})
    print(f"{len(by_seed)} seed(s): {sorted(by_seed)}\n")
    print("step (100 BC steps = 1 episode)  mean  std   per-seed")
    for s in steps:
        vals = [d[s] for d in by_seed.values() if s in d]
        print(f"{s:>6}  ep~{s // 100:<3}  {mean(vals):.2f}  {pstdev(vals):.2f}  "
              + " ".join(f"{v:.2f}" for v in vals))

    last = max(steps)
    final = [d[last] for d in by_seed.values() if last in d]
    first = [d[min(steps)] for d in by_seed.values() if min(steps) in d]
    print(f"\nFirst eval (closest to frozen base): {mean(first):.2f}  (paper base {PAPER['Base']})")
    print(f"Final eval, step {last}: {mean(final):.2f} +/- {pstdev(final):.2f}  "
          f"(paper FlowDAgger {PAPER['FlowDAgger']})")
    print("\nPaper Table 1, Assembly:", ", ".join(f"{k} {v}" for k, v in PAPER.items()))
    print("Note: 25 rollouts/seed gives a standard error of roughly 0.06 near 0.8, "
          "so expect a few points of variation.")


if __name__ == "__main__":
    main()