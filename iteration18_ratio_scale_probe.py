#!/usr/bin/env python3
"""
Iteration 18 - Ratio-at-Scale Probe: does width rescue the 1:3 hybrid on MQAR?

WHY
---
Iteration 17 found that a 1:3 attention:SSM hybrid (SSAS) fails to form the
recall circuit at d=128 under every protocol tested (both placements, 2x
budget), while 1:1 (SASA) reaches transformer-level recall at half the KV. The
open question it left: is the 1:3 failure a small-scale training artifact --
as the success of production 1:3-1:7 hybrids (Jamba arXiv:2403.19887, Zamba
arXiv:2405.16712) suggests -- or does it persist with more capacity?

This probe doubles width (d=256, head_dim 64) and doubles the budget
(8000 steps) for the two informative arms:

  ratio_1to3 (SSAS): the failing configuration -- does scale rescue it?
  ssm4       (SSSS): pure fixed-state control at the same width, so any 1:3
                     gain is attributable to the attention layer, not width.

Protocol otherwise identical to iteration 17 (lr=3e-4, warmup 10%, one
protocol for all models; see the iteration 17 log entry for why lr matters).

REPRODUCE
---------
  python iteration18_ratio_scale_probe.py

Outputs: results/iteration18_ratio_scale.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from iteration16_quality_mqar import train_model, evaluate_model, EVAL_PAIR_COUNTS, N_VALS
from iteration17_hybrid_ratio_mqar import RatioModel

RESULTS_DIR = Path("results")

ARMS = {"ratio_1to3": "SSAS", "ssm4": "SSSS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print("=" * 100)
    print(f"ITERATION 18 - RATIO-AT-SCALE PROBE (d={args.d_model}, {args.steps} steps, "
          f"lr={args.lr})")
    print("=" * 100)
    print(f"[GPU] {torch.cuda.get_device_name(0)} | chance accuracy {1 / N_VALS:.3f}")

    report: Dict[str, Dict] = {}
    for name, pattern in ARMS.items():
        model = RatioModel(pattern, d=args.d_model).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {name} [{pattern}] d={args.d_model} ({n_params:,} params) ---")
        stats = train_model(model, args.steps, args.batch, device, args.seed + 1,
                            lr=args.lr, warmup_frac=args.warmup_frac)
        accs = evaluate_model(model, device, seed=args.seed + 2)
        print(f"    held-out accuracy: " + "  ".join(f"N={n}: {a:.3f}" for n, a in accs.items()))
        report[name] = {"pattern": pattern, "d_model": args.d_model,
                        "params": n_params, **stats,
                        "held_out_accuracy_by_pairs": accs}
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("SUMMARY (held-out accuracy; chance = 0.016)")
    header = "  ".join(f"N={n:>4}" for n in EVAL_PAIR_COUNTS)
    print(f"{'arm':<14} {header}")
    for name, r in report.items():
        row = "  ".join(f"{r['held_out_accuracy_by_pairs'][n]:>6.3f}" for n in EVAL_PAIR_COUNTS)
        print(f"{name:<14} {row}")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 18,
        "purpose": "Test whether doubled width + budget rescues the 1:3 hybrid's "
                   "recall circuit; pure-SSM control isolates the attention effect.",
        "protocol": {"steps": args.steps, "batch": args.batch, "d_model": args.d_model,
                     "lr": args.lr, "warmup_frac": args.warmup_frac, "seed": args.seed},
        "results": report,
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(RESULTS_DIR / "iteration18_ratio_scale.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\n[saved] results/iteration18_ratio_scale.json")


if __name__ == "__main__":
    main()
