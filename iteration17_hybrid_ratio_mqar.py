#!/usr/bin/env python3
"""
Iteration 17 - Hybrid Ratio Sweep on MQAR: quality vs. KV-cache reduction.

WHY
---
Iteration 16 established (non-circularly) that pure fixed-state SSMs lose
in-context recall as the number of stored pairs grows, while a 1 SSM + 1
attention hybrid matched the transformer at half the KV. Production hybrids
(Jamba, Zamba, Granite 4.0 lineage) use sparser ratios (1 attention : 3-7 SSM).
This artifact quantifies that trade-off directly: how much attention is enough,
and how much recall-time memory does each ratio actually buy back?

DESIGN
------
Four-block models over the same MQAR task, training protocol, and evaluation as
iteration 16 (imported from `iteration16_quality_mqar` -- one source of truth):

  attn4     : ATTN ATTN ATTN ATTN   (transformer reference, 4x KV)
  ratio 1:1 : SSM  ATTN SSM  ATTN   (2x KV)
  ratio 1:3 : SSM  SSM  ATTN SSM    (1x KV -- the production-style sparse ratio)
  ssm4      : SSM  SSM  SSM  SSM    (constant state, no KV)

The single attention block in the 1:3 model is placed third (mid-late), matching
the common placement in published hybrids. Recall-time memory per sequence:
n_attn_blocks * 2*T*d floats of KV + n_ssm_blocks * d floats of state.

Every model is trained end-to-end and evaluated held-out; no lookup tables are
written by the harness and no baseline is hardcoded (see iteration 16 retraction
record for why this is stated explicitly).

TRAINING PROTOCOL NOTE (finding from this iteration)
----------------------------------------------------
At iteration 16's lr=1e-3, 4-block models containing 2+ SSM blocks fail to
converge on MQAR (loss plateaus ~2.0) while pure attention converges fully --
which would falsely look like an architectural failure. At lr=3e-4 with 10%
warmup the same hybrids converge to ~100% in-distribution. The sweep therefore
uses lr=3e-4 / warmup 10% for EVERY model (one protocol, all contenders), and
this sensitivity is reported rather than hidden.

ASSUMPTIONS AND LIMITATIONS
---------------------------
- Same as iteration 16: MQAR probes in-context recall, not full LM quality;
  models are small (d=128), so conclusions are about scaling shape and the
  relative cost of removing attention, not absolute 70B quality.
- One attention placement per ratio (no placement sweep); placement is
  documented so the result is interpretable.

REPRODUCE
---------
  python iteration17_hybrid_ratio_mqar.py --steps 4000 --plot

Outputs: results/iteration17_ratio_mqar.json, plots/iteration17_ratio_quality.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from iteration16_quality_mqar import (
    AttentionBlock, SSMBlock, RecallModel, train_model, evaluate_model,
    EVAL_PAIR_COUNTS, MAX_POS, N_VALS, VOCAB,
)

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# block pattern per ratio; "A" = attention, "S" = SSM (fixed state)
RATIOS: Dict[str, str] = {
    "attn4": "AAAA",
    "ratio_1to1": "SASA",
    "ratio_1to3": "SSAS",
    "ssm4": "SSSS",
}


class RatioModel(nn.Module):
    """Four-block MQAR model with a configurable attention/SSM pattern."""

    def __init__(self, pattern: str, d: int = 128):
        super().__init__()
        self.pattern = pattern
        self.d = d
        self.embed = nn.Embedding(VOCAB, d)
        self.uses_positions = "A" in pattern
        if self.uses_positions:
            self.pos = nn.Embedding(MAX_POS, d)
        self.blocks = nn.ModuleList(
            [AttentionBlock(d) if c == "A" else SSMBlock(d) for c in pattern])
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_VALS)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq)
        if self.uses_positions:
            x = x + self.pos(torch.arange(seq.shape[1], device=seq.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))  # [B, T, N_VALS]

    def recall_state_floats(self, context_len: int) -> int:
        """Per-sequence recall-time memory beyond weights (fp32 floats)."""
        n_attn = self.pattern.count("A")
        n_ssm = self.pattern.count("S")
        return n_attn * 2 * context_len * self.d + n_ssm * self.d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="one rate for all models; 1e-3 diverges for deep recurrent hybrids")
    parser.add_argument("--warmup-frac", type=float, default=0.10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required (all quality numbers must be real device runs).")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print("=" * 100)
    print("ITERATION 17 - HYBRID RATIO SWEEP ON MQAR (quality vs. KV reduction, all models trained)")
    print("=" * 100)
    print(f"[GPU] {torch.cuda.get_device_name(0)} | eval N in {EVAL_PAIR_COUNTS} "
          f"(128 = extrapolation) | chance {1 / N_VALS:.3f}")

    report: Dict[str, Dict] = {}
    for name, pattern in RATIOS.items():
        model = RatioModel(pattern, d=args.d_model).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {name} [{pattern}] ({n_params:,} params) ---")
        train_stats = train_model(model, args.steps, args.batch, device, args.seed + 1,
                                  lr=args.lr, warmup_frac=args.warmup_frac)
        accs = evaluate_model(model, device, seed=args.seed + 2)
        kv_128k_mb = model.recall_state_floats(131072) * 4 / 1e6
        print(f"    held-out accuracy: " + "  ".join(f"N={n}: {a:.3f}" for n, a in accs.items()))
        print(f"    recall state @128k ctx: {kv_128k_mb:.2f} MB (fp32)")
        report[name] = {
            "pattern": pattern,
            "params": n_params,
            **train_stats,
            "held_out_accuracy_by_pairs": accs,
            "recall_state_mb_128k_ctx": round(kv_128k_mb, 3),
        }
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("SUMMARY (held-out accuracy; chance = 0.016; N=128 = length extrapolation)")
    header = "  ".join(f"N={n:>4}" for n in EVAL_PAIR_COUNTS)
    print(f"{'model':<14} {'pattern':<7} {header}   state@128k")
    for name, r in report.items():
        row = "  ".join(f"{r['held_out_accuracy_by_pairs'][n]:>6.3f}" for n in EVAL_PAIR_COUNTS)
        print(f"{name:<14} {r['pattern']:<7} {row}   {r['recall_state_mb_128k_ctx']:>9.2f} MB")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 17,
        "purpose": ("Quantify recall quality vs. KV reduction across attention:SSM "
                    "ratios; extends the iteration 16 non-circular MQAR benchmark."),
        "protocol": {"steps": args.steps, "batch": args.batch, "d_model": args.d_model,
                     "lr": args.lr, "warmup_frac": args.warmup_frac,
                     "seed": args.seed, "blocks": 4,
                     "attention_placement": "see RATIOS patterns; single-attn placed third",
                     "note": "lr=1e-3 diverges for 4-block models with 2+ SSM blocks; "
                             "3e-4 converges for all -- documented protocol sensitivity"},
        "results": report,
        "gpu": torch.cuda.get_device_name(0),
    }
    out_path = RESULTS_DIR / "iteration17_ratio_mqar.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {out_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name, r in report.items():
            accs = r["held_out_accuracy_by_pairs"]
            axes[0].plot(EVAL_PAIR_COUNTS, [accs[n] for n in EVAL_PAIR_COUNTS],
                         "o-", label=f"{name} [{r['pattern']}]")
        axes[0].axhline(1 / N_VALS, color="gray", ls="--", lw=1, label="chance")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("key-value pairs in context (N)")
        axes[0].set_ylabel("held-out recall accuracy")
        axes[0].set_title("MQAR recall vs. attention:SSM ratio")
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        names = list(report)
        axes[1].bar(names, [report[n]["recall_state_mb_128k_ctx"] for n in names])
        axes[1].set_ylabel("recall state @128k ctx (MB)")
        axes[1].set_title("Memory bought back by each ratio")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration17_ratio_quality.png", dpi=140)
        print("[plot] plots/iteration17_ratio_quality.png")


if __name__ == "__main__":
    main()
