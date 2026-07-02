#!/usr/bin/env python3
"""
Iteration 16 - Non-Circular Quality Benchmark: Multi-Query Associative Recall (MQAR).

WHAT THIS FIXES
---------------
Prior quality evaluations (iterations 10-15, quality_sustainability_diagnostic.py)
were circular: fact vectors were explicitly written into associative slots via
`adapt_fact`, then "needle retrieval quality" read them back from those same
slots. That tests a 128-entry dictionary, not a model -- ~99% is guaranteed by
construction. Worse, iteration 11's transformer baseline quality was HARDCODED
(`0.031 + np.random.normal(0, 0.01)`); the baseline was never run. Those quality
claims are RETRACTED (see BREAKTHROUGH_PROGRESS_LOG.md, Iteration 16).

THIS BENCHMARK
--------------
MQAR (multi-query associative recall) is the standard synthetic probe for
in-context recall capacity, used in the Mamba/Zoology/Based line of work
(Arora et al. 2023, arXiv:2312.04927; Gu & Dao 2023, arXiv:2312.00752).
A sequence presents N key->value pairs, then a SEP token, then all N keys
again in shuffled order as queries; at each query position the model must
output the value bound to that key ([k1 v1 ... kN vN SEP q1 q2 ... qN],
loss/accuracy over all N query positions). Nothing is written into any
lookup table by the harness -- every model must LEARN retrieval end-to-end
from data.

Three architectures, matched parameter budgets, identical training protocol:
  1. transformer -- 2 blocks of causal multi-head attention + MLP (the
     von Neumann baseline whose inference cache grows linearly with context).
  2. ssm         -- 2 blocks of input-selective gated linear recurrence + MLP
     (constant O(d) inference state per layer, the fixed-state architecture
     this campaign advocates).
  3. hybrid      -- 1 SSM block + 1 attention block (the literature's
     standard compromise).

All training and evaluation are real fp32 forward/backward passes on the GPU.
Held-out evaluation across N in {4..64} pairs (in-distribution) and N=128
(length extrapolation, marked as such). Whatever the numbers come out to be
is what gets reported -- the known literature result is that fixed-state
models DEGRADE as the number of pairs approaches state capacity, and this
benchmark is expected to reproduce that honestly, contradicting the retracted
"quality stays perfect at any context" claims.

INFERENCE-MEMORY ANALYSIS (the actual architectural trade-off)
--------------------------------------------------------------
  transformer: KV cache = 2 * n_layers * T * d floats -> linear in context T
  ssm:         state    = n_layers * d floats         -> constant in context
  hybrid:      one attention layer of KV + one constant state

ASSUMPTIONS AND LIMITATIONS
---------------------------
- MQAR is a proxy for in-context recall, not full language modeling quality.
- Models are small (d=128, 2 blocks); conclusions are about scaling *shape*
  (accuracy vs. number of stored pairs; memory vs. context), not absolute LM
  quality at 70B scale.
- The SSM recurrence is computed with an explicit per-step loop (clear and
  correct rather than fast); training speed is not a claim of this artifact.
- Learned absolute position embeddings (transformer/hybrid) are untrained
  beyond the max training length; the N=128 column is flagged as
  extrapolation for those models.

REPRODUCE
---------
  python iteration16_quality_mqar.py --plot
  python iteration16_quality_mqar.py --steps 3000 --plot   # longer training

Outputs: results/iteration16_quality_mqar.json, plots/iteration16_quality.png
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

N_KEYS = 128  # must cover the largest eval N (unique keys per sequence)
N_VALS = 64
SEP_TOKEN = N_KEYS + N_VALS
VOCAB = N_KEYS + N_VALS + 1
MAX_POS = 1024


# =============================================================================
# Task: [k1 v1 ... kN vN SEP q1 ... qN] where the q_i are the same N keys in
# shuffled order; at each query position predict the value bound to that key.
# Loss/accuracy over all N query positions (standard multi-query AR).
# =============================================================================

def make_mqar_batch(batch: int, n_pairs: int, device: torch.device,
                    generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    assert n_pairs <= N_KEYS, "cannot draw more unique keys than the key vocabulary"
    # Unique keys per sequence (no ambiguous bindings).
    keys = torch.argsort(torch.rand(batch, N_KEYS, generator=generator, device=device),
                         dim=1)[:, :n_pairs]
    vals = torch.randint(N_VALS, (batch, n_pairs), generator=generator,
                         device=device) + N_KEYS
    q_perm = torch.argsort(torch.rand(batch, n_pairs, generator=generator, device=device),
                           dim=1)
    q_keys = torch.gather(keys, 1, q_perm)
    targets = torch.gather(vals, 1, q_perm) - N_KEYS  # class indices over values

    seq = torch.empty(batch, 3 * n_pairs + 1, dtype=torch.long, device=device)
    seq[:, 0:2 * n_pairs:2] = keys
    seq[:, 1:2 * n_pairs:2] = vals
    seq[:, 2 * n_pairs] = SEP_TOKEN
    seq[:, 2 * n_pairs + 1:] = q_keys
    return seq, targets


# =============================================================================
# Blocks (matched budgets: each block ~= 4*d^2 mixing params + 8*d^2 MLP)
# =============================================================================

class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.silu(self.fc1(self.norm(x))))


class AttentionBlock(nn.Module):
    """Pre-LN causal multi-head attention + MLP. Inference cache: O(T) KV."""

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.mlp = MLP(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, d)
        x = x + self.out(y)
        return self.mlp(x)


class SSMBlock(nn.Module):
    """Input-selective gated linear recurrence + MLP. Inference state: O(d), constant.

        h_t = a_t * h_{t-1} + (1 - a_t) * u_t,   a_t = sigmoid(W_a x_t)  (per channel)
        y_t = W_o (h_t * silu(W_g x_t))

    Diagonal selective SSM in the Mamba spirit, kept minimal and explicit.
    The per-step loop is the honest recurrence (no parallel-scan tricks).
    """

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, d)
        self.decay = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.mlp = MLP(d)
        # Bias decay toward remembering (a ~ 0.95) at init so early training
        # does not collapse the state before retrieval circuits can form.
        nn.init.constant_(self.decay.bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        z = self.norm(x)
        u = self.in_proj(z)
        a = torch.sigmoid(self.decay(z))
        g = F.silu(self.gate(z))
        h = torch.zeros(B, d, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = a[:, t] * h + (1 - a[:, t]) * u[:, t]
            ys.append(h)
        y = torch.stack(ys, dim=1) * g
        x = x + self.out(y)
        return self.mlp(x)


class RecallModel(nn.Module):
    def __init__(self, arch: str, d: int = 128):
        super().__init__()
        self.arch = arch
        self.embed = nn.Embedding(VOCAB, d)
        self.uses_positions = arch in ("transformer", "hybrid")
        if self.uses_positions:
            self.pos = nn.Embedding(MAX_POS, d)
        blocks = {
            "transformer": lambda: nn.ModuleList([AttentionBlock(d), AttentionBlock(d)]),
            "ssm": lambda: nn.ModuleList([SSMBlock(d), SSMBlock(d)]),
            "hybrid": lambda: nn.ModuleList([SSMBlock(d), AttentionBlock(d)]),
        }[arch]()
        self.blocks = blocks
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_VALS)  # value-class logits at every position

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.embed(seq)
        if self.uses_positions:
            x = x + self.pos(torch.arange(seq.shape[1], device=seq.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))  # [B, T, N_VALS]

    @staticmethod
    def query_logits(logits: torch.Tensor, n_pairs: int) -> torch.Tensor:
        """Logits at the N query positions (inputs q1..qN sit at 2N+1 .. 3N)."""
        return logits[:, 2 * n_pairs + 1:, :]

    def inference_state_floats(self, context_len: int, d: int = 128) -> int:
        """Per-sequence recall-time memory beyond weights (the KV-vs-state trade-off)."""
        per_attn = 2 * context_len * d  # K and V, one layer
        per_ssm = d                     # h, one layer
        return {
            "transformer": 2 * per_attn,
            "ssm": 2 * per_ssm,
            "hybrid": per_attn + per_ssm,
        }[self.arch]


# =============================================================================
# Train / evaluate -- identical protocol for every architecture. No hardcoded
# numbers anywhere: the baseline is trained and evaluated exactly like the rest.
# =============================================================================

TRAIN_PAIR_COUNTS = [4, 8, 16, 32, 64]
EVAL_PAIR_COUNTS = [4, 8, 16, 32, 64, 128]  # 128 = length extrapolation
MAX_TRAIN_PAIRS = max(TRAIN_PAIR_COUNTS)


def train_model(model: RecallModel, steps: int, batch: int, device: torch.device,
                seed: int) -> Dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps - warmup)],
        milestones=[warmup])
    model.train()
    t0 = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, steps + 1):
        n_pairs = TRAIN_PAIR_COUNTS[
            int(torch.randint(len(TRAIN_PAIR_COUNTS), (1,), generator=generator,
                              device=device).item())]
        seq, targets = make_mqar_batch(batch, n_pairs, device, generator)
        logits = RecallModel.query_logits(model(seq), n_pairs)
        loss = F.cross_entropy(logits.reshape(-1, N_VALS), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last_loss = loss.item()
        if step % max(1, steps // 5) == 0:
            print(f"    step {step:5d}/{steps}  loss {last_loss:.4f}")
    return {"train_seconds": round(time.perf_counter() - t0, 1),
            "final_train_loss": round(last_loss, 4)}


@torch.no_grad()
def evaluate_model(model: RecallModel, device: torch.device, seed: int,
                   n_eval: int = 2048, batch: int = 256) -> Dict[int, float]:
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    accs = {}
    for n_pairs in EVAL_PAIR_COUNTS:
        correct = total = 0
        for _ in range(n_eval // batch):
            seq, targets = make_mqar_batch(batch, n_pairs, device, generator)
            pred = RecallModel.query_logits(model(seq), n_pairs).argmax(dim=-1)
            correct += int((pred == targets).sum().item())
            total += targets.numel()
        accs[n_pairs] = round(correct / total, 4)
    return accs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required (all quality numbers must be real device runs).")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print("=" * 100)
    print("ITERATION 16 - MQAR QUALITY BENCHMARK (non-circular; every model trained; no hardcoded baselines)")
    print("=" * 100)
    print(f"[GPU] {torch.cuda.get_device_name(0)} | task: {N_KEYS} keys x {N_VALS} values, "
          f"train N in {TRAIN_PAIR_COUNTS}, eval N in {EVAL_PAIR_COUNTS} (128 = extrapolation)")
    print(f"[NOTE] chance accuracy = {1 / N_VALS:.3f}")

    report: Dict[str, Dict] = {}
    for arch in ("transformer", "ssm", "hybrid"):
        model = RecallModel(arch, d=args.d_model).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {arch} ({n_params:,} params) ---")
        train_stats = train_model(model, args.steps, args.batch, device, args.seed + 1)
        accs = evaluate_model(model, device, seed=args.seed + 2)
        state_8k = model.inference_state_floats(8192, args.d_model)
        state_128k = model.inference_state_floats(131072, args.d_model)
        print(f"    held-out accuracy: " +
              "  ".join(f"N={n}: {a:.3f}" for n, a in accs.items()))
        print(f"    inference state (beyond weights): {state_8k * 4 / 1e6:.2f} MB @8k ctx, "
              f"{state_128k * 4 / 1e6:.2f} MB @128k ctx (fp32)")
        report[arch] = {
            "params": n_params,
            **train_stats,
            "held_out_accuracy_by_pairs": accs,
            "inference_state_mb_8k_ctx": round(state_8k * 4 / 1e6, 3),
            "inference_state_mb_128k_ctx": round(state_128k * 4 / 1e6, 3),
        }
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("SUMMARY (held-out accuracy; chance = 0.016)")
    header = "  ".join(f"N={n:>4}" for n in EVAL_PAIR_COUNTS)
    print(f"{'arch':<12} {header}   state@128k")
    for arch, r in report.items():
        row = "  ".join(f"{r['held_out_accuracy_by_pairs'][n]:>6.3f}" for n in EVAL_PAIR_COUNTS)
        print(f"{arch:<12} {row}   {r['inference_state_mb_128k_ctx']:>8.2f} MB")
    print("\nN=128 is length extrapolation (untrained positions for transformer/hybrid).")
    print("These are the real trade-offs; no architecture gets asserted quality for free.")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 16,
        "purpose": ("Non-circular quality benchmark; supersedes and retracts the "
                    "slot-readback quality claims and hardcoded baselines of iterations 10-15."),
        "task": {"name": "MQAR", "n_keys": N_KEYS, "n_vals": N_VALS,
                 "chance_accuracy": 1 / N_VALS,
                 "train_pairs": TRAIN_PAIR_COUNTS, "eval_pairs": EVAL_PAIR_COUNTS,
                 "references": ["arXiv:2312.04927 (Zoology)", "arXiv:2312.00752 (Mamba)"]},
        "protocol": {"steps": args.steps, "batch": args.batch, "d_model": args.d_model,
                     "optimizer": "AdamW lr=1e-3 wd=0.01 cosine", "seed": args.seed,
                     "eval_samples_per_setting": 2048},
        "results": report,
        "gpu": torch.cuda.get_device_name(0),
    }
    out_path = RESULTS_DIR / "iteration16_quality_mqar.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {out_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for arch, r in report.items():
            accs = r["held_out_accuracy_by_pairs"]
            axes[0].plot(EVAL_PAIR_COUNTS, [accs[n] for n in EVAL_PAIR_COUNTS], "o-", label=arch)
        axes[0].axhline(1 / N_VALS, color="gray", ls="--", lw=1, label="chance")
        axes[0].axvline(MAX_TRAIN_PAIRS, color="gray", ls=":", lw=1)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("key-value pairs in context (N)")
        axes[0].set_ylabel("held-out recall accuracy")
        axes[0].set_title("MQAR: trained models, held-out eval\n(right of dotted line = extrapolation)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        ctxs = [1024, 8192, 32768, 131072]
        for arch in report:
            m = RecallModel(arch, d=args.d_model)
            axes[1].plot(ctxs, [m.inference_state_floats(c, args.d_model) * 4 / 1e6 for c in ctxs],
                         "o-", label=arch)
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("context length"); axes[1].set_ylabel("inference state (MB)")
        axes[1].set_title("The real architectural trade-off:\nrecall memory vs. context")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration16_quality.png", dpi=140)
        print("[plot] plots/iteration16_quality.png")


if __name__ == "__main__":
    main()
