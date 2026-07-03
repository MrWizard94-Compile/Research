#!/usr/bin/env python3
"""
Iteration 18 - Expert-Prefetch Miss-Penalty Curve: what predictor accuracy is
actually worth on this pipeline.

WHY
---
Iteration 17 measured the two routing bounds for MoE streaming at B=1:
oracle prefetch (179 ms/token) and just-in-time routing (199 ms/token, +11%).
Real systems sit between them, using a predictor that guesses layer i+1's
experts from layer i's state (published hit rates ~80-90%, e.g. Fiddler
arXiv:2402.07033, MoE-Infinity arXiv:2401.14361, Pre-gated MoE
arXiv:2308.12066). Rather than invent a predictor for random weights (whose
routing has no learnable structure -- a predictor trained here would measure
nothing real), this artifact measures the PIPELINE COST of any given accuracy:
tok/s as a function of controlled hit rate h, with real on-demand fetches for
every miss.

METHOD
------
- Record the true routing trace (deterministic given seed) once.
- For each hit rate h in {1.0, 0.9, 0.8, 0.5, 0.25}: during layer i's compute,
  prefetch layer i+1's PREDICTED experts -- each true expert kept with
  probability h, else replaced by a wrong one (seeded). When layer i+1's true
  routing is revealed, missing experts are fetched on demand; compute waits.
  h=1.0 reproduces oracle; h=0.25 approximates a chance predictor (2 of 8).
- Every transfer is a real PCIe copy of a real pinned 67 MB expert; misses are
  counted and reported alongside an analytic check
  (predicted wall ~= oracle wall + misses x expert transfer time).

This turns "we need a good predictor" into a measured requirement: the curve
says exactly how many tok/s each point of hit rate buys on this hardware.

ASSUMPTIONS AND LIMITATIONS
---------------------------
- Same testbed as iteration 17 (16 layers x 8 experts x 67.1 MB, top-2, B=1).
- Miss pattern is i.i.d. per expert; real predictors miss in bursts on topic
  shifts, which would cluster stalls, not change their total.
- Hit rate is imposed, not learned; predictor design is future work on a real
  checkpoint (routing structure requires trained weights).

REPRODUCE
---------
  python iteration18_prefetch_miss_curve.py --plot

Outputs: results/iteration18_prefetch_results.json, plots/iteration18_prefetch.png
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from iteration16_corrected_streaming import HardwareProfile, _rms_norm
from iteration17_moe_streaming import (
    build_moe_host_layers, make_routers, route, expert_forward,
    ExpertSlots, N_EXPERTS, TOP_K, expert_bytes,
)

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

HIT_RATES = [1.0, 0.9, 0.8, 0.5, 0.25]


def record_routing_trace(host, slots, routers, x: torch.Tensor) -> List[List[int]]:
    """True expert ids per layer for this activation (computed layer by layer)."""
    trace = []
    xc = x.clone()
    with torch.no_grad():
        for i in range(len(host)):
            ids_t, gates = route(xc, routers[i])
            ids = [int(v) for v in ids_t.cpu()]
            trace.append(ids)
            # advance activation with the true experts (weights already on GPU? no --
            # copy synchronously; this is trace recording, not a timed path)
            y = torch.zeros_like(xc)
            for k, eid in enumerate(ids):
                slots.up[k].copy_(host[i][eid]["up"])
                slots.down[k].copy_(host[i][eid]["down"])
                y = y + gates[k] * expert_forward(xc, slots.up[k], slots.down[k])
            xc = _rms_norm(xc + y)
    return trace


def predict_ids(true_ids: List[int], hit_rate: float, rng: random.Random) -> List[int]:
    """Each true expert kept with prob h, else swapped for a wrong expert."""
    pred = []
    for eid in true_ids:
        if rng.random() < hit_rate:
            pred.append(eid)
        else:
            wrong = [e for e in range(N_EXPERTS) if e not in true_ids and e not in pred]
            pred.append(rng.choice(wrong) if wrong else eid)
    return pred


def run_predicted_token(host, slots: ExpertSlots, routers, trace: List[List[int]],
                        x: torch.Tensor, hit_rate: float, rng: random.Random,
                        transfer_stream: torch.cuda.Stream,
                        compute_stream: torch.cuda.Stream) -> Dict[str, float]:
    """Prefetch predicted experts one layer ahead; on-demand fetch every miss."""
    L = len(host)
    slot_cursor = 0
    slot_guard: Dict[int, torch.cuda.Event] = {}
    misses = 0
    bytes_moved = 0

    def enqueue(layer: int, ids: List[int]) -> Tuple[Dict[int, int], torch.cuda.Event]:
        nonlocal slot_cursor, bytes_moved
        placed = {}
        ev = torch.cuda.Event()
        with torch.cuda.stream(transfer_stream):
            for eid in ids:
                s = slot_cursor % slots.n_slots
                slot_cursor += 1
                if s in slot_guard:
                    transfer_stream.wait_event(slot_guard[s])
                slots.up[s].copy_(host[layer][eid]["up"], non_blocking=True)
                slots.down[s].copy_(host[layer][eid]["down"], non_blocking=True)
                bytes_moved += 2 * (slots.up[s].numel() + slots.down[s].numel())
                placed[eid] = s
            ev.record(transfer_stream)
        return placed, ev

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Layer 0: routing is known at token start (it depends on the input), so its
    # true experts are prefetched; prediction applies to layers 1..L-1.
    placed, ready = enqueue(0, trace[0])
    for i in range(L):
        # Prefetch predicted experts for layer i+1 while layer i computes.
        next_placed = None
        if i + 1 < L:
            pred = predict_ids(trace[i + 1], hit_rate, rng)
            next_placed = enqueue(i + 1, pred)
        gates = route(x, routers[i])[1]
        # Reveal true routing for layer i; fetch anything the prediction missed.
        missing = [eid for eid in trace[i] if eid not in placed]
        if missing:
            misses += len(missing)
            fill, fill_ev = enqueue(i, missing)
            placed.update(fill)
            ready = fill_ev
        ce = torch.cuda.Event()
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(ready)
            y = torch.zeros_like(x)
            for k, eid in enumerate(trace[i]):
                s = placed[eid]
                y = y + gates[k] * expert_forward(x, slots.up[s], slots.down[s])
            x = _rms_norm(x + y)
            ce.record(compute_stream)
        for eid, s in placed.items():
            slot_guard[s] = ce
        if next_placed is not None:
            placed, ready = next_placed
        torch.cuda.current_stream().wait_event(ce)
        x = x.clone()
    torch.cuda.synchronize()
    return {
        "wall_ms": (time.perf_counter() - t0) * 1000,
        "misses": misses,
        "gb_moved": bytes_moved / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    device = torch.device("cuda")
    torch.manual_seed(0)

    hw = HardwareProfile.capture()
    e_mb = expert_bytes(args.n, args.m) / 1e6
    print("=" * 100)
    print("ITERATION 18 - EXPERT-PREFETCH MISS-PENALTY CURVE (tok/s vs predictor hit rate)")
    print("=" * 100)
    print(f"[HW] {hw.gpu_name} | {args.layers} layers x {N_EXPERTS} experts x {e_mb:.1f} MB, "
          f"top-{TOP_K}, B=1")

    host = build_moe_host_layers(args.layers, args.n, args.m, device)
    routers = make_routers(args.layers, args.n, device)
    slots = ExpertSlots(3 * TOP_K + 2, args.n, args.m, device)  # 2 layers in flight + miss room

    x0 = torch.randn(1, args.n, dtype=torch.float16, device=device)
    trace = record_routing_trace(host, slots, routers, x0)
    expert_ms = e_mb / 1e3 / 12.3 * 1000  # analytic per-expert transfer at measured PCIe

    results = []
    for h in HIT_RATES:
        rng = random.Random(42)
        transfer_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        run_predicted_token(host, slots, routers, trace, x0, h, rng,
                            transfer_stream, compute_stream)  # warmup
        runs = [run_predicted_token(host, slots, routers, trace, x0, h,
                                    random.Random(100 + t), transfer_stream, compute_stream)
                for t in range(args.tokens)]
        wall = sum(r["wall_ms"] for r in runs) / args.tokens
        misses = sum(r["misses"] for r in runs) / args.tokens
        gb = sum(r["gb_moved"] for r in runs) / args.tokens
        res = {
            "hit_rate": h,
            "wall_ms_per_token": round(wall, 1),
            "avg_misses_per_token": round(misses, 1),
            "gb_moved_per_token": round(gb, 2),
            "toks_s": round(1000 / wall, 2),
        }
        results.append(res)
        print(f"  h={h:>4.2f}: {wall:>6.1f} ms/token, {misses:>4.1f} misses, "
              f"{gb:.2f} GB moved, {res['toks_s']:.2f} tok/s")

    oracle_wall = results[0]["wall_ms_per_token"]
    print("\nAnalytic check (oracle + misses x per-expert transfer "
          f"{expert_ms:.1f} ms):")
    for r in results[1:]:
        pred = oracle_wall + r["avg_misses_per_token"] * expert_ms
        err = 100 * (pred - r["wall_ms_per_token"]) / r["wall_ms_per_token"]
        print(f"  h={r['hit_rate']:.2f}: predicted {pred:.0f} ms vs measured "
              f"{r['wall_ms_per_token']:.0f} ms ({err:+.0f}%)")

    lo, hi = results[-1], results[0]
    print(f"\nHeadline: a perfect predictor is worth "
          f"{hi['toks_s'] / lo['toks_s']:.2f}x over a chance one on this pipeline; "
          f"at the published ~80-90% hit rates the cost vs oracle is "
          f"{100 * (hi['toks_s'] - results[2]['toks_s']) / hi['toks_s']:.0f}-"
          f"{100 * (hi['toks_s'] - results[1]['toks_s']) / hi['toks_s']:.0f}%.")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 18,
        "purpose": "Measured cost curve of expert-prefetch accuracy on the MoE "
                   "streaming pipeline; turns predictor requirements into numbers.",
        "hardware": {**hw.__dict__},
        "config": {"layers": args.layers, "n": args.n, "m": args.m,
                   "experts": N_EXPERTS, "top_k": TOP_K,
                   "expert_mb": round(e_mb, 1), "tokens": args.tokens},
        "curve": results,
        "assumptions": [
            "i.i.d. miss pattern (real predictors miss in bursts; same total stall).",
            "Hit rate imposed, not learned; predictor design needs a real checkpoint.",
            "Layer-0 experts always prefetched (routing known at token start).",
        ],
    }
    with open(RESULTS_DIR / "iteration18_prefetch_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("[saved] results/iteration18_prefetch_results.json")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["hit_rate"] for r in results], [r["toks_s"] for r in results], "o-")
        ax.set_xlabel("expert-prefetch hit rate")
        ax.set_ylabel("tok/s (B=1, measured)")
        ax.set_title("MoE streaming throughput vs predictor accuracy\n"
                     f"({args.layers}L x {N_EXPERTS} experts x {e_mb:.0f} MB, top-{TOP_K}, real fetch on miss)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration18_prefetch.png", dpi=140)
        print("[plot] plots/iteration18_prefetch.png")


if __name__ == "__main__":
    main()
