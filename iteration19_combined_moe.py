#!/usr/bin/env python3
"""
Iteration 19 - Combined Full-Stack MoE Pipeline: INT4 row-block experts +
expert prefetch with realistic hit rates + INT8 `_int_mm` batched compute.

WHY
---
Iterations 16-19 measured each lever in isolation:
  - MoE top-2/8 expert streaming: 4.00x fewer bytes (iter. 17)
  - prefetch miss-penalty curve: 80-90% hit rates cost 4-11% (iter. 18)
  - INT4 row-block transfer + folded dequant: 3.15x over fp16 (iter. 19a)
  - torch._int_mm INT8 batched compute: 16.4x over fp16 GEMM (iter. 18)
This artifact runs them TOGETHER in one pipeline and reports single end-to-end
numbers, because levers that each work alone routinely fight each other when
combined (shared PCIe, shared SMs, extra dequant buffers). Whatever the
combined number is, that is the project's honest full-stack state.

PIPELINE
--------
16 layers x 8 experts; each expert stored INT4 row-block packed in pinned host
RAM (16.8 MB vs 67.1 MB fp16), per-input-channel scales folded into the
activation (iteration 19a technique; for the batched path the folded activation
is dynamically quantized to INT8 and weights enter `_int_mm` as (Q-8) int8, so
no weight-side dequant pass to fp16 is needed at all).

Modes: batch 1 (fp16 GEMV compute) and batch 32 (`_int_mm` INT8 compute),
each at prefetch hit rate 1.0 (oracle) and 0.85 (mid published predictor
range), with real on-demand fetches per miss.

ASSUMPTIONS AND LIMITATIONS
---------------------------
- Batch-pooled routing; random weights (systems cost only; quantization and
  routing-prediction ACCURACY need a real checkpoint - iteration 18/19 notes).
- INT8 activations use one dynamic per-tensor scale per matmul; group-wise
  schemes add small gathers, not measured.
- 70B-class MoE numbers are scaled from measured per-byte rates and marked as
  such; the dense attention floor of a real MoE transformer is not modeled.

REPRODUCE
---------
  python iteration19_combined_moe.py --plot

Outputs: results/iteration19_combined_results.json, plots/iteration19_combined.png
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from iteration16_corrected_streaming import HardwareProfile, PowerSampler, _rms_norm
from iteration17_moe_streaming import make_routers, route, N_EXPERTS, TOP_K
from iteration18_prefetch_miss_curve import predict_ids

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


# =============================================================================
# INT4 row-block packed experts (up [n,m] + down [m,n] per expert)
# =============================================================================

def pack_rowblock(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [r, c] fp16 to INT4 row-block packed uint8 [r/2, c] + scales [1, r]."""
    r = w.shape[0]
    s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 7.0
    q = (w / s).round().clamp(-8, 7).to(torch.int16) + 8  # 0..15
    packed = (q[: r // 2].to(torch.uint8) | (q[r // 2:].to(torch.uint8) << 4))
    return packed, s.to(torch.float16).view(1, r)


def build_packed_experts(n_layers: int, n: int, m: int, device: torch.device):
    """host[layer][expert] = packed up/down (pinned); scales resident on GPU."""
    host, scales = [], []
    gen = torch.Generator(device=device).manual_seed(1)
    up_t = torch.empty(n, m, dtype=torch.float16, device=device)
    down_t = torch.empty(m, n, dtype=torch.float16, device=device)
    for _ in range(n_layers):
        lh, ls = [], []
        for _ in range(N_EXPERTS):
            up_t.normal_(0.0, n ** -0.5, generator=gen)
            down_t.normal_(0.0, m ** -0.5, generator=gen)
            up_p, up_s = pack_rowblock(up_t)
            down_p, down_s = pack_rowblock(down_t)
            hu = torch.empty_like(up_p, device="cpu", pin_memory=True); hu.copy_(up_p)
            hd = torch.empty_like(down_p, device="cpu", pin_memory=True); hd.copy_(down_p)
            lh.append({"up": hu, "down": hd})
            ls.append({"up_s": up_s, "down_s": down_s})
        host.append(lh)
        scales.append(ls)
    del up_t, down_t
    torch.cuda.empty_cache()
    return host, scales


class PackedSlots:
    """GPU staging for packed experts + shared unpack work buffers."""

    def __init__(self, n_slots: int, n: int, m: int, device: torch.device):
        self.n, self.m, self.n_slots = n, m, n_slots
        self.up = [torch.empty(n // 2, m, dtype=torch.uint8, device=device)
                   for _ in range(n_slots)]
        self.down = [torch.empty(m // 2, n, dtype=torch.uint8, device=device)
                     for _ in range(n_slots)]
        # fp16 path work buffers (one pair; experts computed sequentially)
        self.wf_up = torch.empty(n, m, dtype=torch.float16, device=device)
        self.wf_down = torch.empty(m, n, dtype=torch.float16, device=device)
        # int8 path work buffers ((Q-8) fits int8)
        self.w8_up = torch.empty(n, m, dtype=torch.int8, device=device)
        self.w8_down = torch.empty(m, n, dtype=torch.int8, device=device)
        self.tmp_up = torch.empty(n // 2, m, dtype=torch.uint8, device=device)
        self.tmp_down = torch.empty(m // 2, n, dtype=torch.uint8, device=device)


def _unpack_into(packed: torch.Tensor, tmp: torch.Tensor, out: torch.Tensor,
                 int8: bool) -> None:
    half = out.shape[0] // 2
    torch.bitwise_and(packed, 15, out=tmp)
    out[:half].copy_(tmp)
    torch.bitwise_right_shift(packed, 4, out=tmp)
    out[half:].copy_(tmp)
    if int8:
        out.sub_(8)          # (Q-8) as int8; no offset correction needed later
    # fp16 path leaves 0..15 in place; offset folded into the GEMV instead


def expert_fp16(x: torch.Tensor, slots: PackedSlots, s_idx: int,
                sc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """B=1 path: unpack to fp16, folded-scale GEMV chain (iteration 19a)."""
    _unpack_into(slots.up[s_idx], slots.tmp_up, slots.wf_up, int8=False)
    xs = x * sc["up_s"]
    h = F.silu(xs @ slots.wf_up - 8.0 * xs.sum(dim=-1, keepdim=True))
    _unpack_into(slots.down[s_idx], slots.tmp_down, slots.wf_down, int8=False)
    hs = h * sc["down_s"]
    return hs @ slots.wf_down - 8.0 * hs.sum(dim=-1, keepdim=True)


def expert_int8(x: torch.Tensor, slots: PackedSlots, s_idx: int,
                sc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """B>=17 path: unpack to (Q-8) int8, dynamic-quant activation, _int_mm."""
    _unpack_into(slots.up[s_idx], slots.tmp_up, slots.w8_up, int8=True)
    xs = x * sc["up_s"]
    sx = xs.abs().amax().clamp_min(1e-6) / 127.0
    x8 = (xs / sx).round().clamp(-127, 127).to(torch.int8)
    h = F.silu(torch._int_mm(x8, slots.w8_up).to(torch.float16) * sx)
    _unpack_into(slots.down[s_idx], slots.tmp_down, slots.w8_down, int8=True)
    hs = h * sc["down_s"]
    sh = hs.abs().amax().clamp_min(1e-6) / 127.0
    h8 = (hs / sh).round().clamp(-127, 127).to(torch.int8)
    return torch._int_mm(h8, slots.w8_down).to(torch.float16) * sh


def record_trace(host, scales, slots: PackedSlots, routers, x: torch.Tensor) -> List[List[int]]:
    trace = []
    xc = x.clone()
    with torch.no_grad():
        for i in range(len(host)):
            ids_t, gates = route(xc, routers[i])
            ids = [int(v) for v in ids_t.cpu()]
            trace.append(ids)
            y = torch.zeros_like(xc)
            for k, eid in enumerate(ids):
                slots.up[k].copy_(host[i][eid]["up"])
                slots.down[k].copy_(host[i][eid]["down"])
                y = y + gates[k] * expert_fp16(xc, slots, k, scales[i][eid])
            xc = _rms_norm(xc + y)
    return trace


def run_combined_token(host, scales, slots: PackedSlots, routers,
                       trace: List[List[int]], x: torch.Tensor, hit_rate: float,
                       rng: random.Random, use_int8: bool,
                       transfer_stream: torch.cuda.Stream,
                       compute_stream: torch.cuda.Stream) -> Dict[str, float]:
    L = len(host)
    slot_cursor = 0
    slot_guard: Dict[int, torch.cuda.Event] = {}
    misses = 0
    bytes_moved = 0
    expert_fn = expert_int8 if use_int8 else expert_fp16

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
                bytes_moved += slots.up[s].numel() + slots.down[s].numel()
                placed[eid] = s
            ev.record(transfer_stream)
        return placed, ev

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    placed, ready = enqueue(0, trace[0])
    for i in range(L):
        next_placed = None
        if i + 1 < L:
            pred = predict_ids(trace[i + 1], hit_rate, rng)
            next_placed = enqueue(i + 1, pred)
        gates = route(x, routers[i])[1]
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
                y = y + gates[k] * expert_fn(x, slots, placed[eid], scales[i][eid])
            x = _rms_norm(x + y)
            ce.record(compute_stream)
        for eid, s in placed.items():
            slot_guard[s] = ce
        if next_placed is not None:
            placed, ready = next_placed
        torch.cuda.current_stream().wait_event(ce)
        x = x.clone()
    torch.cuda.synchronize()
    return {"wall_ms": (time.perf_counter() - t0) * 1000,
            "misses": misses, "gb_moved": bytes_moved / 1e9}


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
    expert_mb = (args.n * args.m + args.m * args.n) / 2 / 1e6  # int4 bytes
    print("=" * 100)
    print("ITERATION 19 - COMBINED FULL-STACK MoE (INT4 row-block + prefetch + _int_mm)")
    print("=" * 100)
    print(f"[HW] {hw.gpu_name} | {args.layers} layers x {N_EXPERTS} experts x "
          f"{expert_mb:.1f} MB int4-packed (vs 67.1 MB fp16), top-{TOP_K}")
    print(f"[REF] iteration 17 fp16 oracle B=1 measured 5.69 tok/s; fp16 dense-equiv 1.41 tok/s")

    t0 = time.perf_counter()
    host, scales = build_packed_experts(args.layers, args.n, args.m, device)
    routers = make_routers(args.layers, args.n, device)
    slots = PackedSlots(3 * TOP_K + 2, args.n, args.m, device)
    print(f"[MODEL] packed host init {time.perf_counter() - t0:.1f} s "
          f"({args.layers * N_EXPERTS * expert_mb / 1e3:.1f} GB pinned)")

    x1 = torch.randn(1, args.n, dtype=torch.float16, device=device)
    trace = record_trace(host, scales, slots, routers, x1)

    results = []
    for batch, use_int8 in ((1, False), (32, True)):
        x = torch.randn(batch, args.n, dtype=torch.float16, device=device)
        for h in (1.0, 0.85):
            transfer_stream = torch.cuda.Stream()
            compute_stream = torch.cuda.Stream()
            run_combined_token(host, scales, slots, routers, trace, x, h,
                               random.Random(0), use_int8, transfer_stream, compute_stream)
            sampler = PowerSampler(interval_s=0.3)
            sampler.start()
            runs = [run_combined_token(host, scales, slots, routers, trace, x, h,
                                       random.Random(100 + t), use_int8,
                                       transfer_stream, compute_stream)
                    for t in range(args.tokens)]
            power = sampler.stop()
            wall = sum(r["wall_ms"] for r in runs) / args.tokens
            res = {
                "batch": batch,
                "compute": "_int_mm int8" if use_int8 else "fp16 GEMV",
                "hit_rate": h,
                "wall_ms_per_token": round(wall, 1),
                "avg_misses": round(sum(r["misses"] for r in runs) / args.tokens, 1),
                "gb_moved_per_token": round(sum(r["gb_moved"] for r in runs) / args.tokens, 3),
                "toks_s_single_stream": round(1000 / wall, 2),
                "toks_s_aggregate": round(batch * 1000 / wall, 1),
            }
            if power:
                res["energy_j_per_token"] = round(power["avg_power_w"] * wall / 1000, 1)
            results.append(res)
            print(f"  B={batch:>2} ({res['compute']:<12}) h={h:.2f}: "
                  f"{wall:>6.1f} ms/token, {res['gb_moved_per_token']:.3f} GB, "
                  f"{res['avg_misses']:.1f} misses, "
                  f"{res['toks_s_single_stream']:.2f} tok/s single / "
                  f"{res['toks_s_aggregate']:.1f} aggregate"
                  + (f", {res['energy_j_per_token']} J/token" if "energy_j_per_token" in res else ""))

    b1 = next(r for r in results if r["batch"] == 1 and r["hit_rate"] == 0.85)
    print(f"\nCombined stack vs iteration-17 fp16 oracle (5.69 tok/s): "
          f"{b1['toks_s_single_stream'] / 5.69:.2f}x at realistic prefetch (h=0.85)")
    print(f"Combined stack vs iteration-17 fp16 dense-equivalent (1.41 tok/s): "
          f"{b1['toks_s_single_stream'] / 1.41:.2f}x")

    # 70B-class MoE (INT4 storage: 35.3 GB total, 25% active = 8.8 GB/token)
    gb_active_70b = 35.3 * (TOP_K / N_EXPERTS)
    per_gb_ms = b1["wall_ms_per_token"] / b1["gb_moved_per_token"]
    tok_s_70b = 1000 / (gb_active_70b * per_gb_ms)
    print(f"\n70B-class MoE INT4 (scaled from measured {per_gb_ms:.1f} ms/GB at h=0.85, B=1): "
          f"{gb_active_70b:.1f} GB/token -> {tok_s_70b:.2f} tok/s single-stream")
    print("(before the dense attention floor of a real MoE transformer; see assumptions)")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 19,
        "purpose": "All measured levers in one pipeline: INT4 row-block expert transfer, "
                   "prefetch with realistic hit rate, _int_mm batched compute.",
        "hardware": {**hw.__dict__},
        "config": {"layers": args.layers, "n": args.n, "m": args.m,
                   "experts": N_EXPERTS, "top_k": TOP_K,
                   "expert_int4_mb": round(expert_mb, 1), "tokens": args.tokens},
        "measured": results,
        "references_measured_previously": {
            "fp16_moe_oracle_b1_toks_s_iter17": 5.69,
            "fp16_dense_equiv_b1_toks_s_iter17": 1.41,
        },
        "seventy_b_moe_int4_scaled": {
            "gb_active_per_token": round(gb_active_70b, 1),
            "toks_s_single_stream_h085": round(tok_s_70b, 2),
        },
        "assumptions": [
            "Batch-pooled routing; random weights; systems cost only.",
            "Per-tensor dynamic activation scales on the int8 path.",
            "70B numbers scaled from measured ms/GB; dense attention floor not modeled.",
        ],
    }
    with open(RESULTS_DIR / "iteration19_combined_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\n[saved] results/iteration19_combined_results.json")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = [f"B={r['batch']}\nh={r['hit_rate']}" for r in results]
        ax.bar(labels, [r["toks_s_aggregate"] for r in results])
        ax.axhline(5.69, color="gray", ls="--", lw=1, label="iter17 fp16 oracle B=1")
        ax.set_ylabel("tok/s aggregate (measured)")
        ax.set_title("Combined stack: INT4 row-block + prefetch + _int_mm")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration19_combined.png", dpi=140)
        print("[plot] plots/iteration19_combined.png")


if __name__ == "__main__":
    main()
