#!/usr/bin/env python3
"""
Iteration 19 - Optimized INT4 Unpack: closing the measured fused-kernel gap in
stock PyTorch via weight layout + dequant-math folding.

WHY
---
Iteration 18 measured that naive INT4 unpack (strided nibble writes + full
elementwise sub/mul passes) runs at 4.8 GB/s packed, making INT4-streamed
decode dequant-bound: dense 70B INT4 = 0.14 tok/s vs the 0.35 tok/s transfer
bound, i.e. fused dequant-GEMV kernels are worth 2.54x. Before reaching for
Triton (WSL2 here has only a docker-desktop distro -- no practical dev
environment) or llama.cpp, this artifact measures how much of that gap plain
PyTorch can recover with two zero-cost-at-inference changes:

1. ROW-BLOCK PACKING (offline repack): byte (i, j) holds the nibbles of
   Wq[i, j] and Wq[i + n/2, j], so unpacking writes two CONTIGUOUS row blocks
   instead of strided columns. Probe measured 2.5x faster unpacking.
2. DEQUANT-MATH FOLDING: with per-row scales s, y = x @ (s * (Wq - 8)) equals
   (x*s) @ Wq - 8*(x*s).sum(). Scaling the [1, n] activation replaces two full
   [n, n] fp16 elementwise passes (sub_, mul_) with O(n) work; the folded-
   scale GEMV measured identical to a plain GEMV.

Measured head-to-head on the same 16-layer streamed-decode testbed as
iteration 18 (fp16 reference and naive INT4 re-measured in the same run).

ASSUMPTIONS AND LIMITATIONS
---------------------------
- Row-block packing is an offline weight-layout choice; real checkpoints can
  be repacked once at load time. Random weights here; quantization accuracy
  is out of scope (established literature -- see iteration 18 header).
- Per-row (input-channel) scales; group-wise scales would add one small
  gather per group, not measured here.
- This is still unfused (unpack materializes fp16 weights in VRAM). A truly
  fused kernel reads packed nibbles inside the GEMM; whatever gap remains
  after this artifact is the honest remaining value of such kernels.

REPRODUCE
---------
  python iteration19_opt_unpack.py --plot

Outputs: results/iteration19_opt_unpack.json, plots/iteration19_opt_unpack.png
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from iteration16_corrected_streaming import HardwareProfile, PowerSampler, layer_dim_for_gb

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


class FP16Format:
    """Reference: fp16 bytes over PCIe, plain GEMV."""
    name = "fp16"

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.staging = [torch.empty(n, n, dtype=torch.float16, device=device)
                        for _ in range(2)]

    def make_host(self, master: torch.Tensor) -> torch.Tensor:
        h = torch.empty_like(master, device="cpu", pin_memory=True)
        h.copy_(master)
        return h

    def step(self, x: torch.Tensor, s: int) -> torch.Tensor:
        return F.silu(x @ self.staging[s])


class INT4NaiveFormat:
    """Iteration 18 layout: column-interleaved nibbles, strided unpack, sub/mul."""
    name = "int4-naive"

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.staging = [torch.empty(n, n // 2, dtype=torch.uint8, device=device)
                        for _ in range(2)]
        self.work = [torch.empty(n, n, dtype=torch.float16, device=device)
                     for _ in range(2)]
        self.tmp = torch.empty(n, n // 2, dtype=torch.uint8, device=device)
        self.scale = None  # set from master stats

    def make_host(self, master: torch.Tensor) -> torch.Tensor:
        s = master.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 7.0
        self.scale = s.to(torch.float16)
        q = (master / s).round().clamp(-8, 7).to(torch.int16) + 8
        packed = (q[:, 0::2].to(torch.uint8) | (q[:, 1::2].to(torch.uint8) << 4))
        h = torch.empty_like(packed, device="cpu", pin_memory=True)
        h.copy_(packed)
        return h

    def step(self, x: torch.Tensor, s: int) -> torch.Tensor:
        st, w, t = self.staging[s], self.work[s], self.tmp
        torch.bitwise_and(st, 15, out=t)
        w[:, 0::2].copy_(t)
        torch.bitwise_right_shift(st, 4, out=t)
        w[:, 1::2].copy_(t)
        w.sub_(8.0).mul_(self.scale)
        return F.silu(x @ w)


class INT4RowBlockFormat:
    """Row-block nibbles (contiguous unpack) + scale/offset folded into activation."""
    name = "int4-rowblock"

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.staging = [torch.empty(n // 2, n, dtype=torch.uint8, device=device)
                        for _ in range(2)]
        self.work = [torch.empty(n, n, dtype=torch.float16, device=device)
                     for _ in range(2)]
        self.tmp = torch.empty(n // 2, n, dtype=torch.uint8, device=device)
        self.srow = None  # [1, n] per-input-channel scales, folded into x

    def make_host(self, master: torch.Tensor) -> torch.Tensor:
        n = self.n
        s = master.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 7.0
        self.srow = s.to(torch.float16).view(1, n)
        q = (master / s).round().clamp(-8, 7).to(torch.int16) + 8  # 0..15
        packed = (q[: n // 2].to(torch.uint8) | (q[n // 2:].to(torch.uint8) << 4))
        h = torch.empty_like(packed, device="cpu", pin_memory=True)
        h.copy_(packed)
        return h

    def step(self, x: torch.Tensor, s: int) -> torch.Tensor:
        st, w, t = self.staging[s], self.work[s], self.tmp
        n2 = self.n // 2
        torch.bitwise_and(st, 15, out=t)
        w[:n2].copy_(t)                       # contiguous uint8 -> fp16 cast
        torch.bitwise_right_shift(st, 4, out=t)
        w[n2:].copy_(t)
        xs = x * self.srow                    # fold per-row scale into activation
        y = xs @ w - 8.0 * xs.sum(dim=-1, keepdim=True)  # fold zero-point
        return F.silu(y)


def run_token(host_layers: List[torch.Tensor], fmt, x: torch.Tensor,
              transfer_stream: torch.cuda.Stream,
              compute_stream: torch.cuda.Stream) -> Dict[str, float]:
    L = len(host_layers)
    t_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(L)]
    c_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(L)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(L):
        s = i % 2
        with torch.cuda.stream(transfer_stream):
            if i >= 2:
                transfer_stream.wait_event(c_ev[i - 2][1])
            t_ev[i][0].record(transfer_stream)
            fmt.staging[s].copy_(host_layers[i], non_blocking=True)
            t_ev[i][1].record(transfer_stream)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(t_ev[i][1])
            c_ev[i][0].record(compute_stream)
            x = fmt.step(x, s)
            x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
            c_ev[i][1].record(compute_stream)
    torch.cuda.synchronize()
    return {
        "wall_ms": (time.perf_counter() - t0) * 1000,
        "transfer_ms": sum(a.elapsed_time(b) for a, b in t_ev),
        "compute_ms": sum(a.elapsed_time(b) for a, b in c_ev),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--layer-gb", type=float, default=0.25)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    device = torch.device("cuda")
    torch.manual_seed(0)

    hw = HardwareProfile.capture()
    n = layer_dim_for_gb(args.layer_gb)
    print("=" * 100)
    print("ITERATION 19 - OPTIMIZED INT4 UNPACK (row-block layout + dequant-math folding)")
    print("=" * 100)
    print(f"[HW] {hw.gpu_name} | {args.layers} layers x fp16 [{n} x {n}] "
          f"({2 * n * n / 1e9:.3f} GB fp16, {n * n / 2 / 1e9:.3f} GB int4-packed per layer)")

    formats = [FP16Format(n, device), INT4NaiveFormat(n, device),
               INT4RowBlockFormat(n, device)]
    hosts: Dict[str, List[torch.Tensor]] = {f.name: [] for f in formats}
    master = torch.empty(n, n, dtype=torch.float16, device=device)
    gen = torch.Generator(device=device).manual_seed(1)
    t0 = time.perf_counter()
    for _ in range(args.layers):
        master.normal_(0.0, n ** -0.5, generator=gen)
        for f in formats:
            hosts[f.name].append(f.make_host(master))
    del master
    torch.cuda.empty_cache()
    print(f"[MODEL] host init {time.perf_counter() - t0:.1f} s "
          f"(fp16 + both int4 layouts pinned)")

    results = []
    for f in formats:
        transfer_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        x = torch.randn(1, n, dtype=torch.float16, device=device)
        run_token(hosts[f.name], f, x, transfer_stream, compute_stream)  # warmup
        sampler = PowerSampler(interval_s=0.3)
        sampler.start()
        runs = [run_token(hosts[f.name], f, x, transfer_stream, compute_stream)
                for _ in range(args.tokens)]
        power = sampler.stop()
        avg = {k: sum(r[k] for r in runs) / args.tokens for k in runs[0]}
        gb = args.layers * hosts[f.name][0].numel() * hosts[f.name][0].element_size() / 1e9
        res = {
            "format": f.name,
            "gb_moved_per_token": round(gb, 3),
            "wall_ms_per_token": round(avg["wall_ms"], 1),
            "transfer_ms": round(avg["transfer_ms"], 1),
            "compute_ms": round(avg["compute_ms"], 1),
            "bound": "pcie" if avg["transfer_ms"] > avg["compute_ms"] else "dequant/compute",
            "toks_s": round(1000 / avg["wall_ms"], 2),
        }
        if power:
            res["energy_j_per_token"] = round(power["avg_power_w"] * avg["wall_ms"] / 1000, 1)
        results.append(res)
        print(f"  {res['format']:<14} {gb:.3f} GB/token | wall {res['wall_ms_per_token']:>6.1f} ms | "
              f"transfer {res['transfer_ms']:>6.1f} | compute {res['compute_ms']:>6.1f} | "
              f"{res['bound']}-bound | {res['toks_s']:.2f} tok/s")

    by = {r["format"]: r for r in results}
    fp16_t, naive_t, opt_t = (by[k]["toks_s"] for k in ("fp16", "int4-naive", "int4-rowblock"))
    print(f"\nUplift vs fp16: naive {naive_t / fp16_t:.2f}x -> row-block {opt_t / fp16_t:.2f}x")

    # 70B dense INT4 with the optimized stock-torch path
    pcie = by["fp16"]["gb_moved_per_token"] / (by["fp16"]["transfer_ms"] / 1000)
    opt_packed_rate = by["int4-rowblock"]["gb_moved_per_token"] / (by["int4-rowblock"]["compute_ms"] / 1000)
    gb70 = 35.3
    transfer_bound = gb70 / pcie
    compute_bound = gb70 / opt_packed_rate
    actual = max(transfer_bound, compute_bound)
    print(f"\n=== 70B dense INT4, stock torch, optimized layout ===")
    print(f"  transfer bound {1 / transfer_bound:.2f} tok/s | optimized-unpack bound "
          f"{1 / compute_bound:.2f} tok/s -> actual {1 / actual:.2f} tok/s")
    print(f"  Iteration 18 naive stock torch: 0.14 tok/s | fused-kernel bound: {1 / transfer_bound:.2f}")
    print(f"  Remaining value of truly fused kernels: {actual / transfer_bound:.2f}x "
          f"(was 2.54x before this optimization)")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 19,
        "purpose": "Close the measured fused-kernel gap in stock PyTorch via row-block "
                   "INT4 packing and dequant-math folding.",
        "hardware": {**hw.__dict__},
        "config": {"layers": args.layers, "n": n, "tokens": args.tokens},
        "measured": results,
        "seventy_b_dense_int4": {
            "transfer_bound_toks_s": round(1 / transfer_bound, 3),
            "optimized_stock_torch_toks_s": round(1 / actual, 3),
            "naive_stock_torch_toks_s_iter18": 0.14,
            "remaining_fused_kernel_value_x": round(actual / transfer_bound, 2),
        },
        "assumptions": [
            "Row-block packing is an offline repack; per-row scales folded into the activation.",
            "Unfused: fp16 weights still materialize in VRAM; fused kernels avoid that entirely.",
            "Quantization accuracy out of scope (see iteration 18 references).",
        ],
    }
    with open(RESULTS_DIR / "iteration19_opt_unpack.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\n[saved] results/iteration19_opt_unpack.json")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        names = [r["format"] for r in results]
        ax.bar(names, [r["toks_s"] for r in results])
        ax.set_ylabel("tok/s (B=1, measured)")
        ax.set_title("Streamed decode: fp16 vs naive INT4 vs row-block INT4\n"
                     "(row-block layout + scale folding, stock PyTorch)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration19_opt_unpack.png", dpi=140)
        print("[plot] plots/iteration19_opt_unpack.png")


if __name__ == "__main__":
    main()
