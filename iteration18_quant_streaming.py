#!/usr/bin/env python3
"""
Iteration 18 - Quantized Transfer Streaming: INT8/INT4 over PCIe, dequant cost
measured, and the INT8 DP4A path for batched decode.

WHY
---
Every 70B projection in iterations 16-17 assumes INT4-packed weights are what
moves over PCIe (35.3 GB, 0.441 GB/layer). The measured testbeds, however,
streamed fp16 and the projections implicitly assumed that turning packed bytes
into computable weights is free. This artifact closes that gap by measuring,
end-to-end on the streaming pipeline:

1. fp16 transfer -> GEMV                       (iteration 16 baseline, re-measured)
2. INT8 transfer -> GPU dequant -> fp16 GEMV   (2x fewer PCIe bytes)
3. INT4-packed transfer -> GPU unpack -> GEMV  (4x fewer PCIe bytes)

and, for batched decode:

4. torch._int_mm (INT8 DP4A) vs the stock fp16 GEMM path at B>=32
   (iteration 17 measured the fp16 path at ~7.5 GB/s weight-read; the probe
   for this iteration measured _int_mm at ~61 GB/s int8 weight-read = ~16x
   more weights consumed per second, in stock PyTorch, no custom kernels).

HONEST FRAMING
--------------
- Dequant/unpack kernels available in stock PyTorch are unfused elementwise
  passes. If their cost exceeds the PCIe bytes they save, the INT4 "win"
  evaporates on this software stack -- whatever the measurement says is what
  the docs will say. Fused dequant-GEMV kernels (llama.cpp-class) read packed
  weights directly and are the known fix; this artifact quantifies exactly how
  much such kernels are worth on this machine.
- Quantization ACCURACY is out of scope here: INT8/INT4 weight quality for
  LLMs is established literature (LLM.int8() arXiv:2208.07339; GPTQ
  arXiv:2210.17323; AWQ arXiv:2306.00978). This artifact measures systems
  cost only; weights are random and per-row scales are carried through the
  math so the compute is representative.
- torch._int_mm requires M > 16 (measured constraint on this build), so B=1/8
  decode cannot use it directly; B=8 is measured with padding to M=17 and
  reported as such.

REPRODUCE
---------
  python iteration18_quant_streaming.py --plot

Outputs: results/iteration18_quant_results.json, plots/iteration18_quant.png
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from iteration16_corrected_streaming import (
    HardwareProfile, PowerSampler, layer_dim_for_gb,
)

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


# =============================================================================
# Kernel micro-benchmarks
# =============================================================================

def bench_ms(fn, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def sweep_kernels(n: int, device: torch.device) -> List[Dict]:
    """Weight-consumption rate (billions of weights/s) per kernel path."""
    out = []
    Wf = torch.randn(n, n, dtype=torch.float16, device=device) * (n ** -0.5)
    W8 = torch.randint(-127, 128, (n, n), dtype=torch.int8, device=device)
    weights = n * n

    for B in (1, 8, 32):
        x = torch.randn(B, n, dtype=torch.float16, device=device)
        ms = bench_ms(lambda: x @ Wf)
        out.append({"path": "fp16 matmul", "batch": B, "ms": round(ms, 3),
                    "gweights_per_s": round(weights / 1e9 / (ms / 1000), 1)})

    for B in (8, 32):
        M = max(B, 17)  # _int_mm constraint: M > 16
        x8 = torch.randint(-127, 128, (M, n), dtype=torch.int8, device=device)
        try:
            ms = bench_ms(lambda: torch._int_mm(x8, W8))
            out.append({"path": f"_int_mm int8 (M={M})", "batch": B, "ms": round(ms, 3),
                        "gweights_per_s": round(weights / 1e9 / (ms / 1000), 1)})
        except Exception as e:
            out.append({"path": f"_int_mm int8 (M={M})", "batch": B,
                        "error": str(e)[:80]})

    del Wf, W8
    torch.cuda.empty_cache()
    return out


# =============================================================================
# Streamed formats. One layer = one [n x n] weight matrix; per-row scales are
# tiny and GPU-resident. Dequant/unpack kernels use preallocated buffers only.
# =============================================================================

class FormatFP16:
    name = "fp16"
    bytes_per_layer_factor = 2.0  # bytes per weight over PCIe

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.staging = [torch.empty(n, n, dtype=torch.float16, device=device)
                        for _ in range(2)]

    def make_host(self, gen_w: torch.Tensor) -> torch.Tensor:
        h = torch.empty_like(gen_w, device="cpu", pin_memory=True)
        h.copy_(gen_w)
        return h

    def dequant(self, slot_idx: int) -> torch.Tensor:
        return self.staging[slot_idx]  # nothing to do


class FormatINT8:
    name = "int8"
    bytes_per_layer_factor = 1.0

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.staging = [torch.empty(n, n, dtype=torch.int8, device=device)
                        for _ in range(2)]
        self.work = [torch.empty(n, n, dtype=torch.float16, device=device)
                     for _ in range(2)]
        self.scale = (torch.rand(n, 1, dtype=torch.float16, device=device) * 0.01)

    def make_host(self, gen_w: torch.Tensor) -> torch.Tensor:
        q = (gen_w / 0.005).round().clamp(-127, 127).to(torch.int8)
        h = torch.empty_like(q, device="cpu", pin_memory=True)
        h.copy_(q)
        return h

    def dequant(self, slot_idx: int) -> torch.Tensor:
        w = self.work[slot_idx]
        w.copy_(self.staging[slot_idx])  # int8 -> fp16 cast into preallocated buf
        w.mul_(self.scale)
        return w


class FormatINT4:
    name = "int4-packed"
    bytes_per_layer_factor = 0.5

    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.staging = [torch.empty(n, n // 2, dtype=torch.uint8, device=device)
                        for _ in range(2)]
        self.work = [torch.empty(n, n, dtype=torch.float16, device=device)
                     for _ in range(2)]
        self.tmp = torch.empty(n, n // 2, dtype=torch.uint8, device=device)
        self.scale = (torch.rand(n, 1, dtype=torch.float16, device=device) * 0.02)

    def make_host(self, gen_w: torch.Tensor) -> torch.Tensor:
        q = (gen_w / 0.01).round().clamp(-8, 7).to(torch.int16) + 8  # 0..15
        lo = q[:, 0::2].to(torch.uint8)
        hi = q[:, 1::2].to(torch.uint8)
        packed = (lo | (hi << 4))
        h = torch.empty_like(packed, device="cpu", pin_memory=True)
        h.copy_(packed)
        return h

    def dequant(self, slot_idx: int) -> torch.Tensor:
        s, w, t = self.staging[slot_idx], self.work[slot_idx], self.tmp
        torch.bitwise_and(s, 15, out=t)
        w[:, 0::2].copy_(t)          # uint8 -> fp16 strided cast
        torch.bitwise_right_shift(s, 4, out=t)
        w[:, 1::2].copy_(t)
        w.sub_(8.0).mul_(self.scale)
        return w


def run_streamed_token(host_layers: List[torch.Tensor], fmt, x: torch.Tensor,
                       transfer_stream: torch.cuda.Stream,
                       compute_stream: torch.cuda.Stream) -> Dict[str, float]:
    """Double-buffered: copy layer i+1 packed bytes while dequant+GEMV layer i."""
    L = len(host_layers)
    t_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(L)]
    c_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(L)]
    bytes_moved = 0

    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    for i in range(L):
        s = i % 2
        with torch.cuda.stream(transfer_stream):
            if i >= 2:
                transfer_stream.wait_event(c_ev[i - 2][1])  # staging+work slot free
            t_ev[i][0].record(transfer_stream)
            fmt.staging[s].copy_(host_layers[i], non_blocking=True)
            bytes_moved += host_layers[i].numel() * host_layers[i].element_size()
            t_ev[i][1].record(transfer_stream)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(t_ev[i][1])
            c_ev[i][0].record(compute_stream)
            W = fmt.dequant(s)
            x = F.silu(x @ W)
            x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
            c_ev[i][1].record(compute_stream)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_t0) * 1000
    return {
        "wall_ms": wall_ms,
        "transfer_ms": sum(a.elapsed_time(b) for a, b in t_ev),
        "compute_ms": sum(a.elapsed_time(b) for a, b in c_ev),
        "gb_moved": bytes_moved / 1e9,
    }


def measure_format(fmt, host_layers: List[torch.Tensor], n: int, n_tokens: int,
                   device: torch.device) -> Dict:
    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    x = torch.randn(1, n, dtype=torch.float16, device=device)
    run_streamed_token(host_layers, fmt, x, transfer_stream, compute_stream)  # warmup
    sampler = PowerSampler(interval_s=0.3)
    sampler.start()
    runs = [run_streamed_token(host_layers, fmt, x, transfer_stream, compute_stream)
            for _ in range(n_tokens)]
    power = sampler.stop()
    avg = {k: sum(r[k] for r in runs) / n_tokens for k in runs[0]}
    res = {
        "format": fmt.name,
        "gb_moved_per_token": round(avg["gb_moved"], 3),
        "wall_ms_per_token": round(avg["wall_ms"], 1),
        "transfer_ms": round(avg["transfer_ms"], 1),
        "compute_ms_dequant_plus_gemv": round(avg["compute_ms"], 1),
        "bound": "pcie" if avg["transfer_ms"] > avg["compute_ms"] else "dequant/compute",
        "toks_s": round(1000 / avg["wall_ms"], 2),
    }
    if power:
        res["energy_j_per_token"] = round(
            power["avg_power_w"] * avg["wall_ms"] / 1000, 1)
    return res


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--layer-gb", type=float, default=0.25, help="fp16 layer size")
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
    print("ITERATION 18 - QUANTIZED TRANSFER STREAMING (INT8/INT4 over PCIe, dequant measured)")
    print("=" * 100)
    print(f"[HW] {hw.gpu_name} | VRAM {hw.vram_gb} GB | host avail {hw.host_avail_gb} GB")
    print(f"[MODEL] {args.layers} layers x fp16 [{n} x {n}] "
          f"({2 * n * n / 1e9:.3f} GB/layer fp16; int8 = half; int4 = quarter)")

    print("\n--- Kernel micro-bench: weights consumed per second ---")
    kernels = sweep_kernels(8192, device)
    for k in kernels:
        if "error" in k:
            print(f"  B={k['batch']:>2} {k['path']:<22} ERROR: {k['error']}")
        else:
            print(f"  B={k['batch']:>2} {k['path']:<22} {k['ms']:>8.3f} ms  "
                  f"{k['gweights_per_s']:>7.1f} Gweights/s")

    # Build one fp16 master per layer; each format derives its host bytes from it.
    formats = [FormatFP16(n, device), FormatINT8(n, device), FormatINT4(n, device)]
    hosts: Dict[str, List[torch.Tensor]] = {f.name: [] for f in formats}
    gen = torch.Generator(device=device).manual_seed(1)
    t0 = time.perf_counter()
    master = torch.empty(n, n, dtype=torch.float16, device=device)
    for _ in range(args.layers):
        master.normal_(0.0, n ** -0.5, generator=gen)
        for f in formats:
            hosts[f.name].append(f.make_host(master))
    del master
    torch.cuda.empty_cache()
    total_pinned = sum(h.numel() * h.element_size()
                       for hs in hosts.values() for h in hs) / 1e9
    print(f"\n[MODEL] host init: {time.perf_counter() - t0:.1f} s, "
          f"{total_pinned:.1f} GB pinned across all three formats")

    print("\n--- Streamed decode, B=1 (transfer || dequant+GEMV, double-buffered) ---")
    results = []
    for f in formats:
        res = measure_format(f, hosts[f.name], n, args.tokens, device)
        results.append(res)
        print(f"  {res['format']:<12} {res['gb_moved_per_token']:.3f} GB/token | "
              f"wall {res['wall_ms_per_token']:>6.1f} ms | transfer {res['transfer_ms']:>6.1f} | "
              f"dequant+GEMV {res['compute_ms_dequant_plus_gemv']:>6.1f} | "
              f"{res['bound']}-bound | {res['toks_s']:.2f} tok/s"
              + (f" | {res['energy_j_per_token']} J/token" if "energy_j_per_token" in res else ""))

    fp16_res = next(r for r in results if r["format"] == "fp16")
    print("\nUplift vs fp16 transfer: " + ", ".join(
        f"{r['format']}: {r['toks_s'] / fp16_res['toks_s']:.2f}x"
        for r in results if r["format"] != "fp16"))

    # 70B implications with these measured (unfused, stock-torch) kernels
    pcie = fp16_res["gb_moved_per_token"] / (fp16_res["transfer_ms"] / 1000)
    int4_res = next(r for r in results if r["format"] == "int4-packed")
    int4_gb_packed = int4_res["gb_moved_per_token"]
    unpack_rate_packed_gbs = int4_gb_packed / (int4_res["compute_ms_dequant_plus_gemv"] / 1000)
    gb70 = 35.3  # 70B INT4-packed
    transfer_bound_s = gb70 / pcie
    dequant_bound_s = gb70 / unpack_rate_packed_gbs
    actual_s = max(transfer_bound_s, dequant_bound_s)
    print("\n=== 70B-class dense INT4 streaming with STOCK-TORCH kernels ===")
    print(f"  transfer bound : {gb70:.1f} GB / {pcie:.1f} GB/s = {transfer_bound_s:.2f} s/token "
          f"({1 / transfer_bound_s:.2f} tok/s)")
    print(f"  unpack bound   : {gb70:.1f} GB / {unpack_rate_packed_gbs:.1f} GB/s = "
          f"{dequant_bound_s:.2f} s/token ({1 / dequant_bound_s:.2f} tok/s)")
    print(f"  -> stock-torch INT4 streaming is {'unpack' if dequant_bound_s > transfer_bound_s else 'transfer'}-bound: "
          f"{1 / actual_s:.2f} tok/s. Fused dequant-GEMV kernels (llama.cpp-class) are "
          f"worth {actual_s / transfer_bound_s:.2f}x here -- that is the quantified kernel gap.")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 18,
        "purpose": "Measure quantized-transfer streaming and dequant kernel costs that "
                   "iterations 16-17 projections implicitly assumed away.",
        "hardware": {**hw.__dict__},
        "config": {"layers": args.layers, "n": n, "tokens": args.tokens},
        "kernel_bench": kernels,
        "streamed_formats_b1": results,
        "seventy_b_dense_int4_stock_torch": {
            "transfer_bound_toks_s": round(1 / transfer_bound_s, 3),
            "unpack_bound_toks_s": round(1 / dequant_bound_s, 3),
            "actual_stock_torch_toks_s": round(1 / actual_s, 3),
            "fused_kernel_value_x": round(actual_s / transfer_bound_s, 2),
        },
        "assumptions": [
            "Random weights; quantization ACCURACY out of scope (established literature: "
            "LLM.int8(), GPTQ, AWQ). Systems cost only.",
            "Per-row scales resident on GPU; dequant into preallocated buffers (no allocs "
            "in the hot loop).",
            "_int_mm requires M>16 on this build; B<=16 uses padding where noted.",
        ],
    }
    with open(RESULTS_DIR / "iteration18_quant_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\n[saved] results/iteration18_quant_results.json")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        names = [r["format"] for r in results]
        axes[0].bar(names, [r["transfer_ms"] for r in results], label="transfer")
        axes[0].bar(names, [r["compute_ms_dequant_plus_gemv"] for r in results],
                    bottom=[r["transfer_ms"] for r in results], label="dequant+GEMV (serialized view)")
        axes[0].set_ylabel("ms per token (component sums)")
        axes[0].set_title("Where the time goes by transfer format")
        axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
        axes[1].bar(names, [r["toks_s"] for r in results])
        axes[1].set_ylabel("tok/s (B=1, measured wall)")
        axes[1].set_title("Streamed decode by format (overlapped pipeline)")
        axes[1].grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration18_quant.png", dpi=140)
        print("[plot] plots/iteration18_quant.png")


if __name__ == "__main__":
    main()
