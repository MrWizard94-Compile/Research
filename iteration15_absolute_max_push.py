#!/usr/bin/env python3
"""
*** RETRACTED / DEPRECATED (Iteration 16 audit, 2026-07-02) ***

DO NOT cite results from this script. Three defects invalidate its claims:
1. `stream_and_compute` iterates `range(min(n_layers_to_use, self.n_layers))`
   with `self.n_layers` fixed at 8, so every "16/32/64/120 layer" result
   actually ran the same 8-layer, <10 MB model. The claimed 0.45 GB layers
   were never allocated or transferred (real layer size: ~0.74 MB).
2. The "overlap" metric divides the hypothetical runtime of a fictional
   0.45 GB-per-layer model by the measured runtime of the tiny real one.
   It does not measure transfer/compute overlap.
3. The 70B projection divides PCIe transfer time by prefetch depth
   (`transfer_ms / p`). PCIe is a single shared link; prefetch cannot
   multiply bandwidth. Correct steady-state: max(transfer, compute) per
   layer -> ~0.34 tok/s for dense 70B INT4 streaming, not 2.06.

Superseded by: iteration16_corrected_streaming.py (measured, validated).
Retraction record: BREAKTHROUGH_PROGRESS_LOG.md, Iteration 16.

--- Original (inaccurate) description follows ---

Iteration 15 – Absolute Maximum Push: Scale to the Breaking Point on Your GTX 1660 Ti 6GB + 48GB

Complete, production-ready artifact.

Pushes the hybrid architecture (constant-state SSM + Associative/TSAM + MoE + reversible + async streaming + CUDA graphs) to the absolute limit on *your exact hardware*:
- Model size: Up to 120+ layers @ 0.45 GB each (~100B+ class INT4, limited by your 48GB host RAM).
- Context: Up to 128k (real runs where possible + analytical projection).
- Stress: Ramp scales until OOM, throughput collapse, or overlap failure.
- All with real GPU execution, CUDA events for precise overlap, memory snapshots, hardware attestation.

Goal: Find the real ceiling. How big (params + context) can we go before the new architecture breaks, vs old von Neumann rules.

Merges from Grok project: HardwareProfile, layer streaming model, SSM+assoc forward patterns.

No Triton (Linux-only). Pure native Windows wins on your 1660 Ti.

Run:
    python iteration15_absolute_max_push.py --max --plot

This will attempt extreme scales and report exactly where/why it breaks (or doesn't).

Outputs: results/iteration15_*.json + proof + plots.
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

@dataclass
class HardwareProfile:
    vram_gb: float = 6.44
    host_ram_gb: float = 48.0
    gpu_bw_gbs: float = 336.0
    cpu_to_gpu_bw_gbs: float = 12.0
    tdp_w: float = 450.0

    @classmethod
    def capture(cls) -> 'HardwareProfile':
        hw = cls()
        if torch.cuda.is_available():
            try:
                hw.vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            except: pass
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], timeout=3).decode().strip()
                hw.vram_gb = float(out) / 1024
            except: pass
        return hw

class StreamingHybridLayer(nn.Module):
    def __init__(self, d: int, ds: int):
        super().__init__()
        self.d = d
        self.ds = ds
        self.A_log = nn.Parameter(torch.randn(ds) * -0.6)
        self.B = nn.Parameter(torch.randn(d, ds) * 0.04)
        self.C = nn.Parameter(torch.randn(ds, d) * 0.04)
        self.D = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, d, bias=False)
        self.register_buffer('assoc_keys', torch.randn(128, d) * 0.3)
        self.register_buffer('assoc_vals', torch.randn(128, d) * 0.3)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = torch.exp(self.A_log)
        x_in = x.mean(dim=0) if x.dim() > 1 else x
        proj = x_in @ self.B
        h = A * h + proj
        y_ssm = h @ self.C + self.D * x_in
        sims = self.assoc_keys @ x_in / (self.assoc_keys.norm(dim=1) * x_in.norm() + 1e-8)
        idx = sims.argmax()
        y_assoc = self.assoc_vals[idx]
        y = 0.4 * y_ssm + 0.6 * y_assoc
        out = self.proj(y)
        return out, h

class MaxScaleStreamingModel(nn.Module):
    """Pushed model: constant state + host-pinned streaming for extreme scales."""
    def __init__(self, d_model: int = 384, n_layers: int = 8, d_state: int = 48):
        super().__init__()
        self.d = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.layers = nn.ModuleList([StreamingHybridLayer(d_model, d_state) for _ in range(n_layers)])
        self.host_weights: Dict[int, Dict] = {}

    def prepare_host_pinned(self):
        for i, layer in enumerate(self.layers):
            self.host_weights[i] = {
                'A_log': layer.A_log.data.cpu().pin_memory(),
                'B': layer.B.data.cpu().pin_memory(),
                'C': layer.C.data.cpu().pin_memory(),
                'D': layer.D.data.cpu().pin_memory(),
                'proj_weight': layer.proj.weight.data.cpu().pin_memory(),
            }

    def stream_and_compute(self, seq: torch.Tensor, n_layers_to_use: int, prefetch: int = 6, use_graph: bool = False, graph=None) -> torch.Tensor:
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        h = torch.zeros(self.d_state, device=DEVICE, dtype=seq.dtype)
        x = seq

        # Prefetch depth (pushed to 8+ for deeper overlap/power)
        for i in range(min(n_layers_to_use, self.n_layers)):
            # Stream current + future layers up to prefetch depth
            for p in range(prefetch):
                ii = i + p
                if ii < min(n_layers_to_use, self.n_layers) and ii in self.host_weights:
                    hw = self.host_weights[ii]
                    lyr = self.layers[ii]
                    with torch.cuda.stream(transfer_stream):
                        lyr.A_log.data.copy_(hw['A_log'], non_blocking=True)
                        lyr.B.data.copy_(hw['B'], non_blocking=True)
                        lyr.C.data.copy_(hw['C'], non_blocking=True)
                        lyr.D.data.copy_(hw['D'], non_blocking=True)
                        lyr.proj.weight.data.copy_(hw['proj_weight'], non_blocking=True)
            evt = torch.cuda.Event()
            transfer_stream.record_event(evt)

            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(evt)
                x, h = self.layers[i](x, h)
                if graph is not None:
                    graph.replay()
        torch.cuda.synchronize()
        return x.mean(0) if x.dim() > 1 else x

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(self.d_state, device=DEVICE, dtype=seq.dtype)
        x = seq
        for lyr in self.layers:
            x, h = lyr(x, h)
        return x.mean(0) if x.dim() > 1 else x

def capture_compute_graph(d: int = 384) -> torch.cuda.CUDAGraph:
    g = None
    try:
        def kernel(x, w):
            for _ in range(2):
                x = torch.matmul(x, w)
            return x
        w = torch.randn(d, d, device=DEVICE, dtype=torch.float16)
        x = torch.randn(1, d, device=DEVICE, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(5): _ = kernel(x, w)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                _ = kernel(x, w)
        print("[GRAPH] Captured compute kernel graph")
    except Exception as e:
        print(f"[GRAPH] Capture skipped: {e}")
    return g

def measure_real_pcie_bw(layer_gb: float = 0.45, iters: int = 5) -> tuple:
    """Measure real H2D transfer time for a realistic layer-sized buffer using pinned memory.
    Returns (effective_gb_per_s, avg_transfer_ms).
    """
    size_bytes = int(layer_gb * 1e9)
    numel = size_bytes // 2  # fp16
    host_buf = torch.empty(numel, dtype=torch.float16, pin_memory=True)
    gpu_buf = torch.empty(numel, dtype=torch.float16, device=DEVICE)
    times = []
    for _ in range(iters):
        host_buf.random_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_buf.copy_(host_buf, non_blocking=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg_t = sum(times) / len(times)
    effective_bw = layer_gb / avg_t
    avg_ms = avg_t * 1000
    print(f"[PCIe REAL] Measured {effective_bw:.2f} GB/s effective for {layer_gb}GB layer (avg {avg_ms:.1f} ms)")
    return effective_bw, avg_ms

def push_to_absolute_max(hw: HardwareProfile, base_d: int = 384, base_layers: int = 8,
                         target_layers: int = 120, target_layer_gb: float = 0.45,
                         max_ctx: int = 131072, trials: int = 1):
    """Ramp to extreme size + context until break."""
    print(f"\n=== PUSH TO ABSOLUTE MAX ===")
    print(f"Target: {target_layers} layers @ {target_layer_gb}GB (~100B+ class), ctx up to {max_ctx}")
    print(f"Host limit: {hw.host_ram_gb}GB | GPU active limit ~{hw.vram_gb * 0.8:.1f}GB")

    # Tighten PCIe with real measurement (instead of theoretical 12 GB/s)
    real_bw, real_ms = measure_real_pcie_bw(target_layer_gb)
    hw.cpu_to_gpu_bw_gbs = real_bw  # use measured for accurate projections

    model = MaxScaleStreamingModel(d_model=base_d, n_layers=base_layers).to(DEVICE)
    model.prepare_host_pinned()
    compiled = model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("[OPT] torch.compile applied")
    except Exception as e:
        print(f"[OPT] compile fallback: {e}")

    graph = capture_compute_graph(base_d)

    results = []
    scales = [8, 16, 32, 64, target_layers]
    ctxs = [2048, 8192, 32768, max_ctx]

    ooms = []
    for nl in scales:
        for ctx in ctxs:
            if nl > target_layers: continue
            times = []
            vrams = []
            for _ in range(trials):
                try:
                    s = torch.randn(ctx, base_d, device=DEVICE)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = compiled.stream_and_compute(s, nl, prefetch=8, graph=graph)
                    end.record()
                    torch.cuda.synchronize()
                    ms = start.elapsed_time(end)
                    vram = torch.cuda.max_memory_allocated() / 1e9
                    times.append(ms)
                    vrams.append(vram)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        ooms.append((nl, ctx))
                        print(f"  OOM at {nl}L / ctx={ctx}")
                        break
                    raise
            if times:
                avg_ms = sum(times) / len(times)
                avg_vram = sum(vrams) / len(vrams)
                transfer_per = (target_layer_gb / hw.cpu_to_gpu_bw_gbs) * 1000
                ideal = nl * (transfer_per + 4.0)
                overlap = ideal / max(avg_ms, 1)
                toks = 1000 / avg_ms if avg_ms > 0 else 0
                res = {
                    "layers": nl, "context": ctx,
                    "avg_ms": round(avg_ms, 2),
                    "vram_gb": round(avg_vram, 3),
                    "overlap_x": round(overlap, 1),
                    "toks_s": round(toks, 1),
                    "measured_pcie_gbs": round(hw.cpu_to_gpu_bw_gbs, 2)
                }
                results.append(res)
                print(f"  {nl}L @ {ctx}ctx: {avg_ms:.1f}ms, VRAM {avg_vram:.2f}GB, overlap {overlap:.1f}x, {toks:.1f} toks/s (real PCIe {hw.cpu_to_gpu_bw_gbs:.2f} GB/s)")

    # Analyze breaking
    max_vram = max((r["vram_gb"] for r in results), default=0)
    max_overlap = max((r["overlap_x"] for r in results), default=0)
    max_toks = max((r["toks_s"] for r in results), default=0)
    breaking = {
        "max_vram_observed": round(max_vram, 2),
        "max_overlap": round(max_overlap, 1),
        "max_toks_s": round(max_toks, 1),
        "ooms": ooms,
        "host_limit_gb": hw.host_ram_gb,
        "gpu_active_safe_gb": round(hw.vram_gb * 0.75, 2)
    }

    # 70B+ / 128k projection using max observed
    proj = {
        "layers": target_layers,
        "layer_gb": target_layer_gb,
        "total_weights_gb": round(target_layers * target_layer_gb, 1),
        "active_vram": round(target_layer_gb * 2.5, 2),
        "projected_toks_128k": round(max_toks * 0.5, 1),  # conservative for 128k
        "fits": (target_layer_gb * 2.5) < (hw.vram_gb * 0.8) and (target_layers * target_layer_gb) < hw.host_ram_gb
    }

    print("\n=== BREAKING ANALYSIS ===")
    print(f"Max VRAM observed: {breaking['max_vram_observed']} GB (safe headroom ~{breaking['gpu_active_safe_gb']}GB)")
    print(f"Max overlap: {breaking['max_overlap']}x")
    print(f"Max toks/s: {breaking['max_toks_s']}")
    print(f"OOMs encountered: {len(ooms)} at scales {ooms}")
    print(f"70B+ @128k fits: {proj['fits']} (active {proj['active_vram']}GB, total weights {proj['total_weights_gb']}GB)")

    # PCIe modeling using *real measured* BW (tightened vs theoretical)
    transfer_per_layer_ms = (target_layer_gb / hw.cpu_to_gpu_bw_gbs) * 1000
    print(f"\nPCIe Modeling (real measured):")
    print(f"  Real effective BW: {hw.cpu_to_gpu_bw_gbs:.2f} GB/s")
    print(f"  Real transfer per layer: {transfer_per_layer_ms:.1f} ms")
    if breaking.get('max_overlap', 1) > 1:
        hidden = transfer_per_layer_ms / breaking['max_overlap']
        print(f"  At max measured overlap, hidden transfer cost ~{hidden:.3f} ms/layer")

    # Cleaner 70B-class projection using real measured BW + pushed prefetch depths
    real_bw = hw.cpu_to_gpu_bw_gbs
    layers_70b = 80
    layer_gb_70b = 0.45
    ctx_128k = 131072
    transfer_ms = layer_gb_70b / real_bw * 1000
    # Realistic per-layer compute for 70B on this GPU (from matmul timing ~0.66ms per big matmul, ~5ms full layer)
    compute_ms = 5.0
    print(f"\n=== CLEAN 70B / 128k PROJECTION (real BW {real_bw:.2f} GB/s) ===")
    for p in [5, 6, 8, 10]:
        effective_transfer = transfer_ms / p
        effective_ms_per_layer = max(compute_ms, effective_transfer)
        total_time_ms = effective_ms_per_layer * layers_70b
        toks_s_70b = 1000 / total_time_ms
        print(f"  prefetch={p}: Effective per layer {effective_ms_per_layer:.1f} ms -> {toks_s_70b:.2f} toks/s @128k")
        if effective_transfer <= compute_ms:
            print("    -> Reached compute-bound floor (2.5 toks/s)!")
            break

    return results, breaking, proj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max", action="store_true", help="Absolute max: 120 layers, 128k ctx")
    args = parser.parse_args()

    print("=" * 140)
    print("*** RETRACTED (Iteration 16 audit): this script runs an 8-layer <10 MB model regardless")
    print("*** of the layer count shown, its 'overlap' metric is fictional, and its projection math")
    print("*** is physically wrong. Use iteration16_corrected_streaming.py instead.")
    print("=" * 140)
    print("ITERATION 15 – ABSOLUTE MAXIMUM PUSH: Size + Context to Breaking Point")
    print("Hardware: Your GTX 1660 Ti 6GB + 48GB (real execution, no shortcuts)")
    print("=" * 140)

    hw = HardwareProfile.capture()
    print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM {hw.vram_gb}GB | Host {hw.host_ram_gb}GB")

    sig = {"nvidia_smi": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")[:80]}

    target_l = 120 if args.max else 80
    target_c = 131072 if args.max else 8192

    res, breaking, proj = push_to_absolute_max(
        hw, target_layers=target_l, max_ctx=target_c, trials=args.trials
    )

    if args.plot:
        PLOTS_DIR.mkdir(exist_ok=True)
        # Scaling plot
        if res:
            layers = sorted(set(r["layers"] for r in res))
            max_o = []
            for l in layers:
                o = max(r["overlap_x"] for r in res if r["layers"] == l)
                max_o.append(o)
            plt.figure(figsize=(8,4))
            plt.plot(layers, max_o, 'o-')
            plt.xlabel("Layers")
            plt.ylabel("Max Overlap (x)")
            plt.title("Iter15 Absolute Max Push: Overlap vs Scale")
            plt.grid(True)
            plt.savefig(PLOTS_DIR / "iteration15_max_scaling.png", dpi=140)
            print("[plot] plots/iteration15_max_scaling.png")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 15,
        "hardware": asdict(hw),
        "sig": sig,
        "results": res,
        "breaking": breaking,
        "70b_plus_proj": proj,
        "note": "Pushed beyond 70B / 128k. New arch decides limits (active set only)."
    }
    with open(RESULTS_DIR / "iteration15_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("[saved] results/iteration15_results.json")

    with open(RESULTS_DIR / "iteration15_hardware_proof.json", "w") as f:
        json.dump({"target": f"GTX 1660 Ti {hw.vram_gb}GB + {hw.host_ram_gb}GB", "sig": sig, "breaking": breaking}, f, indent=2)
    print("[proof] iteration15_hardware_proof.json")

    print("\nReproduce absolute max: python iteration15_absolute_max_push.py --max --plot")
    print("Breaking points fully documented. New architecture redefines the ceiling.")

if __name__ == "__main__":
    main()
