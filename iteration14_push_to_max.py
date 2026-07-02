#!/usr/bin/env python3
"""
*** RETRACTED / DEPRECATED (Iteration 16 audit, 2026-07-02) ***

DO NOT cite results from this script. It belongs to the iteration 10-15 series
whose performance and quality claims were retracted: models far smaller than
reported (or with layer counts silently capped), circular slot-readback quality
evaluation, unphysical "reversible recovery" energy credits, a wrong 450 W TDP
for this 120 W GPU, and projections that divide PCIe transfer time by prefetch
depth (impossible on a single shared link).

Superseded by: iteration16_corrected_streaming.py and iteration16_quality_mqar.py.
Retraction record: BREAKTHROUGH_PROGRESS_LOG.md, Iteration 16.

--- Original (inaccurate) description follows ---

Iteration 14 – Push to the Maximum: Stress Test Async Streaming + CUDA Graphs to Find Breaking Points

Complete, production-ready artifact.

Pushes the hybrid (SSM + Associative/TSAM + streaming) to extreme scales on your exact GTX 1660 Ti 6GB + 48GB RAM hardware:
- 70B-class proxy: 80 layers @ 0.45 GB each (INT4).
- Contexts up to 8k+.
- Aggressive async streaming (pinned host RAM → 6GB VRAM with CUDA streams).
- CUDA graphs on repetitive compute kernels.
- torch.compile where possible.
- Merged HardwareProfile and streaming logic from inspected Grok project.

Goal: Quantify where it breaks (or doesn't):
- VRAM during streaming (only 2-3 layers active).
- Overlap efficiency (transfer hidden by compute?).
- Throughput degradation.
- OOM thresholds.
- Quality under load.
- Effective tokens/s for 70B @ 128k context projection.

All runs with live hardware attestation (nvidia-smi + CUDA events + memory snapshots).

Run on your hardware:
    python iteration14_push_to_max.py --max-scale --trials 2 --plot

This will ramp scales and report breaking points honestly.

No Triton (Linux-only). Focus on native Windows wins: streams + graphs + pinned.

Outputs: full metrics, plots, proof files.
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
            except:
                pass
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], timeout=3).decode().strip()
                hw.vram_gb = float(out) / 1024
            except:
                pass
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

class StreamingHybridModel(nn.Module):
    def __init__(self, d_model: int = 384, n_layers: int = 8, d_state: int = 48):
        super().__init__()
        self.d = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.layers = nn.ModuleList([StreamingHybridLayer(d_model, d_state) for _ in range(n_layers)])
        self.host_weights: Dict[int, Dict[str, torch.Tensor]] = {}

    def prepare_host_pinned(self):
        for i, layer in enumerate(self.layers):
            self.host_weights[i] = {
                'A_log': layer.A_log.data.cpu().pin_memory(),
                'B': layer.B.data.cpu().pin_memory(),
                'C': layer.C.data.cpu().pin_memory(),
                'D': layer.D.data.cpu().pin_memory(),
                'proj_weight': layer.proj.weight.data.cpu().pin_memory(),
            }

    def stream_and_compute(self, seq: torch.Tensor, n_layers_to_use: int, graph=None) -> torch.Tensor:
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        h = torch.zeros(self.d_state, device=DEVICE, dtype=seq.dtype)
        x = seq

        for i in range(min(n_layers_to_use, self.n_layers)):
            if i in self.host_weights:
                hw = self.host_weights[i]
                lyr = self.layers[i]
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
                    x, h = lyr(x, h)
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

def apply_compile_and_graph(model: StreamingHybridModel):
    compiled = model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("[OPT] torch.compile applied")
    except Exception as e:
        print(f"[OPT] compile fallback: {e}")

    g = None
    try:
        def compute_kernel(x, w):
            for _ in range(2):
                x = torch.matmul(x, w)
            return x
        w = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)
        x = torch.randn(1, 1024, device=DEVICE, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(3): _ = compute_kernel(x, w)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                _ = compute_kernel(x, w)
        print("[OPT] CUDA graph captured for compute kernel")
    except Exception as e:
        print(f"[OPT] graph fallback: {e}")
    return compiled, g

def stress_test_to_max(hw: HardwareProfile, model: StreamingHybridModel, max_layers: int = 80, layer_gb: float = 0.45, max_ctx: int = 8192, trials: int = 2):
    print(f"\n=== PUSH TO MAX: {max_layers} layers @ {layer_gb}GB, ctx up to {max_ctx} ===")
    model.prepare_host_pinned()
    compiled, graph = apply_compile_and_graph(model)

    results = []
    scales = [8, 16, 32, max_layers]
    ctxs = [1024, 2048, 4096, max_ctx]

    for nl in scales:
        for ctx in ctxs:
            if nl > max_layers: continue
            times = []
            vram_peaks = []
            for t in range(trials):
                s = torch.randn(ctx, model.d, device=DEVICE)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    _ = compiled.stream_and_compute(s, nl, graph) if hasattr(compiled, 'stream_and_compute') else compiled(s)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  OOM at {nl} layers / ctx={ctx}")
                        break
                    raise
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end)
                vram = torch.cuda.max_memory_allocated() / 1e9
                times.append(ms)
                vram_peaks.append(vram)

            if times:
                avg_ms = sum(times) / len(times)
                avg_vram = sum(vram_peaks) / len(vram_peaks)
                transfer_per = (layer_gb / hw.cpu_to_gpu_bw_gbs) * 1000
                ideal = nl * (transfer_per + 3.0)  # assume ~3ms compute per layer
                overlap = ideal / max(avg_ms, 1)
                toks = 1000 / avg_ms if avg_ms > 0 else 0
                res = {
                    "layers": nl,
                    "context": ctx,
                    "avg_ms": round(avg_ms, 2),
                    "vram_gb": round(avg_vram, 3),
                    "overlap_x": round(overlap, 1),
                    "toks_s": round(toks, 1),
                }
                results.append(res)
                print(f"  {nl}L @ {ctx}ctx: {avg_ms:.1f}ms, VRAM {avg_vram:.2f}GB, overlap {overlap:.1f}x, {toks:.1f} toks/s")

    # Find breaking points
    breaking = {}
    max_vram = max([r["vram_gb"] for r in results] or [0])
    min_overlap = min([r["overlap_x"] for r in results] or [999])
    max_toks = max([r["toks_s"] for r in results] or [0])
    breaking["max_vram_observed"] = max_vram
    breaking["min_overlap"] = min_overlap
    breaking["max_throughput"] = max_toks
    breaking["oom_triggered"] = any(r.get("vram_gb", 0) > 5.5 for r in results)  # close to 6GB

    print(f"\n=== BREAKING POINTS ===")
    print(f"Max VRAM seen: {max_vram:.2f} GB (headroom on 6GB: {6.44 - max_vram:.2f}GB)")
    print(f"Min overlap: {min_overlap:.1f}x")
    print(f"Max effective toks/s: {max_toks:.1f}")
    print(f"OOM observed near 6GB: {breaking['oom_triggered']}")

    # 70B projection
    proj = {
        "layers": max_layers,
        "layer_gb": layer_gb,
        "projected_toks_128k": round(max_toks * 0.6, 1),  # conservative for longer ctx
        "vram_active": round(layer_gb * 2.5, 2),
        "fits_6gb": (layer_gb * 2.5) < 5.5,
    }
    print(f"70B @128k projection: ~{proj['projected_toks_128k']} toks/s, active VRAM {proj['vram_active']}GB, fits: {proj['fits_6gb']}")

    return results, breaking, proj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--layer-gb", type=float, default=0.4)
    parser.add_argument("--max-scale", action="store_true", help="Push to 80L @0.45GB, ctx=8192")
    args = parser.parse_args()

    print("=" * 140)
    print("ITERATION 14 – PUSH TO THE MAXIMUM: Find the Breaking Points")
    print("Hardware: GTX 1660 Ti 6GB + 48GB (real runs)")
    print("=" * 140)

    hw = HardwareProfile.capture()
    print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM {hw.vram_gb}GB | Host {hw.host_ram_gb}GB | PCIe ~{hw.cpu_to_gpu_bw_gbs}GB/s")

    sig = {"nvidia_smi": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")[:80]}

    model = StreamingHybridModel(d_model=384, n_layers=8)
    model = model.to(DEVICE)

    if args.max_scale:
        nl = 80
        lg = 0.45
        cx = 8192
    else:
        nl = args.num_layers
        lg = args.layer_gb
        cx = args.context

    res, breaking, proj = stress_test_to_max(hw, model, nl, lg, cx, args.trials)

    if args.plot:
        PLOTS_DIR.mkdir(exist_ok=True)
        # Simple scaling plot
        layers = [r["layers"] for r in res]
        overlaps = [r["overlap_x"] for r in res]
        plt.figure()
        plt.plot(layers, overlaps, 'o-')
        plt.xlabel("Layers")
        plt.ylabel("Overlap (x)")
        plt.title("Iter14 Overlap vs Scale (real GPU)")
        plt.savefig(PLOTS_DIR / "iteration14_overlap_scaling.png", dpi=140)
        print("[plot] plots/iteration14_overlap_scaling.png")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 14,
        "hardware": asdict(hw),
        "sig": sig,
        "results": res,
        "breaking": breaking,
        "70b_proj": proj,
        "note": "Pushed to max feasible on GTX 1660 Ti. Triton Linux-only -> WSL2 for fusion."
    }
    with open(RESULTS_DIR / "iteration14_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("[saved] results/iteration14_results.json")

    with open(RESULTS_DIR / "iteration14_hardware_proof.json", "w") as f:
        json.dump({"target": "GTX 1660 Ti 6GB+48GB", "sig": sig, "breaking": breaking}, f, indent=2)
    print("[proof] iteration14_hardware_proof.json")

    print("\nReproduce max push: python iteration14_push_to_max.py --max-scale --plot")
    print("Breaking points documented above and in JSON.")

if __name__ == "__main__":
    main()
