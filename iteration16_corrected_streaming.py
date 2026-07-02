#!/usr/bin/env python3
"""
Iteration 16 - Corrected Streaming Benchmark (audit fix for Iterations 10-15).

WHAT THIS FIXES
---------------
Iteration 15 reported "120 layers @ 0.45 GB" results while actually executing an
8-layer, <10 MB model (the layer count was capped by `self.n_layers`), computed a
fictional "overlap" metric (hypothetical big-model time divided by tiny-model
time), and projected 70B throughput by dividing PCIe transfer time by prefetch
depth -- which is physically impossible on a single shared PCIe link.
Those results are RETRACTED (see BREAKTHROUGH_PROGRESS_LOG.md, Iteration 16).

This artifact does what Iteration 15 claimed to do:

1. Actually allocates N layers of the stated size in pinned host RAM
   (each layer is one fp16 [n x n] weight matrix, ~layer_gb bytes).
2. Actually streams every layer over PCIe every token, double/triple-buffered
   on dedicated CUDA streams, and computes a real GEMM against the streamed
   weights for a real activation batch.
3. Measures, with CUDA events on the respective streams:
   - wall time per token,
   - total transfer time and total compute time,
   - overlap efficiency = fraction of theoretically hidable time actually
     hidden, against a genuinely-serial reference run of the same work.
4. Projects 70B-class throughput with the correct steady-state pipeline model:
       per_layer_time = max(transfer_time, compute_time)
   (prefetch depth does NOT divide transfer time; PCIe is one shared link),
   and validates the projection against the measured runs.
5. Reports power/thermal sampled during the actual measured load, against the
   GPU's REAL power limit read from nvidia-smi (~120 W for a GTX 1660 Ti),
   not a hardcoded 450 W.

HONEST PHYSICS THIS DEMONSTRATES
--------------------------------
Dense weight streaming for single-stream decode is PCIe-bound:
  70B INT4 ~= 35 GB of weights per token / ~12 GB/s PCIe ~= ~0.3 tok/s.
Overlap hides the (much smaller) compute almost entirely, but cannot hide
transfer behind itself. The real levers, quantified in the projection section:
  - MoE sparsity (stream only active experts),
  - batching / speculative decoding (amortize each weight pass over B tokens),
  - keeping a resident fraction of weights in VRAM.

ASSUMPTIONS AND LIMITATIONS
---------------------------
- Weights are random (values do not affect transfer/GEMM timing); activations
  are RMS-normalized between layers purely for numerical hygiene.
- One [n x n] GEMM per layer is a simplified proxy for a real block
  (attention/SSM + MLP); it is weight-read-bound like real decode GEMMs,
  which is the regime that matters for this measurement.
- Projections use bandwidths measured on THIS machine during the pipelined
  run (not theoretical peaks) and are validated against measured wall times.
- No "reversible energy recovery" factor is applied anywhere. Prior
  iterations multiplied energy by (1 - 0.55) citing reversible computing;
  no such mechanism exists on shipped CMOS GPUs. Retracted.

REPRODUCE
---------
  python iteration16_corrected_streaming.py --plot            # default: 16 x 0.25 GB
  python iteration16_corrected_streaming.py --stress --plot   # 48 x 0.25 GB (~11.6 GB pinned)

Outputs: results/iteration16_results.json, plots/iteration16_streaming.png
"""

import argparse
import ctypes
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


# =============================================================================
# Hardware capture -- all values read from the machine, none hardcoded.
# =============================================================================

def _windows_host_mem_gb() -> Optional[Dict[str, float]]:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_uint64)
                for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                          "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                          "ullAvailExtendedVirtual")
            ]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return {"total_gb": st.ullTotalPhys / 1e9, "avail_gb": st.ullAvailPhys / 1e9}
    except Exception:
        return None


def _nvidia_smi(fields: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return out or None
    except Exception:
        return None


@dataclass
class HardwareProfile:
    gpu_name: str = "unknown"
    driver: str = "unknown"
    vram_gb: float = 0.0
    power_limit_w: Optional[float] = None  # real board power limit, NOT assumed
    host_total_gb: Optional[float] = None
    host_avail_gb: Optional[float] = None
    torch_version: str = torch.__version__

    @classmethod
    def capture(cls) -> "HardwareProfile":
        hw = cls()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw.gpu_name = props.name
            hw.vram_gb = round(props.total_memory / 1e9, 2)
        smi = _nvidia_smi("driver_version,power.limit")
        if smi:
            parts = [p.strip() for p in smi.split(",")]
            hw.driver = parts[0]
            try:
                hw.power_limit_w = float(parts[1])
            except (IndexError, ValueError):
                pass
        mem = _windows_host_mem_gb()
        if mem:
            hw.host_total_gb = round(mem["total_gb"], 1)
            hw.host_avail_gb = round(mem["avail_gb"], 1)
        return hw


class PowerSampler(threading.Thread):
    """Samples real power/temperature/SM-utilization via nvidia-smi while running."""

    def __init__(self, interval_s: float = 0.4):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.samples: List[Dict[str, float]] = []
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            out = _nvidia_smi("power.draw,temperature.gpu,utilization.gpu")
            if out:
                try:
                    p, t, u = [float(x.strip()) for x in out.split(",")]
                    self.samples.append({"power_w": p, "temp_c": t, "sm_util_pct": u})
                except ValueError:
                    pass
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> Dict[str, float]:
        self._stop_evt.set()
        self.join(timeout=3)
        if not self.samples:
            return {}
        return {
            "avg_power_w": round(sum(s["power_w"] for s in self.samples) / len(self.samples), 1),
            "peak_power_w": round(max(s["power_w"] for s in self.samples), 1),
            "avg_temp_c": round(sum(s["temp_c"] for s in self.samples) / len(self.samples), 1),
            "peak_temp_c": round(max(s["temp_c"] for s in self.samples), 1),
            "avg_sm_util_pct": round(sum(s["sm_util_pct"] for s in self.samples) / len(self.samples), 1),
            "n_samples": len(self.samples),
        }


# =============================================================================
# Layer construction -- the layers claimed are the layers allocated.
# =============================================================================

def layer_dim_for_gb(layer_gb: float) -> int:
    """Largest multiple-of-256 n such that an fp16 [n x n] matrix is <= layer_gb."""
    n = int((layer_gb * 1e9 / 2) ** 0.5)
    return max(256, (n // 256) * 256)


def build_host_layers(n_layers: int, n: int, device: torch.device) -> List[torch.Tensor]:
    """Allocate n_layers pinned host fp16 [n x n] matrices, init via GPU randn.

    Init values only matter for numerical hygiene (avoid NaN garbage in
    uninitialized memory); std = n^-0.5 keeps activations O(1) across layers.
    """
    layers = []
    with torch.no_grad():
        gpu_tmp = torch.empty(n, n, dtype=torch.float16, device=device)
        for i in range(n_layers):
            gpu_tmp.normal_(0.0, n ** -0.5)
            host = torch.empty(n, n, dtype=torch.float16, pin_memory=True)
            host.copy_(gpu_tmp)
            layers.append(host)
        del gpu_tmp
        torch.cuda.empty_cache()
    return layers


# =============================================================================
# Measurement
# =============================================================================

def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


def run_pipelined_token(host_layers: List[torch.Tensor], slots: List[torch.Tensor],
                        x: torch.Tensor, transfer_stream: torch.cuda.Stream,
                        compute_stream: torch.cuda.Stream) -> Dict[str, float]:
    """One 'token': stream every layer H2D and GEMM against it, overlapped.

    Transfer of layer i waits until the compute that last used slot (i mod S)
    has finished; compute of layer i waits until its transfer has finished.
    Timing via CUDA events recorded on the owning streams.
    """
    L, S = len(host_layers), len(slots)
    t_start = [torch.cuda.Event(enable_timing=True) for _ in range(L)]
    t_end = [torch.cuda.Event(enable_timing=True) for _ in range(L)]
    c_start = [torch.cuda.Event(enable_timing=True) for _ in range(L)]
    c_end = [torch.cuda.Event(enable_timing=True) for _ in range(L)]

    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    for i in range(L):
        slot = slots[i % S]
        with torch.cuda.stream(transfer_stream):
            if i >= S:
                transfer_stream.wait_event(c_end[i - S])  # slot must be free
            t_start[i].record(transfer_stream)
            slot.copy_(host_layers[i], non_blocking=True)
            t_end[i].record(transfer_stream)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(t_end[i])
            c_start[i].record(compute_stream)
            x = _rms_norm(F.silu(x @ slot))
            c_end[i].record(compute_stream)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_t0) * 1000

    transfer_ms = sum(t_start[i].elapsed_time(t_end[i]) for i in range(L))
    compute_ms = sum(c_start[i].elapsed_time(c_end[i]) for i in range(L))
    return {"wall_ms": wall_ms, "transfer_ms": transfer_ms, "compute_ms": compute_ms}


def run_serial_token(host_layers: List[torch.Tensor], slot: torch.Tensor,
                     x: torch.Tensor) -> float:
    """Ground-truth serial reference: same work, no overlap (default stream)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for host in host_layers:
        slot.copy_(host, non_blocking=False)
        x = _rms_norm(F.silu(x @ slot))
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def measure_config(host_layers: List[torch.Tensor], n: int, batch: int,
                   n_slots: int, n_tokens: int, device: torch.device) -> Dict:
    L = len(host_layers)
    layer_gb = host_layers[0].numel() * 2 / 1e9
    slots = [torch.empty(n, n, dtype=torch.float16, device=device) for _ in range(n_slots)]
    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    x = torch.randn(batch, n, dtype=torch.float16, device=device)

    # cuBLAS warmup for this shape + one pipeline warmup token
    for _ in range(3):
        _ = x @ slots[0]
    run_pipelined_token(host_layers, slots, x, transfer_stream, compute_stream)

    torch.cuda.reset_peak_memory_stats()
    runs = [run_pipelined_token(host_layers, slots, x, transfer_stream, compute_stream)
            for _ in range(n_tokens)]
    vram_gb = torch.cuda.max_memory_allocated() / 1e9

    serial_ms = run_serial_token(host_layers, slots[0], x)

    wall = sum(r["wall_ms"] for r in runs) / n_tokens
    transfer = sum(r["transfer_ms"] for r in runs) / n_tokens
    compute = sum(r["compute_ms"] for r in runs) / n_tokens
    ideal = max(transfer, compute)  # asymptotic pipeline floor (fill ignored)
    hidable = serial_ms - ideal
    overlap_eff = max(0.0, min(1.0, (serial_ms - wall) / hidable)) if hidable > 1e-3 else 0.0

    total_gb = L * layer_gb
    res = {
        "layers": L,
        "layer_gb_actual": round(layer_gb, 4),
        "total_streamed_gb_per_token": round(total_gb, 2),
        "batch": batch,
        "slots": n_slots,
        "wall_ms_per_token": round(wall, 1),
        "serial_ms_per_token": round(serial_ms, 1),
        "ideal_pipeline_ms": round(ideal, 1),
        "transfer_ms_total": round(transfer, 1),
        "compute_ms_total": round(compute, 1),
        "overlap_efficiency": round(overlap_eff, 3),
        "effective_pcie_gbs_in_pipeline": round(total_gb / (transfer / 1000), 2),
        "effective_compute_gbs_weight_read": round(total_gb / (compute / 1000), 2),
        "vram_peak_gb": round(vram_gb, 3),
        "toks_s_batch1_equiv": round(1000 / wall, 3),
        "toks_s_aggregate": round(batch * 1000 / wall, 2),
    }
    del slots, x
    torch.cuda.empty_cache()
    return res


# =============================================================================
# Projection -- correct pipeline model, validated against measurement.
# =============================================================================

def project_token_time_ms(layers: int, layer_gb: float, pcie_gbs: float,
                          compute_gbs: float, active_frac: float = 1.0,
                          resident_frac: float = 0.0) -> Dict[str, float]:
    """Steady-state pipelined decode time for one weight pass.

    per_layer = max(transfer, compute). Prefetch depth changes NOTHING here:
    PCIe is a single shared link, so queueing more transfers cannot shorten
    the sum of transfer times. Overlap can only hide the smaller phase.

    active_frac   -- fraction of weights touched per token (MoE sparsity).
    resident_frac -- fraction of active weights permanently in VRAM (no PCIe).
    """
    transfer_ms = layer_gb * active_frac * (1 - resident_frac) / pcie_gbs * 1000
    compute_ms = layer_gb * active_frac / compute_gbs * 1000
    per_layer = max(transfer_ms, compute_ms)
    token_ms = layers * per_layer + transfer_ms  # + pipeline fill
    return {
        "transfer_ms_per_layer": round(transfer_ms, 2),
        "compute_ms_per_layer": round(compute_ms, 2),
        "token_ms": round(token_ms, 1),
        "bound": "pcie" if transfer_ms >= compute_ms else "compute",
    }


def build_projections(pcie_gbs: float, compute_rates: Dict[int, float]) -> List[Dict]:
    """70B-class INT4 (80 layers x ~0.44 GB = ~35 GB) on this machine.

    compute_rates maps batch size -> MEASURED weight-read GB/s at that batch on
    this GPU. On the GTX 1660 Ti (no tensor cores) fp16 GEMM at B>1 measures far
    slower than the B=1 GEMV path, so batched cases can flip to compute-bound
    here. Optimized INT4 decode kernels (llama.cpp-class) stay near the
    weight-read bound; each batched case therefore also reports that upper
    bound explicitly rather than silently assuming it.
    """
    L, gb = 80, 0.441
    b1_rate = compute_rates.get(1, min(compute_rates.values()))
    cases = [
        ("dense 70B INT4, B=1", 1.0, 0.0, 1),
        ("dense 70B INT4, B=1, 13% VRAM-resident", 1.0, 0.13, 1),
        ("dense 70B INT4, B=8 (batch/speculative)", 1.0, 0.0, 8),
        ("dense 70B INT4, B=32", 1.0, 0.0, 32),
        ("MoE ~70B total, ~30% active/token, B=1", 0.30, 0.0, 1),
        ("MoE ~30% active + 13% resident, B=8", 0.30, 0.13, 8),
    ]
    out = []
    for name, active, resident, batch in cases:
        rate = compute_rates.get(batch, b1_rate)
        p = project_token_time_ms(L, gb, pcie_gbs, rate, active, resident)
        entry = {
            "case": name,
            **p,
            "batch": batch,
            "compute_gbs_used": round(rate, 1),
            "toks_s_single_stream": round(1000 / p["token_ms"], 3),
            "toks_s_aggregate": round(batch * 1000 / p["token_ms"], 2),
        }
        if batch > 1:
            ub = project_token_time_ms(L, gb, pcie_gbs, b1_rate, active, resident)
            entry["toks_s_aggregate_if_weight_read_bound"] = round(batch * 1000 / ub["token_ms"], 2)
        out.append(entry)
    return out


def validate_projection(measured: Dict, pcie_gbs: float, compute_gbs: float) -> Dict:
    """Predict the measured config from calibrated rates; report the error."""
    p = project_token_time_ms(measured["layers"], measured["layer_gb_actual"],
                              pcie_gbs, compute_gbs)
    err = (p["token_ms"] - measured["wall_ms_per_token"]) / measured["wall_ms_per_token"]
    return {
        "config": f'{measured["layers"]}L x {measured["layer_gb_actual"]}GB B={measured["batch"]}',
        "predicted_token_ms": p["token_ms"],
        "measured_token_ms": measured["wall_ms_per_token"],
        "error_pct": round(100 * err, 1),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--layer-gb", type=float, default=0.25)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--tokens", type=int, default=4, help="measured tokens per config")
    parser.add_argument("--slots", type=int, default=3, help="GPU weight buffers (>=2)")
    parser.add_argument("--stress", action="store_true", help="48 layers (~11.6 GB pinned host)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required for this benchmark.")
    device = torch.device("cuda")
    torch.manual_seed(0)

    if args.stress:
        args.layers = 48

    hw = HardwareProfile.capture()
    print("=" * 100)
    print("ITERATION 16 - CORRECTED STREAMING BENCHMARK (audit fix; retracts Iteration 10-15 claims)")
    print("=" * 100)
    print(f"[HW] {hw.gpu_name} | VRAM {hw.vram_gb} GB | power limit "
          f"{hw.power_limit_w if hw.power_limit_w else 'unknown'} W (real, from nvidia-smi) | "
          f"host RAM {hw.host_total_gb} GB total / {hw.host_avail_gb} GB available | "
          f"torch {hw.torch_version}")

    n = layer_dim_for_gb(args.layer_gb)
    layer_gb_actual = 2 * n * n / 1e9
    pinned_gb = args.layers * layer_gb_actual
    print(f"[MODEL] {args.layers} layers x fp16 [{n} x {n}] = {layer_gb_actual:.3f} GB/layer, "
          f"{pinned_gb:.1f} GB pinned host total -- ACTUALLY ALLOCATED AND STREAMED")

    if hw.host_avail_gb and pinned_gb > 0.6 * hw.host_avail_gb:
        sys.exit(f"Refusing to pin {pinned_gb:.1f} GB with only {hw.host_avail_gb} GB host RAM available.")

    t0 = time.perf_counter()
    host_layers = build_host_layers(args.layers, n, device)
    print(f"[MODEL] host allocation + init took {time.perf_counter() - t0:.1f} s")

    sampler = PowerSampler()
    sampler.start()

    results = []
    for batch in args.batches:
        res = measure_config(host_layers, n, batch, args.slots, args.tokens, device)
        results.append(res)
        print(f"\n  {res['layers']}L x {res['layer_gb_actual']:.3f}GB, batch={batch}: "
              f"wall {res['wall_ms_per_token']:.0f} ms/token "
              f"(serial {res['serial_ms_per_token']:.0f} ms, ideal {res['ideal_pipeline_ms']:.0f} ms)")
        print(f"    transfer {res['transfer_ms_total']:.0f} ms ({res['effective_pcie_gbs_in_pipeline']:.2f} GB/s) | "
              f"compute {res['compute_ms_total']:.0f} ms ({res['effective_compute_gbs_weight_read']:.0f} GB/s weight-read)")
        print(f"    overlap efficiency {res['overlap_efficiency']:.0%} of hidable time | "
              f"VRAM peak {res['vram_peak_gb']:.2f} GB | "
              f"{res['toks_s_batch1_equiv']:.2f} tok/s single-stream, {res['toks_s_aggregate']:.1f} tok/s aggregate")

    power = sampler.stop()
    if power:
        limit = f" (limit {hw.power_limit_w} W)" if hw.power_limit_w else ""
        print(f"\n[POWER] during measured load: avg {power['avg_power_w']} W, "
              f"peak {power['peak_power_w']} W{limit}, "
              f"temp avg {power['avg_temp_c']} C peak {power['peak_temp_c']} C, "
              f"SM util avg {power['avg_sm_util_pct']}% "
              f"(low SM util is EXPECTED: the pipeline is PCIe-bound, not compute-bound)")

    # Calibrate projection rates from the pipelined runs (measured, per batch).
    cal = results[0]
    pcie_gbs = cal["effective_pcie_gbs_in_pipeline"]
    compute_rates = {r["batch"]: r["effective_compute_gbs_weight_read"] for r in results}
    compute_gbs = compute_rates.get(1, cal["effective_compute_gbs_weight_read"])

    print("\n" + "=" * 100)
    print("PROJECTION VALIDATION (formula vs. this run's measured wall times)")
    validations = [validate_projection(r, pcie_gbs, r["effective_compute_gbs_weight_read"]) for r in results]
    for v in validations:
        print(f"  {v['config']}: predicted {v['predicted_token_ms']:.0f} ms, "
              f"measured {v['measured_token_ms']:.0f} ms, error {v['error_pct']:+.1f}%")

    print("\n" + "=" * 100)
    print(f"70B-CLASS PROJECTIONS (80 x 0.441 GB INT4; calibrated PCIe {pcie_gbs:.2f} GB/s, "
          f"compute {compute_gbs:.0f} GB/s weight-read)")
    print("Correct pipeline model: per_layer = max(transfer, compute). Prefetch does NOT divide transfer time.")
    print("Batched cases use the MEASURED (slower) B>1 fp16 GEMM rate on this GPU; the")
    print("'if weight-read-bound' figure is the upper bound with optimized INT4 decode kernels.")
    projections = build_projections(pcie_gbs, compute_rates)
    for p in projections:
        extra = (f" [{p['toks_s_aggregate_if_weight_read_bound']:.2f} tok/s if weight-read-bound]"
                 if "toks_s_aggregate_if_weight_read_bound" in p else "")
        print(f"  {p['case']}: {p['token_ms'] / 1000:.2f} s/weight-pass, "
              f"{p['toks_s_single_stream']:.2f} tok/s single-stream, "
              f"{p['toks_s_aggregate']:.2f} tok/s aggregate ({p['bound']}-bound){extra}")

    print("\nMEMORY MODEL (70B-class @ 128k context on this machine):")
    kv_gb = 2 * 80 * 131072 * (8 * 128) * 2 / 1e9  # GQA: 8 KV heads x 128 dim, fp16
    print(f"  Transformer KV cache (GQA, fp16): {kv_gb:.1f} GB -> does not fit 6 GB VRAM; "
          f"weights + KV ~{35.3 + kv_gb:.0f} GB total")
    print(f"  Streamed SSM/hybrid: VRAM = {args.slots} slots x layer + activations "
          f"(measured peak {max(r['vram_peak_gb'] for r in results):.2f} GB); "
          f"host = 35.3 GB weights; recurrent state = O(MB), constant in context")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 16,
        "purpose": "Corrected benchmark; supersedes and retracts iteration 10-15 performance claims.",
        "hardware": asdict(hw),
        "config": {"layers": args.layers, "layer_dim": n,
                   "layer_gb_actual": round(layer_gb_actual, 4),
                   "pinned_host_gb": round(pinned_gb, 2),
                   "slots": args.slots, "tokens_per_config": args.tokens},
        "measured": results,
        "power_during_load": power,
        "projection_validation": validations,
        "projections_70b": projections,
        "assumptions": [
            "Per-layer compute proxied by one weight-read-bound fp16 GEMM of the full layer.",
            "Projection rates calibrated from in-pipeline measurements, not theoretical peaks.",
            "No reversible-computing energy credit (prior 55% 'recovery' factor retracted).",
            "toks_s_aggregate assumes batch/speculative decoding amortizes each weight pass.",
        ],
    }
    out_path = RESULTS_DIR / ("iteration16_results_stress.json" if args.stress
                              else "iteration16_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {out_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = [f"B={r['batch']}" for r in results]
        xs = range(len(results))
        axes[0].bar([x - 0.25 for x in xs], [r["serial_ms_per_token"] for r in results], 0.25, label="serial")
        axes[0].bar(xs, [r["wall_ms_per_token"] for r in results], 0.25, label="measured pipelined")
        axes[0].bar([x + 0.25 for x in xs], [r["ideal_pipeline_ms"] for r in results], 0.25, label="ideal max(T,C)")
        axes[0].set_xticks(list(xs)); axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("ms per token (weight pass)")
        axes[0].set_title(f"{args.layers}L x {layer_gb_actual:.2f} GB: measured vs bounds")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot([r["batch"] for r in results], [r["toks_s_aggregate"] for r in results], "o-")
        axes[1].set_xlabel("batch"); axes[1].set_ylabel("aggregate tok/s")
        axes[1].set_title("Batching amortizes the PCIe-bound weight pass")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration16_streaming.png", dpi=140)
        print("[plot] plots/iteration16_streaming.png")

    print("\nReproduce: python iteration16_corrected_streaming.py --plot   (add --stress for 48 layers)")


if __name__ == "__main__":
    main()
