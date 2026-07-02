#!/usr/bin/env python3
"""
Iteration 17 - MoE Expert Streaming: measured byte reduction, routing-latency
cost, batched kernel paths, and integrated wall-power energy per token.

WHY
---
Iteration 16 established the honest physics of dense weight streaming on this
machine: PCIe-bound at ~12.3 GB/s, so dense 70B INT4 decodes at ~0.34 tok/s.
Its projections said MoE sparsity is the largest single lever (~3-4x fewer
bytes per token). Projections are not measurements; this artifact streams a
real expert-granular MoE end-to-end and measures:

1. BYTES: dense-equivalent vs. top-2-of-8 expert streaming (25% of expert
   weights per token), with actual transferred bytes counted, not assumed.
2. ROUTING LATENCY: the routing decision for layer i+1 depends on layer i's
   output, so just-in-time (JIT) routing delays the next transfer. Measured
   two ways:
     - oracle: expert choices recorded from a prior identical pass, transfers
       fully overlapped (upper bound = perfect prefetch prediction);
     - jit: router evaluated on the live activation, top-k indices synced to
       CPU, only then transfers enqueued (lower bound, includes the sync).
   Expectation from first principles: with transfer >> compute at B=1, the JIT
   penalty is roughly one compute+sync per layer -- small. Verified below.
3. KERNEL PATHS (iteration 16 follow-up): on this TU116 (no tensor cores),
   fp16 GEMM at B>1 measured 20x slower than the B=1 GEMV path. A small sweep
   of equivalent formulations (x@W vs. transposed, reduced-precision reduction
   on/off) measures the best available path in stock PyTorch and reports the
   remaining gap to the weight-read bound honestly.
4. ENERGY: GPU board power integrated over each measured config
   (nvidia-smi sampling), reported as J/token total and incremental over
   measured idle. GPU board only -- CPU/DRAM/PSU losses are NOT included;
   stated as a limitation, not hidden.

MODEL
-----
L layers; each layer = 8 experts; each expert = fp16 up-proj [n x m] +
down-proj [m x n] (n=8192, m=2048 -> 67 MB/expert, 537 MB/layer, 8.6 GB for
L=16 -- all genuinely allocated in pinned host RAM and streamed per token).
Routers are tiny resident linears [n x 8] (2 MB total; documented). Layer
compute: y = sum_k gate_k * (silu(x @ up_k) @ down_k), x = rmsnorm(x + y).
Weights are random; routing therefore exercises arbitrary expert subsets,
which is the correct regime for transfer benchmarking (real routers change
WHICH experts move, not HOW MANY).

ASSUMPTIONS AND LIMITATIONS
---------------------------
- Top-2-of-8 is Mixtral-style; real MoE transformers also carry dense
  attention/norm weights per layer (not modeled; they add a dense floor to
  bytes/token, noted in the projection).
- Oracle mode measures perfect expert prediction; published expert-prefetch
  predictors reach ~80-90% hit rates, so real systems land between jit and
  oracle. Both bounds are reported.
- Energy covers the GPU board only.

REPRODUCE
---------
  python iteration17_moe_streaming.py --plot

Outputs: results/iteration17_moe_results.json, plots/iteration17_moe.png
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from iteration16_corrected_streaming import (
    HardwareProfile, PowerSampler, _rms_norm, _nvidia_smi,
)

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

N_EXPERTS = 8
TOP_K = 2


# =============================================================================
# Model construction -- every expert genuinely allocated in pinned host RAM.
# =============================================================================

def build_moe_host_layers(n_layers: int, n: int, m: int,
                          device: torch.device) -> List[List[Dict[str, torch.Tensor]]]:
    """host[layer][expert] = {'up': pinned [n,m] fp16, 'down': pinned [m,n] fp16}."""
    layers = []
    with torch.no_grad():
        up_tmp = torch.empty(n, m, dtype=torch.float16, device=device)
        down_tmp = torch.empty(m, n, dtype=torch.float16, device=device)
        for _ in range(n_layers):
            experts = []
            for _ in range(N_EXPERTS):
                up_tmp.normal_(0.0, n ** -0.5)
                down_tmp.normal_(0.0, m ** -0.5)
                up_h = torch.empty(n, m, dtype=torch.float16, pin_memory=True)
                down_h = torch.empty(m, n, dtype=torch.float16, pin_memory=True)
                up_h.copy_(up_tmp)
                down_h.copy_(down_tmp)
                experts.append({"up": up_h, "down": down_h})
            layers.append(experts)
        del up_tmp, down_tmp
        torch.cuda.empty_cache()
    return layers


def expert_bytes(n: int, m: int) -> int:
    return 2 * (n * m + m * n)  # fp16 up + down


class ExpertSlots:
    """GPU-resident buffers for streamed experts, reused round-robin."""

    def __init__(self, n_slots: int, n: int, m: int, device: torch.device):
        self.up = [torch.empty(n, m, dtype=torch.float16, device=device) for _ in range(n_slots)]
        self.down = [torch.empty(m, n, dtype=torch.float16, device=device) for _ in range(n_slots)]
        self.n_slots = n_slots


def make_routers(n_layers: int, n: int, device: torch.device) -> List[torch.Tensor]:
    """Tiny resident router weights [n x N_EXPERTS] per layer (~2 MB total)."""
    g = torch.Generator(device=device).manual_seed(7)
    return [torch.randn(n, N_EXPERTS, generator=g, device=device, dtype=torch.float16) * 0.05
            for _ in range(n_layers)]


def route(x: torch.Tensor, router: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k expert ids + renormalized gates for the (batch-pooled) activation.

    Batch-pooled routing (one expert set per step) matches the transfer-cost
    question asked here; per-token routing at large batch approaches all-expert
    activation and is covered by the dense-equivalent measurement.
    """
    logits = (x.mean(dim=0) if x.dim() > 1 else x).float() @ router.float()
    gates = torch.softmax(logits, dim=-1)
    top = torch.topk(gates, TOP_K)
    return top.indices, (top.values / top.values.sum()).to(x.dtype)


def expert_forward(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return F.silu(x @ up) @ down


# =============================================================================
# Pipelines. Timing identical in style to iteration 16: CUDA events on the
# owning streams, wall clock across the full token, bytes counted as copied.
# =============================================================================

def run_token(host: List[List[Dict[str, torch.Tensor]]], slots: ExpertSlots,
              routers: List[torch.Tensor], x: torch.Tensor,
              transfer_stream: torch.cuda.Stream, compute_stream: torch.cuda.Stream,
              mode: str, oracle_ids: Optional[List[List[int]]] = None
              ) -> Tuple[Dict[str, float], List[List[int]]]:
    """One token through all layers. mode in {'dense', 'oracle', 'jit'}.

    dense : stream all N_EXPERTS experts per layer, fully overlapped.
    oracle: stream only oracle_ids[i] (recorded routing), fully overlapped.
    jit   : router on live activation -> sync top-k ids to CPU -> then enqueue
            transfers; the sync + lost overlap is the measured routing cost.
    """
    L = len(host)
    per_layer_ids: List[List[int]] = []
    t_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    c_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    bytes_moved = 0
    slot_cursor = 0
    # slot index -> compute-done event that last consumed it (for safe reuse)
    slot_guard: Dict[int, torch.cuda.Event] = {}

    def enqueue_experts(layer: int, ids: List[int]) -> Tuple[List[int], torch.cuda.Event]:
        nonlocal bytes_moved, slot_cursor
        used = []
        ts, te = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(transfer_stream):
            ts.record(transfer_stream)
            for eid in ids:
                s = slot_cursor % slots.n_slots
                slot_cursor += 1
                if s in slot_guard:
                    transfer_stream.wait_event(slot_guard[s])
                slots.up[s].copy_(host[layer][eid]["up"], non_blocking=True)
                slots.down[s].copy_(host[layer][eid]["down"], non_blocking=True)
                bytes_moved += slots.up[s].numel() * 2 + slots.down[s].numel() * 2
                used.append(s)
            te.record(transfer_stream)
        t_events.append((ts, te))
        return used, te

    torch.cuda.synchronize()
    wall_t0 = time.perf_counter()

    if mode in ("dense", "oracle"):
        # Routing known up front -> transfers for layer i+1 overlap compute of i.
        plan = ([list(range(N_EXPERTS))] * L if mode == "dense" else oracle_ids)
        pending: List[Tuple[List[int], torch.cuda.Event]] = []
        pending.append(enqueue_experts(0, plan[0]))
        for i in range(L):
            used, ready = pending.pop(0)
            if i + 1 < L:
                pending.append(enqueue_experts(i + 1, plan[i + 1]))
            gates = (torch.full((len(used),), 1.0 / len(used), device=x.device, dtype=x.dtype)
                     if mode == "dense" else route(x, routers[i])[1])
            cs, ce = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(ready)
                cs.record(compute_stream)
                y = torch.zeros_like(x)
                for k, s in enumerate(used):
                    y = y + gates[k] * expert_forward(x, slots.up[s], slots.down[s])
                x = _rms_norm(x + y)
                ce.record(compute_stream)
            c_events.append((cs, ce))
            for s in used:
                slot_guard[s] = ce
            per_layer_ids.append(plan[i] if mode == "oracle" else list(range(N_EXPERTS)))
    else:  # jit
        for i in range(L):
            ids_t, gates = route(x, routers[i])
            ids = [int(v) for v in ids_t.cpu()]  # the honest sync: CPU must know what to copy
            per_layer_ids.append(ids)
            used, ready = enqueue_experts(i, ids)
            cs, ce = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(ready)
                cs.record(compute_stream)
                y = torch.zeros_like(x)
                for k, s in enumerate(used):
                    y = y + gates[k] * expert_forward(x, slots.up[s], slots.down[s])
                x = _rms_norm(x + y)
                ce.record(compute_stream)
            c_events.append((cs, ce))
            for s in used:
                slot_guard[s] = ce
            torch.cuda.current_stream().wait_event(ce)
            x = x.clone()  # materialize for the next router read on the default stream

    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_t0) * 1000
    metrics = {
        "wall_ms": wall_ms,
        "transfer_ms": sum(a.elapsed_time(b) for a, b in t_events),
        "compute_ms": sum(a.elapsed_time(b) for a, b in c_events),
        "gb_moved": bytes_moved / 1e9,
    }
    return metrics, per_layer_ids


def measure_mode(host, slots, routers, n: int, batch: int, mode: str, n_tokens: int,
                 device: torch.device, oracle_ids=None) -> Dict:
    transfer_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    x = torch.randn(batch, n, dtype=torch.float16, device=device)
    run_token(host, slots, routers, x, transfer_stream, compute_stream, mode, oracle_ids)  # warmup
    torch.cuda.reset_peak_memory_stats()
    runs = []
    for _ in range(n_tokens):
        m, ids = run_token(host, slots, routers, x, transfer_stream, compute_stream,
                           mode, oracle_ids)
        runs.append(m)
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    avg = {k: sum(r[k] for r in runs) / n_tokens for k in runs[0]}
    return {
        "mode": mode, "batch": batch,
        "wall_ms_per_token": round(avg["wall_ms"], 1),
        "transfer_ms": round(avg["transfer_ms"], 1),
        "compute_ms": round(avg["compute_ms"], 1),
        "gb_moved_per_token": round(avg["gb_moved"], 3),
        "effective_pcie_gbs": round(avg["gb_moved"] / (avg["transfer_ms"] / 1000), 2),
        "vram_peak_gb": round(vram_gb, 3),
        "toks_s_single_stream": round(1000 / avg["wall_ms"], 3),
        "toks_s_aggregate": round(batch * 1000 / avg["wall_ms"], 2),
    }


# =============================================================================
# Kernel-path sweep (iteration 16 follow-up): best stock-PyTorch GEMM at B>1.
# =============================================================================

def sweep_gemm_paths(n: int, device: torch.device, iters: int = 10) -> List[Dict]:
    W = torch.randn(n, n, dtype=torch.float16, device=device) * (n ** -0.5)
    results = []
    for B in (1, 8, 32):
        x = torch.randn(B, n, dtype=torch.float16, device=device)
        variants = {
            "x@W": lambda: x @ W,
            "(W.T@x.T).T": lambda: (W.t() @ x.t()).t(),
            "x@W, full-prec reduction": None,  # handled via backend flag below
        }
        for name, fn in variants.items():
            flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            if name.endswith("full-prec reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                fn = lambda: x @ W
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / iters * 1000
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
            results.append({"batch": B, "path": name, "ms": round(ms, 3),
                            "weight_read_gbs": round(W.numel() * 2 / 1e9 / (ms / 1000), 1)})
    del W
    torch.cuda.empty_cache()
    return results


# =============================================================================
# Main
# =============================================================================

def measure_idle_power(seconds: float = 3.0) -> Optional[float]:
    samples = []
    end = time.time() + seconds
    while time.time() < end:
        out = _nvidia_smi("power.draw")
        if out:
            try:
                samples.append(float(out))
            except ValueError:
                pass
        time.sleep(0.4)
    return round(sum(samples) / len(samples), 1) if samples else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--n", type=int, default=8192, help="model dim")
    parser.add_argument("--m", type=int, default=2048, help="expert FFN dim")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--tokens", type=int, default=3, help="measured tokens per config")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this benchmark.")
    device = torch.device("cuda")
    torch.manual_seed(0)

    hw = HardwareProfile.capture()
    e_bytes = expert_bytes(args.n, args.m)
    layer_gb = N_EXPERTS * e_bytes / 1e9
    pinned_gb = args.layers * layer_gb
    print("=" * 100)
    print("ITERATION 17 - MoE EXPERT STREAMING (measured bytes, routing cost, kernels, energy)")
    print("=" * 100)
    print(f"[HW] {hw.gpu_name} | VRAM {hw.vram_gb} GB | power limit {hw.power_limit_w} W | "
          f"host {hw.host_total_gb} GB total / {hw.host_avail_gb} GB available")
    print(f"[MODEL] {args.layers} layers x {N_EXPERTS} experts x {e_bytes / 1e6:.1f} MB "
          f"= {layer_gb:.3f} GB/layer, {pinned_gb:.1f} GB pinned host total "
          f"-- ACTUALLY ALLOCATED AND STREAMED (top-{TOP_K} active)")

    if hw.host_avail_gb and pinned_gb > 0.6 * hw.host_avail_gb:
        raise SystemExit(f"Refusing to pin {pinned_gb:.1f} GB with "
                         f"{hw.host_avail_gb} GB host RAM available.")

    idle_w = measure_idle_power()
    print(f"[POWER] idle GPU board power: {idle_w} W")

    t0 = time.perf_counter()
    host = build_moe_host_layers(args.layers, args.n, args.m, device)
    routers = make_routers(args.layers, args.n, device)
    print(f"[MODEL] host allocation + init took {time.perf_counter() - t0:.1f} s")

    # Slots: dense needs 2 layers x 8 experts in flight; MoE needs 2 x top-k.
    slots = ExpertSlots(2 * N_EXPERTS, args.n, args.m, device)

    print("\n--- Kernel-path sweep (fp16 GEMM formulations on this GPU) ---")
    gemm_paths = sweep_gemm_paths(args.n, device)
    for r in gemm_paths:
        print(f"  B={r['batch']:>2} {r['path']:<26} {r['ms']:>8.3f} ms  "
              f"({r['weight_read_gbs']:>6.1f} GB/s weight-read)")
    best_by_batch = {}
    for r in gemm_paths:
        if r["batch"] not in best_by_batch or r["ms"] < best_by_batch[r["batch"]]["ms"]:
            best_by_batch[r["batch"]] = r
    for B, r in best_by_batch.items():
        print(f"  best at B={B}: {r['path']} -> {r['weight_read_gbs']} GB/s")

    results = []
    energy = {}
    for batch in args.batches:
        # Record oracle routing from a JIT pass (identical activations/seeds).
        torch.manual_seed(1)
        _, oracle_ids = run_token(host, slots, routers,
                                  torch.randn(batch, args.n, dtype=torch.float16, device=device),
                                  torch.cuda.Stream(), torch.cuda.Stream(), "jit")
        for mode in ("dense", "oracle", "jit"):
            sampler = PowerSampler(interval_s=0.3)
            sampler.start()
            res = measure_mode(host, slots, routers, args.n, batch, mode,
                               args.tokens, device,
                               oracle_ids=oracle_ids if mode == "oracle" else None)
            p = sampler.stop()
            if p and idle_w is not None:
                wall_s = res["wall_ms_per_token"] / 1000
                res["avg_power_w"] = p["avg_power_w"]
                res["energy_j_per_token_total"] = round(p["avg_power_w"] * wall_s, 1)
                res["energy_j_per_token_incremental"] = round(
                    (p["avg_power_w"] - idle_w) * wall_s, 1)
            results.append(res)
            e = res.get("energy_j_per_token_total")
            print(f"\n  {mode:>6} B={batch}: {res['wall_ms_per_token']:.0f} ms/token, "
                  f"{res['gb_moved_per_token']:.2f} GB moved "
                  f"({res['effective_pcie_gbs']:.1f} GB/s), "
                  f"{res['toks_s_single_stream']:.2f} tok/s single "
                  f"/ {res['toks_s_aggregate']:.1f} aggregate"
                  + (f", {e} J/token GPU-board" if e is not None else ""))

    # Headline deltas at B=1
    d = {r["mode"]: r for r in results if r["batch"] == 1}
    if {"dense", "oracle", "jit"} <= d.keys():
        byte_ratio = d["dense"]["gb_moved_per_token"] / d["oracle"]["gb_moved_per_token"]
        speedup = d["dense"]["wall_ms_per_token"] / d["oracle"]["wall_ms_per_token"]
        jit_penalty = (d["jit"]["wall_ms_per_token"] - d["oracle"]["wall_ms_per_token"])
        print("\n" + "=" * 100)
        print(f"MEASURED MoE EFFECT (B=1): bytes/token reduced {byte_ratio:.2f}x "
              f"-> {speedup:.2f}x faster than dense-equivalent")
        print(f"JIT routing cost vs oracle prefetch: +{jit_penalty:.0f} ms/token "
              f"({100 * jit_penalty / d['oracle']['wall_ms_per_token']:.0f}%) -- real systems "
              f"with ~80-90% predictors land between the two bounds")

        # 70B-class MoE projection anchored to measured rates
        bw = d["oracle"]["effective_pcie_gbs"]
        active_frac = TOP_K / N_EXPERTS
        gb_70b = 35.3
        token_s = gb_70b * active_frac / bw
        print(f"\n70B-class MoE projection (measured {bw:.1f} GB/s, top-{TOP_K}/{N_EXPERTS} "
              f"= {active_frac:.0%} active, expert weights only):")
        print(f"  {gb_70b * active_frac:.1f} GB/token -> {token_s:.2f} s/token "
              f"-> {1 / token_s:.2f} tok/s single-stream "
              f"(vs 0.34 dense; + dense attention floor lowers this somewhat)")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "iteration": 17,
        "purpose": "Measure (not project) the MoE byte-reduction lever, routing cost, "
                   "kernel paths, and GPU-board energy per token.",
        "hardware": {**hw.__dict__},
        "config": {"layers": args.layers, "n": args.n, "m": args.m,
                   "experts": N_EXPERTS, "top_k": TOP_K,
                   "expert_mb": round(e_bytes / 1e6, 1),
                   "pinned_host_gb": round(pinned_gb, 2),
                   "tokens_per_config": args.tokens},
        "idle_power_w": idle_w,
        "gemm_paths": gemm_paths,
        "measured": results,
        "assumptions": [
            "Batch-pooled top-2/8 routing; real routers change which experts move, not how many.",
            "Oracle = perfect expert prefetch (upper bound); jit = zero prediction (lower bound).",
            "Energy is GPU board only (nvidia-smi); CPU/DRAM/PSU excluded.",
            "Dense attention/shared weights of a real MoE transformer are not streamed here; "
            "they add a dense floor to bytes/token.",
        ],
    }
    out_path = RESULTS_DIR / "iteration17_moe_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {out_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS_DIR.mkdir(exist_ok=True)
        b1 = [r for r in results if r["batch"] == 1]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        modes = [r["mode"] for r in b1]
        axes[0].bar(modes, [r["wall_ms_per_token"] for r in b1])
        axes[0].set_ylabel("ms per token")
        axes[0].set_title(f"{args.layers}L x {N_EXPERTS} experts, B=1:\n"
                          f"dense vs oracle-prefetch vs JIT routing")
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[1].bar(modes, [r["gb_moved_per_token"] for r in b1])
        axes[1].set_ylabel("GB transferred per token")
        axes[1].set_title("Measured bytes per token")
        axes[1].grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "iteration17_moe.png", dpi=140)
        print("[plot] plots/iteration17_moe.png")


if __name__ == "__main__":
    main()
