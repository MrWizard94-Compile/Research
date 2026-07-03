# CURRENT ARCHITECTURE (Iterations 16–18 — Corrected Streaming, Measured MoE, Quantized Transfer)

**Iterations**: 16 (audit + corrected dense streaming) · 17 (MoE expert streaming, kernels, energy, hybrid ratios) · 18 (quantized transfer, prefetch economics, ratio at scale)
**Date**: 2026-07-03
**Artifacts**: `iteration16_corrected_streaming.py`, `iteration16_quality_mqar.py`,
`iteration17_moe_streaming.py`, `iteration17_hybrid_ratio_mqar.py`,
`iteration18_quant_streaming.py`, `iteration18_prefetch_miss_curve.py`,
`iteration18_ratio_scale_probe.py`

> **Correction notice**: The Iteration 15 version of this document reported results
> from a benchmark that never ran the model it claimed (an 8-layer, <10 MB proxy was
> measured while 120 × 0.45 GB layers were reported), used a fictional "overlap"
> metric, and projected throughput with physically impossible bandwidth math.
> All Iteration 10–15 performance/quality claims are retracted; see
> `BREAKTHROUGH_PROGRESS_LOG.md` (Iteration 16) for the itemized retraction.

## Core Principles (unchanged in intent, now honestly measured)

- Weights live in pinned host RAM; only a small working set (2–3 layer-sized
  buffers) resides in VRAM.
- A dedicated transfer stream copies layer *i+1* over PCIe while a compute stream
  runs layer *i* — classic double-buffered streaming (the same family of technique
  as llama.cpp offload / FlexGen / DeepSpeed-Inference).
- Fixed-state (SSM-style) context handling keeps recall-time memory constant in
  context length, versus a transformer KV cache that grows linearly.

## What Is Actually Measured (GTX 1660 Ti 6 GB, 51.4 GB host, torch 2.12.1+cu126)

Every layer below was genuinely allocated in pinned host RAM and genuinely
streamed over PCIe every token. Timing via CUDA events on the owning streams;
validation against a truly-serial reference run of identical work.

**16 layers × 0.242 GB (3.9 GB streamed per token):**

| batch | wall/token | serial | ideal max(T,C) | overlap eff. | VRAM peak | throughput |
|------:|-----------:|-------:|---------------:|-------------:|----------:|-----------:|
| 1     | 317 ms     | 340 ms | 315 ms         | 93%          | 0.75 GB   | 3.15 tok/s |
| 8     | 580 ms     | 910 ms | 560 ms         | 94%          | 0.76 GB   | 13.8 tok/s aggregate |
| 32    | 683 ms     | 979 ms | 663 ms         | 94%          | 0.77 GB   | 46.8 tok/s aggregate |

**48 layers × 0.242 GB stress (11.6 GB streamed per token):**

| batch | wall/token | serial  | ideal   | overlap eff. | throughput |
|------:|-----------:|--------:|--------:|-------------:|-----------:|
| 1     | 946 ms     | 1026 ms | 945 ms  | 98%          | 1.06 tok/s |
| 8     | 2131 ms    | 3055 ms | 2110 ms | 98%          | 3.8 tok/s aggregate |

Key measured rates: **PCIe 12.3 GB/s** effective in-pipeline; compute (fp16 GEMV,
B=1) reads weights at **~149 GB/s**; at B=8/32 this GPU's fp16 GEMM path drops to
**6–7 GB/s** (Turing TU116 has no tensor cores) — real batched deployments need
optimized INT4 decode kernels to stay near the weight-read bound.

Wall time is within 2% of the ideal pipeline floor `layers × max(transfer, compute)`:
the streaming implementation is essentially perfect; **the bound itself is PCIe**.

## 70B-Class Projection (correct pipeline model, validated ±0–6% vs measurement)

`per_layer = max(transfer, compute)`. Prefetch depth does **not** divide transfer
time — PCIe is a single shared link. (Iteration 15's `transfer/prefetch` model was
wrong by ~6×.)

Dense 70B INT4 = 80 layers × 0.441 GB = 35.3 GB streamed per token:

| case | single-stream | aggregate |
|------|--------------:|----------:|
| dense, B=1 | **0.34 tok/s** | 0.34 tok/s |
| dense, B=1, 13% VRAM-resident | 0.40 tok/s | 0.40 tok/s |
| dense, B=8 (batch/speculative) | 0.20 tok/s | 1.56 tok/s (2.76 if weight-read-bound) |
| dense, B=32 | 0.17 tok/s | 5.27 tok/s (11.0 if weight-read-bound) |
| MoE ~30% active, B=1 | **1.15 tok/s** | 1.15 tok/s |
| MoE 30% + 13% resident, B=8 | 0.65 tok/s | 5.20 tok/s (10.6 if weight-read-bound) |

**Honest bottom line**: dense 70B streaming on this machine is ~0.3–0.4 tok/s,
period. The real levers are MoE sparsity (stream only active experts), batching /
speculative decoding (amortize each weight pass), residency, and faster host links
(PCIe 4/5 hardware scales these numbers roughly linearly).

## MoE Expert Streaming — the biggest lever, now measured (Iteration 17)

`iteration17_moe_streaming.py`: 16 layers × 8 experts × 67.1 MB (8.6 GB pinned
host), expert-granular transfers, tiny resident routers, top-2 routing. Bytes
counted as copied, not assumed:

| mode (B=1) | GB/token | wall/token | tok/s | J/token (GPU board) |
|---|---:|---:|---:|---:|
| dense-equivalent | 8.59 | 711 ms | 1.41 | 26.9 |
| MoE, oracle prefetch | 2.15 | 179 ms | **5.58** | 6.5 |
| MoE, just-in-time routing | 2.15 | 199 ms | 5.02 | 6.9 |

Measured: **4.00× fewer bytes → 3.97× faster and ~4× less energy per token.**
The JIT-vs-oracle spread (+11% at B=1) is the true cost of routing depending on
the previous layer's output; real expert-prefetch predictors land between the
bounds. 70B-class MoE anchor at measured PCIe: **~1.37 tok/s** single-stream
(vs. 0.34 dense), before a real MoE transformer's dense attention-weight floor.

The kernel-path sweep in the same artifact confirms the batched fp16 GEMM gap on
this GPU is real across stock formulations (204 GB/s at B=1 vs ~7.5 GB/s at
B=8/32) — custom INT4/fused kernels are the identified path back to the
weight-read bound for batched decode.

## Quantized Transfer + Prefetch Economics (Iteration 18)

Two measured refinements close the remaining assumption gaps:

**Quantized transfer** (`iteration18_quant_streaming.py`): streaming INT8 bytes
and dequantizing on-GPU is the stock-PyTorch sweet spot — **1.89× end-to-end
(3.15 → 5.96 tok/s on the 16-layer testbed)**, still PCIe-bound. INT4-packed
transfer moves 4× fewer bytes but the stock unpack kernel (4.8 GB/s packed)
makes the pipeline dequant-bound: 1.54× only, and dense 70B INT4 lands at
0.14 tok/s instead of the 0.35 transfer bound. **Fused dequant-GEMV kernels are
worth a measured 2.54×** — that, plus `torch._int_mm` for batched decode
(62.2 vs 3.7 Gweights/s, 16.4×), defines the kernel workstream precisely.

**Prefetch economics** (`iteration18_prefetch_miss_curve.py`): tok/s vs expert-
predictor hit rate, with real on-demand fetches per miss, matches
`oracle + misses × 5.5 ms` within 1%. At published ~80–90% hit rates the cost
vs perfect prefetch is only 4–11%: **predictor quality is not the bottleneck on
this pipeline; bytes are.**

## Memory Model @ 128k Context (the part of the original thesis that survives)

- Transformer KV cache (70B-class GQA, fp16): **42.9 GB** — cannot fit 6 GB VRAM;
  weights + KV ≈ 78 GB.
- Streamed fixed-state model: VRAM peak **measured 0.77 GB**, weights 35.3 GB in
  host RAM, recurrent state O(MB) and constant in context.

## Power / Thermal (real limits)

GTX 1660 Ti board power limit: **120 W** (read from nvidia-smi; the 450 W figure in
prior docs was wrong). During the measured PCIe-bound load: avg 37 W, peak 39 W,
~39 °C. Low SM power is expected — the workload waits on transfers; it is *not*
evidence of architectural energy efficiency.

## Quality

Assessed non-circularly in `iteration16_quality_mqar.py`: transformer, SSM, and
hybrid models of matched size are trained end-to-end on multi-query associative
recall and evaluated held-out. No harness-written lookup tables, no hardcoded
baselines. See `BASELINE_COMPARISON.md` for the numbers and the honest trade-off
(fixed-state recall capacity vs. constant memory).

## Reproduction

```powershell
python iteration16_corrected_streaming.py --plot            # 16 x 0.242 GB
python iteration16_corrected_streaming.py --stress --plot   # 48 x 0.242 GB
python iteration16_quality_mqar.py --steps 4000 --plot
```
