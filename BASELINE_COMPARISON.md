# BASELINE_COMPARISON (Iteration 16 — Corrected, Measured)

**Iteration**: 16
**Date**: 2026-07-02

> The Iteration 15 version of this file claimed "2.06 tok/s, fits: yes" while the
> project's own saved results said `"fits": false`, and rested on a benchmark that
> never ran the model it reported. Retracted; see `BREAKTHROUGH_PROGRESS_LOG.md`.
> Everything below is measured on the GTX 1660 Ti 6 GB + 51.4 GB host machine or
> projected with the validated pipeline model (`ENERGY_MEMORY_MODEL.md`).

## Contenders (70B-class @ 128k context, this machine)

- **Baseline**: von Neumann transformer, full KV cache, GPU-resident.
- **Streamed hybrid**: weights in pinned host RAM, layers streamed over PCIe with
  double-buffered CUDA streams; fixed-state (SSM) layers for most of the stack,
  sparse attention layers for recall (hybrid), MoE sparsity where applicable.

## Memory Footprint @ 128k

| | weights | recall cache/state | VRAM needed | fits 6 GB + 48 GB? |
|---|---:|---:|---:|---|
| Transformer (GQA, fp16 KV) | 35.3 GB (INT4) | 42.9 GB, linear in ctx | ~78 GB | **No** |
| Streamed hybrid | 35.3 GB (host RAM) | O(MB) constant + KV only for sparse attn layers | **0.77 GB measured peak** | **Yes** |

This is the campaign's real, surviving win: the model *fits*, and VRAM stays
under 1 GB regardless of model size, because only the active working set is
resident.

## Throughput (measured + validated projection)

| case | tok/s |
|---|---:|
| Baseline transformer, GPU-resident | does not fit — N/A on this machine |
| Streamed dense 70B INT4, B=1 | **0.34** (PCIe-bound; hard bound = 35.3 GB/token ÷ 12.3 GB/s) |
| + 13% VRAM residency | 0.40 |
| MoE ~30% active per token, B=1 | 1.15 |
| Batch/speculative B=8 | 1.6 aggregate (2.8 if weight-read-bound kernels) |
| Batch/speculative B=32 | 5.3 aggregate (11.0 if weight-read-bound kernels) |

Measured anchors: 16 × 0.242 GB layers → 3.15 tok/s; 48 × 0.242 GB → 1.06 tok/s;
overlap efficiency 93–98%; projection error vs. measurement 0–6%.
**There is no configuration on this hardware in which dense 70B streaming is
"usable-interactive" single-stream; claiming otherwise was the Iteration 15
error.** The levers that change the picture: MoE sparsity, batching, residency,
and a PCIe 4/5 host (≈2–4× the 12.3 GB/s measured here).

## Quality (trained, held-out — `iteration16_quality_mqar.py`)

Multi-query associative recall, matched-size models, identical training
(chance = 1.6%; N = pairs stored in context; N=128 is length extrapolation):

| arch | N=16 | N=64 | N=128 | recall state @128k |
|---|---:|---:|---:|---:|
| transformer | 100% | 100% | 48.9% | 268 MB (linear) |
| pure fixed-state SSM | 12.5% | 5.6% | 3.8% | ~0 MB (constant) |
| hybrid (1 SSM + 1 attn) | 100% | 100% | 32.2% | 134 MB (half) |

Honest reading: constant-state recall capacity is real but small; hybrids keep
attention-class recall while cutting KV (here 2×; production 1:7 hybrid ratios
cut deeper). The retracted "constant state with perfect quality" claim is
replaced by this measured trade-off.

## Power / Thermal

Board power limit 120 W (real, from nvidia-smi — prior docs' 450 W was wrong).
During PCIe-bound streaming: 37 W avg / 39 W peak, ~39 °C. Low power reflects
the GPU waiting on transfers; it is not an efficiency claim.

## Verdict

- **Fits on consumer 6 GB + 48 GB**: yes — measured, the central feasibility result.
- **Usable interactive speed for dense 70B**: no — ~0.34 tok/s, PCIe-bound.
- **Credible path to usable**: MoE (~3×) × batching/speculative (×B) × residency
  (~1.2×) × PCIe 4/5 host (~2–4×), and hybrid layer ratios for quality — each
  lever quantified in `results/iteration16_results.json`.

Reproduction: `python iteration16_corrected_streaming.py --plot` and
`python iteration16_quality_mqar.py --steps 4000 --plot`.
