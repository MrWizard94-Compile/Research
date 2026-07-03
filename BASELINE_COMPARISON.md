# BASELINE_COMPARISON (Iterations 16–17 — Corrected, Measured)

**Iterations**: 16 (dense streaming + MQAR) · 17 (MoE, kernels, energy, ratios)
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

| case | tok/s | basis |
|---|---:|---|
| Baseline transformer, GPU-resident | does not fit | — |
| Streamed dense 70B INT4, fused kernels | 0.34 (PCIe bound) | validated projection |
| Streamed dense 70B INT4, stock torch naive | 0.14 (unpack-bound) | measured (iter. 18) |
| Streamed dense 70B INT4, **stock torch row-block** | **0.28** | **measured** (iter. 19; residual fused-kernel value 1.21×) |
| + 13% VRAM residency | 0.40 | validated projection |
| **70B-class MoE INT4, combined stack, h=0.85** | **1.07** | **measured ms/GB, scaled** (iter. 19; attention floor not modeled) |
| Batched decode, `_int_mm` INT8 | 62.2 vs 3.7 Gweights/s fp16 (16.4×) | measured kernel |

Measured testbed progression (16 layers, B=1, same machine):
fp16 dense-equivalent 1.41 tok/s / 26.9 J → +MoE 5.58 / 6.5 → **+INT4 row-block
+ prefetch h=0.85: 16.25 tok/s / 3.3 J (11.5× faster, 8× less energy)**;
B=32 `_int_mm`: 463 tok/s aggregate. Dense path: fp16 3.10 → INT8 5.96 →
INT4 row-block 9.76 tok/s.

Measured anchors: 16 × 0.242 GB layers → 3.15 tok/s; 48 × 0.242 GB → 1.06 tok/s;
overlap efficiency 93–98%; projection error vs. measurement 0–6%. MoE testbed
(16 L × 8 experts, B=1): dense-equivalent 1.41 → MoE 5.58 tok/s measured, energy
26.9 → 6.5 J/token (GPU board).
**There is no configuration on this hardware in which dense 70B streaming is
"usable-interactive" single-stream; claiming otherwise was the Iteration 15
error.** The levers that change the picture: MoE sparsity, batching, residency,
and a PCIe 4/5 host (≈2–4× the 12.3 GB/s measured here).

## Quality (trained, held-out — `iteration16_quality_mqar.py` + iteration 17 ratio sweep)

Multi-query associative recall, matched-size models, one training protocol per
sweep (chance = 1.6%; N = pairs stored in context; N=128 is length extrapolation).

2-block models (iteration 16):

| arch | N=16 | N=64 | N=128 | recall state @128k |
|---|---:|---:|---:|---:|
| transformer | 100% | 100% | 48.9% | 268 MB (linear) |
| pure fixed-state SSM | 12.5% | 5.6% | 3.8% | ~0 MB (constant) |
| hybrid (1 SSM + 1 attn) | 100% | 100% | 32.2% | 134 MB (half) |

4-block ratio sweep (iteration 17; lr=3e-4 protocol — see log for why):

| model | pattern | N=64 | N=128 | recall state @128k |
|---|---|---:|---:|---:|
| attn4 | AAAA | 100% | 41.7% | 537 MB |
| ratio 1:1 | SASA | **100%** | 28.6% | **268 MB (half)** |
| ratio 1:3 | SSAS | 5.4% † | 3.1% | 134 MB |
| ssm4 | SSSS | 5.5% | 3.7% | ~0 MB |

† Did not converge; investigated — attention-last placement and a 2× training
budget both fail the same way (`results/iter17_ssas_diagnostic.txt`), and the
iteration 18 scale probe (d=256, 8000 steps, pure-SSM control) shows the 1:3
hybrid remains **indistinguishable from pure SSM** even at doubled width. At
this testbed's scale one attention layer forms the recall circuit behind one
SSM block (iteration 16) but not behind three. Production hybrids run sparser
ratios at far larger scale; **on this testbed, 1:1 is the measured frontier:
transformer-level recall at half the KV.** The retracted "constant state with
perfect quality" claim stays replaced by this measured trade-off.

## Power / Thermal

Board power limit 120 W (real, from nvidia-smi — prior docs' 450 W was wrong).
During PCIe-bound streaming: 37 W avg / 39 W peak, ~39 °C. Low power reflects
the GPU waiting on transfers; it is not an efficiency claim.

## Verdict

- **Fits on consumer 6 GB + 48 GB**: yes — measured, the central feasibility result.
- **Usable interactive speed for dense 70B**: no — ~0.34 tok/s, PCIe-bound.
- **MoE lever: now measured, not projected** — 4.00× fewer bytes → 3.97× faster
  and ~4× less energy per token; JIT routing costs +11%. 70B-class MoE ≈ 1.37
  tok/s single-stream on this machine.
- **Quality at reduced memory**: 1:1 SSM:attention hybrid = transformer recall
  at half KV (measured); sparser ratios need scale this testbed doesn't have.
- **Remaining path to usable**: custom INT4 kernels for batched decode (measured
  26× gap), expert-prefetch prediction, PCIe 4/5 host (~2–4×).

Reproduction: `python iteration16_corrected_streaming.py --plot`,
`python iteration16_quality_mqar.py --steps 4000 --plot`,
`python iteration17_moe_streaming.py --plot`,
`python iteration17_hybrid_ratio_mqar.py --steps 4000 --plot`.
