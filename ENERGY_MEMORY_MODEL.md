# ENERGY / MEMORY / PERFORMANCE MODEL (Iterations 16–17 — Corrected, Measured)

**Iterations**: 16 (dense streaming, corrected model) + 17 (MoE, kernels, energy)
**Date**: 2026-07-02

> **Correction notice**: The Iteration 15 version of this model divided PCIe
> transfer time by prefetch depth (physically impossible on a shared link) and the
> Iteration 10–14 energy models multiplied compute energy by (1 − 0.55) as
> "reversible recovery" — a fudge factor with no physical mechanism on shipped CMOS
> GPUs. Both are removed. All constants below are measured on this machine or
> derived from the correct pipeline model, and the model is validated against
> measured wall times.

## Throughput Model (streamed decode, weights in host RAM)

For a model of `L` layers of `G` GB each, PCIe bandwidth `BW_pcie`, and GPU
weight-read rate `BW_comp` (how fast the GPU can consume weights in GEMM/GEMV):

```
transfer_ms(layer) = G * active_frac * (1 - resident_frac) / BW_pcie * 1000
compute_ms(layer)  = G * active_frac / BW_comp * 1000
token_ms           = L * max(transfer_ms, compute_ms) + transfer_ms   # + pipeline fill
tok/s (aggregate)  = batch * 1000 / token_ms
```

- `active_frac`: fraction of weights touched per token (1.0 dense; ~0.3 typical MoE).
- `resident_frac`: fraction of active weights kept permanently in VRAM.
- **Prefetch depth does not appear in this model.** Overlap can hide the smaller
  of (transfer, compute) behind the larger — never shrink the larger. Measured
  overlap efficiency on this machine: 93–98% of the hidable time.

## Measured Constants (GTX 1660 Ti, 2026-07-02, `results/iteration16_results.json`)

| constant | value | how measured |
|---|---|---|
| PCIe H2D effective | **12.3 GB/s** | CUDA events on transfer stream, pinned buffers, in-pipeline |
| Weight-read rate, fp16 GEMV (B=1) | **149 GB/s** | CUDA events on compute stream |
| Weight-read rate, fp16 GEMM (B=8/32) | **6–7 GB/s** | same (TU116 has no tensor cores; batched fp16 GEMM path is slow on this GPU) |
| GPU board power limit | **120 W** | nvidia-smi power.limit (not 450 W) |
| Power during PCIe-bound streaming | 37 W avg / 39 W peak, ~39 °C | nvidia-smi sampled during the measured run |

## Validation (model vs. measurement, same machine, same run)

| config | predicted | measured | error |
|---|---:|---:|---:|
| 16L × 0.242 GB, B=1 | 335 ms | 317 ms | +5.7% |
| 16L × 0.242 GB, B=8 | 580 ms | 580 ms | −0.0% |
| 16L × 0.242 GB, B=32 | 684 ms | 683 ms | +0.1% |
| 48L × 0.242 GB, B=1 (stress) | 964 ms | 946 ms | +1.9% |

## Memory Model

```
Transformer (GQA, fp16):  KV_bytes = 2 * L * ctx * (kv_heads * head_dim) * 2
  70B-class @ 128k: 2 * 80 * 131072 * 1024 * 2 = 42.9 GB   (grows linearly with ctx)

Streamed fixed-state model:
  VRAM   = slots * G + activations          (measured peak: 0.77 GB)
  Host   = L * G weights                    (35.3 GB for 70B INT4)
  State  = O(L * d_model) floats            (MBs, constant in ctx)
```

This memory contrast — linear KV vs. constant state — is the surviving, correct
core of the campaign's thesis, and it is what makes 70B-class *fit* at all on a
6 GB + 48 GB machine. Throughput, however, is governed by the pipeline model
above and is PCIe-bound at ~0.34 tok/s dense (B=1).

## MoE Extension (Iteration 17 — measured, not projected)

`active_frac` is now a measured quantity. 16 layers × 8 experts × 67.1 MB
(8.6 GB pinned host), top-2 routing, real expert-granular transfers
(`iteration17_moe_streaming.py`, `results/iteration17_moe_results.json`):

| mode (B=1) | GB moved/token | wall/token | tok/s | J/token (GPU board) |
|---|---:|---:|---:|---:|
| dense-equivalent (all 8 experts) | 8.59 | 711 ms | 1.41 | 26.9 |
| MoE oracle prefetch (top-2) | 2.15 | 179 ms | 5.58 | 6.5 |
| MoE just-in-time routing | 2.15 | 199 ms | 5.02 | 6.9 |

- Byte reduction measured **4.00×** → speedup **3.97×**: the pipeline model holds
  under expert-granular streaming.
- **JIT routing cost is +11%** at B=1 (routing for layer *i+1* depends on layer
  *i*'s output, forcing a CPU sync before each transfer). Oracle = perfect
  prefetch; published expert predictors (~80–90% hit) land between the bounds.
  At B=8 the penalty grows to ~65% because compute (slow batched GEMM path on
  this GPU) no longer hides behind transfer.
- 70B-class MoE anchor: 35.3 GB × 25% active = 8.8 GB/token → **1.37 tok/s**
  single-stream at measured PCIe (vs. 0.34 dense), before the dense
  attention-weight floor of a real MoE transformer.

## Kernel-Path Sweep (Iteration 17 — the B>1 gap is real in stock PyTorch)

fp16 [8192×8192] GEMM on this TU116 (no tensor cores), best of three
formulations (`x@W`, transposed, full-precision-reduction toggle):

| batch | best path | weight-read rate |
|---:|---|---:|
| 1 | `x@W` (GEMV path) | 204 GB/s |
| 8 | `x@W` | 7.7 GB/s |
| 32 | `(W.T@x.T).T` | 7.5 GB/s |

No stock formulation closes the ~26× gap; batched-decode projections on this
GPU must use the measured slow rate (as iteration 16 does), with custom
INT4/fused kernels (llama.cpp-class, or Triton under WSL2) as the identified
path back to the weight-read bound.

## Energy Accounting (Iteration 17 methodology — GPU board, integrated)

Method: sample `nvidia-smi power.draw` (0.3 s) during each measured config;
J/token = avg power × wall time; idle board power (measured 19.9 W) reported
alongside for incremental figures. **Scope: GPU board only — CPU, host DRAM,
and PSU losses are excluded**; this is stated, not hidden.

Measured at B=1: dense-equivalent streaming 26.9 J/token → MoE 6.5–6.9 J/token
(**~4× energy reduction per token**, tracking bytes moved). Per-MAC/pJ-level
claims from iterations 10–15 (including all "reversible recovery" credits)
remain retracted; whole-system energy (wall-plug) is future work.

## Reproduction

```powershell
python iteration16_corrected_streaming.py --plot
python iteration16_corrected_streaming.py --stress --plot
python iteration17_moe_streaming.py --plot
```
