# ENERGY / MEMORY / PERFORMANCE MODEL (Iteration 16 — Corrected)

**Iteration**: 16
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

## Energy Accounting (honest scope)

At 37 W and 0.34 tok/s dense-streamed, energy is roughly 110 J/token — dominated
by wall-clock time at near-idle board power, not by MAC energy. Per-MAC/pJ-level
energy claims from iterations 10–15 (including all "reversible recovery" credits)
are retracted; a defensible energy comparison requires either wall-power
integration over a real inference workload or a first-principles model with
published per-byte movement costs, cleanly separated from these throughput
measurements. That is future work, stated as such.

## Reproduction

```powershell
python iteration16_corrected_streaming.py --plot
python iteration16_corrected_streaming.py --stress --plot
```
