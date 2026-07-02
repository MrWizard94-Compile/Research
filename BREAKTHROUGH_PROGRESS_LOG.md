# BREAKTHROUGH PROGRESS LOG

**Project**: Recurrsive — Von Neumann Bypass for Practical Local Frontier AI
**Start Date**: 2026-07-02
**Mode**: Strict recursive self-improvement. Complete, production-ready artifacts only.

---

## Iteration 16 — Audit, Retraction, and Corrected Baselines

**Date**: 2026-07-02

### Retraction of Iterations 10–15

A code audit found that the headline results of iterations 10–15 were artifacts
of measurement and modeling errors, not breakthroughs. Per the project's own
rules (CLAUDE.md §2, §3, §15), the record is corrected as follows.

| Retracted claim (iter. 10–15) | What was actually true | Corrected result (iter. 16, measured) |
|---|---|---|
| "120 layers @ 0.45 GB (~100B class) ran at 85–102 tok/s" | `stream_and_compute` capped at `self.n_layers = 8`; an 8-layer, <10 MB model ran; claimed layers never allocated | 48 × 0.242 GB layers genuinely streamed: 1.06 tok/s (B=1); 16 × 0.242 GB: 3.15 tok/s |
| "Overlap 413–675×" | Metric divided a hypothetical big-model time by the tiny model's time; measured nothing | Overlap efficiency 93–98% of hidable time (CUDA events, vs. true serial reference); wall within 2% of the physical floor |
| "70B @ 128k projected 2.06 tok/s (prefetch=6), 2.5 compute-bound" | Projection divided PCIe transfer time by prefetch depth — impossible on one shared link | Dense 70B INT4: **0.34 tok/s** (PCIe-bound). MoE ~30% active: 1.15 tok/s. Batch B=32: up to ~5–11 tok/s aggregate. Validated ±0–6% vs. measurement |
| "Needle/state quality ≥95–99%, stable at any context" | Circular: harness wrote facts into slots, then read them back; iter. 11 transformer baseline was hardcoded (`0.031 + noise`), never run | MQAR, all models trained end-to-end, held-out: transformer 100% (N≤64), pure fixed-state SSM 27%→6% as N grows, hybrid 100% at half the KV. See below |
| "Reversible emulation recovers 55% of compute energy" | Multiplier with no physical mechanism on shipped CMOS GPUs | Removed. No energy credits without a mechanism |
| "36.5 W under heavy sustained load (450 W TDP)" | GPU board limit is 120 W; low power showed the GPU was waiting on PCIe, not efficiency | Reported as-is with the real 120 W limit and the correct interpretation |
| "70B fits: yes" (BASELINE_COMPARISON, iter. 15) | The saved results JSON itself said `"fits": false` (54 GB > 48 GB host) | 80 × 0.441 GB = 35.3 GB **does** fit host RAM; measured VRAM peak 0.77 GB |

Deprecation banners were added to `iteration10–15*.py` and
`quality_sustainability_diagnostic.py`; they remain in the repo as historical
record only.

### Corrected Artifact 1: `iteration16_corrected_streaming.py`

Actually allocates N pinned-host layers of the stated size, streams every layer
over PCIe every token on dedicated CUDA streams (double/triple-buffered), runs a
real weight-read-bound GEMM per layer, and times everything with CUDA events
against a genuinely-serial reference.

**Measured (GTX 1660 Ti 6 GB, 51.4 GB host, torch 2.12.1+cu126):**
- PCIe effective: 12.3 GB/s. Weight-read compute: 149 GB/s (fp16 GEMV, B=1); 6–7 GB/s at B=8/32 (TU116 has no tensor cores).
- 16 L × 0.242 GB (3.9 GB/token): 317 ms/token B=1 (3.15 tok/s), overlap 93–94%, VRAM peak 0.75–0.77 GB. B=32: 46.8 tok/s aggregate.
- 48 L × 0.242 GB stress (11.6 GB/token): 946 ms/token B=1 (1.06 tok/s), overlap 98%. No OOM; scaling exactly linear in transferred bytes, as physics requires.
- Power during load: 37 W avg / 39 W peak (limit 120 W), ~39 °C.

**Projection (correct model `per_layer = max(transfer, compute)`, validated +5.7%/−0.0%/+0.1%/+1.9% against the four measured configs):** dense 70B INT4 ≈ 0.34 tok/s; +13% VRAM residency → 0.40; MoE ~30% active → 1.15; batch/speculative B=8–32 → 1.6–5.3 tok/s aggregate measured-kernel, 2.8–11 tok/s if weight-read-bound (optimized INT4 kernels).

### Corrected Artifact 2: `iteration16_quality_mqar.py`

Non-circular quality benchmark: multi-query associative recall (Zoology,
arXiv:2312.04927; Mamba, arXiv:2312.00752). Transformer, fixed-state SSM, and
hybrid (1 SSM + 1 attention) of matched size are all trained end-to-end
(4000 steps, identical protocol) and evaluated held-out. No harness-written
lookup tables; no hardcoded baselines.

**Held-out accuracy (chance = 0.016; N = key-value pairs in context; N=128 is length extrapolation):**

| arch | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 | recall state @128k ctx |
|---|---|---|---|---|---|---|---|
| transformer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.489 | 268 MB (linear) |
| ssm (pure fixed-state) | 0.272 | 0.171 | 0.125 | 0.082 | 0.056 | 0.038 | ~0 MB (constant) |
| hybrid (1 SSM + 1 attn) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.322 | 134 MB (half) |

**Honest reading**: exact in-context recall does not come free with a constant
state — a small fixed state measurably loses bindings as N grows (consistent
with the published MQAR literature). Hybrids recover attention-class recall at
reduced KV. This replaces the retracted "constant state, perfect quality" story
with the real, quantified trade-off.

### What survives of the campaign thesis

1. **Memory**: 70B-class fits a 6 GB + 48 GB machine via host-resident weights
   (35.3 GB) + streaming (VRAM peak measured 0.77 GB); a 128k transformer KV
   cache (42.9 GB) does not fit. Constant-state layers keep recall memory flat.
2. **Throughput is PCIe-bound and modest**: ~0.34 tok/s dense; the honest levers
   are MoE sparsity, batching/speculative decoding, residency, and faster links
   (PCIe 4/5 scale these ~linearly).
3. **Quality**: hybrid ratios are the credible path (quantified above), not pure
   fixed state.

### Reproduction

```powershell
python iteration16_corrected_streaming.py --plot
python iteration16_corrected_streaming.py --stress --plot
python iteration16_quality_mqar.py --steps 4000 --plot
```

---

## Iterations 1–15 (historical)

Iteration 1 (`iteration1_sim.py`) remains a valid first-order analytical model
of memory scaling and data-movement dominance. Iterations 10–15 are retracted
per the table above; their entries and scripts are preserved unedited (with
banners) in `RECURSIVE_STATE.md` and the repository history for the record.
