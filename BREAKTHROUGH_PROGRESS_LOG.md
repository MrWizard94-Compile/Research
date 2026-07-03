# BREAKTHROUGH PROGRESS LOG

**Project**: Recurrsive — Von Neumann Bypass for Practical Local Frontier AI
**Start Date**: 2026-07-02
**Mode**: Strict recursive self-improvement. Complete, production-ready artifacts only.

---

## Iteration 19 — Optimized INT4 Unpack + Combined Full-Stack Pipeline

**Date**: 2026-07-03
**Artifacts**: `iteration19_opt_unpack.py`, `iteration19_combined_moe.py`

### 1. Row-block INT4 + dequant-math folding — most of the fused-kernel prize, in stock PyTorch

Two zero-inference-cost changes attack iteration 18's dequant bottleneck:
row-block nibble packing (offline repack → contiguous unpack writes instead of
strided) and folding the per-row scale/zero-point into the activation
(`y = (x·s) @ Q − 8·(x·s).sum()` replaces two full [n×n] fp16 passes with O(n)
work). Measured on the 16-layer dense streaming testbed (B=1):

| format | wall/token | compute | bound | tok/s |
|---|---:|---:|---|---:|
| fp16 | 322 ms | 26 ms | PCIe | 3.10 |
| INT4 naive (iter. 18) | 206 ms | 200 ms | dequant | 4.87 |
| **INT4 row-block** | **103 ms** | 97 ms | dequant (barely) | **9.76 (3.15× vs fp16)** |

70B dense INT4 in stock torch rises from 0.14 → **0.28 tok/s**, against the
0.34 fused-kernel bound: **the remaining value of truly fused kernels is now a
measured 1.21×** (was 2.54×). WSL2 here has only a docker-desktop distro, so
the Triton path stays future work — now with a much smaller prize attached.

### 2. Combined full-stack pipeline — the levers compound

INT4 row-block experts (16.8 MB vs 67.1 MB fp16) + expert prefetch with real
on-demand miss fetches + `_int_mm` INT8 batched compute, all in one pipeline:

| mode | wall/token | GB moved | tok/s | J/token |
|---|---:|---:|---:|---:|
| B=1, fp16 GEMV, h=1.00 | 60.8 ms | 0.537 | 16.46 | 2.9 |
| **B=1, fp16 GEMV, h=0.85** | **61.5 ms** | 0.582 | **16.25** | **3.3** |
| B=32, `_int_mm`, h=1.00 | 68.5 ms | 0.537 | 466.9 aggregate | 3.7 |
| B=32, `_int_mm`, h=0.85 | 69.1 ms | 0.582 | 462.9 aggregate | 4.0 |

- vs iteration-17 fp16 MoE oracle (5.69 tok/s): **2.86×**.
- vs iteration-17 fp16 dense-equivalent (1.41 tok/s): **11.52×**, with energy
  26.9 → 3.3 J/token (**8× less energy per token**, GPU board).
- Realistic prefetch (h=0.85) costs ~1% in the combined stack — transfer is
  small enough that misses hide almost entirely.
- **70B-class MoE INT4, scaled from measured ms/GB: 1.07 tok/s single-stream**
  (before a real MoE transformer's dense attention floor; stated).

### Campaign scoreboard after 19 iterations (same machine, all measured)

| configuration | tok/s (B=1) | J/token |
|---|---:|---:|
| fp16 dense-equivalent streaming (the honest starting point) | 1.41 | 26.9 |
| + MoE top-2/8 (iter. 17) | 5.58 | 6.5 |
| + INT4 row-block transfer + folded dequant (iter. 19) | 16.46 | 2.9 |
| + realistic prefetch h=0.85 | 16.25 | 3.3 |
| + `_int_mm` at B=32 | 462.9 aggregate | 4.0 |

### Next steps (Iteration 20 candidates)

1. Real-checkpoint milestone (Mixtral-class INT4 via this pipeline or
   llama.cpp): unlocks real routing traces (true predictor hit rates), real
   quality, and closes the "random weights" caveat on every number above.
2. Triton fused dequant-GEMV under a real WSL2 dev distro — measured residual
   prize: 1.21× on dense INT4.
3. Wall-plug energy metering (hardware meter required).
4. Dense attention floor modeling for MoE transformers (the one stated gap in
   the 70B MoE numbers).

---

## Iteration 18 — Quantized Transfer, Prefetch Economics, Ratio at Scale

**Date**: 2026-07-03
**Artifacts**: `iteration18_quant_streaming.py`, `iteration18_prefetch_miss_curve.py`,
`iteration18_ratio_scale_probe.py`

### 1. Quantized transfer streaming — the dequant term, measured

Every prior 70B projection assumed INT4-packed bytes over PCIe with free
dequantization. Measured on the 16-layer testbed (B=1, overlapped pipeline,
preallocated dequant buffers):

| format | GB/token | transfer | dequant+GEMV | bound | tok/s | J/token |
|---|---:|---:|---:|---|---:|---:|
| fp16 | 3.88 | 316 ms | 26 ms | PCIe | 3.15 | 11.6 |
| INT8 → dequant | 1.94 | 160 ms | 132 ms | PCIe | **5.96 (1.89×)** | 9.0 |
| INT4 → unpack | 0.97 | 80 ms | 200 ms | **dequant** | 4.86 (1.54×) | 11.8 |

- **INT8 is the stock-PyTorch sweet spot**: dequant (dtype-cast `copy_` +
  `mul_`) nearly keeps pace with PCIe → 1.89× end-to-end.
- **INT4's 4× byte win is eaten by the stock unpack kernel** (4.8 GB/s packed):
  dense 70B INT4 on stock torch is **0.14 tok/s** (unpack-bound), not the 0.35
  transfer bound. **Fused dequant-GEMV kernels are worth a measured 2.54×** —
  the kernel workstream now has a precise value.
- Batched decode fix found in stock torch: `torch._int_mm` (INT8 DP4A) consumes
  **62.2 Gweights/s at B=8–32 vs 3.7–3.8 fp16** (16.4×; constraint M>16,
  B≤16 needs padding). The throughput model gains a third term:
  `per_layer = max(transfer, dequant + gemm)`.

### 2. Expert-prefetch miss-penalty curve — predictor accuracy priced

Real on-demand fetch per miss on the iteration 17 MoE testbed; hit rate
imposed (predictor design needs a real checkpoint — random weights have no
learnable routing structure, and pretending otherwise would measure nothing):

| hit rate | ms/token | misses | tok/s |
|---:|---:|---:|---:|
| 1.00 | 176 | 0 | 5.69 |
| 0.90 | 183 | 1.3 | 5.46 |
| 0.80 | 197 | 3.7 | 5.08 |
| 0.50 | 255 | 14.3 | 3.92 |
| 0.25 (chance) | 298 | 22.0 | 3.35 |

Analytic model `oracle + misses × 5.5 ms` matches within 1%. At published
80–90% hit rates (Pre-gated MoE, Fiddler, MoE-Infinity) the cost vs perfect
prefetch is **4–11%: predictor quality is not the bottleneck; bytes are.**

### 3. Ratio at scale — 1:3 is not rescued by width or budget

Doubled width (d=256, ~3.5M params) and budget (8000 steps), pure-SSM control
at the same width (`results/iteration18_ratio_scale.json`):

| arm | N=4 | N=64 | N=128 |
|---|---:|---:|---:|
| ratio 1:3 (SSAS, d=256) | 0.270 | 0.061 | 0.034 |
| ssm4 (SSSS, d=256) | 0.273 | 0.061 | 0.043 |

The 1:3 hybrid remains **indistinguishable from pure SSM** — its attention
layer contributes nothing measurable. Both arms also reproduce the late-
training destabilization (loss 2.02 → 3.17 after step ~5k) seen at d=128, so
it is a consistent optimization pathology of SSM-heavy stacks under this
protocol, not noise. Bounded conclusion across iterations 16–18: on this
testbed (d ≤ 256, ≤ 8000 steps, both placements), **the measured hybrid
frontier is 1:1 — transformer recall at half KV**; sparser ratios' published
success at production scale remains outside this testbed's envelope and is
recorded as an open question, not contradicted and not confirmed.

### Updated 70B-class picture on this machine (all measured or validated)

- Dense INT4, stock-torch kernels: 0.14 tok/s (unpack-bound — measured).
- Dense INT4, fused kernels: 0.35 tok/s (transfer bound — the kernel prize).
- MoE top-2/8 + 80–90% prefetch: ~1.2–1.3 tok/s.
- MoE + INT8 transfer for experts (fits: 2× bytes of INT4 but only active
  experts move): combinable with the above; INT8 measured 1.89× on dense.
- Batched decode: `_int_mm` restores 16.4× of the batched kernel gap.

### Next steps (Iteration 19 candidates)

1. Fused dequant-GEMV via Triton under WSL2 (measured prize: 2.54×) or
   llama.cpp kernel integration.
2. Combine levers in one pipeline: INT8 expert streaming + prefetch at
   realistic hit rates + `_int_mm` batched compute — one measured end-to-end
   number for the full stack.
3. Real-checkpoint milestone (Mixtral-class INT4): real routing traces enable
   an actual predictor and quality measurement on the streamed stack.
4. Wall-plug energy metering (needs hardware meter).

---

## Iteration 17 — MoE Streaming Measured, Kernel Paths, Energy, Hybrid Ratios

**Date**: 2026-07-02
**Artifacts**: `iteration17_moe_streaming.py`, `iteration17_hybrid_ratio_mqar.py`

Executes the four next steps set at the end of Iteration 16. All numbers below
are measured on the GTX 1660 Ti with the GPU otherwise idle.

### 1. MoE expert streaming — the biggest lever, measured end-to-end

16 layers × 8 experts × 67.1 MB (8.6 GB pinned host, every expert genuinely
allocated and streamed), tiny resident routers, top-2 routing, bytes counted as
copied:

| mode (B=1) | GB/token | wall/token | tok/s | J/token (GPU board) |
|---|---:|---:|---:|---:|
| dense-equivalent (all 8 experts) | 8.59 | 711 ms | 1.41 | 26.9 |
| MoE, oracle prefetch | 2.15 | 179 ms | **5.58** | **6.5** |
| MoE, just-in-time routing | 2.15 | 199 ms | 5.02 | 6.9 |

- **4.00× fewer bytes → 3.97× faster, ~4× less energy/token.** The iteration 16
  pipeline model holds under expert-granular streaming.
- **JIT routing costs +11% at B=1** (routing for layer i+1 needs layer i's
  output → CPU sync before each transfer). Oracle = perfect prefetch; published
  expert predictors (~80–90% hit) land between the bounds. At B=8 the JIT
  penalty grows to ~65% because the slow batched GEMM path no longer hides.
- 70B-class MoE anchor at measured PCIe: **1.37 tok/s** single-stream (vs. 0.34
  dense), before the dense attention-weight floor of a real MoE transformer.

### 2. Kernel-path sweep (iteration 16 follow-up)

Best stock-PyTorch fp16 GEMM on this TU116: 204 GB/s weight-read at B=1 (GEMV
path) vs. ~7.5 GB/s at B=8/32 across all tested formulations (`x@W`,
transposed, reduced-precision-reduction toggle). The ~26× batched-decode gap is
real in stock PyTorch; custom INT4/fused kernels (llama.cpp-class, or Triton
under WSL2) are the identified path back to the weight-read bound.

### 3. Energy methodology (GPU board, integrated)

J/token = sampled `nvidia-smi` power × wall time, idle (19.9 W) reported
alongside. Dense-equivalent 26.9 J/token → MoE 6.5–6.9 J/token at B=1. Scope is
the GPU board only (CPU/DRAM/PSU excluded — stated). Whole-system wall-plug
metering remains future work.

### 4. Hybrid ratio sweep on MQAR (4-block models, one protocol for all)

Protocol finding first: at iteration 16's lr=1e-3, every 4-block model with ≥2
SSM blocks fails to converge (loss ~2.0) while pure attention converges — which
would falsely read as an architectural failure. At **lr=3e-4, warmup 10%** the
1:1 hybrid converges fully; the sweep uses that one protocol for every model.

| model | pattern | N=64 held-out | N=128 (extrap.) | recall state @128k |
|---|---|---:|---:|---:|
| attn4 | AAAA | **1.000** | 0.417 | 537 MB |
| ratio 1:1 | SASA | **1.000** | 0.286 | **268 MB (half)** |
| ratio 1:3 | SSAS | 0.054 † | 0.031 | 134 MB |
| ssm4 | SSSS | 0.055 | 0.037 | ~0 MB |

† **Did not converge — and this was investigated, not assumed** (see
`results/iter17_ssas_diagnostic.txt`): attention-last placement (SSSA, 4000
steps) fails identically, and doubling the budget (SSAS, 8000 steps)
destabilizes late rather than converging. Combined with iteration 16's 2-block
[SSM→ATTN] reaching 100%, the bounded finding is: at d=128 scale, one attention
layer forms the recall circuit behind **one** SSM block but not behind three,
under every protocol tested. Production hybrids run 1:3–1:7 at vastly larger
width/depth and training budgets, so this flags small-scale training
difficulty, not a fundamental cap — but at this scale, **1:1 is the measured
frontier: transformer-level recall at half the KV.**

### Success-criteria movement

- Data-movement reduction: MoE streaming measured at 4× bytes/energy per token
  (on top of the fit-enabling architecture) — real progress toward the ≥10×
  criterion, honestly short of it on throughput-critical dense paths.
- Quality ≥95% with reduced memory: met by the 1:1 hybrid at half KV (through
  N=64, in-distribution); constant-memory (pure SSM) recall remains capacity-
  limited, as iteration 16 found.

### Next steps (carried into Iteration 18)

1. Custom INT4/fused decode kernels (WSL2 Triton or llama.cpp integration) to
   close the 26× batched-GEMM gap.
2. Expert-prefetch predictor to close the JIT→oracle routing gap; measure hit
   rate against recorded routing traces.
3. Scale the hybrid-ratio question (wider d_model, more heads/steps) to test
   whether sparse ratios (1:3+) recover recall with scale, as published hybrids
   indicate.
4. Wall-plug (whole-system) energy metering.
5. Milestone: stream a real MoE checkpoint (Mixtral-class INT4) through this
   pipeline end-to-end.

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
