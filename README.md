# RECURSIVE — Von Neumann Bypass for Practical Local Frontier AI

**Mission**: Deliver real, measurable, reproducible progress toward eliminating the
von Neumann data-movement energy tax and VRAM scaling walls that prevent
frontier-scale AI (70B+ class) from running locally on consumer/edge hardware.

This is a long-term recursive engineering campaign. Every claim must be backed by
a runnable artifact and validated measurement.

## Current Status (2026-07-03, Iteration 18)

**Iteration 18 (latest)** — quantized transfer, prefetch economics, and the
ratio-at-scale answer, all measured (`results/iteration18_*.json`):

- **INT8 streaming is the stock-PyTorch sweet spot**: 1.89× end-to-end
  (3.15 → 5.96 tok/s on the 16-layer testbed), still PCIe-bound. INT4-packed
  transfer is dequant-bound in stock torch (unpack 4.8 GB/s), putting dense 70B
  at 0.14 tok/s — **fused dequant-GEMV kernels are worth a measured 2.54×**,
  and `torch._int_mm` (INT8 DP4A) restores 16.4× of the batched-decode gap.
  The throughput model gains a third term: `max(transfer, dequant + gemm)`.
- **Prefetch economics measured**: tok/s vs expert-predictor hit rate matches
  `oracle + misses × 5.5 ms` within 1%; published 80–90% predictors cost only
  4–11% vs perfect prefetch. Predictor quality is not the bottleneck — bytes are.
- **Ratio at scale**: doubling width and budget does not rescue the 1:3 hybrid —
  it stays indistinguishable from pure SSM (control at same width). The measured
  hybrid frontier on this testbed remains **1:1 = transformer recall at half KV**;
  sparser ratios' production-scale success is an open question here, recorded
  with evidence, neither contradicted nor confirmed.

**Iteration 17** executed the four next steps from Iteration 16, all
measured on the target machine (GTX 1660 Ti 6 GB + 51.4 GB host):

- **MoE expert streaming measured end-to-end** (`iteration17_moe_streaming.py`):
  16 layers × 8 experts × 67.1 MB genuinely streamed; top-2 routing moves
  **4.00× fewer bytes → 3.97× faster (1.41 → 5.58 tok/s) and ~4× less energy
  (26.9 → 6.5 J/token, GPU board)** than the dense equivalent. Just-in-time
  routing costs only +11% vs. perfect prefetch at B=1. 70B-class MoE anchor:
  **~1.37 tok/s** single-stream (vs. 0.34 dense).
- **Kernel-path sweep**: the batched fp16 GEMM slowdown on this GPU (204 GB/s at
  B=1 vs ~7.5 GB/s at B=8/32) is real across stock PyTorch formulations; custom
  INT4/fused kernels are the identified fix.
- **Energy methodology**: integrated GPU-board power per token (idle reported
  separately; CPU/DRAM/PSU excluded — stated scope).
- **Hybrid ratio sweep on MQAR** (`iteration17_hybrid_ratio_mqar.py`): the 1:1
  SSM:attention hybrid reaches **transformer-level recall (100% through N=64) at
  half the KV**. The sparser 1:3 ratio did not converge at this small scale —
  investigated across placement and 2× budget, documented rather than hidden.
  Protocol finding: deep recurrent hybrids need lr 3e-4 where lr 1e-3 falsely
  looks like architectural failure.

## Iteration 16 — Audit and Corrected Baselines

**⚠️ Iterations 10–15 are retracted.** An audit (recorded in
`BREAKTHROUGH_PROGRESS_LOG.md`) found their headline results were measurement and
modeling artifacts: benchmarks that silently ran an 8-layer <10 MB model while
reporting 120 × 0.45 GB layers, a fictional "overlap" metric, throughput
projections that divided PCIe transfer time by prefetch depth (physically
impossible), circular quality evaluations with a hardcoded baseline, an
unphysical "reversible energy recovery" credit, and a wrong 450 W TDP for a
120 W GPU. The retracted scripts carry deprecation banners.

**Current ground truth** comes from two corrected, validated artifacts:

### 1. `iteration16_corrected_streaming.py` — performance

Genuinely allocates pinned-host layers of the stated size, streams every layer
over PCIe every token (double-buffered CUDA streams), computes real GEMMs against
the streamed weights, and times everything with CUDA events against a
truly-serial reference.

Measured on GTX 1660 Ti 6 GB + 51.4 GB host (PCIe ≈ 12.3 GB/s effective):

- 16 × 0.242 GB layers (3.9 GB streamed/token): **3.15 tok/s**, overlap 93–94%, VRAM peak 0.77 GB.
- 48 × 0.242 GB stress (11.6 GB streamed/token): **1.06 tok/s**, overlap 98%. No OOM.
- Wall time sits within 2% of the physical floor `layers × max(transfer, compute)` —
  the implementation is essentially perfect; **the floor itself is PCIe**.

Validated projection for dense 70B INT4 (35.3 GB streamed/token): **≈0.34 tok/s**
single-stream. Levers, each quantified in the artifact: MoE ~30% active → 1.15
tok/s; batching/speculative B=32 → up to ~5–11 tok/s aggregate; VRAM residency
→ +18%; PCIe 4/5 hosts scale ~linearly. Projection error vs. measurement: 0–6%.

### 2. `iteration16_quality_mqar.py` — quality

Non-circular recall benchmark (MQAR; Zoology arXiv:2312.04927, Mamba
arXiv:2312.00752). Transformer, fixed-state SSM, and hybrid models of matched
size are all trained end-to-end and evaluated held-out — no harness-written
lookup tables, no hardcoded baselines. Result (chance 1.6%):

- Transformer: 100% recall through N=64 pairs; cache grows linearly (268 MB @128k).
- Pure fixed-state SSM: 27% → 6% as N grows; state constant (~0 MB).
- Hybrid (1 SSM + 1 attention): 100% through N=64 at half the KV.

Constant-state recall is not free — hybrids are the credible quality path, and
the trade-off is now measured instead of asserted.

## What Survives of the Thesis

1. **70B-class fits consumer hardware** (the feasibility result): 35.3 GB of INT4
   weights in host RAM + <1 GB VRAM working set, measured. A 128k transformer KV
   cache (42.9 GB) does not fit — constant/reduced-state architectures matter.
2. **Dense streaming throughput is PCIe-bound and modest** (~0.34 tok/s here).
   Anything faster must reduce bytes moved per token (MoE, residency, hybrid
   ratios) or amortize them (batching, speculative decoding).
3. **Iteration 1's analytical simulator** (`iteration1_sim.py`) remains a valid
   first-order model of memory scaling and data-movement dominance.

## Success Criteria (unchanged, honestly scored)

1. Complete, runnable, documented artifacts — **met** (iteration 16).
2. ≥10–100× movement-energy reduction / constant-memory ≥95% quality / 70B on
   ≤48 GB end-to-end — **partially met**: memory criterion demonstrated
   (fits, constant state); quality ≥95% holds for hybrids on MQAR, not pure
   fixed state; energy claims are withdrawn pending a defensible methodology.
3. Reproducible benchmarks + baseline comparison — **met** (validated ±0–6%).
4. Honest assumptions, limitations, maturity — **restored by this iteration**.
5. Extensible, clean code — **met** for iteration 16 artifacts.

## Reproduce

```powershell
# System Python 3.10 with torch 2.12.1+cu126 (see requirements.txt notes)
python iteration16_corrected_streaming.py --plot            # 16 x 0.242 GB
python iteration16_corrected_streaming.py --stress --plot   # 48 x 0.242 GB
python iteration16_quality_mqar.py --steps 4000 --plot
python iteration17_moe_streaming.py --plot                  # MoE lever, measured
python iteration17_hybrid_ratio_mqar.py --steps 4000 --plot # ratio sweep
python iteration18_quant_streaming.py --plot                # INT8/INT4 transfer
python iteration18_prefetch_miss_curve.py --plot            # predictor economics
python iteration18_ratio_scale_probe.py                     # 1:3 at d=256
python iteration1_sim.py                                    # analytical model
```

Outputs land in `results/` (JSON) and `plots/` (PNG); both are gitignored —
regenerate them by running the artifacts.

## Documentation Map

- `BREAKTHROUGH_PROGRESS_LOG.md` — iteration log; Iteration 16 contains the
  itemized retraction table and corrected results.
- `CURRENT_ARCHITECTURE.md` — the streaming architecture and measured behavior.
- `ENERGY_MEMORY_MODEL.md` — the validated pipeline/memory model and measured
  constants; explains why prefetch cannot divide transfer time.
- `BASELINE_COMPARISON.md` — honest head-to-head vs. the von Neumann baseline.
- `RECURSIVE_STATE.md` — full campaign history (correction notice at top;
  historical entries preserved unedited).
- `CLAUDE.md` — the working rules this audit enforced.

## Next Steps (Iteration 19 candidates)

1. Fused dequant-GEMV kernels — Triton under WSL2 or llama.cpp integration.
   The prize is now measured: 2.54× on dense INT4 streaming.
2. Combine the measured levers in one pipeline (INT8 expert streaming +
   realistic prefetch + `_int_mm` batched compute) for a single end-to-end
   number for the full stack.
3. Real-checkpoint milestone (Mixtral-class INT4): real routing traces enable
   an actual predictor and quality measurement on the streamed stack.
4. Whole-system (wall-plug) energy metering — requires a hardware meter.

**Rule going forward (CLAUDE.md §15): claim only what the artifact demonstrates.**
