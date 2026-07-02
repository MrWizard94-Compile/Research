# Iteration 4 Plan — Scale Quality Fidelity + Unified Artifact

**Date**: 2026-07-02  
**Campaign Context**: Recurrsive — Von Neumann Bypass for Practical Local Frontier AI  
**Decision Record**: Primary focus = Option 1 from the 2–4 options analysis.

## 1. Objective
Deliver **one unified, complete, runnable artifact** that, for the first time in this campaign, demonstrates in a single execution:

- Constant (or near-constant) memory/state at 128k+ context
- Significant data-movement energy reduction (target 20–100× modeled vs strong vN transformer baseline)
- Credible quality/robustness preservation (target ≥90–95% relative to baseline on relevant long-context tasks, or clear quantification of the gap)

The artifact must meet all production-ready standards (clean code, full documentation, validation, reproducible outputs).

## 2. Scope & Deliverables
- One primary Python script (e.g. `iteration4_unified_hybrid_sim.py`)
- Full integration of:
  - Refined first-principles energy/memory model (movement vs compute separation)
  - Hybrid architecture (improved SSM + fixed-capacity associative memory)
  - Scaled reference implementation (larger dimensions or small real proxy model)
  - Real long-context quality task (needle retrieval + at least one additional metric)
- Outputs:
  - Console + saved table with all required metrics (accuracy, memory vs context, energy per token broken down, throughput proxy, deltas)
  - Plots (joint Pareto, scaling curves, energy breakdown)
  - JSON/CSV results
  - Clear "Success Criteria Dashboard" printout
- Updated documentation in code + RECURSIVE_STATE.md

## 3. Key Technical Steps (in priority order)
1. Scale the compute model (D_MODEL 256–512, N_LAYERS 8–12, larger d_state/associative slots).
2. Improve quality evaluation:
   - Longer, more realistic needle-in-haystack (multi-fact, variable difficulty)
   - Add state-tracking or multi-hop recall metric
   - Report mean + std + confidence
3. Wire the refined energy model on every evaluation point using realistic 70B-class parameters for energy while running the scaled reference forward pass.
4. Add explicit PIM and reversible modes with clear labeling.
5. Produce unified reporting + comparison to strong transformer baseline.
6. Add sensitivity (e.g. number of associative slots, context length).
7. Validation notes against published Mamba-3/hybrid and PIM results.

## 4. Success Targets for This Iteration
- Runnable script that a user executes and sees:
  - Quality score documented relative to baseline (aim for credible path to or demonstration of ≥95% relative on the proxy tasks)
  - Memory flat or logarithmic
  - Movement energy reduction clearly in the 20–100×+ range at 128k
- Honest quantification of remaining gaps (scale, real weights, etc.)
- All code complete, no stubs, documented, reproducible.

## 5. Risks & Mitigations
- Runtime: Provide `--quick` flag (fewer trials, capped context 32k).
- Quality still below target: Be transparent in the dashboard and plots; show the curve.
- No real weights available: Use significantly larger synthetic dimensions + note the limitation.

## 6. Why This Over Other Options
This option has the highest leverage on the current dominant blocker (quality demonstration at scale) while maximally reusing the strongest previous deliverables (refined energy model + hybrid architecture). It directly moves the needle on the "or" conditions in the Breakthrough Success Criteria.

## 7. Timeline Within Campaign
- Primary deliverable: Complete runnable unified artifact + results.
- Secondary (if time): Lightweight reversible primitive sketch inside the energy model.

## 8. Success Criteria Tie-in
This iteration is explicitly designed to strengthen evidence for Breakthrough Success Criterion 2 while maintaining full compliance with 1, 3, 4, and 5.

---

**Status**: Planning complete. Ready for implementation of Iteration 4 artifact upon user confirmation.