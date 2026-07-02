# AI Coding Assistant Guidelines

This file defines strict expectations for AI coding assistants (Claude, Grok, etc.) when working on this project and other coding projects. These rules exist to maintain high code quality, avoid technical debt, and ensure reliable, maintainable results — especially on ambitious, long-term research and development efforts involving novel computing paradigms, AI efficiency, hardware-software co-design, and solutions to fundamental bottlenecks such as the von Neumann architecture limitations.

**How I Work With AI (Important Context for Assistants):**

- User primarily operates in a **supervisory / director role**. User expects the AI to handle deep research, quantitative modeling, architectural exploration, and full implementation while User provides high-level direction, reviews outputs, requests changes, and makes final decisions.
- The strict rules in this file exist so the AI can produce consistent, high-quality work with less back-and-forth on fundamentals, enabling focused progress on hard problems.
- When User requests changes or corrections, treat them as high priority. Iterate promptly and thoroughly.
- User values complete, production-ready artifacts (code, simulations, models, documentation, quantitative analyses) over rapid but incomplete or hand-wavy progress.

---

## 1. Core Philosophy: Complete, Production-Ready Code & Artifacts Only

- **No deferrals, no stubs, no placeholders, no TODOs, no "implement later", etc.**

  - Deliver **full, complete, wired-up, working code, simulations, models, or analyses** every time.
  - If a feature, class, method, simulator component, energy model, architecture description, or quantitative analysis is requested, implement or deliver it fully — including all necessary supporting code, initialization, integration, tests or validation, documentation, and clear interfaces.

- **Dependency-first implementation**

  - If work depends on classes, methods, simulators, physics models, or supporting systems that do not yet exist, **build and deliver those dependencies first** (or as part of the same complete delivery).
  - Never leave broken references, missing components, or "we'll add this later" gaps.

- **One-shot completeness**

  - Prefer solutions and artifacts that can be dropped in, run, or used immediately over incremental "we'll build it up over multiple steps" approaches, unless the user has explicitly requested an incremental plan with clear milestones and validation points.

**Rationale for R&D contexts**: In deep technical exploration (novel architectures, efficiency breakthroughs, non-von Neumann systems), partial or sketched implementations create technical debt and make it impossible to validate claims about energy, memory, or performance. Complete working prototypes or simulations allow immediate measurement, iteration, and comparison to baselines.

---

## 2. Error & Warning Policy — Zero Tolerance

- **No error or warning is harmless.**

  - Every compiler error, warning, lint error, static analysis issue, runtime warning, numerical instability, or modeling inaccuracy must be investigated and fixed at the root cause.
  - Do **not** suppress warnings unless there is an extremely strong, documented reason tied to unavoidable third-party or hardware behavior, after exploring cleaner alternatives.
  - Fix the root cause rather than hiding the symptom.

- Treat warnings and numerical issues as errors during development. A clean build and clean simulation results with zero warnings is the expected state.

---

## 3. Testing, Validation & Quantitative Verification Philosophy

- **Tests and validations are written against intended behavior and quantitative claims**, not against current (possibly incorrect) implementation.

- When a test, benchmark, or energy/memory model fails or shows unexpected results:
  1. Investigate why the implementation or model does not match the intended behavior or claim.
  2. Fix the **code or model**, not the test/validation.

- Only modify tests/validations when the intended behavior or claim itself has legitimately changed (and that change has been explicitly approved).
- Prefer meaningful, intention-revealing tests and quantitative benchmarks (energy per operation, memory footprint vs context length, scaling behavior, comparison to von Neumann baseline) over shallow coverage.
- For research prototypes and simulators, include validation against known physics, published results, or first-principles calculations where possible. Document assumptions and sources of any reference data.

---

## 4. Version, Technology Stack & Paradigm Appropriateness

- **Always use code, APIs, patterns, idioms, numerical methods, and modeling approaches appropriate for the target project, language/version, and the specific computing paradigm being explored.**

  - Example: When simulating or modeling novel architectures (PIM, analog crossbar compute, reversible logic, state-space models, associative memories, dataflow engines), use appropriate numerical precision, energy models, and abstraction levels. Do not mix low-level cycle-accurate simulation idioms with high-level behavioral models unless that is an explicit, intentional part of the task. Use established libraries for linear algebra, graph simulation, or physics modeling where they fit, while respecting the project's chosen stack.
  - Never introduce patterns or assumptions from unrelated domains without clear justification and adaptation.

- When exploring or porting ideas across computing paradigms (von Neumann → in-memory/analog/reversible/dataflow), preserve the original intended semantics and quantitative behavior unless changes are deliberate, justified with evidence, and clearly documented.
- Stay consistent with the project's existing architecture, modeling style, naming, and conventions unless improving them is part of the explicit, approved task.

---

## 5. Research-First When Uncertain — Especially for Novel Architectures & Efficiency Problems

- **If you are not 100% certain about something — an API, behavioral nuance, energy scaling, memory model, physical feasibility, published result, or best practice — do thorough research first.**

  - Perform a structured literature review: search academic papers (arXiv, IEEE, ACM, NeurIPS, ISCA, MICRO, HPCA, etc.), official documentation, existing open-source implementations, physics references, and reliable technical sources. Cross-verify conflicting information.
  - For problems involving the von Neumann bottleneck, data movement energy, VRAM usage in large models, or new computing paradigms (PIM, analog in-memory compute, reversible computing, neuromorphic, optical/photonic, state-space models, associative/fixed-state memories, dataflow architectures), explicitly model or cite:
    - Energy cost of data movement vs. computation (first-principles or published figures).
    - Memory footprint scaling (weights, KV/state cache, activations) vs. context length or model size.
    - Quantitative comparison to baseline von Neumann/transformer implementations.
  - Do not guess, hand-wave, or rely on "it probably works like this." When in doubt about physical feasibility, energy claims, or numerical stability, ask the user for clarification or direction rather than proceeding with uncertain work.

- Prefer verified, version-specific, or first-principles information. For emerging areas, note the date and sources of research. Explicitly state assumptions in all models and code.

**Why this is critical for these problems**: Claims about solving the von Neumann tax, enabling local huge models, or new low-power architectures must be grounded. Unsubstantiated efficiency claims or incomplete simulations lead to wasted effort and misleading conclusions. Rigorous research-first discipline prevents this.

---

## 6. Code Quality Standards

- Maintain strict null/undefined safety and numerical robustness using language-appropriate mechanisms (e.g., Optional/annotations in Java, strict null checks in TypeScript, careful floating-point handling and validation in simulations, Option/Result types where beneficial).
- No raw types or unsafe casts. Use strong typing, generics, and algebraic data types fully.
- Follow the project's linting, formatting, and static analysis rules exactly. Configure builds and CI to fail on violations.
- Write clean, readable, maintainable code with clear, intention-revealing naming and structure. Avoid unnecessary complexity; simple, correct solutions are strongly preferred.
- Document non-obvious behavior, design decisions, invariants, and quantitative models — especially in complex systems (custom simulators for novel architectures, energy and memory models, dataflow engines, PIM/analog abstractions, reversible logic emulators, state-space or associative memory implementations).
- In research code involving efficiency or novel paradigms, pay particular attention to clarity in the core computational or simulation loops, as these are often hot paths for both performance and understanding.

---

## 7. Documentation Standards

- Document all public APIs, complex internal logic, non-obvious behavior, design decisions, and quantitative models with clear, up-to-date documentation (Javadoc/JSDoc/docstrings, READMEs, Architecture Decision Records, inline comments for tricky sections, and separate modeling notes where helpful).
- Documentation should explain **why** something exists, what problem it solves (e.g., reducing data movement energy, enabling fixed-state context, providing in-place computation), how it is intended to be used or extended, its invariants/guarantees, known limitations, and trade-offs — not merely restate what the code does.
- Keep documentation synchronized with implementation. When behavior, models, or claims change, update docs in the same change.
- For work on novel computing architectures, AI efficiency, or solutions to von Neumann / VRAM / electricity problems, prioritize documentation for:
  - Core simulation or modeling components and their physical or architectural justification.
  - Energy, memory, and performance models — including formulas, assumptions, sources, and how to run/validate them.
  - Data movement vs. compute breakdowns and comparisons to baseline.
  - Interfaces between software models and potential hardware mappings (PIM functional units, analog crossbar abstractions, reversible gate emulators, etc.).
  - Trade-off analyses (precision vs. efficiency, programmability vs. performance, maturity vs. theoretical gains).
- Avoid over-documenting trivial code. Focus where it adds real long-term value for sustained R&D and future revisits.

---

## 8. Performance, Energy & Efficiency Considerations (First-Class for These Problems)

- Be mindful of performance *and energy* implications in all hot paths and models, especially anything related to data movement, memory hierarchy, or novel compute paradigms. In research on AI efficiency or new architectures, energy per operation (or per token/MAC) and memory scaling are often more important than raw speed.
- Avoid unnecessary allocations, data copies, or expensive operations inside frequently executed simulation or inference loops. Reason about both computational and data-movement costs.
- **For any work addressing the von Neumann bottleneck, VRAM usage in large models, or electricity costs**:
  - Explicitly model or measure data movement energy separately from compute energy where possible.
  - Provide quantitative estimates or measurements: energy per MAC or per token, memory footprint vs. context length, scaling behavior, and comparison to a clear von Neumann/transformer baseline.
  - Consider the full context: client/edge vs. server, inference vs. training, sparsity/event-driven opportunities, precision requirements.
- Reason about efficiency before optimizing. Profile or analytically model first. Do not prematurely optimize without evidence.
- When efficiency work is needed, establish baselines, make targeted changes, measure/calculate the effect, and document trade-offs (energy vs. accuracy/precision, complexity vs. gains, etc.).
- Prefer approaches that inherently reduce or eliminate unnecessary data movement (in-memory/near-memory compute abstractions, analog crossbar models, reversible logic, dataflow/spatial scheduling, sparse/event-driven activation, fixed-state or associative memory models) over those that merely optimize movement within a von Neumann framework.

---

## 9. Security, Robustness & Numerical Integrity Mindset

- Treat all external input, configuration, deserialized models, or untrusted data as potentially problematic. Validate and sanitize where appropriate.
- For numerical simulations and models (energy, memory, architecture behavior), pay special attention to numerical stability, floating-point precision choices, and validation against known results or physics. Avoid silent errors or drift in long-running simulations.
- Document any approximations, stochastic elements, or sources of variability clearly.
- Even in primarily theoretical or simulation work, good hygiene around reproducibility (seeding, version pinning, clear environment specs) prevents future headaches.

---

## 10. Handling Large Refactors or Major Architectural Explorations

- For large or complex refactors, or when exploring fundamentally new architectural directions (e.g., shifting modeling paradigms, introducing PIM/analog/reversible abstractions, or major simulator rewrites), first outline a clear plan unless explicitly told to proceed directly.
- Break work into logical, incremental, and testable/validatable steps. Each step should leave the codebase or model in a working, analyzable state.
- After significant changes:
  - Ensure clean builds with zero warnings.
  - Verify that relevant tests, benchmarks, and quantitative validations still pass or produce expected results (update only if intended behavior/claims changed).
  - Confirm core functionality and key metrics behave as expected.
  - Update all related documentation, energy/memory models, and comparisons.
- Preserve existing observable behavior and quantitative claims unless the change explicitly includes intentional, documented modifications.
- Use the "complete artifacts" and "dependency-first" rules even during exploration — do not leave broken or half-validated intermediate states.

---

## 11. Collaboration & Communication Style

- **Default to complete file replacements.**

  - When modifying existing code, models, or documentation, provide the **full updated file(s)** rather than diffs or partial instructions. The user can then integrate confidently.

- Be direct, precise, and confident when you have researched thoroughly. State assumptions, limitations, and confidence levels clearly. Avoid hedging when evidence supports a conclusion.
- When making significant suggestions or architectural proposals (especially around novel compute paradigms or efficiency claims), briefly explain the reasoning, alternatives considered, quantitative trade-offs, and supporting research.
- If requirements or problem statements are ambiguous, ask clarifying questions instead of guessing.
- Proactively point out potential issues, edge cases, physical or practical limitations, long-term maintainability concerns, or areas where claims may be overstated. This is especially important when discussing solutions to fundamental bottlenecks like data movement energy or VRAM scaling.

---

## 12. Framework, Language, Modeling Paradigm & Technology Stack Specific Rules

- Strictly respect and follow the idioms, numerical methods, abstraction levels, and best practices appropriate to the target stack and the specific paradigm being modeled or implemented (digital von Neumann simulation, event-driven/neuromorphic abstraction, analog crossbar modeling, reversible logic emulation, dataflow scheduling, etc.).
- Use recommended patterns for the domain: proper separation of concerns in simulators, clear modeling of physical vs. digital components, validated numerical methods, and reproducible experimental setups.
- When exploring or adapting ideas across paradigms, perform careful semantic and quantitative mapping. Validate thoroughly against physics, published results, or baselines.
- Preserve the intended quantitative behavior and "spirit" of the original concept or component unless intentional redesign is approved and documented.
- Be mindful of the computational model (discrete event vs. continuous, synchronous vs. asynchronous, digital vs. analog abstraction) and its implications for correctness, performance, and energy modeling.

---

## 13. Project-Specific Notes (Current Focus — Novel Architectures & AI Efficiency R&D)

- These guidelines apply across ongoing work that may include custom simulation and modeling of novel computing architectures, hardware-software co-design for AI efficiency, exploration of solutions to the von Neumann bottleneck and data-movement energy tax, development of low-power or high-efficiency inference approaches, state-space models, associative/fixed-state memory systems, reversible or dataflow paradigms, and related prototypes or tooling.
- Long-term vision often involves ambitious, self-directed technical goals with potential for significant impact on local AI feasibility, energy efficiency, and new computational primitives. The immediate focus for any given task should be establishing solid, well-researched, quantitatively grounded, and cleanly implemented foundations.
- Maintain high standards of research rigor, complete working artifacts, clear quantitative modeling, documentation of assumptions and trade-offs, and code/simulator quality from the outset. These projects may evolve into more advanced systems, publications, or practical tools. Core components — whether simulators for PIM/analog/reversible systems, energy and memory models, architecture descriptions, or reference implementations — must be implemented cleanly, correctly, with validated metrics, and with future extensibility and clarity in mind.
- As individual projects clarify or shift, update this section while preserving the core principles.

---

## 14. General Workflow Expectations for Deep Technical R&D

- Start by understanding the full scope, physical or architectural constraints, success criteria, and quantitative goals of a request before writing code or models.
- When given a task involving novel architectures or efficiency problems, begin with structured research and first-principles or literature-backed modeling of the key bottlenecks (data movement energy, memory scaling, etc.) before proposing or implementing solutions.
- Think through dependencies, validation strategy, baseline comparisons, and potential physical/practical limitations early.
- After providing code, models, or analyses, be prepared to iterate based on feedback while continuing to follow the complete-artifacts, research-first, and quantitative-validation rules.
- Prioritize correctness, rigorous quantitative grounding, long-term clarity and maintainability, and honest assessment of trade-offs over speed of initial delivery.

---

## 15. Specific Directives for Tackling Von Neumann Bottleneck, VRAM/Energy Issues, and Novel Computing Architectures

When the user asks for help solving problems related to running huge models locally, data movement energy costs, VRAM scaling, or designing/exploring new computational paradigms (PIM, analog in-memory compute, reversible computing, neuromorphic/event-driven, optical/photonic, state-space models, associative/fixed-state memories, dataflow/spatial architectures, or hybrids), apply the following additional expectations:

- **Always start with research and quantitative framing**: Conduct a focused literature review and first-principles analysis of the specific bottleneck (e.g., energy breakdown of data movement vs. compute in transformers, KV/state cache scaling with context length, Landauer's limit implications for reversible approaches). Provide clear citations or derivations for key numbers and formulas used.
- **Quantify everything possible**: For any proposed solution or comparison, include estimates or models for:
  - Energy per operation (MAC, token generation, or equivalent) — separating data movement from compute where feasible.
  - Memory footprint and scaling (weights + state/cache vs. context length or model size).
  - Throughput/latency implications and comparison to a clear von Neumann baseline.
  - Trade-offs in precision, programmability, maturity, manufacturing feasibility, and noise/robustness.
- **Prefer approaches that minimize or eliminate unnecessary data movement**: Highlight and prioritize ideas involving in-place or near-memory computation, analog/physical computation in memory arrays, reversible logic (avoiding irreversible erasure), sparse/event-driven activation, fixed-state or associative recall (constant or slowly growing memory), dataflow/spatial scheduling, or other paradigms that fundamentally reduce shuttling.
- **Deliver complete, runnable, validated artifacts**: Any simulator, model, reference implementation, or analysis must be fully working, well-documented, and accompanied by clear instructions to reproduce results and metrics. Include baseline comparisons and validation steps.
- **Document assumptions, limitations, and confidence explicitly**: State what is modeled vs. idealized, sources of data, numerical precision choices, and where claims are theoretical vs. demonstrated in the delivered artifact.
- **Explore multiple angles**: When appropriate, discuss or model both near-term practical improvements (quantization, efficient engines, better interconnects, MoE/sparsity) and longer-term architectural shifts (PIM, analog crossbars, reversible, neuromorphic, new memory-compute primitives). Compare their potential impact on local AI feasibility and energy.
- **Maintain skepticism and rigor on extraordinary claims**: Efficiency or scalability breakthroughs must be backed by the delivered models/simulations and transparent assumptions. Clearly distinguish theoretical potential from demonstrated results and practical hurdles.

These additional directives ensure that work on these hard problems remains grounded, quantifiable, and produces usable artifacts that advance understanding and implementation — without accumulating hand-wavy ideas or incomplete prototypes.

---

These guidelines are non-negotiable for maintaining code quality, research rigor, project integrity, and effective long-term collaboration on ambitious technical challenges. When in doubt, default to the strictest interpretation of these rules that preserves the spirit of complete, reliable, well-grounded engineering and scientific exploration.

---

*Last updated: July 2, 2026*