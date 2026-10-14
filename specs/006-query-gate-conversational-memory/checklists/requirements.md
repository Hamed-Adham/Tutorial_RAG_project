# Specification Quality Checklist: Feature 006 QueryGate & Conversational Memory Routing

**Purpose**: Validate specification completeness and quality before proceeding to planning  
**Created**: 2026-10-14  
**Last Updated**: 2026-10-14  
**Feature**: [spec.md](../spec.md)  

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) in user stories or functional requirements
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain (All 3 architectural decisions and user refinements resolved)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified (cold-start turn-0 fallback, SubQ request chunking, missing summary backfill timeout guard, zero-needed fallback)
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows (intent gating, need dispatching, history evaluation, retrieval augmentation, incremental summary caching, and UI manual selection)
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- All clarifications and refinements finalized:
  1. Cold-start Turn 0: Auto-coerce Option 1 to Option 3 (`needs_retrieve_only`) and Option 2 to friendly prompt ("هنوز گفتگویی انجام نشده است، لطفاً پرسش خود را مطرح کنید").
  2. Citation policy for `needs_history_only`: Returns empty citations list (`citations: []`) and displays a distinct conversational badge: "پاسخ بر اساس گفتگوی پیشین".
  3. Missing summary backfill: Deterministically check existence of `q_and_a_summary` across the last 20 turns and call the summary generator LLM once if any are missing.
  4. Decision Model 2 SubQ Architecture: Uses SubQ Decisions API (`https://api.subq.ai/v1/systemone`, model `subq-snap-preview`) with a 10-10 parallel batch chunking structure.
  5. Context fidelity & cap: No artificial numerical limit on Decision Model 2 needed turns; needed Q&As in the recent 5 messages include full Q&A text; older needed turns (turns 6–20) include concise summaries.
  6. Qwen-style UI Reply & Context Selection: When history is OFF, each message card includes a "Reply / Select as Context" button with a dismissible floating preview chip above the input box, passing the selected turn directly as context and bypassing Model 2.
- Specification is 100% complete and validated. Ready for `/speckit-plan`.
