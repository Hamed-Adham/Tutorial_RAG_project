---
name: langfuse-decision-models
description: Build and operate decision-model evaluators in Langfuse (Choice, Score, and Yes/no questions answered by OpenAI gpt-6-luna or TypeSafe Jev) that write typed scores instead of free-form verdicts. Use when classifying traces, routing intents, rating rubrics, flagging guardrails, or when an LLM judge is too slow, too expensive, or too inconsistent for the metric.
metadata:
  required_access:
    - LANGFUSE_PROJECT_INTERFACE
---

# Decision-Model Evaluators

Decision models answer **typed questions** and return one typed score per question.
They skip free-form generation, so they are cheap, fast, calibrated, and consistent —
but they return no written rationale.

> **Experimental:** the setup is stable, but UI details and score format may change.
> Verify against the live docs before building on specifics.

## When to use one

Use a decision model when the possible answers are known **before** the call:
classification, routing, rubric levels, guardrail flags, and yes/no policy checks.

Use an LLM judge instead when you need written reasoning, or when the answer space
is too open-ended to define upfront. Combining both works well: run decision models
broadly and cheaply, then send a sample or the flagged subset to an LLM judge.

| Situation | Use |
| --- | --- |
| "Which topic is this?" from a fixed taxonomy | Decision model (Choice) |
| "How frustrated is the user?" on 4 ordered levels | Decision model (Score) |
| "Does this output contain PII?" | Decision model (Yes/no) |
| "Why did this answer go wrong?" | LLM judge |

## Anatomy of an evaluator

1. **Model input** — observation data mapped to the provider's expected state.
   OpenAI receives one general state named `input`; Jev receives a JSON state with
   named fields.
2. **One or more questions** — each has a type, instructions, and allowed answers.
   All questions are answered in **one call**.
3. **One score per question** — each typed answer becomes a score on the
   observation. Probabilities and confidence go to score metadata, never to the value.

## Question types

| Type | Returns | Score | Good for |
| --- | --- | --- | --- |
| **Choice** | 1 option from a fixed set (2–255), with probabilities | `CATEGORICAL` = selected option | Topic detection, intent, routing, failure mode, which tool should have been called |
| **Score** | Position on 2–10 ordered levels, with probabilities | `NUMERIC` = expected level `0`…`levels-1` | Rubrics: severity, frustration, completeness |
| **Yes/no** | `P(true)` (TypeSafe calls this a **Noul**) | `NUMERIC` between 0 and 1 | Guardrails: out-of-scope, PII, policy violation |

Rules that decide most outcomes:

- **Choice** — give every option a short description and always include an escape
  hatch (`other`, `unclear`). Jev must pick one option and cannot signal "none fit".
- **Score** — describe every level in plain language, ordered low to high. The value
  is probability-weighted, so `1.7` means "between level 1 and 2, closer to 2".
- **Yes/no** — optionally describe what counts as `true` and what counts as `false`.
  Because you get a probability, pick a threshold matching your risk tolerance, or
  use three bands: act / review / ignore.
- **Keep each question atomic** — one decision per question. "Is the answer correct
  and polite?" splits into two questions. Both are answered in the same call anyway,
  so splitting costs nothing.

## OpenAI vs. TypeSafe Jev

| | OpenAI (`gpt-6-luna`) | TypeSafe Jev |
| --- | --- | --- |
| State | One general state named `input` | JSON state with named fields |
| Question instructions | Plain text; cannot reference parts of the state | Can reference fields with backticks: ``Compare `output` with `expectedOutput`.`` |
| Score levels | Label required, description optional | Description required |
| Refusals | Can refuse; one refusal **fails the whole run** and writes no scores | Always picks an answer |

Connections:

- **OpenAI** — Settings → LLM Connections, add an OpenAI connection with access to
  `gpt-6-luna`. Langfuse calls the OpenAI Decisions API through it.
- **TypeSafe** — adapter `typesafe`, choose the upstream, paste the key. Upstreams:
  TypeSafe (billed by TypeSafe), Vercel AI Gateway, or OpenRouter (billed through them).
  Vercel and OpenRouter expose the same TypeSafe API, so no base URL or extra headers
  are needed. Use `jev-latest` to follow releases, or pin e.g. `jev-1.13.0` when an
  evaluation threshold must stay stable. TypeSafe connections are available **only**
  to decision-model evaluators because Jev cannot generate text.

## Mapping the model input

Map only what the decision needs: observation input, output, metadata, tool calls,
**Expected Output**, or **Experiment Item Metadata**. Jev state fields are referenced
by name in backticks; OpenAI's single `input` state is evaluated as a whole.

## Setup steps

1. **Connect credentials** — Settings → LLM Connections.
2. **Create the evaluator** — Evaluators page → New evaluator → New decision model
   evaluator, then pick connection and model. Templates prefill input and questions:
   - *Assign Input Topic* (Choice) — topic taxonomy classification
   - *Flag Out-of-Scope Request* (Yes/no) — off-role requests
   - *Rate Customer Frustration* (Score) — four frustration levels
   - *User Conversation Signal* (Yes/no × 7) — rephrases, corrections, hand-off
     requests, repeats, error quotes, frustration, success confirmation in one call
3. **Define the questions** — one per criterion: pick type, write instructions,
   define allowed answers, set the score name. Langfuse suggests a score name from the
   question.
4. **Map the model input** from observation data.
5. **Test** — filter to representative samples, run on one, and read the per-question
   probability distribution, confidence, and estimated cost. Use **Raw output** to see
   the exact provider request. Iterate until samples look right.
6. **Save, then attach** — create a rule from the test filters (or attach to an
   existing rule) to run on incoming observations. Without a rule you can still use it
   for batch evaluation and prompt experiments.

Score names should name the **metric**, not the evaluator (`refund_requested`, not
`refund judge`) — the name lands on every observation the evaluator touches.

## Scores written by decision models

| Field | Content |
| --- | --- |
| `name` | The score name set on the question |
| `value` / `dataType` | Choice → selected option as `CATEGORICAL`. Score → expected level as `NUMERIC`. Yes/no → `P(true)` as `NUMERIC` in `[0, 1]` |
| `comment` | Short summary, e.g. `ready (p=0.91); confidence 0.82; runner-up needs_revision (0.09)` |
| `metadata.openai` / `.typesafe` | `questionId`, `type`, resolved `model`, and where available `probabilities`, `confidence`, and the Score level legend |

Filter, chart, and alert on these scores like any other. A Yes/no score stays a
probability on purpose — that is the calibration. Threshold at filter/alert time
(e.g. `>= 0.7`). If you truly need a boolean, use a Choice question with `yes`/`no`
options instead.

## Debugging and limits

Every run creates a trace in the internal environment `langfuse-llm-as-a-judge`, named
`Execute evaluator: <evaluator name>`, containing one generation with the exact
request, answers, token usage, and cost. Internal environments are hidden from the
default tracing view — filter the tracing table by `environment = langfuse-llm-as-a-judge`
or open the trace from the score itself.

When a score looks wrong: inspect the mapped input and criteria first, then tighten the
question or add an option.

| Constraint | Limit |
| --- | --- |
| Questions per evaluator | 1–50, all answered in one call |
| Choice options | 2–255, unique values |
| Score levels | 2–10, ordered |
| Input size | TypeSafe accepts ~32k tokens for state plus the longest question |
| Rationale | None — typed verdicts only; use an LLM judge if you need explanation |
| Refusals | OpenAI refusals fail the run with no scores written; Jev always answers |
| Data handling | Input and questions go through the selected OpenAI/TypeSafe connection — review provider and upstream terms for sensitive data, and use Langfuse masking |

## Cost and speed reference

Both providers price by **input tokens only**. TypeSafe reports Jev at 40–400× cheaper
and 20–200× faster than frontier models on classification; OpenAI reports the Decisions
API at ~10× the speed of the Responses API. Treat these as reported figures, then
measure on your own workload before committing a threshold.

## Related

- `references/setting-up-evals.md` — choose the metric set before choosing the method
- `references/judge-calibration.md` — validate the judge/evaluator against ground truth
- `references/error-analysis.md` — diagnose qualitative disagreements
- `references/ci-cd.md` — gate experiments in CI/CD
- Langfuse: [Decision model evaluators](https://langfuse.com/docs/evaluation/evaluation-methods/decision-models) · [Writing good evaluators](https://langfuse.com/academy/evaluate/writing-evaluators) · TypeSafe: [Jev primitives](https://docs.typesafe.ai/primitives)
