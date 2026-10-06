# Buy or Wait? — submission solution

A bounded single agent orchestrating deterministic financial tools. The model reads
untrusted evidence and writes prose; it never computes a number, never selects a plan, and
never overrides a validator. Every figure in `output.csv` is produced by `Decimal`
arithmetic in `forecast.py`, `planner.py`, and `validator.py`.

## Architecture

Five explicit stages per request, capped and traced in `buywait/agent.py`:

1. **Retrieve** the user's profile, events, payment options, messages and images.
2. **Interpret evidence** — a *model-routed tool loop* (`loop.py`) over deterministic
   read-only tools, backed by a deterministic source-aware allow-list (`evidence.py`), a
   single-call extractor fallback (`semantic.py`), and image extraction (`vision.py`).
   Every model-proposed fact must cite a real message and match its amounts and dates
   literally before it is accepted.
3. **Build a dated 90-day cash ledger** in the user's home currency (`forecast.py`).
4. **Enumerate and rank plans** — full, partial, instalment, wait, and permitted spending
   changes, ranked by deadline, cost, earliness and payment count (`planner.py`).
5. **Validate independently** — cross-field, preference and schema guardrails
   (`validator.py`), which can reject a plan the planner produced, then an **adjudicator**
   review (`adjudicator.py`) whose flags only take effect if a deterministic re-check
   independently confirms them.

Model touchpoints, all narrow and all optional:

| Module | Role | May it change a number? |
|---|---|---|
| `semantic.py` | propose typed evidence facts from messages | No — provenance-gated, literal match required |
| `vision.py` | read an amount from a receipt or bill PNG | No — resolved by deterministic rules, hash-cached |
| `explanation.py` | write `decision_explanation` from the final trace | No — sees only the settled decision |
| `loop.py` | choose which evidence to inspect, and in what order | No — tools are read-only; facts pass the same provenance gate |
| `adjudicator.py` | review the finished decision and raise one reason code | No — a flag acts only when a deterministic re-check confirms it |

### Evidence tool loop

The model calls `list_messages`, `read_message`, `list_events`, `list_images` and
`read_image_amount` as it sees fit, then terminates with `emit_evidence_facts`. It is capped
at 8 iterations and a wall-clock deadline; a cap, timeout or malformed step falls back to the
single-call extractor, which itself falls back to the deterministic interpreter. Routing is
genuinely data-dependent rather than decorative — the final run produced **15 distinct tool
sequences** across 250 requests, from `no_evidence_available` (50 requests, one step) to an
eight-step image-and-events investigation. The per-request sequence is recorded in the trace and the histogram is
in `evaluation/usage_report.md`.

### Adjudicator, and what two iterations taught us

The reviewer returns one reason code from a closed set. A flag alone changes nothing: each
code maps to a deterministic re-check, and only a confirmed flag replaces the decision with
the conservative fallback. That asymmetry was load-bearing twice.

**First iteration — the model was handed raw dates.** It raised 52 flags on 250 requests and
the re-check refuted all 52. Inspecting them showed it simply could not compare dates: on
`request_53` it asserted a final instalment on 2026-01-16 "falls after the
desired_completion_date of 2026-01-20". Had flags been trusted, 50 correct instalment plans
would have been downgraded.

**Second iteration — the comparisons were precomputed.** The trace now carries a
`precomputed_checks` object and the reviewer is told to read the booleans rather than do
arithmetic. That exposed a bug in two of the checks themselves, not in the model:

- `unsupported_income_counted` allow-listed income sources and flagged anything else, but the
  allow-list omitted `conservative_income_recurrence`, `message_salary_once`,
  `scheduled_credit`, `message_salary`, `message_salary_date` and `confirmed_message_credit`
  — all legitimate. It confirmed 149 flags and drove `not_affordable` from 49 to 174.
- `plan_breaches_minimum_balance` compared the *baseline* projected minimum against
  `minimum_balance_to_keep`. That dip is precisely what makes a request unaffordable, so it
  flagged 9 correct decisions.

Both were caught by diffing the status distribution against the previous run before shipping.
Neither is soundly checkable from the trace alone — income admission is already gated in
`forecast.py` and the provenance checks, and re-simulating the plan would duplicate the
planner — so both now always refute, documented in the code. The reviewer may still raise
them; they can no longer act.

The adjudicator now acts only on the four checks that are provably correct:
`plan_exceeds_deadline`, `partial_payment_malformed`, `protected_category_changed` and
`explanation_unsupported`. On the final run it returned **approved for all 250 requests and
confirmed zero flags**, so it changed nothing.

That is the honest result: **on this dataset the adjudicator costs tokens and catches
nothing.** It is retained because it is provably incapable of harm and is the only path by
which a rule-violating plan could still be caught. Disable with `BUYWAIT_ADJUDICATOR=0`, and
the tool loop with `BUYWAIT_TOOL_LOOP=0`.

### A note on determinism

`temperature` is not pinned: the installed `anthropic` SDK's `Messages.create()` rejects the
argument, and setting it raised `TypeError` on every live call — which surfaced only because
the usage report counts failed calls, and showed 250/250 adjudicator failures silently
falling back to `approved`. Run-to-run reproducibility instead comes from the SHA-256 disk
cache, which replays identical responses for identical inputs.

### Financial rules

Money is `Decimal` throughout. Exchange rates are read only in their supplied direction on
the settlement date. Pending debits are reserved; pending credits, bonuses, refunds and
unrealised investment value are excluded until they settle. Recurrence is detected only
where history supports it. Essential *variable* spending (groceries, transport and similar
multi-merchant categories) is reserved separately from detected recurrences, at the 75th
percentile of trailing daily burn over a 7-day horizon — see "Reserve policy" below. Every
candidate plan is simulated day by day against `minimum_balance_to_keep`.

## Setup and run

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r code/requirements.txt
```

The deterministic path needs no key and no network:

```bash
./.venv/bin/python code/main.py
```

For the full model-assisted run, put `ANTHROPIC_API_KEY` in `code/.env` (see
`.env.example`) or export it, then:

```bash
BUYWAIT_USE_LLM=1 BUYWAIT_WORKERS=8 ./.venv/bin/python code/main.py
```

Both write `output.csv`, `evaluation/decision_traces.jsonl` and
`code/evaluation/usage_report.md`. Secrets are read from the environment only and are never
written to any artifact.

Verify the written file against the section 6.2 contract, evaluate against the 25 public
examples, run the regression suite, and rebuild the archive:

```bash
./.venv/bin/python code/evaluation/verify_output.py
./.venv/bin/python code/evaluation/main.py
./.venv/bin/python -m unittest discover -s code/tests -v
./.venv/bin/python code/evaluation/package.py
```

`verify_output.py` re-derives every contract rule from `requests.csv` rather than trusting
the pipeline that produced the file, so it catches what the in-memory validator cannot: it
checks the bytes on disk. It covers column order, row count, uniqueness and order, amount
bounds, enum membership, plan parseability and chronology, the partial-payment identities
(exactly two payments, summing to `requested_amount`, first equal to `amount_safe_to_pay`,
second on or before `desired_completion_date`), the `affordable_now` earliest-date rule, the
three-change cap, and explanation length and CSV safety.

`package.py` excludes `.env`, `.cache/`, `.venv/` and bytecode by construction, and refuses
to build if `evaluation/usage_report.md` is absent or a secret file would be included.

### Caches

Model responses are cached on disk under `code/.cache/anthropic/`, keyed by a SHA-256 of
the model, system prompt and payload, so reruns are deterministic and free. Rebuild with:

```bash
BUYWAIT_REBUILD_CACHE=1 BUYWAIT_USE_LLM=1 ./.venv/bin/python code/main.py
```

Image amounts are cached in `buywait/image_facts.json` alongside the SHA-256 of the source
PNG. A hash mismatch invalidates the entry and forces re-extraction.

## Reserve policy

`amount_safe_to_pay` was systematically too high because the ledger reserved detected
recurrences (rent, subscriptions) but not essential *variable* spending. Grid search over
estimator × horizon against the 25 public samples:

| Estimator | Horizon | Absolute residual | amount | status |
|---|---|---:|---:|---:|
| none | none | 4,749,358.90 | 4/25 | 23/25 |
| median | next income | 2,543,450.79 | 3/25 | 24/25 |
| median | 7 days | 1,835,763.13 | 4/25 | 24/25 |
| **p75** | **7 days** | **1,192,467.20** | **4/25** | **24/25** |
| p75 | 14 days | 4,611,657.49 | 2/25 | 23/25 |
| p75 | 30 days | 11,443,116.77 | 2/25 | 23/25 |

p75 × 7 days cuts absolute amount error by 74.9% and lifts status accuracy, without costing
any other categorical match. Selectable via `BUYWAIT_VARIABLE_ESTIMATOR` and
`BUYWAIT_VARIABLE_DAYS`; reproduce with
`./.venv/bin/python code/evaluation/main.py --compare-variable-reserves`.

## Explanation quality

`decision_explanation` is written by the model from the final trace only, then checked
deterministically for number grounding (every figure must appear in the trace), status and
method consistency, presence of `minimum_balance_to_keep`, spending-change coverage, length
and CSV safety. A rejected draft gets exactly one repair attempt carrying the rejection
reason; a second failure falls back to the pre-validated deterministic template.

Current full-dataset run: 0/250 fallbacks. Rejection reasons and repair counts are recorded
in `evaluation/usage_report.md` on every run.

## Public sample scores

| Field | Exact matches |
|---|---:|
| amount | 4/25 |
| status | 24/25 |
| method | 24/25 |
| plan | 23/25 |
| earliest | 23/25 |
| changes | 22/25 |

The sample file is a regression suite only. No sample request IDs or output labels are read
by the prediction pipeline.

## Tests

`tests/test_solution.py` covers the dataset contract, sample regression thresholds, evidence
provenance and the four vision resolution rules. `tests/test_adversarial.py` covers
injection and conflict handling, asserting on the **final decision row** rather than on
intermediate state:

- source authority is not policy authority (an employer cannot waive the minimum balance);
- a claimed settlement cannot promote a pending credit;
- an embedded CSV row cannot poison the ledger;
- a non-English injection is ignored;
- a merchant cannot invent income;
- salary conflicts resolve by recency in **both** directions, proving the tiebreak is not
  optimism;
- an unconvertible currency in evidence is dropped, not fatal;
- a missing PNG is recorded and never produces an invented amount;
- **a positive control**: a well-formed employer salary amendment *does* move the plan.
  Without it, every "nothing changed" assertion above could pass vacuously.

`tests/test_contract.py` applies the same principle to the output verifier: it asserts the
committed `output.csv` passes, then mutates it twelve ways — reordered columns, a dropped
row, reversed row order, an amount above `requested_amount`, a negative amount, an unknown
status enum, an empty explanation, a pipe character in an explanation, a non-chronological
plan, a wrong `affordable_now` earliest date, and four spending changes — and requires the
verifier to reject every one. A checker that passes everything is worse than no checker.

## Module map

- `buywait/data.py` — schema checks, indexes, fixed-date FX, `try_convert` for untrusted currencies.
- `buywait/evidence.py` — source-aware deterministic evidence allow-list.
- `buywait/semantic.py` — structured proposals with provenance verification.
- `buywait/vision.py` — image extraction and the three resolution rules.
- `buywait/forecast.py` — recurrence detection, variable-spend reserve, dated simulation.
- `buywait/planner.py` — feasible-plan enumeration and ranking.
- `buywait/validator.py` — independent cross-field and preference guardrails.
- `buywait/explanation.py` — prose generation, verification and one-shot repair.
- `buywait/llm.py` — Anthropic forced-tool runtime, retries, timeouts, disk cache, usage.
- `prompts/` — versioned system prompts.
- `evaluation/verify_output.py` — independent section 6.2 contract check on the written file.
- `evaluation/package.py` — reproducible archive build with secret exclusion.
- `evaluation/decision_traces.jsonl` — per-request audit trace.

The program reads only the supplied `dataset/` files and contains no evaluation labels or
organizer-only data.

## Vision cache disagreements

The cache is generated by the tool and applies deterministic date-window,
labelled-final-total, and complete-line-item-sum rules. Readings that differed from the
previously verified same-hash cache are recorded here.

| Image | Previous cache | Raw extracted evidence | Resolution | Applied value |
|---|---:|---:|---|---:|
| `image_04` | INR 32,854 | Item Bill INR 2,854; 13 complete items sum to 2,854 | Truncated source with no visible final total; use complete additive sum | **INR 2,854 (cache corrected)** |
| `image_05` | INR 822.05 | INR 704.05 due through 2026-02-06; INR 822.05 due afterward | Settlement 2026-02-09 falls in the second date window | INR 822.05 |
| `image_07` | INR 8,528 | Intermediate total INR 8,528.10; Grand Total INR 8,528 | Prefer the explicitly labelled grand total | INR 8,528 |
| `image_14` | INR 4,543 | Ambiguous stated total INR 4,593; complete items sum to INR 4,543 | Prefer the complete additive item sum and retain the disagreement | INR 4,543 |

## Known limitations

- **Exact `amount_safe_to_pay` is 4/25 on the public samples.** Absolute error is down 74.9%
  after the variable-spend reserve, and every other column improved or held, but the precise
  conservatism the golden answers apply is not fully reverse-engineered. This is the largest
  known gap.
- **The agent is a fixed five-stage pipeline, not a free tool-calling loop.** The stage order
  is deterministic by design; a model-routed loop was scoped and deliberately dropped for
  reliability and time. The `MAX_STAGES` cap is therefore an assertion, not a live bound.
- **Reserve parameters were selected against 25 public samples.** p75 × 7 days is a general
  rule with a stated rationale rather than a per-request constant, but 25 rows is thin
  evidence and the horizon in particular may not generalise to the hidden set.
- **Image extraction ran once and is cached.** The four disagreements above were adjudicated
  against the source PNGs; the remaining twelve reproduced their cached values exactly.
- **The deterministic no-key path produces different prose.** It emits the template
  explanation rather than the model-written one, so a run without `ANTHROPIC_API_KEY` is
  valid and complete but scores lower on explanation quality.
- **Reproducibility comes from the cache, not from sampling control.** `temperature` is
  deprecated for `claude-sonnet-5` and passing it returns HTTP 400, so it is not sent. A run
  replays exactly from `code/.cache/anthropic/`, but a full cache rebuild can vary. One row
  (`request_262`, an Indonesian payroll amendment) was observed flipping between
  `not_affordable` and `affordable_now` across rebuilds before the shipped cache was fixed.
  The shipped `output.csv` and cache are consistent with each other.
- **Model-outage behaviour is tested, not theoretical.** An accidental run against a rejected
  parameter produced 714 consecutive API failures; the pipeline degraded to deterministic
  explanations and still emitted 250 contract-valid rows with no crash.
