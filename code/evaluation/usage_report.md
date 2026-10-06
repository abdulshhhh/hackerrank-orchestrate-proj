# Token Usage Report

## Final full-dataset run

- Requests processed: 250
- Model providers and names: Anthropic / claude-sonnet-5
- Model calls: 789 (0 failed and fell back safely; 345 cache hits)
- Input tokens: 1475214
- Output tokens: 109410
- Total tokens: 1584624
- Average tokens per request: 6338.50
- Estimated total model cost: USD 6.066792
- Estimated model cost per request: USD 0.024267

- Explanation drafts rejected on first pass: 0/250
- Explanation drafts recovered by one repair attempt: 0
- Explanation fallbacks to the deterministic template: 0/250 (0.00%)


## Evidence tool loop

- Distinct tool sequences across 250 requests: 17
- Fell back to the single-call extractor: 6

| Tool sequence | Requests |
|---|---:|
| `list_messages -> read_message -> emit_evidence_facts` | 181 |
| `no_evidence_available` | 50 |
| `list_messages -> read_message -> list_events -> emit_evidence_facts` | 4 |
| `list_messages -> list_images -> read_message -> read_image_amount -> list_events -> emit_evidence_facts` | 2 |
| `list_messages -> list_images -> read_message -> read_image_amount -> list_events -> fallback_single_call` | 1 |
| `list_messages -> list_images -> read_message -> read_image_amount -> list_events -> list_events -> list_events -> emit_evidence_facts` | 1 |
| `list_images -> list_events -> read_image_amount -> emit_evidence_facts` | 1 |
| `list_messages -> list_images -> read_message -> read_image_amount -> list_events -> list_events -> list_events -> list_events -> list_events -> list_events -> list_events -> list_events -> fallback_single_call` | 1 |
| `list_messages -> list_images -> list_events -> read_image_amount -> emit_evidence_facts` | 1 |
| `list_messages -> list_images -> read_message -> read_image_amount -> list_events -> list_events -> list_events -> fallback_single_call` | 1 |
| `list_messages -> list_images -> read_message -> read_image_amount -> fallback_single_call` | 1 |
| `list_messages -> list_images -> read_message -> read_image_amount -> emit_evidence_facts` | 1 |
| `list_messages -> list_images -> read_message -> read_image_amount -> list_events -> list_events -> fallback_single_call` | 1 |
| `list_messages -> read_message -> list_events -> list_events -> list_events -> emit_evidence_facts` | 1 |
| `list_messages -> read_message -> list_events -> list_events -> list_events -> list_events -> emit_evidence_facts` | 1 |
| `list_messages -> read_message -> fallback_single_call` | 1 |
| `list_messages -> read_message -> list_events -> list_events -> emit_evidence_facts` | 1 |

## Adjudicator

- Reviews: 250
- Flags raised by the model: 0
- Flags confirmed by the deterministic re-check (these changed a decision): 0
- Flags refuted by the deterministic re-check (these changed nothing): 0

| Verdict | Count |
|---|---:|
| `approved` | 250 |

## Calls by type

| Call type | Model calls | Cache hits | Failed calls | Input tokens | Output tokens |
|---|---:|---:|---:|---:|---:|
| adjudicator | 159 | 91 | 0 | 298313 | 35699 |
| decision_explanation | 3 | 247 | 0 | 6929 | 219 |
| evidence_interpreter | 0 | 6 | 0 | 0 | 0 |
| evidence_tool_loop | 627 | 0 | 0 | 1169972 | 73492 |
| vision_extractor | 0 | 1 | 0 | 0 | 0 |

Pricing is supplied through BUYWAIT_INPUT_USD_PER_MILLION and BUYWAIT_OUTPUT_USD_PER_MILLION; zero means no model was used or no pricing was configured. Image amounts use a hash-verified OCR cache. Arithmetic, forecasting, plan generation, and validation always run locally.
