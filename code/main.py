#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from buywait import AffordabilityAgent, DataRepository
from buywait.validator import OUTPUT_COLUMNS


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parent
    repo_root = code_root if (code_root / "dataset" / "requests.csv").is_file() else code_root.parent
    parser = argparse.ArgumentParser(description="Generate Buy or Wait? decisions")
    parser.add_argument("--dataset", type=Path, default=repo_root / "dataset")
    parser.add_argument("--output", type=Path, default=repo_root / "output.csv")
    parser.add_argument("--trace", type=Path, default=repo_root / "evaluation" / "decision_traces.jsonl")
    parser.add_argument("--usage-report", type=Path, default=code_root / "evaluation" / "usage_report.md")
    return parser.parse_args()


def atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_traces(path: Path, traces: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(trace, sort_keys=True) + "\n" for trace in traces), encoding="utf-8")


def write_usage_report(path: Path, request_count: int, usages: list, explanation=None, tool_loop=None, adjudicator=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    call_types: dict[str, dict[str, int]] = {}
    providers: set[str] = set()
    models: set[str] = set()
    for usage in usages:
        providers.add(usage.provider)
        if usage.model != "none":
            models.add(usage.model)
        for call_type, bucket in usage.by_call_type.items():
            aggregate = call_types.setdefault(
                call_type,
                {"calls": 0, "failed_calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0},
            )
            for field in aggregate:
                aggregate[field] += getattr(bucket, field)

    calls = sum(item["calls"] for item in call_types.values())
    failed_calls = sum(item["failed_calls"] for item in call_types.values())
    cache_hits = sum(item["cache_hits"] for item in call_types.values())
    input_tokens = sum(item["input_tokens"] for item in call_types.values())
    output_tokens = sum(item["output_tokens"] for item in call_types.values())
    total_tokens = input_tokens + output_tokens
    average_tokens = Decimal(total_tokens) / Decimal(request_count) if request_count else Decimal("0")
    input_rate = Decimal(os.getenv("BUYWAIT_INPUT_USD_PER_MILLION", "0"))
    output_rate = Decimal(os.getenv("BUYWAIT_OUTPUT_USD_PER_MILLION", "0"))
    estimated_cost = (
        Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
    ) / Decimal("1000000")
    average_cost = estimated_cost / Decimal(request_count) if request_count else Decimal("0")
    type_rows = "\n".join(
        f"| {name} | {values['calls']} | {values['cache_hits']} | {values['failed_calls']} | "
        f"{values['input_tokens']} | {values['output_tokens']} |"
        for name, values in sorted(call_types.items())
    )
    if not type_rows:
        type_rows = "| none | 0 | 0 | 0 | 0 | 0 |"
    fallback_line = ""
    if explanation is not None:
        fallback_rate = Decimal(explanation.fallback_count) * Decimal("100") / Decimal(request_count or 1)
        first_pass = getattr(explanation, "first_pass_rejections", {})
        rejected = sum(first_pass.values())
        reasons = ", ".join(f"{name} {count}" for name, count in sorted(first_pass.items(), key=lambda item: -item[1]))
        fallback_line = (
            f"- Explanation drafts rejected on first pass: {rejected}/{request_count}"
            f"{f' ({reasons})' if reasons else ''}\n"
            f"- Explanation drafts recovered by one repair attempt: {explanation.repaired_count}\n"
            f"- Explanation fallbacks to the deterministic template: {explanation.fallback_count}/{request_count} "
            f"({fallback_rate.quantize(Decimal('0.01'))}%)\n"
        )
    agent_lines = ""
    if tool_loop is not None and tool_loop.sequences:
        ordered = sorted(tool_loop.sequences.items(), key=lambda item: -item[1])
        rows_text = "\n".join(f"| `{name}` | {count} |" for name, count in ordered)
        agent_lines += (
            f"\n## Evidence tool loop\n\n"
            f"- Distinct tool sequences across {request_count} requests: {len(tool_loop.sequences)}\n"
            f"- Fell back to the single-call extractor: {tool_loop.fallback_count}\n\n"
            f"| Tool sequence | Requests |\n|---|---:|\n{rows_text}\n"
        )
    if adjudicator is not None and adjudicator.total_count:
        reasons = "\n".join(
            f"| `{name}` | {count} |" for name, count in sorted(adjudicator.reason_counts.items(), key=lambda item: -item[1])
        )
        agent_lines += (
            f"\n## Adjudicator\n\n"
            f"- Reviews: {adjudicator.total_count}\n"
            f"- Flags raised by the model: {adjudicator.flagged_count}\n"
            f"- Flags confirmed by the deterministic re-check (these changed a decision): {adjudicator.confirmed_count}\n"
            f"- Flags refuted by the deterministic re-check (these changed nothing): {adjudicator.refuted_count}\n\n"
            f"| Verdict | Count |\n|---|---:|\n{reasons}\n"
        )
    path.write_text(
        "# Token Usage Report\n\n"
        "## Final full-dataset run\n\n"
        f"- Requests processed: {request_count}\n"
        f"- Model providers and names: {', '.join(sorted(providers))} / {', '.join(sorted(models)) or 'none'}\n"
        f"- Model calls: {calls} ({failed_calls} failed and fell back safely; {cache_hits} cache hits)\n"
        f"- Input tokens: {input_tokens}\n"
        f"- Output tokens: {output_tokens}\n"
        f"- Total tokens: {total_tokens}\n"
        f"- Average tokens per request: {average_tokens.quantize(Decimal('0.01'))}\n"
        f"- Estimated total model cost: USD {estimated_cost.quantize(Decimal('0.000001'))}\n"
        f"- Estimated model cost per request: USD {average_cost.quantize(Decimal('0.000001'))}\n\n"
        f"{fallback_line}\n"
        f"{agent_lines}\n"
        "## Calls by type\n\n"
        "| Call type | Model calls | Cache hits | Failed calls | Input tokens | Output tokens |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"{type_rows}\n\n"
        "Pricing is supplied through BUYWAIT_INPUT_USD_PER_MILLION and BUYWAIT_OUTPUT_USD_PER_MILLION; zero means no model was used "
        "or no pricing was configured. Image amounts use a hash-verified OCR cache. Arithmetic, forecasting, plan generation, and "
        "validation always run locally.\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    repository = DataRepository(args.dataset)
    agent = AffordabilityAgent(repository)
    workers = max(1, int(os.getenv("BUYWAIT_WORKERS", "8")))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(agent.decide, repository.requests))
    rows = [decision.as_row() for decision, _ in results]
    traces = [trace for _, trace in results]
    if len(rows) != len(repository.requests) or len({row["request_id"] for row in rows}) != len(rows):
        raise RuntimeError("output request coverage check failed")
    atomic_write_csv(args.output.resolve(), rows)
    write_traces(args.trace.resolve(), traces)
    usages = [repository.vision_usage, agent.semantic.usage, agent.adjudicator.usage]
    if hasattr(agent, "explanation"):
        usages.append(agent.explanation.usage)
    write_usage_report(
        args.usage_report.resolve(), len(rows), usages, getattr(agent, "explanation", None),
        tool_loop=agent.tool_loop, adjudicator=agent.adjudicator,
    )
    if hasattr(agent, "explanation"):
        rate = Decimal(agent.explanation.fallback_count) * Decimal("100") / Decimal(len(rows) or 1)
        print(f"Explanation fallbacks: {agent.explanation.fallback_count}/{len(rows)} ({rate.quantize(Decimal('0.01'))}%)")
    print(f"Wrote {len(rows)} decisions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
