"""Export evaluation results in multiple formats."""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Dict, Optional

from agenteval.evaluator import EvalResult


def to_json(results: Dict[str, EvalResult], path: Optional[str] = None) -> str:
    """Export evaluation results as JSON.

    Args:
        results: Dict mapping agent names to EvalResult.
        path: Optional file path to write.

    Returns:
        JSON string.
    """
    data = {
        name: {
            "metrics": er.metrics.to_dict(),
            "safety": er.safety.to_dict(),
            "per_task": [r.to_dict() for r in er.results],
        }
        for name, er in results.items()
    }
    output = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(output)
    return output


def to_csv(results: Dict[str, EvalResult], path: Optional[str] = None) -> str:
    """Export evaluation metrics as CSV.

    One row per agent with summary metrics.

    Args:
        results: Dict mapping agent names to EvalResult.
        path: Optional file path to write.

    Returns:
        CSV string.
    """
    buf = StringIO()
    fieldnames = [
        "agent",
        "accuracy",
        "success_rate",
        "total_runs",
        "correct",
        "failed",
        "latency_mean_ms",
        "latency_p95_ms",
        "tokens_mean",
        "cost_total",
        "cost_per_run",
        "safety_score",
        "violations",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for name, er in results.items():
        m = er.metrics
        s = er.safety
        writer.writerow({
            "agent": name,
            "accuracy": f"{m.accuracy:.4f}",
            "success_rate": f"{m.success_rate:.4f}",
            "total_runs": m.total_runs,
            "correct": m.correct,
            "failed": m.failed,
            "latency_mean_ms": f"{m.latency_mean:.1f}",
            "latency_p95_ms": f"{m.latency_p95:.1f}",
            "tokens_mean": f"{m.tokens_mean:.0f}",
            "cost_total": f"{m.cost_total:.4f}",
            "cost_per_run": f"{m.cost_per_run:.4f}",
            "safety_score": f"{s.safety_score:.4f}",
            "violations": len(s.violations),
        })

    output = buf.getvalue()
    if path:
        with open(path, "w") as f:
            f.write(output)
    return output


def to_markdown(results: Dict[str, EvalResult]) -> str:
    """Export evaluation results as a Markdown table.

    Args:
        results: Dict mapping agent names to EvalResult.

    Returns:
        Markdown string with comparison table.
    """
    lines = ["# Evaluation Results", ""]

    # Summary table
    lines.append("| Agent | Accuracy | Success Rate | Latency (p95) | Cost/Run | Safety |")
    lines.append("|-------|----------|-------------|---------------|----------|--------|")

    for name, er in results.items():
        m = er.metrics
        s = er.safety
        lines.append(
            f"| {name} | {m.accuracy:.1%} | {m.success_rate:.1%} | "
            f"{m.latency_p95:.0f}ms | ${m.cost_per_run:.4f} | {s.safety_score:.1%} |"
        )

    # Per-agent details
    for name, er in results.items():
        m = er.metrics
        lines.append("")
        lines.append(f"## {name}")
        lines.append(f"- Runs: {m.total_runs} (correct: {m.correct}, failed: {m.failed})")
        lines.append(f"- Latency: mean={m.latency_mean:.0f}ms, p50={m.latency_p50:.0f}ms, "
                      f"p95={m.latency_p95:.0f}ms, p99={m.latency_p99:.0f}ms")
        lines.append(f"- Tokens: mean={m.tokens_mean:.0f}, total={m.tokens_total}")

        if er.safety.violations:
            lines.append("")
            lines.append("### Safety Violations")
            for v in er.safety.violations:
                lines.append(f"- [{v.severity}] {v.check}: {v.detail}")

    return "\n".join(lines)
