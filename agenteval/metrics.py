"""Accuracy, latency, cost, and reliability metrics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agenteval.runner import RunResult


@dataclass
class MetricsReport:
    """Aggregated metrics for a set of agent runs."""

    agent_name: str
    total_runs: int
    succeeded: int
    failed: int

    # Accuracy
    accuracy: float  # fraction of correct answers
    correct: int
    incorrect: int
    unanswered: int  # runs that errored

    # Latency (ms)
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float

    # Tokens
    tokens_mean: float
    tokens_total: int

    # Cost
    cost_total: float
    cost_per_run: float

    # Reliability
    success_rate: float  # fraction of runs that didn't error

    # Per-category breakdown
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_runs": self.total_runs,
            "accuracy": self.accuracy,
            "success_rate": self.success_rate,
            "latency_mean_ms": self.latency_mean,
            "latency_p95_ms": self.latency_p95,
            "tokens_mean": self.tokens_mean,
            "cost_total": self.cost_total,
            "cost_per_run": self.cost_per_run,
            "by_category": self.by_category,
        }


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile of a list."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    low = int(idx)
    high = min(low + 1, len(sorted_v) - 1)
    frac = idx - low
    return sorted_v[low] + frac * (sorted_v[high] - sorted_v[low])


def _check_correct(
    result: RunResult,
    judge_fn: Optional[Callable[[str, str], bool]] = None,
) -> bool:
    """Check if an agent's output matches the expected answer."""
    if result.expected is None:
        return True  # No expected answer = auto-pass
    if result.error:
        return False

    output = result.agent_output.strip().lower()
    expected = result.expected.strip().lower()

    if judge_fn:
        return judge_fn(result.agent_output, result.expected)

    # Exact match
    if output == expected:
        return True

    # Containment match (expected is contained in output)
    if expected in output:
        return True

    return False


def compute_metrics(
    results: List[RunResult],
    agent_name: str = "agent",
    judge_fn: Optional[Callable[[str, str], bool]] = None,
    cost_per_1k_input: float = 0.0,
    cost_per_1k_output: float = 0.0,
) -> MetricsReport:
    """Compute aggregated metrics from a list of run results.

    Args:
        results: List of RunResult from agent runs.
        agent_name: Name of the agent being evaluated.
        judge_fn: Optional custom function(output, expected) -> bool.
        cost_per_1k_input: Cost per 1000 input tokens.
        cost_per_1k_output: Cost per 1000 output tokens.

    Returns:
        MetricsReport with accuracy, latency, cost, and reliability metrics.
    """
    if not results:
        return MetricsReport(
            agent_name=agent_name, total_runs=0, succeeded=0, failed=0,
            accuracy=0, correct=0, incorrect=0, unanswered=0,
            latency_mean=0, latency_p50=0, latency_p95=0, latency_p99=0,
            tokens_mean=0, tokens_total=0, cost_total=0, cost_per_run=0,
            success_rate=0,
        )

    total = len(results)
    succeeded = sum(1 for r in results if r.succeeded)
    failed = total - succeeded

    # Accuracy
    correct = sum(1 for r in results if _check_correct(r, judge_fn))
    unanswered = sum(1 for r in results if r.error is not None)
    answerable = total - unanswered
    accuracy = correct / answerable if answerable > 0 else 0.0

    # Latency
    latencies = [r.total_duration_ms for r in results if r.succeeded]
    latency_mean = statistics.mean(latencies) if latencies else 0.0

    # Tokens
    all_tokens = [r.total_tokens for r in results]
    tokens_total = sum(all_tokens)
    tokens_mean = statistics.mean(all_tokens) if all_tokens else 0.0

    # Cost
    cost_total = sum(
        (r.total_tokens_in / 1000) * cost_per_1k_input
        + (r.total_tokens_out / 1000) * cost_per_1k_output
        for r in results
    )
    cost_per_run = cost_total / total if total > 0 else 0.0

    # Per-category
    categories: Dict[str, List[RunResult]] = {}
    for r in results:
        # Get category from metadata if available
        cat = r.metadata.get("category", "general")
        categories.setdefault(cat, []).append(r)

    by_category = {}
    for cat, cat_results in categories.items():
        cat_correct = sum(1 for r in cat_results if _check_correct(r, judge_fn))
        cat_total = len(cat_results)
        by_category[cat] = {
            "accuracy": cat_correct / cat_total if cat_total > 0 else 0.0,
            "total": cat_total,
            "correct": cat_correct,
        }

    return MetricsReport(
        agent_name=agent_name,
        total_runs=total,
        succeeded=succeeded,
        failed=failed,
        accuracy=accuracy,
        correct=correct,
        incorrect=answerable - correct,
        unanswered=unanswered,
        latency_mean=latency_mean,
        latency_p50=_percentile(latencies, 50),
        latency_p95=_percentile(latencies, 95),
        latency_p99=_percentile(latencies, 99),
        tokens_mean=tokens_mean,
        tokens_total=tokens_total,
        cost_total=cost_total,
        cost_per_run=cost_per_run,
        success_rate=succeeded / total if total > 0 else 0.0,
        by_category=by_category,
    )
