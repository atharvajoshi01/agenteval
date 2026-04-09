"""agenteval: Lightweight evaluation framework for AI agents."""

__version__ = "0.1.0"

from agenteval.runner import AgentRunner, RunResult, StepTrace
from agenteval.task import Task, TaskSuite
from agenteval.metrics import MetricsReport, compute_metrics
from agenteval.safety import SafetyReport, SafetyChecker
from agenteval.evaluator import AgentEvaluator, EvalResult
from agenteval.compare import ComparisonReport, compare
from agenteval.async_runner import AsyncAgentRunner
from agenteval.judges import (
    exact_match,
    contains_match,
    numeric_match,
    llm_judge,
    anthropic_judge,
    custom_judge,
)

__all__ = [
    "AgentRunner",
    "AsyncAgentRunner",
    "RunResult",
    "StepTrace",
    "Task",
    "TaskSuite",
    "MetricsReport",
    "compute_metrics",
    "SafetyReport",
    "SafetyChecker",
    "AgentEvaluator",
    "EvalResult",
    "ComparisonReport",
    "compare",
    "exact_match",
    "contains_match",
    "numeric_match",
    "llm_judge",
    "anthropic_judge",
    "custom_judge",
]
