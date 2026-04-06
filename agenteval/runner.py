"""Agent runner — wraps any callable agent and captures execution traces."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agenteval.task import Task


@dataclass
class StepTrace:
    """A single step in an agent's execution."""

    step: int
    action: str
    input: str
    output: str
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of running an agent on a single task."""

    task_name: str
    agent_output: str
    expected: Optional[str]
    steps: List[StepTrace]
    total_duration_ms: float
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "agent_output": self.agent_output,
            "expected": self.expected,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "succeeded": self.succeeded,
            "error": self.error,
            "n_steps": len(self.steps),
        }


class AgentRunner:
    """Wraps an agent callable and captures execution traces.

    The agent callable must accept a string (prompt) and return a string (response).
    For agents that return richer output, provide a ``parse_fn`` to extract the
    response string and optional token counts.

    Parameters:
        agent_fn: Callable that takes a prompt string and returns a response.
        name: Human-readable agent name.
        parse_fn: Optional function to parse agent output into
            (response_str, tokens_in, tokens_out). If None, output is
            used as-is with zero token counts.
        cost_per_1k_input: Cost per 1000 input tokens (for cost estimation).
        cost_per_1k_output: Cost per 1000 output tokens.
    """

    def __init__(
        self,
        agent_fn: Callable[[str], Any],
        name: str = "agent",
        parse_fn: Optional[Callable[[Any], tuple]] = None,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ) -> None:
        self.agent_fn = agent_fn
        self.name = name
        self.parse_fn = parse_fn
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def run(self, task: Task) -> RunResult:
        """Run the agent on a single task and capture the trace."""
        start = time.perf_counter()
        error = None
        output_str = ""
        tokens_in = 0
        tokens_out = 0

        try:
            raw_output = self.agent_fn(task.prompt)

            if self.parse_fn:
                parsed = self.parse_fn(raw_output)
                output_str = str(parsed[0])
                tokens_in = parsed[1] if len(parsed) > 1 else 0
                tokens_out = parsed[2] if len(parsed) > 2 else 0
            else:
                output_str = str(raw_output)
                # Estimate tokens if tiktoken available
                tokens_in, tokens_out = _estimate_tokens(task.prompt, output_str)

        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        elapsed = (time.perf_counter() - start) * 1000

        step = StepTrace(
            step=1,
            action="agent_call",
            input=task.prompt,
            output=output_str,
            duration_ms=elapsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        return RunResult(
            task_name=task.name,
            agent_output=output_str,
            expected=task.expected,
            steps=[step],
            total_duration_ms=elapsed,
            total_tokens_in=tokens_in,
            total_tokens_out=tokens_out,
            error=error,
        )

    def run_many(self, tasks: List[Task], runs_per_task: int = 1) -> List[RunResult]:
        """Run the agent on multiple tasks, optionally multiple times each."""
        results = []
        for task in tasks:
            for _ in range(runs_per_task):
                results.append(self.run(task))
        return results

    def estimate_cost(self, result: RunResult) -> float:
        """Estimate the dollar cost of a run."""
        input_cost = (result.total_tokens_in / 1000) * self.cost_per_1k_input
        output_cost = (result.total_tokens_out / 1000) * self.cost_per_1k_output
        return input_cost + output_cost


def _estimate_tokens(input_text: str, output_text: str) -> tuple:
    """Estimate token counts using tiktoken if available, else word-based."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(input_text)), len(enc.encode(output_text))
    except ImportError:
        # Rough estimate: 1 token per 4 characters
        return len(input_text) // 4, len(output_text) // 4
