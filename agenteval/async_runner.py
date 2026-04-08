"""Async agent runner for concurrent evaluation."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, List, Optional

from agenteval.runner import RunResult, StepTrace, _estimate_tokens
from agenteval.task import Task


class AsyncAgentRunner:
    """Wraps an async agent callable and captures execution traces.

    For agents that use async APIs (OpenAI, Anthropic, LangChain async),
    this runner evaluates tasks concurrently for faster evaluation.

    Parameters:
        agent_fn: Async callable that takes a prompt string and returns a response.
        name: Human-readable agent name.
        parse_fn: Optional function to parse agent output into
            (response_str, tokens_in, tokens_out).
        max_concurrency: Maximum number of concurrent agent calls.
        cost_per_1k_input: Cost per 1000 input tokens.
        cost_per_1k_output: Cost per 1000 output tokens.
    """

    def __init__(
        self,
        agent_fn: Callable[[str], Coroutine[Any, Any, Any]],
        name: str = "async_agent",
        parse_fn: Optional[Callable[[Any], tuple]] = None,
        max_concurrency: int = 5,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ) -> None:
        self.agent_fn = agent_fn
        self.name = name
        self.parse_fn = parse_fn
        self.max_concurrency = max_concurrency
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    async def run(self, task: Task) -> RunResult:
        """Run the agent on a single task asynchronously."""
        start = time.perf_counter()
        error = None
        output_str = ""
        tokens_in = 0
        tokens_out = 0

        try:
            raw_output = await self.agent_fn(task.prompt)

            if self.parse_fn:
                parsed = self.parse_fn(raw_output)
                output_str = str(parsed[0])
                tokens_in = parsed[1] if len(parsed) > 1 else 0
                tokens_out = parsed[2] if len(parsed) > 2 else 0
            else:
                output_str = str(raw_output)
                tokens_in, tokens_out = _estimate_tokens(task.prompt, output_str)

        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        elapsed = (time.perf_counter() - start) * 1000

        step = StepTrace(
            step=1,
            action="async_agent_call",
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

    async def run_many(
        self,
        tasks: List[Task],
        runs_per_task: int = 1,
    ) -> List[RunResult]:
        """Run the agent on multiple tasks concurrently.

        Uses a semaphore to limit concurrency to max_concurrency.
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _run_with_limit(task: Task) -> RunResult:
            async with semaphore:
                return await self.run(task)

        coros = []
        for task in tasks:
            for _ in range(runs_per_task):
                coros.append(_run_with_limit(task))

        return await asyncio.gather(*coros)
