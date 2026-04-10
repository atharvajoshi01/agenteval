"""Retry logic for flaky agents."""

from __future__ import annotations

import time
from typing import Callable, Optional

from agenteval.runner import AgentRunner, RunResult
from agenteval.task import Task


class RetryRunner:
    """Wraps an AgentRunner with automatic retry on failure.

    Useful for evaluating agents that depend on external APIs
    which may intermittently fail (rate limits, timeouts, etc.).

    Parameters:
        runner: The underlying AgentRunner to wrap.
        max_retries: Maximum number of retry attempts per task.
        retry_delay: Seconds to wait between retries.
        retry_on: Optional callable that takes an error string and returns
            True if the error is retryable. If None, all errors are retried.
    """

    def __init__(
        self,
        runner: AgentRunner,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_on: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.runner = runner
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_on = retry_on

    @property
    def name(self) -> str:
        return self.runner.name

    def run(self, task: Task) -> RunResult:
        """Run with retries on failure."""
        last_result = None

        for attempt in range(self.max_retries + 1):
            result = self.runner.run(task)

            if result.succeeded:
                result.metadata["attempts"] = attempt + 1
                return result

            last_result = result

            # Check if error is retryable
            if self.retry_on and result.error and not self.retry_on(result.error):
                result.metadata["attempts"] = attempt + 1
                result.metadata["retryable"] = False
                return result

            # Wait before retrying (skip on last attempt)
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        last_result.metadata["attempts"] = self.max_retries + 1
        last_result.metadata["exhausted_retries"] = True
        return last_result

    def run_many(self, tasks: list, runs_per_task: int = 1) -> list:
        """Run multiple tasks with retry logic."""
        results = []
        for task in tasks:
            for _ in range(runs_per_task):
                results.append(self.run(task))
        return results
