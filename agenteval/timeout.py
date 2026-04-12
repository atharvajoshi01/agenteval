"""Timeout wrapper for agents that may hang."""

from __future__ import annotations

import threading

from agenteval.runner import AgentRunner, RunResult
from agenteval.task import Task


class TimeoutError(Exception):
    """Raised when an agent exceeds the allowed execution time."""
    pass


class TimeoutRunner:
    """Wraps an AgentRunner with a per-task timeout.

    Uses threading to enforce a time limit on agent execution.
    If the agent doesn't respond within the timeout, the run is
    marked as failed with a TimeoutError.

    Parameters:
        runner: The underlying AgentRunner to wrap.
        timeout_seconds: Maximum seconds allowed per task.
    """

    def __init__(
        self,
        runner: AgentRunner,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return self.runner.name

    def run(self, task: Task) -> RunResult:
        """Run the agent with a timeout."""
        result_holder: list = []
        error_holder: list = []

        def _target():
            try:
                result = self.runner.run(task)
                result_holder.append(result)
            except Exception as e:
                error_holder.append(e)

        thread = threading.Thread(target=_target)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            # Agent is still running — timed out
            return RunResult(
                task_name=task.name,
                agent_output="",
                expected=task.expected,
                steps=[],
                total_duration_ms=self.timeout_seconds * 1000,
                error=f"TimeoutError: Agent exceeded {self.timeout_seconds}s limit",
                metadata={"timed_out": True},
            )

        if error_holder:
            return RunResult(
                task_name=task.name,
                agent_output="",
                expected=task.expected,
                steps=[],
                total_duration_ms=0,
                error=f"{type(error_holder[0]).__name__}: {error_holder[0]}",
            )

        if result_holder:
            return result_holder[0]

        return RunResult(
            task_name=task.name,
            agent_output="",
            expected=task.expected,
            steps=[],
            total_duration_ms=0,
            error="Unknown error: no result captured",
        )

    def run_many(self, tasks: list, runs_per_task: int = 1) -> list:
        """Run multiple tasks with timeout."""
        results = []
        for task in tasks:
            for _ in range(runs_per_task):
                results.append(self.run(task))
        return results
