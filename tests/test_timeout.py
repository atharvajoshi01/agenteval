"""Tests for timeout runner."""

import time

from agenteval.runner import AgentRunner
from agenteval.task import Task
from agenteval.timeout import TimeoutRunner


def fast_agent(prompt: str) -> str:
    return prompt


def slow_agent(prompt: str) -> str:
    time.sleep(10)
    return prompt


class TestTimeoutRunner:
    def test_fast_agent_succeeds(self):
        runner = AgentRunner(fast_agent, name="fast")
        timeout = TimeoutRunner(runner, timeout_seconds=5)
        task = Task(name="t", prompt="hello", expected="hello")
        result = timeout.run(task)
        assert result.succeeded
        assert result.agent_output == "hello"

    def test_slow_agent_times_out(self):
        runner = AgentRunner(slow_agent, name="slow")
        timeout = TimeoutRunner(runner, timeout_seconds=0.5)
        task = Task(name="t", prompt="hello")
        result = timeout.run(task)
        assert not result.succeeded
        assert "TimeoutError" in result.error
        assert result.metadata.get("timed_out") is True

    def test_run_many(self):
        runner = AgentRunner(fast_agent, name="fast")
        timeout = TimeoutRunner(runner, timeout_seconds=5)
        tasks = [Task(name=f"t{i}", prompt=f"p{i}") for i in range(3)]
        results = timeout.run_many(tasks)
        assert len(results) == 3
        assert all(r.succeeded for r in results)

    def test_error_agent(self):
        def error_agent(prompt):
            raise ValueError("bad input")

        runner = AgentRunner(error_agent, name="err")
        timeout = TimeoutRunner(runner, timeout_seconds=5)
        task = Task(name="t", prompt="hello")
        result = timeout.run(task)
        assert not result.succeeded
