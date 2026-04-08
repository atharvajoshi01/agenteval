"""Tests for async agent runner."""

import asyncio

import pytest

from agenteval.async_runner import AsyncAgentRunner
from agenteval.task import Task


async def echo_async(prompt: str) -> str:
    await asyncio.sleep(0.01)
    return prompt


async def broken_async(prompt: str) -> str:
    raise RuntimeError("async failure")


class TestAsyncRunner:
    @pytest.mark.asyncio
    async def test_run_single(self):
        runner = AsyncAgentRunner(echo_async, name="echo")
        task = Task(name="t", prompt="hello", expected="hello")
        result = await runner.run(task)
        assert result.agent_output == "hello"
        assert result.succeeded
        assert result.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_run_error(self):
        runner = AsyncAgentRunner(broken_async, name="broken")
        task = Task(name="t", prompt="hello")
        result = await runner.run(task)
        assert not result.succeeded
        assert "RuntimeError" in result.error

    @pytest.mark.asyncio
    async def test_run_many_concurrent(self):
        runner = AsyncAgentRunner(echo_async, name="echo", max_concurrency=3)
        tasks = [Task(name=f"t{i}", prompt=f"p{i}") for i in range(10)]
        results = await runner.run_many(tasks)
        assert len(results) == 10
        assert all(r.succeeded for r in results)

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Verify the semaphore limits concurrent calls."""
        active = {"count": 0, "max": 0}

        async def tracking_agent(prompt: str) -> str:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.05)
            active["count"] -= 1
            return prompt

        runner = AsyncAgentRunner(tracking_agent, name="tracker", max_concurrency=2)
        tasks = [Task(name=f"t{i}", prompt=f"p{i}") for i in range(6)]
        await runner.run_many(tasks)
        assert active["max"] <= 2

    @pytest.mark.asyncio
    async def test_runs_per_task(self):
        runner = AsyncAgentRunner(echo_async, name="echo")
        tasks = [Task(name="t1", prompt="a"), Task(name="t2", prompt="b")]
        results = await runner.run_many(tasks, runs_per_task=3)
        assert len(results) == 6
