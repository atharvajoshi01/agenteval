"""Tests for retry runner."""

from agenteval.runner import AgentRunner
from agenteval.retry import RetryRunner
from agenteval.task import Task


class TestRetryRunner:
    def test_no_retry_on_success(self):
        runner = AgentRunner(lambda p: p, name="echo")
        retry = RetryRunner(runner, max_retries=3, retry_delay=0)
        task = Task(name="t", prompt="hello", expected="hello")
        result = retry.run(task)
        assert result.succeeded
        assert result.metadata["attempts"] == 1

    def test_retries_on_failure(self):
        call_count = {"n": 0}

        def flaky(prompt):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("temporary failure")
            return prompt

        runner = AgentRunner(flaky, name="flaky")
        retry = RetryRunner(runner, max_retries=3, retry_delay=0)
        task = Task(name="t", prompt="hello", expected="hello")
        result = retry.run(task)
        assert result.succeeded
        assert result.metadata["attempts"] == 3

    def test_exhausts_retries(self):
        def always_fail(prompt):
            raise RuntimeError("permanent failure")

        runner = AgentRunner(always_fail, name="broken")
        retry = RetryRunner(runner, max_retries=2, retry_delay=0)
        task = Task(name="t", prompt="hello")
        result = retry.run(task)
        assert not result.succeeded
        assert result.metadata["exhausted_retries"]
        assert result.metadata["attempts"] == 3

    def test_retry_on_filter(self):
        def fail_with_timeout(prompt):
            raise TimeoutError("connection timed out")

        runner = AgentRunner(fail_with_timeout, name="timeout")
        retry = RetryRunner(
            runner,
            max_retries=3,
            retry_delay=0,
            retry_on=lambda err: "timed out" in err,
        )
        task = Task(name="t", prompt="hello")
        result = retry.run(task)
        assert not result.succeeded
        assert result.metadata["attempts"] == 4  # 1 + 3 retries

    def test_non_retryable_error(self):
        def fail_with_auth(prompt):
            raise PermissionError("unauthorized")

        runner = AgentRunner(fail_with_auth, name="auth")
        retry = RetryRunner(
            runner,
            max_retries=3,
            retry_delay=0,
            retry_on=lambda err: "timed out" in err,
        )
        task = Task(name="t", prompt="hello")
        result = retry.run(task)
        assert not result.succeeded
        assert result.metadata.get("retryable") is False
        assert result.metadata["attempts"] == 1

    def test_run_many(self):
        runner = AgentRunner(lambda p: p, name="echo")
        retry = RetryRunner(runner, max_retries=2, retry_delay=0)
        tasks = [Task(name=f"t{i}", prompt=f"p{i}") for i in range(3)]
        results = retry.run_many(tasks, runs_per_task=2)
        assert len(results) == 6
        assert all(r.succeeded for r in results)
