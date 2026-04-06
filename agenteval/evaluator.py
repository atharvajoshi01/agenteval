"""High-level evaluator — ties runner, metrics, and safety together."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from agenteval.compare import ComparisonReport, compare
from agenteval.metrics import MetricsReport, compute_metrics
from agenteval.runner import AgentRunner, RunResult
from agenteval.safety import SafetyChecker, SafetyReport
from agenteval.task import Task, TaskSuite


@dataclass
class EvalResult:
    """Complete evaluation result for a single agent."""

    agent_name: str
    metrics: MetricsReport
    safety: SafetyReport
    results: List[RunResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "metrics": self.metrics.to_dict(),
            "safety": self.safety.to_dict(),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        output = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output


class AgentEvaluator:
    """Evaluate one or more agents against a task suite.

    Parameters:
        agents: Dict mapping agent names to callables or AgentRunner instances.
        metrics: List of metric names to compute (currently all are always computed).
        safety_checker: SafetyChecker instance. If None, default checks are used.
        judge_fn: Custom accuracy judge function(output, expected) -> bool.
        runs_per_task: Number of times to run each task (for reliability measurement).
        cost_per_1k_input: Default cost per 1000 input tokens.
        cost_per_1k_output: Default cost per 1000 output tokens.

    Example::

        evaluator = AgentEvaluator(
            agents={
                "my_agent": my_agent_fn,
                "baseline": baseline_fn,
            },
            runs_per_task=3,
        )
        results = evaluator.run(suite)
        results["my_agent"].metrics.accuracy
        evaluator.compare(results).print_table()
    """

    def __init__(
        self,
        agents: Dict[str, Union[Callable, AgentRunner]],
        metrics: Optional[List[str]] = None,
        safety_checker: Optional[SafetyChecker] = None,
        judge_fn: Optional[Callable[[str, str], bool]] = None,
        runs_per_task: int = 1,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ) -> None:
        self.runners: Dict[str, AgentRunner] = {}
        for name, agent in agents.items():
            if isinstance(agent, AgentRunner):
                self.runners[name] = agent
            else:
                self.runners[name] = AgentRunner(
                    agent_fn=agent,
                    name=name,
                    cost_per_1k_input=cost_per_1k_input,
                    cost_per_1k_output=cost_per_1k_output,
                )

        self.safety_checker = safety_checker or SafetyChecker()
        self.judge_fn = judge_fn
        self.runs_per_task = runs_per_task
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def run(self, suite: TaskSuite) -> Dict[str, EvalResult]:
        """Run all agents against the task suite.

        Args:
            suite: TaskSuite containing the evaluation tasks.

        Returns:
            Dict mapping agent names to EvalResult.
        """
        eval_results: Dict[str, EvalResult] = {}

        for name, runner in self.runners.items():
            # Tag results with task category
            run_results = []
            for task in suite.tasks:
                for _ in range(self.runs_per_task):
                    result = runner.run(task)
                    result.metadata["category"] = task.category
                    run_results.append(result)

            metrics = compute_metrics(
                run_results,
                agent_name=name,
                judge_fn=self.judge_fn,
                cost_per_1k_input=self.cost_per_1k_input,
                cost_per_1k_output=self.cost_per_1k_output,
            )

            safety = self.safety_checker.check(run_results)

            eval_results[name] = EvalResult(
                agent_name=name,
                metrics=metrics,
                safety=safety,
                results=run_results,
            )

        return eval_results

    def compare_results(self, eval_results: Dict[str, EvalResult]) -> ComparisonReport:
        """Compare evaluation results across agents."""
        reports = [er.metrics for er in eval_results.values()]
        return compare(reports)
