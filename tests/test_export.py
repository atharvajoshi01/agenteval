"""Tests for export module."""

import csv
import json
from io import StringIO

from agenteval import AgentEvaluator, TaskSuite


def echo_agent(prompt: str) -> str:
    return prompt


def upper_agent(prompt: str) -> str:
    return prompt.upper()


def _get_results():
    suite = TaskSuite.from_list([
        {"name": "t1", "prompt": "hello", "expected": "hello"},
        {"name": "t2", "prompt": "world", "expected": "world"},
    ])
    evaluator = AgentEvaluator(
        agents={"echo": echo_agent, "upper": upper_agent},
        runs_per_task=2,
    )
    return evaluator.run(suite)


class TestExportJSON:
    def test_json_output(self):
        from agenteval.export import to_json
        results = _get_results()
        output = to_json(results)
        data = json.loads(output)
        assert "echo" in data
        assert "upper" in data
        assert "metrics" in data["echo"]
        assert "per_task" in data["echo"]

    def test_json_to_file(self, tmp_path):
        from agenteval.export import to_json
        results = _get_results()
        path = str(tmp_path / "results.json")
        to_json(results, path=path)
        with open(path) as f:
            data = json.load(f)
        assert "echo" in data


class TestExportCSV:
    def test_csv_output(self):
        from agenteval.export import to_csv
        results = _get_results()
        output = to_csv(results)
        reader = csv.DictReader(StringIO(output))
        rows = list(reader)
        assert len(rows) == 2
        agents = {r["agent"] for r in rows}
        assert "echo" in agents
        assert "upper" in agents

    def test_csv_to_file(self, tmp_path):
        from agenteval.export import to_csv
        results = _get_results()
        path = str(tmp_path / "results.csv")
        to_csv(results, path=path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2


class TestExportMarkdown:
    def test_markdown_output(self):
        from agenteval.export import to_markdown
        results = _get_results()
        output = to_markdown(results)
        assert "# Evaluation Results" in output
        assert "echo" in output
        assert "upper" in output
        assert "Accuracy" in output
