"""Tests for CLI."""

import json
import subprocess
import sys

import pytest


@pytest.fixture
def task_file(tmp_path):
    data = {
        "name": "test_suite",
        "tasks": [
            {"name": "t1", "prompt": "What is 2+2?", "expected": "4", "category": "math"},
            {"name": "t2", "prompt": "Capital of France?", "expected": "Paris", "category": "geo"},
        ],
    }
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(data))
    return path


class TestCLI:
    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "agenteval.cli", "version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "agenteval" in result.stdout

    def test_validate(self, task_file):
        result = subprocess.run(
            [sys.executable, "-m", "agenteval.cli", "validate", str(task_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Valid task suite" in result.stdout
        assert "Tasks: 2" in result.stdout

    def test_validate_missing_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "agenteval.cli", "validate", "/nonexistent.json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_info(self, task_file):
        result = subprocess.run(
            [sys.executable, "-m", "agenteval.cli", "info", str(task_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "test_suite" in result.stdout
        assert "math" in result.stdout
        assert "geo" in result.stdout
