# agenteval

[![CI](https://github.com/atharvajoshi01/agenteval/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvajoshi01/agenteval/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Lightweight evaluation framework for AI agents. Measure accuracy, cost, latency, and safety across any agent architecture.

Works with any agent that accepts a string and returns a string — LangChain, CrewAI, AutoGen, OpenAI Assistants, or plain functions.

## Installation

```bash
pip install agenteval
```

## Quick Start

```python
from agenteval import AgentEvaluator, TaskSuite

# Define tasks
suite = TaskSuite.from_list([
    {"name": "math", "prompt": "What is 2+2?", "expected": "4", "category": "math"},
    {"name": "capital", "prompt": "Capital of France?", "expected": "Paris", "category": "geo"},
    {"name": "code", "prompt": "Write hello world in Python", "expected": "print", "category": "code"},
])

# Evaluate agents
evaluator = AgentEvaluator(
    agents={
        "agent_a": my_agent_a,  # any callable(str) -> str
        "agent_b": my_agent_b,
    },
    runs_per_task=3,  # run each task 3x for reliability measurement
)

results = evaluator.run(suite)

# Check metrics
print(results["agent_a"].metrics.accuracy)
print(results["agent_a"].metrics.latency_p95)
print(results["agent_a"].safety.safety_score)

# Compare side-by-side
evaluator.compare_results(results).print_table()
# agent   | accuracy | success_rate | latency_mean | latency_p95 | tokens_mean | cost_per_run
# agent_a | 91.1%    | 100.0%       | 2800ms       | 3200ms      | 450         | $0.0135
# agent_b | 87.3%    | 97.5%        | 3100ms       | 4500ms      | 520         | $0.0156
# Winner: agent_a
```

## What It Measures

| Module | Metrics |
|--------|---------|
| **Accuracy** | Exact match, containment match, custom judge functions |
| **Latency** | Mean, p50, p95, p99 (per-run, in ms) |
| **Cost** | Token-based cost estimation (configurable per-model pricing) |
| **Reliability** | Success rate across runs, error categorization |
| **Safety** | PII leakage (email, phone, SSN, credit card), prompt injection detection, custom forbidden patterns |

## Task Suites

Define tasks in code, JSON, or YAML:

```yaml
# tasks.yaml
name: customer_support
tasks:
  - name: greeting
    prompt: "Hi, I need help with my order"
    expected: "help"
    category: greeting
  - name: refund
    prompt: "I want a refund for order #1234"
    expected: "refund"
    category: transactions
```

```python
suite = TaskSuite.from_yaml("tasks.yaml")
```

## Safety Checks

```python
from agenteval import SafetyChecker

checker = SafetyChecker(
    check_pii=True,           # emails, phones, SSNs, credit cards, IPs
    check_injection=True,      # prompt injection leak detection
    forbidden_patterns=[       # custom regex patterns
        r"SECRET_KEY",
        r"password\s*[:=]",
    ],
)

report = checker.check(run_results)
print(report.safety_score)  # 0.0 - 1.0
print(report.violations)    # list of SafetyViolation
```

## Custom Judges

For tasks where exact/fuzzy match isn't enough:

```python
def semantic_judge(output: str, expected: str) -> bool:
    """Use an LLM to judge semantic equivalence."""
    # your LLM-as-judge logic here
    return llm_says_equivalent(output, expected)

evaluator = AgentEvaluator(
    agents={"my_agent": agent_fn},
    judge_fn=semantic_judge,
)
```

## Development

```bash
git clone https://github.com/atharvajoshi01/agenteval.git
cd agenteval
pip install -e ".[dev]"
pytest
```

## License

MIT
