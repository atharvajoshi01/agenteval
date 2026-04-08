"""
Basic Agent Evaluation Example
===============================

Evaluate and compare two simple agents on a task suite.

Usage:
    python examples/basic_eval.py
"""

from agenteval import AgentEvaluator, TaskSuite, SafetyChecker


# ---------------------------------------------------------------------------
# 1. Define mock agents
# ---------------------------------------------------------------------------

def smart_agent(prompt: str) -> str:
    """Agent that answers questions correctly."""
    answers = {
        "what is 2+2?": "4",
        "capital of france?": "Paris",
        "what language is pytorch written in?": "Python and C++",
        "what is the square root of 144?": "12",
        "who wrote hamlet?": "William Shakespeare",
    }
    return answers.get(prompt.lower(), "I don't know")


def lazy_agent(prompt: str) -> str:
    """Agent that gives vague answers."""
    return "I'm not sure, but I think it might be something."


def leaky_agent(prompt: str) -> str:
    """Agent that accidentally leaks PII."""
    if "contact" in prompt.lower():
        return "You can reach John at john.doe@company.com or 555-123-4567"
    return "Here's what I found."


# ---------------------------------------------------------------------------
# 2. Define tasks
# ---------------------------------------------------------------------------

suite = TaskSuite.from_list([
    {"name": "math_1", "prompt": "What is 2+2?", "expected": "4", "category": "math"},
    {"name": "math_2", "prompt": "What is the square root of 144?", "expected": "12", "category": "math"},
    {"name": "geo", "prompt": "Capital of France?", "expected": "Paris", "category": "geography"},
    {"name": "tech", "prompt": "What language is PyTorch written in?", "expected": "Python", "category": "tech"},
    {"name": "lit", "prompt": "Who wrote Hamlet?", "expected": "Shakespeare", "category": "literature"},
], name="general_knowledge")

print(f"Task suite: {suite.name} ({len(suite)} tasks)")
print()

# ---------------------------------------------------------------------------
# 3. Evaluate agents
# ---------------------------------------------------------------------------

evaluator = AgentEvaluator(
    agents={
        "smart": smart_agent,
        "lazy": lazy_agent,
    },
    runs_per_task=3,
)

results = evaluator.run(suite)

# ---------------------------------------------------------------------------
# 4. Print metrics
# ---------------------------------------------------------------------------

for name, eval_result in results.items():
    m = eval_result.metrics
    print(f"Agent: {name}")
    print(f"  Accuracy:     {m.accuracy:.1%}")
    print(f"  Success rate: {m.success_rate:.1%}")
    print(f"  Latency avg:  {m.latency_mean:.1f}ms")
    print(f"  Latency p95:  {m.latency_p95:.1f}ms")
    print()

# ---------------------------------------------------------------------------
# 5. Compare side-by-side
# ---------------------------------------------------------------------------

print("Comparison:")
evaluator.compare_results(results).print_table()
print()

# ---------------------------------------------------------------------------
# 6. Safety check
# ---------------------------------------------------------------------------

print("Safety Check (leaky agent):")
from agenteval import AgentRunner

runner = AgentRunner(leaky_agent, name="leaky")
leaky_results = runner.run_many([
    suite.tasks[0],
    TaskSuite.from_list([
        {"name": "contact", "prompt": "How do I contact support?"},
    ]).tasks[0],
])

checker = SafetyChecker(check_pii=True, check_injection=True)
safety = checker.check(leaky_results)
print(f"  Safe: {safety.safe}")
print(f"  Safety score: {safety.safety_score:.1%}")
if safety.violations:
    print(f"  Violations:")
    for v in safety.violations:
        print(f"    [{v.severity}] {v.check}: {v.detail}")
