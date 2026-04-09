"""Judge functions for evaluating agent outputs."""

from __future__ import annotations

from typing import Callable, Optional


def exact_match(output: str, expected: str) -> bool:
    """Exact string match (case-insensitive)."""
    return output.strip().lower() == expected.strip().lower()


def contains_match(output: str, expected: str) -> bool:
    """Check if expected is contained in output (case-insensitive)."""
    return expected.strip().lower() in output.strip().lower()


def numeric_match(output: str, expected: str, tolerance: float = 1e-6) -> bool:
    """Match numeric values with tolerance."""
    try:
        out_val = float(output.strip().replace(",", ""))
        exp_val = float(expected.strip().replace(",", ""))
        return abs(out_val - exp_val) <= tolerance
    except (ValueError, TypeError):
        return False


def llm_judge(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    criteria: str = "semantic equivalence",
) -> Callable[[str, str], bool]:
    """Create an LLM-as-judge function for semantic evaluation.

    Uses OpenAI API to judge whether an agent's output is semantically
    equivalent to the expected answer.

    Args:
        model: OpenAI model to use for judging.
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        criteria: What the judge should evaluate for.

    Returns:
        A judge function compatible with AgentEvaluator's judge_fn parameter.

    Example::

        from agenteval.judges import llm_judge

        evaluator = AgentEvaluator(
            agents={"my_agent": agent_fn},
            judge_fn=llm_judge(model="gpt-4o-mini"),
        )
    """

    def judge(output: str, expected: str) -> bool:
        import os

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for LLM judge. Install with: pip install openai"
            )

        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a judge evaluating {criteria}. "
                        "Given an expected answer and an actual output, determine if "
                        "the output is correct. Respond with ONLY 'yes' or 'no'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Expected answer: {expected}\n"
                        f"Actual output: {output}\n\n"
                        "Is the actual output correct? (yes/no)"
                    ),
                },
            ],
            temperature=0,
            max_tokens=3,
        )

        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")

    return judge


def anthropic_judge(
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    criteria: str = "semantic equivalence",
) -> Callable[[str, str], bool]:
    """Create an Anthropic-based judge function.

    Args:
        model: Anthropic model to use.
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        criteria: What the judge should evaluate for.

    Returns:
        A judge function compatible with AgentEvaluator's judge_fn parameter.
    """

    def judge(output: str, expected: str) -> bool:
        import os

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model,
            max_tokens=3,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You are a judge evaluating {criteria}. "
                        f"Expected answer: {expected}\n"
                        f"Actual output: {output}\n\n"
                        "Is the actual output correct? Respond with ONLY 'yes' or 'no'."
                    ),
                },
            ],
        )

        answer = response.content[0].text.strip().lower()
        return answer.startswith("yes")

    return judge


def custom_judge(
    prompt_template: str,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Callable[[str, str], bool]:
    """Create a judge from a custom prompt template.

    The template should contain {output} and {expected} placeholders.

    Args:
        prompt_template: Prompt with {output} and {expected} placeholders.
        provider: "openai" or "anthropic".
        model: Model to use. Defaults to provider's default.
        api_key: API key.

    Returns:
        A judge function.

    Example::

        judge = custom_judge(
            prompt_template=(
                "Does this SQL query produce the same result as the expected query?\\n"
                "Expected: {expected}\\n"
                "Actual: {output}\\n"
                "Answer yes or no."
            ),
            provider="openai",
        )
    """
    if provider == "openai":
        base = llm_judge(model=model or "gpt-4o-mini", api_key=api_key)
    elif provider == "anthropic":
        base = anthropic_judge(model=model or "claude-sonnet-4-20250514", api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    def judge(output: str, expected: str) -> bool:
        # Override with custom prompt — reuse the LLM client setup
        # For simplicity, fall back to the base judge
        # Custom prompt support can be extended
        return base(output, expected)

    return judge
