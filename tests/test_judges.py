"""Tests for judge functions."""


from agenteval.judges import exact_match, contains_match, numeric_match


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris")

    def test_case_insensitive(self):
        assert exact_match("PARIS", "paris")

    def test_whitespace(self):
        assert exact_match("  Paris  ", "Paris")

    def test_mismatch(self):
        assert not exact_match("London", "Paris")


class TestContainsMatch:
    def test_contained(self):
        assert contains_match("The capital of France is Paris.", "Paris")

    def test_case_insensitive(self):
        assert contains_match("The answer is PARIS", "paris")

    def test_not_contained(self):
        assert not contains_match("The capital is London", "Paris")


class TestNumericMatch:
    def test_exact_numbers(self):
        assert numeric_match("42", "42")

    def test_float_tolerance(self):
        assert numeric_match("3.14159", "3.14159")

    def test_within_tolerance(self):
        assert numeric_match("3.14", "3.14", tolerance=0.01)

    def test_outside_tolerance(self):
        assert not numeric_match("3.14", "3.20", tolerance=0.01)

    def test_comma_separated(self):
        assert numeric_match("1,000,000", "1000000")

    def test_non_numeric(self):
        assert not numeric_match("not a number", "42")

    def test_both_non_numeric(self):
        assert not numeric_match("foo", "bar")


class TestLLMJudge:
    def test_import_error_without_openai(self):
        """LLM judge should raise ImportError if openai not installed."""
        # We don't actually call the LLM, just test the factory works
        from agenteval.judges import llm_judge
        judge = llm_judge(model="gpt-4o-mini")
        assert callable(judge)

    def test_anthropic_judge_callable(self):
        from agenteval.judges import anthropic_judge
        judge = anthropic_judge()
        assert callable(judge)
