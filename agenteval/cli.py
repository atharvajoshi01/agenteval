"""Command-line interface for agenteval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agenteval.task import TaskSuite


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agenteval",
        description="Lightweight evaluation framework for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command")

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a task suite file"
    )
    validate_parser.add_argument("file", help="Path to task suite JSON/YAML file")

    # info command
    info_parser = subparsers.add_parser(
        "info", help="Show info about a task suite"
    )
    info_parser.add_argument("file", help="Path to task suite JSON/YAML file")

    # version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "validate":
        _validate(args.file)
    elif args.command == "info":
        _info(args.file)
    elif args.command == "version":
        from agenteval import __version__
        print(f"agenteval {__version__}")
    else:
        parser.print_help()


def _validate(filepath: str) -> None:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    try:
        if path.suffix in (".yaml", ".yml"):
            suite = TaskSuite.from_yaml(path)
        else:
            suite = TaskSuite.from_json(path)

        print(f"Valid task suite: {suite.name}")
        print(f"  Tasks: {len(suite)}")
        categories = set(t.category for t in suite.tasks)
        print(f"  Categories: {', '.join(sorted(categories))}")
        has_expected = sum(1 for t in suite.tasks if t.expected is not None)
        print(f"  With expected answers: {has_expected}/{len(suite)}")

    except Exception as e:
        print(f"Invalid task suite: {e}", file=sys.stderr)
        sys.exit(1)


def _info(filepath: str) -> None:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    try:
        if path.suffix in (".yaml", ".yml"):
            suite = TaskSuite.from_yaml(path)
        else:
            suite = TaskSuite.from_json(path)
    except Exception as e:
        print(f"Error loading suite: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Suite: {suite.name}")
    if suite.description:
        print(f"Description: {suite.description}")
    print(f"Total tasks: {len(suite)}")
    print()

    # Category breakdown
    categories: dict[str, int] = {}
    for t in suite.tasks:
        categories[t.category] = categories.get(t.category, 0) + 1

    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} tasks")

    print()
    print("Tasks:")
    for t in suite.tasks:
        expected_str = f" (expected: {t.expected[:30]}...)" if t.expected and len(t.expected) > 30 else f" (expected: {t.expected})" if t.expected else ""
        print(f"  [{t.category}] {t.name}: {t.prompt[:60]}...{expected_str}" if len(t.prompt) > 60 else f"  [{t.category}] {t.name}: {t.prompt}{expected_str}")


if __name__ == "__main__":
    main()
