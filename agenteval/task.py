"""Task definitions for agent evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single evaluation task for an agent.

    Attributes:
        name: Short identifier for the task.
        prompt: The input prompt / instruction to send to the agent.
        expected: Expected output for accuracy checking.
        category: Optional grouping (e.g., "math", "retrieval", "tool_use").
        metadata: Arbitrary metadata attached to the task.
        judge: Optional custom judge function name. If None, exact/fuzzy match is used.
    """

    name: str
    prompt: str
    expected: Optional[str] = None
    category: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    judge: Optional[str] = None


class TaskSuite(BaseModel):
    """A collection of evaluation tasks."""

    name: str = "default"
    description: str = ""
    tasks: List[Task] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def filter(self, category: str) -> TaskSuite:
        """Return a new suite with only tasks matching the category."""
        return TaskSuite(
            name=f"{self.name}_{category}",
            description=f"Filtered: {category}",
            tasks=[t for t in self.tasks if t.category == category],
        )

    @classmethod
    def from_list(cls, tasks: List[Dict[str, Any]], name: str = "default") -> TaskSuite:
        """Create a suite from a list of task dicts."""
        return cls(name=name, tasks=[Task(**t) for t in tasks])

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> TaskSuite:
        """Load a task suite from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls.from_list(data)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> TaskSuite:
        """Load a task suite from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return cls.from_list(data)
        return cls(**data)

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        output = self.model_dump_json(indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output
