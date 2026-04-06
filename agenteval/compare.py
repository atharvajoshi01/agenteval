"""Side-by-side comparison of multiple agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agenteval.metrics import MetricsReport


@dataclass
class ComparisonReport:
    """Side-by-side comparison of agent evaluation results."""

    agents: List[MetricsReport]
    winner: Optional[str] = None

    def __post_init__(self) -> None:
        if self.agents and not self.winner:
            # Winner = highest accuracy, tie-break by lowest latency
            self.winner = max(
                self.agents,
                key=lambda a: (a.accuracy, -a.latency_mean),
            ).agent_name

    def summary_table(self) -> List[Dict[str, Any]]:
        """Return a list of dicts suitable for tabular display."""
        rows = []
        for a in self.agents:
            rows.append({
                "agent": a.agent_name,
                "accuracy": f"{a.accuracy:.1%}",
                "success_rate": f"{a.success_rate:.1%}",
                "latency_mean": f"{a.latency_mean:.0f}ms",
                "latency_p95": f"{a.latency_p95:.0f}ms",
                "tokens_mean": f"{a.tokens_mean:.0f}",
                "cost_per_run": f"${a.cost_per_run:.4f}",
                "cost_total": f"${a.cost_total:.4f}",
            })
        return rows

    def print_table(self) -> None:
        """Print a formatted comparison table."""
        rows = self.summary_table()
        if not rows:
            print("No agents to compare.")
            return

        headers = list(rows[0].keys())
        col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        separator = "-+-".join("-" * col_widths[h] for h in headers)

        print(header_line)
        print(separator)
        for row in rows:
            line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            print(line)

        if self.winner:
            print(f"\nWinner: {self.winner}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "agents": [a.to_dict() for a in self.agents],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        output = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output


def compare(reports: List[MetricsReport]) -> ComparisonReport:
    """Compare multiple agent metrics reports side by side."""
    return ComparisonReport(agents=reports)
