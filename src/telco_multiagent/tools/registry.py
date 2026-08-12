"""Tool registry: 7 README core tools + 2 data-only correctives + 6 distractors."""

from __future__ import annotations

from dataclasses import dataclass

from ..vocab import CORE_TOOLS, DISTRACTOR_TOOLS, tool_segment


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    segment: str  # diagnostic | corrective | distractor
    is_distractor: bool


_DESCRIPTIONS: dict[str, str] = {
    "oss_query": "Retrieves RAN KPI details for a cell and time window.",
    "kpi_timeseries": "Returns time-series values for a target KPI.",
    "coverage_map": "Generates coverage information for coverage-hole/beam/interference analysis.",
    "neighbor_audit": "Checks neighboring cells for misconfiguration or anomalies.",
    "recommend_param_change": "Produces optimization or corrective parameter recommendations.",
    "optimize_cell": "Applies a cell-level optimization action (tilt/power/mobility params).",
    "optimize_slice": "Applies a slice-level optimization action (admission/QoS params).",
    "push_config": "Simulates a configuration push to the network.",
    "create_ticket": "Creates an OSS/NOC issue ticket with issue summary and actions.",
    "subscriber_insight": "Retrieves subscriber usage profiles for a specific cell.",
    "device_cap_lookup": "Returns UE/device capability information.",
    "sla_policy_fetch": "Retrieves SLA compliance policies.",
    "traffic_forecast": "Provides predicted traffic load.",
    "complaint_trend_analysis": "Returns complaint and sentiment trends.",
    "spectrum_license_info": "Retrieves spectrum licensing or regulatory information.",
}


def build_registry() -> dict[str, ToolSpec]:
    reg: dict[str, ToolSpec] = {}
    for name in CORE_TOOLS + DISTRACTOR_TOOLS:
        reg[name] = ToolSpec(
            name=name,
            description=_DESCRIPTIONS.get(name, ""),
            segment=tool_segment(name),
            is_distractor=name in DISTRACTOR_TOOLS,
        )
    return reg


REGISTRY: dict[str, ToolSpec] = build_registry()


def registry_markdown() -> str:
    """Render the registry as a markdown table for embedding in prompts."""
    lines = ["| tool | segment | description |", "|---|---|---|"]
    for spec in REGISTRY.values():
        seg = "distractor" if spec.is_distractor else spec.segment
        lines.append(f"| `{spec.name}` | {seg} | {spec.description} |")
    return "\n".join(lines)
