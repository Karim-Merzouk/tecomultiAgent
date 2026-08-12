"""Controlled vocabulary derived from the real TelcoAgent-Bench repository.

Ground truth for these enums is the dataset itself (20 intents, 49 blueprints,
8 gold-path tools) plus the README tool table. See NOTES.md for the three
reconciliations against the design prompt:
  * 20 intents, not 15;
  * corrective tools `optimize_cell`/`optimize_slice` appear in gold data while
    the README core table lists `recommend_param_change` -> the ToolName enum is
    the union of both, so no real gold path is ever rejected at parse time;
  * the 7 taxonomy categories are assigned here (the dataset carries no category
    field) and are documented as a curated mapping.
"""

from __future__ import annotations

from enum import StrEnum

# --- Tools -----------------------------------------------------------------

DIAGNOSTIC_TOOLS: tuple[str, ...] = (
    "oss_query",
    "kpi_timeseries",
    "coverage_map",
    "neighbor_audit",
)

CORRECTIVE_TOOLS: tuple[str, ...] = (
    "recommend_param_change",  # README core name
    "optimize_cell",           # appears in gold data
    "optimize_slice",          # appears in gold data
    "push_config",
    "create_ticket",
)

DISTRACTOR_TOOLS: tuple[str, ...] = (
    "subscriber_insight",
    "device_cap_lookup",
    "sla_policy_fetch",
    "traffic_forecast",
    "complaint_trend_analysis",
    "spectrum_license_info",
)

CORE_TOOLS: tuple[str, ...] = DIAGNOSTIC_TOOLS + CORRECTIVE_TOOLS
ALL_TOOLS: tuple[str, ...] = CORE_TOOLS + DISTRACTOR_TOOLS


class ToolName(StrEnum):
    oss_query = "oss_query"
    kpi_timeseries = "kpi_timeseries"
    coverage_map = "coverage_map"
    neighbor_audit = "neighbor_audit"
    recommend_param_change = "recommend_param_change"
    optimize_cell = "optimize_cell"
    optimize_slice = "optimize_slice"
    push_config = "push_config"
    create_ticket = "create_ticket"
    subscriber_insight = "subscriber_insight"
    device_cap_lookup = "device_cap_lookup"
    sla_policy_fetch = "sla_policy_fetch"
    traffic_forecast = "traffic_forecast"
    complaint_trend_analysis = "complaint_trend_analysis"
    spectrum_license_info = "spectrum_license_info"


def tool_segment(tool: str) -> str:
    """Return 'diagnostic', 'corrective', or 'distractor' for a tool name."""
    if tool in DIAGNOSTIC_TOOLS:
        return "diagnostic"
    if tool in CORRECTIVE_TOOLS:
        return "corrective"
    return "distractor"


# --- Intents ---------------------------------------------------------------

# intent_key -> (human label, category)
INTENTS: dict[str, tuple[str, str]] = {
    "COVERAGE_HOLE": ("Coverage Hole", "Coverage & Beam"),
    "OVERSHOOTING_CELL": ("Overshooting Cell", "Coverage & Beam"),
    "BEAM_MISALIGNMENT": ("Beam Misalignment", "Coverage & Beam"),
    "HO_FAILURE_HOTSPOT": ("Handover Failure Hotspot", "Mobility & Handover"),
    "PINGPONG_HANDOVERS": ("Ping-Pong Handovers", "Mobility & Handover"),
    "TA_DISTRIBUTION_DRIFT": ("Timing Advance Distribution Drift", "Mobility & Handover"),
    "HIGH_PRB_UTILIZATION": ("High PRB Utilization", "Capacity & Load"),
    "LOAD_BALANCING_NEEDED": ("Load Balancing Needed", "Capacity & Load"),
    "RESOURCE_SCHED_ANOMALY": ("Resource Scheduling Anomaly", "Capacity & Load"),
    "THROUGHPUT_DROP_DL": ("Downlink Throughput Drop", "Throughput & Quality"),
    "BLER_ANOMALY": ("BLER Anomaly", "Throughput & Quality"),
    "HQOS_LATENCY_VIOLATION": ("HQoS Latency Violation", "Throughput & Quality"),
    "CELL_OUTAGE_DETECTION": ("Cell Outage", "Reliability & Outage"),
    "RLF_SPIKE": ("RLF Spike", "Reliability & Outage"),
    "DEGRADED_BACKHAUL": ("Degraded Backhaul", "Reliability & Outage"),
    "SLICE_ADMISSION_FAILURE": ("Network Slice Admission Failure", "Network Slicing"),
    "SLICE_QOS_DEGRADATION": ("Network Slice QoS Degradation", "Network Slicing"),
    "CONFIG_MISMATCH": ("Configuration Mismatch", "Configuration & SLA"),
    "PCI_COLLISION": ("PCI Collision", "Configuration & SLA"),
    "SLA_VIOLATION_REPORT": ("SLA Violation", "Configuration & SLA"),
}

INTENT_KEYS: tuple[str, ...] = tuple(INTENTS.keys())
CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(v[1] for v in INTENTS.values()))


# Built via the functional API so the 20 real intent keys stay a single source
# of truth (no 20-line hand-written block) plus the sentinel ``unknown``.
IntentLabel = StrEnum(
    "IntentLabel",
    {"unknown": "unknown", **{k: k for k in INTENT_KEYS}},
)


def intent_label_str(key: str) -> str:
    return INTENTS.get(key, (key, ""))[0]


def intent_category(key: str) -> str:
    return INTENTS.get(key, ("", "Unknown"))[1]
