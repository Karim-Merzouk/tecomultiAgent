"""Blueprint-driven, seed-deterministic KPI simulator.

Ground-truth KPI values are sampled *once* per case (from the blueprint's
constraint ranges); every tool returns a structured response derived from those
values so results are internally consistent. Distractor tools return plausible
but diagnostically useless data. The coordinator is the only caller.
"""

from __future__ import annotations

import random
from typing import Any

from ..data.records import Blueprint

_ANOMALY_BY_INTENT: dict[str, list[str]] = {
    "COVERAGE_HOLE": ["coverage_hole"],
    "OVERSHOOTING_CELL": ["overshoot"],
    "BEAM_MISALIGNMENT": ["beam_azimuth_offset"],
    "HO_FAILURE_HOTSPOT": ["neighbor_misconfig"],
    "PINGPONG_HANDOVERS": ["neighbor_misconfig"],
    "TA_DISTRIBUTION_DRIFT": ["overshoot"],
    "HIGH_PRB_UTILIZATION": ["high_prb_util"],
    "LOAD_BALANCING_NEEDED": ["high_prb_util"],
    "RESOURCE_SCHED_ANOMALY": ["high_prb_util"],
    "THROUGHPUT_DROP_DL": ["low_throughput"],
    "BLER_ANOMALY": ["high_bler"],
    "HQOS_LATENCY_VIOLATION": ["latency_violation"],
    "CELL_OUTAGE_DETECTION": ["outage"],
    "RLF_SPIKE": ["rlf_spike"],
    "DEGRADED_BACKHAUL": ["backhaul_degraded"],
    "SLICE_ADMISSION_FAILURE": ["slice_congestion"],
    "SLICE_QOS_DEGRADATION": ["slice_congestion"],
    "CONFIG_MISMATCH": ["neighbor_misconfig"],
    "PCI_COLLISION": ["pci_conflict"],
    "SLA_VIOLATION_REPORT": ["latency_violation"],
}


def _sample_range(spec: Any, rng: random.Random, default: tuple[float, float]) -> float:
    if isinstance(spec, dict) and "min" in spec and "max" in spec:
        lo, hi = float(spec["min"]), float(spec["max"])
    else:
        lo, hi = default
    return round(rng.uniform(lo, hi), 3)


class ToolSimulator:
    """One instance per case; deterministic given (blueprint, seed, case_id)."""

    def __init__(self, blueprint: Blueprint, seed: int, case_id: str):
        self.blueprint = blueprint
        self.rng = random.Random(f"{seed}:{case_id}:{blueprint.blueprint_id}")
        self.intent = blueprint.intent
        self.cell_id = f"{blueprint.intent[:3]}_{self.rng.randint(1, 99):02d}"
        c = blueprint.constraints
        self.kpis: dict[str, float] = {
            "prb_util": _sample_range(c.get("PRB_util"), self.rng, (0.2, 0.6)),
            "dl_throughput_mbps": _sample_range(
                c.get("throughput_p50_mbps"), self.rng, (2.0, 40.0)
            ),
            "ul_throughput_mbps": round(
                _sample_range(c.get("throughput_p50_mbps"), self.rng, (1.0, 15.0)) * 0.4, 3
            ),
            "sinr_db": _sample_range(c.get("sinr_dB"), self.rng, (-5.0, 20.0)),
            "rsrp_dbm": _sample_range(c.get("rsrp_dBm"), self.rng, (-120.0, -85.0)),
            "bler": round(self.rng.uniform(0.02, 0.25), 3),
            "rlf_rate": round(self.rng.uniform(0.0, 0.08), 3),
            "user_count": float(int(_sample_range(c.get("users"), self.rng, (10, 200)))),
        }
        self.anomaly_flags = _ANOMALY_BY_INTENT.get(self.intent, ["none"])

    # --- dispatch ----------------------------------------------------------

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_t_{tool_name}", None)
        if handler is None:
            return {"ok": True, "note": f"stub response for {tool_name}", "useful": False}
        return handler(arguments)

    # --- core diagnostic ---------------------------------------------------

    def _t_oss_query(self, args: dict) -> dict:
        return {
            "cell_id": self.cell_id,
            "window": args.get("window", "24h"),
            "prb_util": self.kpis["prb_util"],
            "dl_throughput_mbps": self.kpis["dl_throughput_mbps"],
            "ul_throughput_mbps": self.kpis["ul_throughput_mbps"],
            "bler": self.kpis["bler"],
            "rlf_rate": self.kpis["rlf_rate"],
            "user_count": self.kpis["user_count"],
            "useful": True,
        }

    def _t_kpi_timeseries(self, args: dict) -> dict:
        kpi = str(args.get("kpi", "dl_throughput_mbps"))
        base = self.kpis.get(kpi, self.kpis["dl_throughput_mbps"])
        series = [
            (t, round(base * (1 + self.rng.uniform(-0.1, 0.1)), 3)) for t in range(6)
        ]
        return {"kpi": kpi, "series": series, "mean": base, "useful": True}

    def _t_coverage_map(self, args: dict) -> dict:
        return {
            "area": args.get("area", self.blueprint.context.get("area", "urban")),
            "rsrp_dbm": self.kpis["rsrp_dbm"],
            "sinr_db": self.kpis["sinr_db"],
            "flags": self.anomaly_flags,
            "useful": True,
        }

    def _t_neighbor_audit(self, args: dict) -> dict:
        neighbors = [
            {"pci": self.rng.randint(0, 503), "anomaly": self.rng.random() < 0.3}
            for _ in range(3)
        ]
        return {
            "cell_id": args.get("cell_id", self.cell_id),
            "neighbors": neighbors,
            "flags": self.anomaly_flags,
            "useful": True,
        }

    # --- core corrective (side-effect acknowledgments) ---------------------

    def _ack(self, name: str, args: dict) -> dict:
        return {"ok": True, "action": name, "applied_args": args, "useful": True}

    def _t_recommend_param_change(self, args: dict) -> dict:
        return self._ack("recommend_param_change", args)

    def _t_optimize_cell(self, args: dict) -> dict:
        return self._ack("optimize_cell", args)

    def _t_optimize_slice(self, args: dict) -> dict:
        return self._ack("optimize_slice", args)

    def _t_push_config(self, args: dict) -> dict:
        return self._ack("push_config", args)

    def _t_create_ticket(self, args: dict) -> dict:
        tid = f"TT-{self.rng.randint(10000, 99999)}"
        return {"ok": True, "ticket_id": tid, "summary": args.get("summary", ""), "useful": True}

    # --- distractors (plausible but useless) -------------------------------

    def _distractor(self, name: str) -> dict:
        return {"ok": True, "tool": name, "data": {"value": self.rng.random()}, "useful": False}

    def _t_subscriber_insight(self, args: dict) -> dict:
        return self._distractor("subscriber_insight")

    def _t_device_cap_lookup(self, args: dict) -> dict:
        return self._distractor("device_cap_lookup")

    def _t_sla_policy_fetch(self, args: dict) -> dict:
        return self._distractor("sla_policy_fetch")

    def _t_traffic_forecast(self, args: dict) -> dict:
        return self._distractor("traffic_forecast")

    def _t_complaint_trend_analysis(self, args: dict) -> dict:
        return self._distractor("complaint_trend_analysis")

    def _t_spectrum_license_info(self, args: dict) -> dict:
        return self._distractor("spectrum_license_info")
