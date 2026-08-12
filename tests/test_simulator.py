from telco_multiagent.data.records import Blueprint
from telco_multiagent.tools.simulator import ToolSimulator


def _bp():
    return Blueprint(
        intent="HIGH_PRB_UTILIZATION",
        blueprint_id="BP1",
        constraints={
            "PRB_util": {"min": 0.7, "max": 0.95},
            "throughput_p50_mbps": {"min": 2, "max": 10},
            "sinr_dB": {"min": -5, "max": 15},
            "rsrp_dBm": {"min": -120, "max": -90},
        },
    )


def test_kpis_within_ranges():
    sim = ToolSimulator(_bp(), seed=42, case_id="S1")
    assert 0.7 <= sim.kpis["prb_util"] <= 0.95
    assert 2 <= sim.kpis["dl_throughput_mbps"] <= 10
    assert -5 <= sim.kpis["sinr_db"] <= 15
    assert -120 <= sim.kpis["rsrp_dbm"] <= -90


def test_deterministic_given_seed():
    a = ToolSimulator(_bp(), 42, "S1").kpis
    b = ToolSimulator(_bp(), 42, "S1").kpis
    assert a == b


def test_oss_query_consistent_with_kpis():
    sim = ToolSimulator(_bp(), 42, "S1")
    out = sim.execute("oss_query", {"window": "24h"})
    assert out["prb_util"] == sim.kpis["prb_util"]
    assert out["useful"] is True


def test_distractor_marked_useless():
    sim = ToolSimulator(_bp(), 42, "S1")
    assert sim.execute("subscriber_insight", {})["useful"] is False


def test_create_ticket_returns_id():
    sim = ToolSimulator(_bp(), 42, "S1")
    assert sim.execute("create_ticket", {"summary": "x"})["ticket_id"].startswith("TT-")
