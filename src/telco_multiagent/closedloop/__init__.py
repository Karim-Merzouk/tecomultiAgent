"""TelcoAgent-Bench/CL: closed-loop extension.

Turns the benchmark from "did you follow the runbook" into "did you fix the
network", which is what the benchmark paper names as its own open problem:

    "the current evaluation does not model fully closed-loop operational
     reasoning, where agents interpret tool outputs, implement configuration
     changes, and re-evaluate network behavior before reaching a final
     resolution."   -- TelcoAgent-Bench, Sec. IV

Critically, outcome here depends on the *parameter values* of a corrective
action, not on the tool name. A runbook lookup emits correct tool names and
therefore scores 1.0 on every structural metric, yet cannot score at all here.
"""

from .twin import NetworkTwin, FaultModel, Outcome, FAULTS
from .metrics import ClosedLoopMetrics, aggregate_cl

__all__ = ["NetworkTwin", "FaultModel", "Outcome", "FAULTS",
           "ClosedLoopMetrics", "aggregate_cl"]
