"""Contract layer: assume-guarantee contracts over agent handoffs.

Public surface:
    Runbook               -- intent -> canonical tool chain (operator artifact)
    ContractSet           -- C1..C4 admissibility predicates
    project               -- the projection operator  Pi : Sigma* -> C(i_hat)
    ProjectionReport      -- what Pi changed, for certificates and CVR
"""

from .runbook import Runbook, CollisionClasses
from .contracts import ContractSet, Violation, PolicyEnvelope
from .projection import project, ProjectionReport

__all__ = [
    "Runbook",
    "CollisionClasses",
    "ContractSet",
    "Violation",
    "PolicyEnvelope",
    "project",
    "ProjectionReport",
]
