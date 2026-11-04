"""Wire schemas.

Kept in their own package rather than beside the routes so the orchestrator can build a validated
result without importing FastAPI. The one-directional dependency (routes -> schemas, orchestrator ->
schemas, never schemas -> routes) is what lets the fusion smoke script assert against the same models
the API serves.
"""

from app.schemas.analysis import (
    AnalysisResult,
    C2paBlock,
    MetadataBlock,
    ReliabilityBlock,
    Timings,
    VerdictBlock,
)
from app.schemas.common import Error, Health, NotFound
from app.schemas.job import BatchItem, BatchResult, JobCreated, JobStatus

__all__ = [
    "AnalysisResult",
    "BatchItem",
    "BatchResult",
    "C2paBlock",
    "Error",
    "Health",
    "JobCreated",
    "JobStatus",
    "MetadataBlock",
    "NotFound",
    "ReliabilityBlock",
    "Timings",
    "VerdictBlock",
]
