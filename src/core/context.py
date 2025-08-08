# core/context.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


class PipelineContext(BaseModel):
    """
    Central data container for pipeline execution.

    This context flows through all processors, accumulating data and results.
    Processors should treat this as immutable by convention: read from the context,
    return a modified copy, and avoid in-place mutation where possible.

    What is stable and reusable (shared across all pipelines):
    - Core data: `raw_data`, `processed_data`
    - Metadata: `pipeline_name`, `correlation_id`, `start_time`, `config`
    - Results/logging: `stages`, `execution_log`, `errors`, `warnings`

    How to extend for a particular use case (preferred options, in order):
    1) Use `config`, `stages`, and `processed_data` for most needs
       - `config`: inputs and runtime settings for processors (e.g., IDs, flags, thresholds)
       - `stages`: small, structured outputs per processor (metrics, counts, statuses)
       - `processed_data`: additional DataFrames keyed by name (e.g., "hvac.cleaned", "hvac.features")

       Example (inside a processor):
           context.config["alert_multiplier"] = 1.5
           context.stages["variance_analysis"] = {"cooling_stage_1": {"rcv": 0.31, "variance": "Low"}}
           context.processed_data["hvac.cleaned"] = cleaned_df

    2) Add optional domain sub-models when you have several related fields
       - Use a Pydantic sub-model to group cohesive domain results.
       - Keep the field optional so the core API remains stable.

       Example:
           from pydantic import BaseModel, Field
           from typing import Dict, Any, Optional

           class HVACResults(BaseModel):
               thresholds: Dict[str, float] = Field(default_factory=dict)
               variance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

           # In this file (PipelineContext), you would add:
           # hvac: Optional[HVACResults] = None

       This keeps domain-specific data organized without bloating the core API.

    3) Subclass only when absolutely necessary
       - Prefer additive optional fields or sub-models.
       - Subclass when a pipeline truly needs different lifecycle or invariants.
       - If you subclass, maintain compatibility with the base context contract.

    Versioning and evolution guidelines:
    - Prefer additive changes (optional fields) over breaking changes.
    - Namescape keys in `stages`/`processed_data` (e.g., "hvac.*") to avoid collisions.
    - If a domain evolves significantly, consider a dedicated sub-model version field.

    Summary:
    - Treat `PipelineContext` as a reusable core with explicit extension points.
    - Start with `config`/`stages`/`processed_data`; introduce optional sub-models for cohesive domains.
    - Avoid rebuilding or forking the core model for each pipeline.
    """

    # ========== Core Data ==========
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Dict[str, pd.DataFrame] = Field(default_factory=dict)

    # ========== Pipeline Metadata ==========
    pipeline_name: str
    correlation_id: str  # Unique ID for this pipeline run
    start_time: datetime = Field(default_factory=datetime.now)
    config: Dict[str, Any] = Field(default_factory=dict)

    # ========== Processing Results ==========
    # Store small, structured outputs from each processor; keys should be namespaced (e.g., "hvac.filter")
    stages: Dict[str, Any] = Field(default_factory=dict)

    # Domain-specific results (customize per use case; consider grouping into optional sub-models as domain grows)
    issues: Dict[str, List[str]] = Field(default_factory=dict)
    thresholds: Dict[str, Optional[float]] = Field(default_factory=dict)
    variance_analysis: Dict[str, Dict] = Field(default_factory=dict)
    ai_analysis: Optional[str] = None

    # ========== Execution Tracking ==========
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True  # Allow pandas DataFrames

    def log_execution(self, processor_name: str, metadata: Dict = None):
        """
        Append an execution entry for a processor run.

        Parameters:
        - processor_name (str): The processor's name.
        - metadata (Optional[Dict[str, Any]]): Extra details (e.g., execution_time, skipped flags).

        Returns:
        - PipelineContext: The same context instance (for chaining).
        """
        self.execution_log.append({
            'processor': processor_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        return self

    def add_warning(self, message: str):
        """
        Record a warning message with a timestamp.

        Parameters:
        - message (str): Human-readable warning message.

        Returns:
        - PipelineContext: The same context instance (for chaining).
        """
        self.warnings.append(f"[{datetime.now().isoformat()}] {message}")
        return self
