# src/core/pipeline.py
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass
import logging
from src.core.processor import Processor
from src.core.context import PipelineContext

@dataclass
class PipelineConfig:
    """
    Configuration for pipeline execution.

    Attributes:
    - stop_on_error (bool): If True, the pipeline re-raises exceptions from a processor
      and aborts at that step (fail-fast). If False, the error is logged and the
      pipeline attempts to continue with subsequent processors.
    - enable_checkpoints (bool): If True, saves a deep copy of the `PipelineContext`
      after each successful processor into `self.checkpoints[processor.name]`.
      Useful for debugging, replay, and inspection.
    - log_level (str): Logging level for the pipeline logger (e.g., "DEBUG", "INFO").
    """
    stop_on_error: bool = True
    enable_checkpoints: bool = False
    log_level: str = "INFO"


class Pipeline:
    """
    Pipeline orchestrator that executes processors sequentially.

    Core behaviors:
    - Conditional execution: Optional `conditions` mapping from processor name to
      `Callable[[PipelineContext], bool]` determines whether a processor runs.
    - Checkpointing: Optional deep copies of context after each successful step for
      debugging and recovery.
    - Execution logging: Uses the `pipeline` logger and appends to
      `PipelineContext.execution_log` via `context.log_execution(...)`.
    - Error handling: If `stop_on_error` is True, exceptions are re-raised (fail-fast).
      Otherwise, errors are logged and the pipeline continues.

    Notes:
    - The pipeline operates on a shared `PipelineContext` instance that flows through all
      processors. Processors should be stateless and treat the context as immutable by
      convention (read → compute → return an updated copy).
    """

    def __init__(
        self,
        processors: List[Processor],
        config: Optional[PipelineConfig] = None,
        conditions: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the pipeline.

        Parameters:
        - processors (List[Processor]): Ordered list of processors to execute.
        - config (Optional[PipelineConfig]): Execution configuration. Defaults to `PipelineConfig()`.
        - conditions (Optional[Dict[str, Callable[[PipelineContext], bool]]]): Optional mapping of
          processor `name` → predicate that receives the current `PipelineContext` and returns
          True to run or False to skip.

        Side effects:
        - Configures the `pipeline` logger to `config.log_level`.

        Example:
        - Only run "CalculateBaselines" when variance is low:
            conditions = {
                "CalculateBaselines": lambda ctx: all(
                    v.get("variance") == "Low" for v in ctx.variance_analysis.values()
                )
            }
        """
        self.processors = processors
        self.config = config or PipelineConfig()
        self.conditions = conditions or {}
        self.checkpoints = {}
        self.logger = logging.getLogger("pipeline")

        # Set logging level
        logging.basicConfig(level=getattr(logging, self.config.log_level))

    def should_execute(self, processor: Processor, context: PipelineContext) -> bool:
        """
        Determine whether a processor should execute based on its configured condition.

        Parameters:
        - processor (Processor): The processor under consideration.
        - context (PipelineContext): The current context.

        Returns:
        - bool: True if no condition is defined for the processor or if the condition
          evaluates to True; otherwise False.
        """
        if processor.name not in self.conditions:
            return True

        condition_func = self.conditions[processor.name]
        return condition_func(context)

    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute all processors in order, applying conditions and error policy.

        Parameters:
        - context (PipelineContext): The initial pipeline context.

        Returns:
        - PipelineContext: The final context after executing the configured processors.

        Behavior:
        - Skips a processor when its condition evaluates to False, and logs the skip via
          `context.log_execution(..., {'skipped': True})`.
        - On processor exceptions: if `config.stop_on_error` is True, re-raises the error
          and aborts; otherwise logs a warning and continues.
        - When `config.enable_checkpoints` is True, saves a deep copy of the context after
          each successful processor into `self.checkpoints[processor.name]`.
        """
        self.logger.info(f"Starting pipeline: {context.pipeline_name}")

        for i, processor in enumerate(self.processors):
            # Check execution condition
            if not self.should_execute(processor, context):
                self.logger.info(f"Skipping {processor.name} due to condition")
                context.log_execution(processor.name, {'skipped': True})
                continue

            # Execute processor
            try:
                context = processor(context)
            except Exception as e:
                if self.config.stop_on_error:
                    self.logger.error(f"Pipeline failed at {processor.name}")
                    raise
                self.logger.warning(f"Processor {processor.name} failed, continuing...")

            # Save checkpoint if enabled
            if self.config.enable_checkpoints:
                self.checkpoints[processor.name] = context.model_copy(deep=True)
                self.logger.debug(f"Saved checkpoint: {processor.name}")

        self.logger.info(f"Pipeline complete: {context.pipeline_name}")
        return context

    def run_partial(self, context: PipelineContext, start: str, end: str) -> PipelineContext:
        """
        Execute a contiguous subset of processors from `start` to `end` (inclusive).

        Parameters:
        - context (PipelineContext): The initial pipeline context.
        - start (str): Name of the first processor to run (inclusive).
        - end (str): Name of the last processor to run (inclusive).

        Returns:
        - PipelineContext: The context after executing the specified range.

        Behavior:
        - Applies `should_execute` for each processor in the slice.
        - Uses the same error policy and checkpoint behavior as `run`.

        Raises:
        - StopIteration: If `start` or `end` processor names are not found in the pipeline.
        """
        start_idx = next(i for i, p in enumerate(self.processors) if p.name == start)
        end_idx = next(i for i, p in enumerate(self.processors) if p.name == end)

        for processor in self.processors[start_idx:end_idx + 1]:
            if self.should_execute(processor, context):
                context = processor(context)

        return context
