# core/processor.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
from datetime import datetime
from src.core.context import PipelineContext
import time
from functools import wraps

class Processor(ABC):
    """
    Base class for all pipeline processors.

    Role in the architecture:
    - A processor is a single, reusable, stateless step that reads from a `PipelineContext`,
      performs a focused transformation, and returns an updated context.
    - Processors should be side-effect free and treat the context as immutable by convention:
      read → compute → return a modified copy (avoid in-place mutation where possible).
    - Public API is intentionally small: initialize, `process(...)`, optional validation hooks,
      and `__call__` which orchestrates logging, validation, and error handling.

    Error handling:
    - Exceptions raised inside `process(...)` (or validations) are logged.
    - If `self.config.get("stop_on_error", True)` is True, the exception is re-raised (fail-fast).
      Otherwise, the error is recorded in `context.errors`, a warning is added, and the original
      context is returned unchanged.

    Usage example:
        class MyProcessor(Processor):
            def process(self, context: PipelineContext) -> PipelineContext:
                # Prefer creating a modified copy when changing fields
                new_ctx = context.model_copy(deep=True)
                new_ctx.stages["my_step"] = {"status": "ok"}
                return new_ctx

        step = MyProcessor(name="MyStep", config={"stop_on_error": True})
        result_ctx = step(context)
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor.

        Parameters:
        - name (Optional[str]): Human-friendly processor name. Defaults to the class name.
        - config (Optional[Dict[str, Any]]): Per-processor configuration (e.g., thresholds,
          feature flags, stop_on_error). Access via `self.config`.

        Returns:
        - None
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(f"pipeline.{self.name}")

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Perform the processor's main work and return an updated context.

        Contract:
        - Read required inputs from `context`.
        - Compute the result (stateless; no external side effects).
        - Write outputs back to the context (prefer returning a modified copy).
        - Return the updated `PipelineContext`.

        Parameters:
        - context (PipelineContext): The current pipeline context.

        Returns:
        - PipelineContext: The modified context reflecting this step's results.

        Raises:
        - Exception: Implementations may raise domain-specific errors which will be
          handled by `__call__` according to `stop_on_error`.
        """
        pass

    def validate_prerequisites(self, context: PipelineContext):
        """
        Validate that all required inputs exist in the context before processing.

        Guidance:
        - Check for required DataFrames/columns, configuration keys, or prior stage outputs.
        - Keep this fast and deterministic; raise early on invalid inputs.

        Parameters:
        - context (PipelineContext): The current pipeline context.

        Raises:
        - ValueError: When required inputs are missing or malformed.
        """
        pass

    def validate_output(self, context: PipelineContext):
        """
        Validate that this processor produced a correct/complete output.

        Guidance:
        - Verify invariants introduced by `process(...)` (e.g., non-empty results,
          expected keys present in `stages` or `processed_data`).
        - Keep this focused on this processor's responsibilities only.

        Parameters:
        - context (PipelineContext): The context after `process(...)`.

        Raises:
        - ValueError: When the processor's output is invalid or incomplete.
        """
        pass

    def __call__(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the processor with logging, validation, and error handling.

        Execution steps:
        1. Log start and capture start time.
        2. Run `validate_prerequisites(context)`.
        3. Invoke `process(context)` to produce `result`.
        4. Run `validate_output(result)`.
        5. Append execution metadata to `result.execution_log` via `log_execution(...)`.
        6. Return `result`.

        Error policy:
        - On any exception:
            * Log the error (with stack trace).
            * Record the error in `context.errors`.
            * If `self.config.get("stop_on_error", True)` is True: re-raise the exception.
              Otherwise: warn via `context.add_warning(...)` and return the original context.

        Parameters:
        - context (PipelineContext): The current pipeline context.

        Returns:
        - PipelineContext: The updated context on success, or the original context on
          handled failure when `stop_on_error` is False.
        """
        start_time = time.time()
        self.logger.info(f"Starting {self.name}")

        try:
            # Validate prerequisites
            self.validate_prerequisites(context)

            # Process
            result = self.process(context)

            # Validate output
            self.validate_output(result)

            # Log execution
            execution_time = time.time() - start_time
            result.log_execution(self.name, {'execution_time': execution_time})

            self.logger.info(f"Completed {self.name} in {execution_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}", exc_info=True)

            # Log error to context
            context.errors.append({
                'processor': self.name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

            # Re-raise if configured to stop on error
            if self.config.get('stop_on_error', True):
                raise

            # Otherwise return context unchanged
            context.add_warning(f"Processor {self.name} failed: {str(e)}")
            return context
