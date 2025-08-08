# MeshInsights Pipeline Architecture

## Overview

This document describes the **Context + Processor** architectural pattern for building composable, testable, and maintainable data analytics pipelines in the MeshInsights platform. This architecture enables engineers to build production-ready pipelines on Day 1 while maintaining flexibility for complex analytics workflows.

## Core Concepts

### The Three Components

1. **Context**: A shared data container (Pydantic model) that flows through the pipeline, accumulating results at each step
2. **Processor**: A single, reusable processing step that reads from and writes to the context
3. **Pipeline**: An orchestrator that executes processors sequentially with support for conditional execution

### Design Principles

- **Immutable-by-Convention**: Context is treated as immutable; processors return modified copies
- **Single Responsibility**: Each processor does one thing well
- **Explicit Dependencies**: All data dependencies are visible in the context
- **Fail-Fast**: Validation happens early and explicitly
- **Testability First**: Every component is independently testable

## Architecture

### Directory Structure

```
meshinsights_pipeline/
├── core/
│   ├── __init__.py
│   ├── context.py          # PipelineContext definition
│   ├── processor.py        # Base Processor class
│   └── pipeline.py          # Pipeline orchestrator
├── processors/
│   ├── __init__.py
│   ├── data_loading/        # Data access processors
│   ├── data_quality/        # Validation & cleaning
│   ├── algorithms/          # Core algorithms
│   └── ai_analysis/         # GenAI processors
├── factories/
│   └── pipeline_factory.py  # Pre-configured pipelines
├── utils/
│   └── testing.py           # Testing utilities
└── examples/
    └── hvac_pipeline.py     # Example implementation
```

## Implementation Guide

### 1. Context Definition

```python
# core/context.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

class PipelineContext(BaseModel):
    """
    Central data container for pipeline execution.
    
    This context flows through all processors, accumulating data and results.
    Processors should treat this as immutable, returning modified copies.
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
    # Store intermediate results from each processor
    stages: Dict[str, Any] = Field(default_factory=dict)
    
    # Domain-specific results (customize per use case)
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
        """Log processor execution for debugging"""
        self.execution_log.append({
            'processor': processor_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        return self
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(f"[{datetime.now().isoformat()}] {message}")
        return self
```

### 2. Base Processor

```python
# core/processor.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
import time
from functools import wraps

class Processor(ABC):
    """
    Base class for all pipeline processors.
    
    Each processor represents a single, reusable processing step.
    Processors should be stateless and side-effect free.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor.
        
        Args:
            name: Processor name (defaults to class name)
            config: Configuration dictionary for this processor
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(f"pipeline.{self.name}")
    
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process the context and return updated context.
        
        This method should:
        1. Read required data from context
        2. Perform processing
        3. Write results back to context
        4. Return the modified context
        
        Args:
            context: Current pipeline context
            
        Returns:
            Modified pipeline context
        """
        pass
    
    def validate_prerequisites(self, context: PipelineContext):
        """
        Validate that required data exists in context.
        Override this to add specific validation logic.
        
        Raises:
            ValueError: If prerequisites are not met
        """
        pass
    
    def validate_output(self, context: PipelineContext):
        """
        Validate processor output.
        Override this to add output validation.
        
        Raises:
            ValueError: If output validation fails
        """
        pass
    
    def __call__(self, context: PipelineContext) -> PipelineContext:
        """
        Execute processor with logging and error handling.
        
        This wrapper method:
        1. Logs execution start
        2. Validates prerequisites
        3. Executes process()
        4. Validates output
        5. Logs execution complete
        6. Handles errors gracefully
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
```

### 3. Pipeline Orchestrator

```python
# core/pipeline.py
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass
import logging

@dataclass
class PipelineConfig:
    """Configuration for pipeline execution"""
    stop_on_error: bool = True
    enable_checkpoints: bool = False
    log_level: str = "INFO"

class Pipeline:
    """
    Pipeline orchestrator that executes processors sequentially.
    
    Supports:
    - Conditional execution based on context state
    - Checkpointing for debugging and recovery
    - Execution logging
    """
    
    def __init__(
        self, 
        processors: List[Processor],
        config: Optional[PipelineConfig] = None,
        conditions: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            processors: List of processors to execute
            config: Pipeline configuration
            conditions: Dict mapping processor names to condition functions
                       Function signature: (context) -> bool
        """
        self.processors = processors
        self.config = config or PipelineConfig()
        self.conditions = conditions or {}
        self.checkpoints = {}
        self.logger = logging.getLogger("pipeline")
        
        # Set logging level
        logging.basicConfig(level=getattr(logging, self.config.log_level))
    
    def should_execute(self, processor: Processor, context: PipelineContext) -> bool:
        """Check if processor should execute based on conditions"""
        if processor.name not in self.conditions:
            return True
        
        condition_func = self.conditions[processor.name]
        return condition_func(context)
    
    def run(self, context: PipelineContext) -> PipelineContext:
        """
        Execute pipeline.
        
        Args:
            context: Initial pipeline context
            
        Returns:
            Final pipeline context after all processors
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
        """Run pipeline from start processor to end processor (inclusive)"""
        start_idx = next(i for i, p in enumerate(self.processors) if p.name == start)
        end_idx = next(i for i, p in enumerate(self.processors) if p.name == end)
        
        for processor in self.processors[start_idx:end_idx + 1]:
            if self.should_execute(processor, context):
                context = processor(context)
        
        return context
```

### 4. Example Processors

```python
# processors/data_loading/adx_loader.py
from core.processor import Processor
from core.context import PipelineContext
import pandas as pd

class ADXDataLoader(Processor):
    """Load data from Azure Data Explorer"""
    
    def __init__(self, location_id: str, control_id: str, start_date: str, end_date: str):
        super().__init__()
        self.location_id = location_id
        self.control_id = control_id
        self.start_date = start_date
        self.end_date = end_date
    
    def process(self, context: PipelineContext) -> PipelineContext:
        # Simulate data loading (replace with actual ADX query)
        self.logger.info(f"Loading data for location={self.location_id}, control={self.control_id}")
        
        # Your existing data loading logic here
        # data = AdxDataRetriever(self.location_id, self.control_id, 
        #                         self.start_date, self.end_date).get_data()
        
        # For example purposes:
        data = pd.DataFrame({
            'timestamp': pd.date_range(self.start_date, self.end_date, freq='H'),
            'energy': np.random.randint(100, 1000, size=24*30),
            'tstate': np.random.choice([1, 2, 3, 4], size=24*30)
        })
        
        context.raw_data = data
        context.stages['data_loading'] = {
            'rows_loaded': len(data),
            'date_range': f"{self.start_date} to {self.end_date}"
        }
        
        return context

# processors/data_quality/filter_cycles.py
class FilterValidCycles(Processor):
    """Filter out invalid cycles from the data"""
    
    def __init__(self, min_cycle_length: int = 4, min_median_energy: float = 0):
        super().__init__()
        self.min_cycle_length = min_cycle_length
        self.min_median_energy = min_median_energy
    
    def validate_prerequisites(self, context: PipelineContext):
        if context.raw_data is None or context.raw_data.empty:
            raise ValueError("No raw data available for filtering")
        
        required_columns = ['cycle', 'energy']
        missing = set(required_columns) - set(context.raw_data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def process(self, context: PipelineContext) -> PipelineContext:
        df = context.raw_data.copy()
        original_cycles = df['cycle'].nunique()
        
        # Group by cycle and filter
        cycle_stats = df.groupby('cycle').agg(
            count=('cycle', 'size'),
            median_energy=('energy', 'median')
        )
        
        valid_cycles = cycle_stats[
            (cycle_stats['count'] >= self.min_cycle_length) &
            (cycle_stats['median_energy'] > self.min_median_energy)
        ].index
        
        filtered_df = df[df['cycle'].isin(valid_cycles)]
        
        # Update context
        context.raw_data = filtered_df
        context.stages['filter_cycles'] = {
            'original_cycles': original_cycles,
            'valid_cycles': len(valid_cycles),
            'cycles_removed': original_cycles - len(valid_cycles)
        }
        
        if len(valid_cycles) < 10:
            context.add_warning("Low number of valid cycles detected")
        
        return context

# processors/algorithms/variance_analysis.py
class VarianceAnalysis(Processor):
    """Analyze power variance per stage"""
    
    def __init__(self, rcv_threshold: float = 0.35):
        super().__init__()
        self.rcv_threshold = rcv_threshold
    
    def process(self, context: PipelineContext) -> PipelineContext:
        df = context.raw_data
        
        for stage_name in ['cooling_stage_1', 'cooling_stage_2', 'heating_stage_1']:
            stage_data = df[df['stage'] == stage_name]['energy'].values
            
            if len(stage_data) < 50:
                variance = 'Low'
                reason = 'Insufficient data'
            else:
                # Calculate robust coefficient of variation
                median = np.median(stage_data)
                mad = np.median(np.abs(stage_data - median))
                rcv = mad / median if median > 0 else 0
                
                if rcv > self.rcv_threshold:
                    variance = 'High'
                    reason = f'RCV={rcv:.2f} exceeds threshold'
                else:
                    variance = 'Low'
                    reason = f'RCV={rcv:.2f} within threshold'
            
            context.variance_analysis[stage_name] = {
                'variance': variance,
                'reason': reason
            }
        
        return context
```

### 5. Pipeline Factory

```python
# factories/pipeline_factory.py
from typing import Optional
from datetime import datetime
from core.pipeline import Pipeline, PipelineConfig
from processors.data_loading import ADXDataLoader
from processors.data_quality import FilterValidCycles
from processors.algorithms import VarianceAnalysis
from processors.ai_analysis import AIAnalysisProcessor

class PipelineFactory:
    """Factory for creating pre-configured pipelines"""
    
    @staticmethod
    def create_hvac_pipeline(
        location_id: str,
        control_id: str,
        start_date: datetime,
        end_date: datetime,
        enable_ai: bool = True,
        min_cycle_length: int = 4
    ) -> Pipeline:
        """
        Create a complete HVAC analysis pipeline.
        
        This pipeline:
        1. Loads data from ADX
        2. Filters invalid cycles
        3. Analyzes power variance
        4. Calculates baselines (if variance is low)
        5. Runs AI analysis (if variance is high and enabled)
        """
        
        # Build processor list
        processors = [
            ADXDataLoader(location_id, control_id, start_date, end_date),
            FilterValidCycles(min_cycle_length=min_cycle_length),
            VarianceAnalysis(rcv_threshold=0.35),
            IdentifyIssues(),
            CalculateBaselines(),
        ]
        
        if enable_ai:
            processors.append(AIAnalysisProcessor(
                model_provider="azure_openai",
                model_name="gpt-4.1"
            ))
        
        # Define conditional execution rules
        conditions = {
            # Only calculate baselines if all stages have low variance
            'CalculateBaselines': lambda ctx: all(
                stage.get('variance') == 'Low' 
                for stage in ctx.variance_analysis.values()
            ),
            
            # Only run AI if any stage has high variance
            'AIAnalysisProcessor': lambda ctx: any(
                stage.get('variance') == 'High' 
                for stage in ctx.variance_analysis.values()
            )
        }
        
        # Configure pipeline
        config = PipelineConfig(
            stop_on_error=False,  # Continue on non-critical errors
            enable_checkpoints=True,  # Save state after each processor
            log_level="INFO"
        )
        
        return Pipeline(processors, config, conditions)
    
    @staticmethod
    def create_simple_pipeline(data_source: str = "csv") -> Pipeline:
        """
        Create a minimal pipeline for Day 1 testing.
        
        This pipeline:
        1. Loads data
        2. Calculates basic statistics
        3. Displays results
        """
        processors = [
            CSVDataLoader("sample_data.csv"),
            BasicStatistics(),
            ConsoleDisplay()
        ]
        
        return Pipeline(processors)
```

## Usage Examples

### Basic Usage (Day 1)

```python
# examples/quickstart.py
from datetime import datetime
from meshinsights_pipeline.core import PipelineContext
from meshinsights_pipeline.factories import PipelineFactory

# Create pipeline
pipeline = PipelineFactory.create_simple_pipeline()

# Initialize context
context = PipelineContext(
    pipeline_name="quickstart",
    correlation_id="test-001"
)

# Run pipeline
result = pipeline.run(context)

# Access results
print(f"Loaded {len(result.raw_data)} rows")
print(f"Statistics: {result.stages['basic_statistics']}")
```

### Production Usage

```python
# examples/production.py
import streamlit as st
from datetime import datetime, timedelta
from meshinsights_pipeline.core import PipelineContext
from meshinsights_pipeline.factories import PipelineFactory

def run_hvac_analysis(location_id: str, control_id: str):
    """Run HVAC analysis pipeline for a site"""
    
    # Create pipeline
    pipeline = PipelineFactory.create_hvac_pipeline(
        location_id=location_id,
        control_id=control_id,
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now(),
        enable_ai=True
    )
    
    # Initialize context with configuration
    context = PipelineContext(
        pipeline_name="hvac_analysis",
        correlation_id=f"{location_id}-{datetime.now().isoformat()}",
        config={
            'location_id': location_id,
            'control_id': control_id,
            'alert_threshold_multiplier': 1.5
        }
    )
    
    # Run pipeline
    try:
        result = pipeline.run(context)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.json(context.errors)
        return None
    
    # Display results in Streamlit
    st.success("Pipeline completed successfully")
    
    # Show warnings if any
    if result.warnings:
        st.warning("Warnings:")
        for warning in result.warnings:
            st.write(f"- {warning}")
    
    # Display data
    st.subheader("Raw Data")
    st.dataframe(result.raw_data.head(100))
    
    # Display variance analysis
    st.subheader("Variance Analysis")
    st.json(result.variance_analysis)
    
    # Display AI analysis if available
    if result.ai_analysis:
        st.subheader("AI Analysis")
        st.write(result.ai_analysis)
    
    # Display thresholds
    if result.thresholds:
        st.subheader("Calculated Thresholds")
        st.json(result.thresholds)
    
    return result

# Streamlit UI
st.title("HVAC Power Analysis")

location_id = st.text_input("Location ID", "site-001")
control_id = st.text_input("Control ID", "ctrl-001")

if st.button("Run Analysis"):
    with st.spinner("Running pipeline..."):
        result = run_hvac_analysis(location_id, control_id)
```

### Testing Individual Processors

```python
# tests/test_processors.py
import pytest
import pandas as pd
from meshinsights_pipeline.core import PipelineContext
from meshinsights_pipeline.processors.data_quality import FilterValidCycles

def test_filter_cycles():
    """Test cycle filtering processor"""
    
    # Create test data
    test_data = pd.DataFrame({
        'cycle': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
        'energy': [100, 150, 120, 0, 0, 200, 250, 230, 240, 220],
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='H')
    })
    
    # Create context
    context = PipelineContext(
        pipeline_name="test",
        correlation_id="test-001",
        raw_data=test_data
    )
    
    # Create and run processor
    processor = FilterValidCycles(min_cycle_length=3, min_median_energy=10)
    result = processor(context)
    
    # Assertions
    assert len(result.raw_data) == 5  # Only cycle 3 should remain
    assert result.stages['filter_cycles']['cycles_removed'] == 2
    assert result.stages['filter_cycles']['valid_cycles'] == 1

def test_processor_error_handling():
    """Test processor error handling"""
    
    # Create context with no data
    context = PipelineContext(
        pipeline_name="test",
        correlation_id="test-002"
    )
    
    # Processor should raise error due to missing data
    processor = FilterValidCycles()
    
    with pytest.raises(ValueError, match="No raw data available"):
        processor(context)
```

## Tradeoffs

### Advantages

1. **Simplicity**: Simple mental model - data flows through processors
2. **Testability**: Each processor can be tested in isolation
3. **Flexibility**: Easy to add/remove/reorder processors
4. **Debugging**: Execution logs and checkpoints make debugging straightforward
5. **Gradual Adoption**: Can migrate existing code incrementally
6. **Type Safety**: Pydantic provides runtime validation and IDE support
7. **Low Learning Curve**: Standard Python patterns, no complex abstractions

### Disadvantages

1. **Memory Usage**: Context accumulates data throughout pipeline
2. **Sequential Execution**: No built-in parallelism (can be added)
3. **Tight Coupling to Context**: Processors depend on context structure
4. **Versioning Challenges**: Context schema changes affect all processors
5. **Limited Reusability**: Processors tied to specific context fields

### Mitigation Strategies

- **Memory**: Implement data cleanup processors or use references for large datasets
- **Parallelism**: Use `concurrent.futures` for independent processors
- **Coupling**: Use context sub-models for domain separation
- **Versioning**: Implement context versioning with migration logic
- **Reusability**: Create generic processors with configurable field mappings


## Getting Started

```bash
# Install dependencies
pip install pydantic pandas streamlit

# Create your first processor
cat > my_processor.py << 'EOF'
from meshinsights_pipeline.core import Processor, PipelineContext

class MyProcessor(Processor):
    def process(self, context: PipelineContext) -> PipelineContext:
        # Your logic here
        context.stages['my_processor'] = {'status': 'complete'}
        return context
EOF

# Create and run pipeline
cat > run_pipeline.py << 'EOF'
from meshinsights_pipeline.core import Pipeline, PipelineContext
from my_processor import MyProcessor

pipeline = Pipeline([MyProcessor()])
context = PipelineContext(pipeline_name="test", correlation_id="001")
result = pipeline.run(context)
print(result.stages)
EOF

python run_pipeline.py
```
