# MeshInsights Data Pipeline

## Overview
A lightweight, composable pipeline framework for analytics:
- Context: shared state model flowing through the pipeline
- Processor: a single, reusable step operating on the context
- Pipeline: orchestrates processors, conditions, checkpoints, and logging


## Key Concepts
- `PipelineContext`: Core reusable data model with explicit extension points (`config`, `stages`, `processed_data`, optional domain sub-models).
- `Processor`: Stateless step; reads from context and returns an updated context (immutability by convention).
- `Pipeline`: Executes processors in order, supports conditional execution, error policy, and optional checkpoints.




# uv quickstart
## Initial Setup
1. Initialize uv project: `uv init`
2. Set Python version: `uv python pin 3.12.10` (or `uv venv --python 3.12.10`)
3. Create virtual environment: `uv venv` (creates .venv folder)

## Project Structure for `src` Layout
If you want to use `from src.module import ...` style imports, add this to your `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]
include = ["*"]

[tool.uv]
package = true  # Ensures your package is installed when running uv sync
```

## Adding Dependencies
Add python libraries in the 'dependencies' section in `pyproject.toml`:
```toml
[project]
dependencies = [
    "langchain>=0.3.17",
    "openai==1.62.2",
    # ... more dependencies
]
```

After making changes, run `uv sync` to install the dependencies.

## Running Code

### For scripts with src imports
- Run Python modules: `uv run python -m src.apps.main`
- Run Streamlit apps: `uv run python -m streamlit run src/streamlit_apps/app.py`