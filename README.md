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

### Why `python -m` is needed
When using Streamlit or other tools that spawn subprocesses, use `python -m` to ensure the Python environment is properly configured. Direct commands like `uv run streamlit run` may fail with import errors.

## Common Issues

### "ModuleNotFoundError: No module named 'src'"
This means your package isn't installed in editable mode. Fix with:
```bash
uv pip install -e .
# or
uv sync --reinstall-package your-package-name
```

### Slow first run
The first run after `uv sync` can be slow due to:
- Package installation in editable mode
- Python bytecode compilation
- Heavy dependency imports (pandas, scikit-learn, etc.)