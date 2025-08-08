## MeshInsights Pipeline: Architectural Options (GPT‑5)

### Goal
Create a composable, testable Python pipeline that gets data flowing on day 1 (raw → simple analysis → simple GenAI → Streamlit), while establishing a structure that scales across projects.

---

### Option 1 — Minimal in‑repo composable pipeline (custom orchestrator)
- **Summary**: Single Python package with a small `Pipeline` facade and pluggable `Step` classes. Mirrors current examples while formalizing interfaces and test seams.
- **Key components**
  - **Data model**: `PipelineContext` (pydantic/dataclass) holding `data`, `cleaned_data`, `identified_issues`, `power_variance`, `baseline_power_thresholds`, dates, metadata.
  - **Steps** (each `execute(ctx) -> ctx`): `LoadRawData`, `NormalizeRawData`, `FilterValidCycles`, `ClassifyVariance`, `IdentifyIssues`, `CurateStageData`, `ComputeBaselines`, `GenAIAssessment`.
  - **Orchestrator**: `Pipeline.run(steps)` with timing/logging and error wrapping.
  - **Data access**: Reuse `DBConnector`/`PluginManager`.
- **Pros**
  - Minimal deps; fastest to deliver; easy testing; clear interfaces; no framework lock‑in.
- **Cons**
  - DIY scheduling/observability/parallelism.
- **When to choose**
  - Need day‑1 value and clean interfaces with minimal operational overhead.
- **Day‑1 deliverables**
  - Running pipeline with demo query → avg temp calc → simple GenAI call; Streamlit charts; pytest for each step; Dockerfile.

---

### Option 2 — Prefect‑first task/flow pipeline (moderate framework)
- **Summary**: Wrap each step as a Prefect task; compose with a Prefect flow for retries, caching, scheduling, and observability.
- **Key components**
  - **Data model**: Same `PipelineContext` (serialized via pydantic).
  - **Tasks/Flow**: Thin `@task` wrappers around deep step classes; single `@flow` composes tasks; simple concurrency.
  - **Secrets/creds**: Prefect blocks for DB/LLM keys.
- **Pros**
  - Built‑in UI, retries, scheduling, parametrized runs, distributed execution.
- **Cons**
  - New runtime/service; some framework‑specific wiring.
- **When to choose**
  - Team wants production‑ish ergonomics without heavy platform investment.
- **Day‑1 deliverables**
  - Local Prefect flow mirroring steps; Prefect UI showing runs; Streamlit reads latest outputs; containerized agent.

---

### Option 3 — Dagster asset‑based pipeline (strong lineage/teams)
- **Summary**: Model artifacts as Dagster assets (e.g., `raw_data`, `normalized_data`, `curated_data_by_stage`, `power_variance`, `thresholds`, `ai_assessment`) with typed IO and lineage.
- **Key components**
  - **Data model**: Deep algorithm classes operate on `PipelineContext`; assets are thin wrappers persisting Parquet/DuckDB (IO managers can later target S3/ADLS).
  - **Graph**: Software‑defined assets, sensors, schedules, partitions by date/control.
- **Pros**
  - First‑class lineage/metadata, great observability, partitioning, collaboration at scale.
- **Cons**
  - Higher setup/learning curve; heavier for small teams.
- **When to choose**
  - Multiple pipelines, compliance/lineage needs, multi‑team governance.
- **Day‑1 deliverables**
  - Minimal asset graph: raw → normalized → variance → curated/baselines → ai_assessment; Dagster UI; containerized repo.

---

### Cross‑cutting decisions (applies to all options)
- **Data object mutability**
  - Prefer functional flow: each step returns a new `PipelineContext`. Private helpers may mutate local structures.
- **Public API shape**
  - Intent‑revealing methods: `loadRawData()`, `classifyVariance()`, `identifyIssues()`, `computeBaselines()`, `generateReport()`.
- **Testing**
  - Unit tests per step on `PipelineContext` input/output; integration test for golden path with small sample data; snapshot key metrics.
- **Streamlit**
  - Display raw/curated charts, variance results, thresholds, AI summary; button to trigger pipeline with params.
- **Containerization**
  - Single Dockerfile; `.env` for creds; dev `make run/test/build`.

---

### Selection guidance
- **Small team, fast start, minimal ops**: Option 1.
- **Need retries/schedules/observability soon**: Option 2.
- **Scale, lineage, multiple teams/pipelines**: Option 3.
