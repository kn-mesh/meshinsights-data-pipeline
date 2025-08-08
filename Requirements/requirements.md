# Overview
Our existing approach to developing a MeshInsights data analytics pipeline is to start from scratch and/or copy and paste code from existing pipelines. This makes it difficult to maintain, modify, test, and collaborate on these pipelines.

# MeshInsights Pipeline: Key Attributes
- Language: Python
- Analytics Approach: Batch Processing
- Data: Time Series data from IoT devices (telemetry, events, alarms, logs)
- Production Deployment: Containerized
- Key Layers: (1) Data Access + Normalization (2) Data model/object (3) Composable Data Processing Steps [python algorithms] (4) GenAI Analysis (5) Pipeline orchestration [setting the steps, running the pipeline, etc.]

# Scope
- Update the base data processing pipeline to be more composable, testable and reduce the burden on engineers to spin up a new pipeline and modify existing ones written by others.
- When a new project is started, the developer should be able to spin up a new pipeline using the base components.


# Outcomes
1. A single developer will have data flowing through the pipeline on day 1 and be displayed in a simple streamlit app. The idea isn't to have a meaningful analysis, but to have the bones of a pipeline working with limited effort.
- Ex: we receive credentials from a database (where we already have a plugin), perform simplest possible query + single analysis step (i.e. avg temperature) + simplest possible GenAI analysis, and display the raw data in streamlit.
2. Each transformation step in the pipeline is separated and composable passing the data object as a parameter. This will enable developers to focus on specific sections and work with a team effectively. The developer needs to have deep flexibilty in a given pipeline transformation step to apply various algorithms / data curation methods.
3. Each section of the pipeline is testable and can be run independently.


# Key Decisions (non-exhaustive)
- How and where to define the data object / model?
- Is the data object modified in place as it flows through the pipeline?


# Existing Approach
1. **Data access plugin:** provides classes to authenticate to a database, query data and convert to a pandas dataframe. (i.e. SQL, ADX)
- Base class: `DBConnector`
- Manager class: `PluginManager`
- Specific plugin classes (example): `ADXConnectionManager` + `ADXConnector` 
2. **Custom Data Pipeline Class:** retrieves data via queries, processes data in a pipeline and saves relevant raw and curated data.
- Custom built for each use case (ex: PowerAnalysisPipeline).
- Complex algorithms are pulled out of the class and into 'algorithms' directory (to keep some complex responsibilites separate and reduce noise).
- Leverages the data access plugin to query data from a database and convert to a pandas dataframe.
3. **Custom GenAI Analysis Class:** uses the Custom Data Pipeline Class as a parameter, then packages and/or processes relevant data for GenAI API calls.
- Stithces together the system and user prompts which include curated data in string format.


# Example Pipeline: Powerhouse Dynamics HVAC Power Analysis
- **Overview:** The pipeline retrieves + normalizes the most recent 3 months of data for a given machine from MySQL, then performs a series analysis to determine the power variance, check for well known issues, calculate power alarm thresholds, and if the data is complex enough -- analyze with GenAI.
- **Downstream Uses of the Pipeline Class:** 
1. Display raw and curated data in a streamlit app (time series plotly charts)
2. Display the GenAI analysis
3. Update the alert threshold database with the calculated thresholds (if the power variance is low enough)