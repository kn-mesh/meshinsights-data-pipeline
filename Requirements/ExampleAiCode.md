# UnitPowerAiClassification.py

```python
"""
Utilizes the UnitPowerAnalysis object curate data and create the user message for AI issue classification

Functionality:
1. Private methods to perform calculations on the data
2. Pair each calculation method with associated user message snipper
3. Combine all snippets into a single user message
4. Trigger AI inference API call, returning the parsed string final output
"""
import openai
import os
from openai import OpenAI, AzureOpenAI
from google import genai
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from typing import Dict, Any
import pandas as pd


class UnitPowerAiClassification:
    """
    Context object for the UnitPowerAiClassification class

    NOTE: This is only utilized if a cycle has experienced high power variance without a clear issue being identified
    """
    def __init__(self, data_obj, model_provider: str = "azure_openai", model_name: str = "gpt-4.1"):
        """
        Parameters
        ----------
        data_obj : PowerAnalysisPipeline (from src.classes.UnitPowerAnalysis)
            Object containing the data to be used for the AI inference
        model_provider : str
            The AI model provider to use ('openai', 'google', 'anthropic')
        model_name : str
            The specific model name to use for the chosen provider
        """
        self.data_obj = data_obj
        self.model_provider = model_provider
        self.model_name = model_name
        self.user_message = ""
        self.system_message = self._system_message()

        self.curated_data = {
            "power_consumption_over_time": self._curate_power_consumption_over_time(),
            #"nature_of_power_consumption_variance": self._curate_nature_of_power_consumption_variance(),
        }

        self._generate_full_user_message()


    def _system_message(self) -> str:
        """
        Create a system message for the AI inference
        """

        system_message = f"""
<task>
- Analyze the provided data and classify potential issues with the HVAC unit.
- Your audience understands HVAC systems and data analytics, but they appreciate simple language and concise explanations.
- Primarily focus on the stage(s) that exhibit "High" power variance over the analysis period.
- Don't ignore the other stages, see if the behavior of the other stages hint at issues that help to classify the issue(s) in the stage(s) that exhibit "High" power variance.
</task>

<goals>
1. Determine issue(s) causing the "High" power variance in the stage(s).
2. Any major issues present in the other stages that may not have been identified in the upstream analysis?
3. Provide enough detail to help the user resolve the issue(s) identified. (i.e. stages experiencing issues, when and how the issue is occurring, etc.)
</goals>

<output_format>
2 concise paragraphs (~2 sentences each) that summarize:
1. The issue(s) identified and what trends in the data lead to your conclusion.
2. What additional data and/or investigation would you recommend an expert to analyze/perform to validate your findings.
</output_format>

<output_content>
Don't include:
- Reference to the specific algorithms used (i.e. "Gaussian Mixture model revealed...")
- Overly technical and verbose language (i.e. "...conduct temporal analysis with finer resolution...")

Include:
- Concise examples of trends in the data to provide a simple explanation of the issue(s) identified.
- Drastic changes in power consumption over time (i.e. "In Feb-April the median power consumption in cooling phase 1 was ~2,500W, then in May it dropped to ~170W.)
- Key date ranges where changes occured and where to focus subsequent analysis.
</output_content>

<use_case_context>
- This is a commercial HVAC unit that is monitored by Powerhouse Dynamics's SiteSage solution.
- This unit's data was routed to you because at least one of the stages exhibited "High" power variance over the analysis period without an obvious root cause based on a simplistic rules-based approach.
- What is "High" power variance as opposed to "Low" power variance:
    > Low Variance: The vast majority of power consumption is within a relatively narrow range (not in absolute numbers given that power can range from ~200-13,000W) and often follows a normal distribution or skewed distribution (there's a clear clustering of power values).
    > High Variance: Power consumption varies widely and there is no clear primary clustering of power values.
- Low Variance Example in a Cooling Stage for a single unit:
    > Mean: 293 W
    > Std: 10.9 W
    > Min-Max: 251-324 W
    > 25%: 287W
    > 75%: 300W
    > Power Frequency Distribution: Normal Distribution
- High Variance Example in a Cooling Stage for a single unit:
    > Mean: 3987 W
    > Std: 2530 W
    > Min-Max: 1722-7957 W
    > 25%: 1724 W
    > 75%: 6973 W
    > Power Frequency Distribution: Binomial Distribution
</use_case_context>

<make_sure_to_consider_the_following_when_classifying_issues>
- Is there a time based trend in the power consumption?
    > Is the average power consumption increasing or decreasing drastically over time?
    > Is the variance increasing or decreasing drastically over time?
- Are there any patterns that are present across multiple stages?
- Is the machine idling during a heating or cooling cycle as shown by a very low power consumption over many cycles relative to previous cycles?
- Is the machine short cycling as shown by rapid changes in power consumption over a given cycle?
</make_sure_to_consider_the_following_when_classifying_issues>


<data_context>
- Analysis period: {self.data_obj.start_date} - {self.data_obj.end_date}
- Data resolution: hourly measurements
- Raw datapoints: timestamp [datetime], power [watts], stage [str], cycle [int]...
- Stages in the dataset: cooling stage 1, cooling stage 2, heating stage 1, heating stage 2, and fan stage.
    > All data from other stages such as "Fan Only" and "Idle" are deleted.
    > Trust that the data and the stages are correct.
- Cycle is an index that keeps track of distinct cycles which are identified as successive measures where the stage is constant.
</data_context>
        """
        return system_message


    def _generate_full_user_message(self) -> None:
        """
        Compose individual user messages into the single full user message used in the AI inference APIs
        """
        # User Messages
        user_message_power_consumption_over_time = self._user_message_power_consumption_over_time()
        user_message_nature_of_power_consumption_variance = self._user_message_nature_of_power_consumption_variance()
        
        # Full User Message
        user_message = f"""
{user_message_nature_of_power_consumption_variance}

{user_message_power_consumption_over_time}
        """

        self.user_message = user_message



    def _curate_power_consumption_over_time(self) -> Dict[str, Dict[str, float]]:
        """
        Simplify the power consumption over time data into a single structure.

        For each full Sunday–Saturday week in the requested date range, compute:
        • count – number of telemetry rows available for the stage
        • mean – mean energy value (rounded to 0 decimals)
        • median – median energy value (rounded to 0 decimals)
        • std_dev – population standard deviation (rounded to 0 decimals)

        Additional business rules
        -------------------------
        1. Only analyse stages that exhibit "High" power variance. If one
        stage in a hot/cold pair is "High", its sibling stage is analysed
        as well.
        2. Weeks with *zero* datapoints across all analysed stages are omitted
        from the returned dictionary.
        3. A stage is included for a given week **only if it contains at least
        one datapoint in that week**.
        """
        # ---------- 1. Determine stages to analyse ----------
        high_variance_stages = {
            stage for stage, meta in self.data_obj.power_variance.items()
            if meta.get("variance") == "High"
        }

        stage_pairs = [
            {"cooling_stage_1", "cooling_stage_2"},
            {"heating_stage_1", "heating_stage_2"},
            {"fan_stage"},
        ]

        stages_to_analyse = set(high_variance_stages)
        for pair in stage_pairs:
            if high_variance_stages & pair:
                stages_to_analyse |= pair

        if not stages_to_analyse:
            return {}

        # ---------- 2. Helper look-ups ----------
        from src.constants import TSTATE_TO_STAGE  # avoid circular import
        stage_to_tstates: Dict[str, set[int]] = {}
        for tstate, stage in TSTATE_TO_STAGE.items():
            stage_to_tstates.setdefault(stage, set()).add(tstate)

        # ---------- 3. Build week boundaries ----------
        df = self.data_obj.data.copy()
        df["date"] = df["timeStamp"].dt.date

        start = self.data_obj.start_date
        end = self.data_obj.end_date
        start -= pd.Timedelta(days=start.weekday() + 1) if start.weekday() != 6 else pd.Timedelta(0)
        end += pd.Timedelta(days=(5 - end.weekday()) % 7 + 1)

        week_starts = pd.date_range(start=start, end=end, freq="W-SUN")

        # ---------- 4. Aggregate ----------
        weekly_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

        for week_start in week_starts:
            week_end = week_start + pd.Timedelta(days=6)
            label = f"{week_start.date()} - {week_end.date()}"

            week_mask = (df["date"] >= week_start.date()) & (df["date"] <= week_end.date())
            week_df = df[week_mask]

            stage_stats: Dict[str, Dict[str, float]] = {}
            total_week_count = 0  # Track total datapoints for this week

            for stage in stages_to_analyse:
                tstates = stage_to_tstates.get(stage, set())
                stage_df = week_df[week_df["tstate"].isin(tstates)]
                count = int(len(stage_df))

                # Skip stages that have *no* datapoints this week
                if count == 0:
                    continue

                total_week_count += count
                stage_stats[stage] = {
                    "count": count,
                    "mean": int(round(stage_df["energy"].mean(), 0)),
                    "median": int(round(stage_df["energy"].median(), 0)),
                    "std_dev": int(round(stage_df["energy"].std(ddof=0), 0)),
                }

            # Skip weeks that ended up with no stage entries
            if total_week_count == 0:
                continue
            weekly_summary[label] = stage_stats

        return weekly_summary 

    def _user_message_power_consumption_over_time(self) -> str:
        """
        Create a user message for the power consumption over time data
        """
        user_message = f"""
<power_consumption_calculations_over_time>

<power_consumption_calculations_over_time_context>
- Calculates the count, mean, median, and standard deviation of power consumption measurements over the analysis period for each "relevant" stage on a weekly basis.
- "relevant" stages are those that exhibit "High" power variance over the analysis period and any higher/lower stages of the same type.
    > heating | cooling | fan stages are the broad distinct types.
    > ex: if cooling stage 1 is "High" variance then cooling stage 2 is also included in the calculations.
- Data format: dictionary with keys as week labels and values as dictionaries with keys as stage names and values as dictionaries with keys as count, mean, median, and standard deviation.
</power_consumption_calculations_over_time_context>

<power_consumption_calculations_over_time_data>
{self.curated_data["power_consumption_over_time"]}
</power_consumption_calculations_over_time_data>

</power_consumption_calculations_over_time>
        """
        return user_message



    def _user_message_nature_of_power_consumption_variance(self) -> str:
        """
        Create a user message for the nature of power consumption variance data
        """
        user_message = f"""
<power_variance>
<power_variance_context>
- Power variance was calculated in a previous phase of the pipeline using the same data and time period.
- The only options for power variance are "Low" or "High".
- The detailed docstrings for the python code that calculated the power variance are provided below:
'''
Determines if power variance for an HVAC stage is "Low" or "High".

Methodology
-----------
The function employs a sophisticated two-stage process to ensure both
robustness against data glitches and accuracy in identifying complex operational
patterns. A stage is classified as "High" variance if it meets the criteria of
either stage.

1.  **Robust Dispersion Check (Safety Net)**:
    First, a fast and outlier-resistant check is performed using a "Robust
    Coefficient of Variation" (rCV), calculated as `Median Absolute Deviation / Median`.
    This metric is designed to catch stages with broadly and persistently
    spread-out power readings (high dispersion).
    - **Purpose**: This acts as a crucial safety net. A GMM might fail to
        identify high variance in a distribution that is uniformly random
        (i.e., has no distinct clusters), incorrectly modeling it as one wide
        component. The rCV check catches these cases.
    - **Robustness**: By using the median and MAD instead of the mean and
        standard deviation, this check is immune to spurious, short-lived
        sensor spikes, preventing false positives from data glitches.

2.  **Multi-Modal Analysis (GMM)**:
    If the data is not flagged for high dispersion, the function proceeds to
    a more detailed analysis using a Gaussian Mixture Model (GMM). This
    stage tests the hypothesis that "High" variance may be caused by the
    equipment operating in multiple distinct power modes (e.g., a bimodal
    distribution).
    - The GMM fits multiple Gaussian distributions to the data and uses the
        Bayesian Information Criterion (BIC) to find the optimal number of
        underlying components (modes).
    - A stage is only classified as "High" variance by the GMM if it finds
        at least two modes that are both **statistically significant** and
        **meaningfully separated**, based on the following post-processing rules:

        a. **Significance Filtering**: Components representing a tiny fraction
            of the data (see `MIN_SIGNIFICANT_WEIGHT`) are discarded as noise.
        b. **Separation Filtering**: The remaining components' means must be
            sufficiently far apart relative to the overall median power level
            (see `SEPARATION_THRESHOLD`) to be considered practically distinct.

Tuning Parameters
-----------------
The behavior of the algorithm is controlled by several constants that can be
adjusted for different equipment or environments.

- `RCV_THRESHOLD` (Default: 0.35):
    - **Purpose**: The cutoff for the initial robust dispersion check. If
        `rCV > RCV_THRESHOLD`, the stage is immediately flagged as "High"
        variance.

- `MIN_SAMPLES_FOR_TEST` (Default: 50):
    - **Purpose**: Prevents analysis on sparse data where results would be
        unreliable. Stages with fewer samples are defaulted to "Low" variance.

- `MAX_COMPONENTS` (Default: 3):
    - **Purpose**: Limits the complexity of the GMM to prevent overfitting.
        It restricts the search to simple, common cases (e.g., unimodal vs.
        bimodal).

- `MIN_SIGNIFICANT_WEIGHT` (Default: 0.10):
    - **Purpose**: The "noise filter" for GMM components. Defines the minimum
        proportion of data a cluster must represent to be considered a
        significant operational mode.

- `SEPARATION_THRESHOLD` (Default: 0.20):
    - **Purpose**: The "practical difference" filter. Ensures that multiple
        identified modes are distinct enough to matter in practice.

Parameters
----------
df : pd.DataFrame
    A DataFrame containing time-series energy data. Must include 'tstate'
    and 'energy' columns.

Returns
-------
Dict[str, Dict[str, Any]]
    A dictionary mapping each stage name to a sub-dictionary containing the
    analysis results:
    - 'variance': "Low" or "High" classification.
    - 'reason': A brief explanation for the classification.
    - 'n_components': The optimal number of Gaussian components found.
    - 'means': List of mean power values for each component.
    - 'weights': List of weights (proportions) for each component.
    - 'covariances': List of covariances for each component.
'''
</power_variance_context>


<power_variance_classification>
{self.data_obj.power_variance}
</power_variance_classification>

</power_variance>
        """

        return user_message


    @traceable(name="issue_classification")
    def trigger_ai_issue_classification(self, model_provider: str = "azure_openai", model_name: str = "gpt-4.1") -> str:
        """
        Trigger AI Inference API call, returning the parsed string final output

        Parameters
        ----------
        data_obj : PowerAnalysisPipeline
            Object containing the data to be used for the AI inference
        model_provider : str
            The AI model provider to use ('openai', 'google', 'anthropic')
        model_name : str
            The specific model name to use for the chosen provider

        Returns
        -------
        str
            The parsed string final output
        """
        load_dotenv()

        if model_provider == "openai":
            return self._openai_analysis(model_name)
        
        elif model_provider == "azure_openai":
            return self._azure_openai_analysis(model_name)

        elif model_provider == "google":
            return self._google_analysis(model_name)

        elif model_provider == "anthropic":
            return self._anthropic_analysis(model_name)

    
    ### Private methods - AI Inference API calls ###
    def _openai_analysis(self, model_name: str) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()

        response = client.responses.create(
            model=model_name,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": self.user_message
                }
            ]
        )

        return response.output_text


    def _azure_openai_analysis(self, model_name: str) -> str:
        if model_name == "gpt-4.1":
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT_4_1")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = "2025-01-01-preview"

        client = AzureOpenAI(
            azure_endpoint = azure_endpoint, 
            api_key=api_key,  
            api_version=api_version
        )

        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.user_message}
            ]
        )


        return response.choices[0].message.content

    def _anthropic_analysis(self, model_name: str) -> str:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model_name,
            max_tokens=6000,
            thinking={
                "type": "enabled",
                "budget_tokens": 3000
            },
            system=self.system_message,
            messages=[
                {"role": "user", "content": self.user_message}
            ]
        )

        return response.content[1].text


    def _google_analysis(self, model_name: str) -> str:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        prompt = f"{self.system_message}\n\n{self.user_message}"

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        return response.text
```