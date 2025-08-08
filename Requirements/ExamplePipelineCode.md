# UnitPowerAnalysis.py
```python
import pandas as pd
from datetime import date
import numpy as np
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.classes.AdxDataRetrieval import AdxDataRetriever
from src.algorithms.inner_cycle_algs import check_for_short_cycling
from src.constants import COOLING_STAGE_1, COOLING_STAGE_2, HEATING_STAGE_1, HEATING_STAGE_2, FAN_STAGE, TSTATE_TO_STAGE
from src.classes.UnitPowerAiClassification import UnitPowerAiClassification
from src.algorithms.algorithms import classify_power_variance


class PowerAnalysisPipeline:
    """
    End-to-end pipeline for deriving per-stage energy baselines that become the
    "HVAC Stage Failure" alarm thresholds.

    Workflow
    --------
    1. Fetch minute-level thermostat (`control_minutedata`) and energy data with `AdxDataRetriever`.
    2. Merge the feeds and label consecutive compressor/furnace runs as `cycle`.
    3. Remove obviously bad cycles.
    4. Identify issues and power variance per stage.
    5. Calculate baseline power thresholds for each stage.

    Key Attributes (used by downstream code)
    ----------
    data : pd.DataFrame
        Raw, merged dataframe of control and energy data, with a `cycle`
        column added. Produced by `_get_data`.
    cleaned_data : Dict[str, pd.DataFrame]
        A dictionary mapping each HVAC stage to a curated dataframe of its
        cycles. Populated by `_curate_stage_data`.
    baseline_power_thresholds : Dict[str, Optional[float]]
        A dictionary mapping each HVAC stage to its calculated baseline power
        threshold. Populated by `_calculate_baseline_power_thresholds`.
    identified_issues : Dict[str, list]
        A dictionary mapping each HVAC stage to a list of identified issues,
        such as "short_cycling". Populated by `_identify_issues_per_stage`.
    power_variance : Dict[str, Dict[str, Any]]
        A dictionary mapping each HVAC stage to the results of its power
        variance analysis. Must contain a 'variance' key with a value of 'Low' or 'High'.
        - NOTE Additional keys may be added to provide further details about the variance analysis
    """
    def __init__(self, locationId: str, controlId: str, start_date: date, end_date: date, merged_raw_data_df: pd.DataFrame = None):
        """
        Instantiate the history object and immediately run the entire pipeline.

        Parameters
        ----------
        locationId : str
            Site identifier (column `locationId` in the raw text files).
        controlId : str
            HVAC controller / thermostat identifier (`controlId`).
        start_date, end_date : datetime.date
            Inclusive date range delimiting the historical look-back window.
        merged_raw_data_df : pd.DataFrame | None
            (Optional) Dataframe of merged control and energy data (produced by AdxDataRetriever)
        """
        self.locationId = locationId
        self.controlId = controlId
        self.start_date = start_date
        self.end_date = end_date
        
        # Set data placeholders
        self.data = pd.DataFrame()
        self.cleaned_data = {
            COOLING_STAGE_1: pd.DataFrame(),
            COOLING_STAGE_2: pd.DataFrame(),
            HEATING_STAGE_1: pd.DataFrame(),
            HEATING_STAGE_2: pd.DataFrame(),
            #"fan_cycle": pd.DataFrame(), # NOTE: No current fan cycles in the dataset
        }
        self.power_variance: Dict[str, Dict[str, Any]] = {
            COOLING_STAGE_1: {"variance": "Low"},
            COOLING_STAGE_2: {"variance": "Low"},
            HEATING_STAGE_1: {"variance": "Low"},
            HEATING_STAGE_2: {"variance": "Low"},
            #"fan_cycle": {"variance": "Low"}, # NOTE: No current fan cycles in the dataset
        }
        self.identified_issues = {
            COOLING_STAGE_1: [],
            COOLING_STAGE_2: [],
            HEATING_STAGE_1: [],
            HEATING_STAGE_2: [],
            #"fan_cycle": [], # NOTE: No current fan cycles in the dataset
        }
        self.baseline_power_thresholds = {
            COOLING_STAGE_1: None,
            COOLING_STAGE_2: None,
            HEATING_STAGE_1: None,
            HEATING_STAGE_2: None,
            #"fan_cycle": None, # NOTE: No current fan cycles in the dataset
        }
        
        
        self._run_data_transformation_pipeline(merged_raw_data_df)

    def _run_data_transformation_pipeline(self, merged_data_df: pd.DataFrame = None) -> None:
        """
        Run the entire data transformation pipeline
        """
        # 1) Fetch raw telemetry 
        if merged_data_df is None:
            self.data = self._get_data()
        else:
            self.data = merged_data_df

        # 2) Filter out invalid cycles
        self.data = self._filter_valid_cycles(self.data)

        # 3) Evaluate power variance on raw data
        self._identify_power_variance_per_stage_raw(self.data)

        # 4) Detect issues (e.g., short-cycling)
        self._identify_issues_per_stage(self.data)

        # 5) Curate per-stage / per-cycle dataframes
        self.cleaned_data = self._curate_stage_data(self.data)

        # 6) Check power variance on curated dataframes
        ## Issue to check for: long cycles are skewing the raw data variance results, checking to make sure curated cycles are not high variance
        self._identify_power_variance_per_stage_curated(self.cleaned_data)

        # 7) Derive baseline thresholds
        self.baseline_power_thresholds = self._calculate_baseline_power_thresholds()

        
    def _get_data(self) -> pd.DataFrame:
        """
        Retrieve, merge and pre-process raw telemetry for the requested period.

        Steps
        -----
        1. Load and merge minute-level control and energy data using `AdxDataRetriever`.
        2. Retain the minimal set of columns required by downstream logic.

        Returns
        -------
        pd.DataFrame
            Processed dataframe with columns: timeStamp, tstate, energy, cycle
        """
        retriever = AdxDataRetriever(self.locationId, self.controlId, self.start_date, self.end_date)
        
        # Data retrieval and merging now handled by AdxDataRetriever
        merged_df = retriever.get_data()

        return merged_df

    
    def _filter_valid_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new dataframe containing only cycles that satisfy:
        • ≥ 4 rows
        • median(energy) > 0
        • mode(energy) > 0

        This implementation is vectorized to improve performance by avoiding
        Python-level iteration over grouped data, especially for calculating
        the mode.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``cycle`` and ``energy`` columns.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe with only valid cycles
        """
        MIN_VALID_CYCLE_LENGTH = 4

        if df.empty or 'cycle' not in df.columns or 'energy' not in df.columns:
            return pd.DataFrame(columns=df.columns)

        # Step 1: Calculate count and median in a single, efficient pass.
        cycle_stats = df.groupby("cycle").agg(
            count=("cycle", "size"),
            median_energy=("energy", "median"),
        )

        # Step 2: Calculate the mode in a vectorized way to avoid a slow lambda.
        # This works by counting (cycle, energy) pairs and then finding the
        # energy with the maximum count for each cycle.
        if not df.empty:
            value_counts = df.groupby(['cycle', 'energy'], observed=True).size()
            loc_of_max_counts = value_counts.groupby('cycle', observed=True).idxmax()
            
            if not loc_of_max_counts.empty:
                # Extract mode values from the multi-index and join back.
                modes = pd.Series(
                    loc_of_max_counts.apply(lambda x: x[1]),
                    index=loc_of_max_counts.index,
                    name="mode_energy"
                )
                cycle_stats = cycle_stats.join(modes)

        # Ensure mode_energy column exists to prevent errors, even if no modes were found.
        if 'mode_energy' not in cycle_stats:
            cycle_stats['mode_energy'] = np.nan

        # Step 3: Filter for valid cycles based on the aggregated statistics.
        # We drop cycles where a mode couldn't be determined.
        cycle_stats.dropna(subset=['mode_energy'], inplace=True)
        valid_cycles = cycle_stats[
            (cycle_stats["count"] >= MIN_VALID_CYCLE_LENGTH)
            & (cycle_stats["median_energy"] > 0)
            & (cycle_stats["mode_energy"] > 0)
        ].index

        # Step 4: Return a filtered copy of the original dataframe.
        return df[df["cycle"].isin(valid_cycles)].copy()


    def _identify_power_variance_per_stage_raw(self, df: pd.DataFrame) -> None:
        """
        Populate `self.power_variance` with the power variance per stage.

        Parameters
        ----------
        df : pd.DataFrame
            Output of `_filter_valid_cycles` (must contain 'tstate', and 'energy' columns).
            - Dataframe must be sorted chronologically 
        """
        # --------------------------- tunable parameters --------------------------- #
        MAX_RAW_SAMPLES_PER_STAGE = 5000  # Cap to most recent N samples for performance
        RCV_THRESHOLD = 0.35          # robust CV cut-off (MAD / median)
        MIN_SAMPLES_FOR_TEST = 50     # samples required before running analysis
        MAX_COMPONENTS = 3            # 1–3 mixture components
        MIN_SIGNIFICANT_WEIGHT = 0.10 # drop tiny GMM components (<10 % of points)
        SEPARATION_THRESHOLD = 0.20   # min relative gap (Δμ / overall median)
        # ------------------------------------------------------------------------- #

        # Pre-compute power vectors for each stage, using only the most recent data
        # to cap GMM fitting time. Assumes `df` is sorted chronologically.
        stage_payloads = []
        for tstate, stage_name in TSTATE_TO_STAGE.items():
            power_values = df.loc[df["tstate"] == tstate, "energy"].values

            if len(power_values) > MAX_RAW_SAMPLES_PER_STAGE:
                power_values = power_values[-MAX_RAW_SAMPLES_PER_STAGE:]

            stage_payloads.append(
                (stage_name, power_values.astype(float))
            )

        # Helper executed in parallel
        def _classify_stage_parallel(stage_name: str, power: np.ndarray):
            result = classify_power_variance(
                power=power,
                rcv_threshold=RCV_THRESHOLD,
                min_samples=MIN_SAMPLES_FOR_TEST,
                max_components=MAX_COMPONENTS,
                min_significant_weight=MIN_SIGNIFICANT_WEIGHT,
                separation_threshold=SEPARATION_THRESHOLD,
            )
            return stage_name, result

        # Parallel map – one worker per HVAC stage (4–5 tasks max)
        pairs = Parallel(n_jobs=-1, backend="threading")(
            delayed(_classify_stage_parallel)(stage_name, power)
            for stage_name, power in stage_payloads
        )

        # Update stored variance results
        self.power_variance.update({stage: res for stage, res in pairs})


    def _identify_power_variance_per_stage_curated(
        self,
        data_by_stage: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Re-check power variance after cycle curation.

        Why?
        ----
        Extremely long / anomalous cycles can inflate the raw-data variance
        calculation.  By re-evaluating the already-curated cycle data we ensure
        that stages originally labelled as "Low" variance truly remain so.

        Logic
        -----
        • Only stages whose initial variance == "Low" are re-checked.  
        • Empty curated dataframes are skipped (no new information).  
        • If the curated check returns "High", we overwrite the existing entry
          in `self.power_variance` for that stage.

        Parameters
        ----------
        data_by_stage : Dict[str, pd.DataFrame]
            Mapping of stage name → curated dataframe produced by
            `_curate_stage_data`.
        """
        # --------------------------- tunable parameters --------------------------- #
        RCV_THRESHOLD = 0.35          # robust CV cut-off (MAD / median)
        MIN_SAMPLES_FOR_TEST = 20     # cycles required before running analysis
        MAX_COMPONENTS = 3            # 1–3 mixture components
        MIN_SIGNIFICANT_WEIGHT = 0.10 # drop tiny GMM components (<10 % of points)
        SEPARATION_THRESHOLD = 0.20   # min relative gap (Δμ / overall median)
        # ------------------------------------------------------------------------- #

        # NOTE Parallel execution did not speed up the process, so keeping it sequential for now
        for stage, stage_df in data_by_stage.items():
            # Skip stages that were already "High" or have no curated data
            if self.power_variance.get(stage, {}).get("variance") != "Low":
                continue
            if stage_df.empty or "median_energy_cycle" not in stage_df.columns:
                continue

            # Extract power (energy) values for analysis
            power = stage_df["median_energy_cycle"].values.astype(float)

            # Use the shared variance classification algorithm
            result = classify_power_variance(
                power=power,
                rcv_threshold=RCV_THRESHOLD,
                min_samples=MIN_SAMPLES_FOR_TEST,
                max_components=MAX_COMPONENTS,
                min_significant_weight=MIN_SIGNIFICANT_WEIGHT,
                separation_threshold=SEPARATION_THRESHOLD,
            )

            # Upgrade variance level if curated data shows "High"
            if result.get("variance") == "High":
                self.power_variance[stage].update(result)


    def _identify_issues_per_stage(self, df: pd.DataFrame) -> None:
        """
        Populate `self.identified_issues` with issues present in the data.
        Currently `low_cycle_count` and `short_cycling` are evaluated. Short
        cycling is only evaluated for stages that exhibit "High" power
        variance and have adequate cycles for analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Output of `_filter_valid_cycles` (must contain 'cycle', 'tstate', and 'energy' columns).
        """
        MIN_CYCLES_FOR_VALID_BASELINE = 10

        for tstate, stage in TSTATE_TO_STAGE.items():
            stage_df = df[df["tstate"] == tstate]

            # Check for a minimum number of cycles to be able to curate the data
            if stage_df['cycle'].nunique() < MIN_CYCLES_FOR_VALID_BASELINE:
                self.identified_issues[stage].append("low_cycle_count")

            # Only check for short cycling if power variance is high and there are enough cycles to curate the data
            if (self.power_variance.get(stage, {}).get('variance') == "High" and 
                "low_cycle_count" not in self.identified_issues[stage]):
                if check_for_short_cycling(stage_df):
                    self.identified_issues[stage].append("short_cycling")


    def _curate_stage_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Build a curated dataframe for each HVAC stage.

        Logic:
        1. Split the raw dataframe by `tstate` (one slice per stage).
        2. If any issues were identified for a stage, return an empty
           dataframe for that stage (no attempt at summarisation).
        3. Otherwise choose the appropriate analysis based on the power
           variance classification for that stage.
           # NOTE All stages have the same analytics for now (if any stage-specific logic is needed, it can be added later)

        Parameters
        ----------
        df : pd.DataFrame
            Output of `_filter_valid_cycles`.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping of stage name → curated dataframe with the schema
            ['cycle', 'timeStamp', 'tstate', 'median_energy_cycle'].
        """
        # Step-1: slice master df by tstate
        data_by_stage: Dict[str, pd.DataFrame] = {
            stage: df[df["tstate"] == tstate].copy()
            for tstate, stage in TSTATE_TO_STAGE.items()
        }

        # Canonical empty frame used when issues exist (to avoid downstream errors)
        EMPTY_FRAME = pd.DataFrame(
            columns=["cycle", "timeStamp", "tstate", "median_energy_cycle"]
        )

        curated_data: Dict[str, pd.DataFrame] = {}

        for stage in [HEATING_STAGE_1, HEATING_STAGE_2,
                      COOLING_STAGE_1, COOLING_STAGE_2]: # TODO Add support for fan stage
            # Step-2: skip summarisation entirely when any issue is present
            if self.identified_issues.get(stage):
                curated_data[stage] = EMPTY_FRAME
                continue

            # Step-3: summarise based on variance level
            variance_level = self.power_variance.get(stage, {}).get("variance", "Low")
            stage_df = data_by_stage.get(stage, pd.DataFrame())

            if variance_level == "High":
                curated_data[stage] = self._curate_cycle_data_high_variance(stage_df)
            else:
                curated_data[stage] = self._curate_cycle_data_low_variance(stage_df)

        return curated_data


    def _curate_cycle_data_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return one row per `cycle` with the median value per cycle (reduces noise and simplifies the data)
        - Cycles are consecutive time series measurements that are marked by a change in the tstate value

        Parameters
        ----------
        df : pd.DataFrame
            Stage-specific dataframe

        Returns
        -------
        pd.DataFrame
            Schema: ['cycle', 'timeStamp', 'tstate', 'median_energy_cycle']
        """
        if df.empty:
            return pd.DataFrame(
                columns=['cycle', 'timeStamp', 'tstate', 'median_energy_cycle']
            )

        return (df.groupby('cycle', as_index=False)
                  .agg(timeStamp=('timeStamp', 'first'),
                       tstate=('tstate', 'first'),
                       median_energy_cycle=('energy', 'median')))
    

    def _curate_cycle_data_high_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns an empty dataframe to avoid calculating the baseline power thresholds that are inaccurate.
        - Reason: Don't want to calculate the baseline power threshold when there's high variance
        - FUTURE: Probably want to find discrete 'issues' that cause the high variance

        NOTE: Previous approach was to simply utilize the algorithm in the _curate_cycle_data_low_variance function
        the issue was that the simple act of calcluating the baseline power threshold makes the prototype look innacurate  
        """
        return pd.DataFrame(columns=['cycle', 'timeStamp', 'tstate', 'median_energy_cycle'])  


    def _calculate_baseline_power_thresholds(self) -> Dict[str, Optional[float]]:
        """
        Calculate the baseline power thresholds for each HVAC stage (which will be used to calculate the alert thresholds)
        Only stages that still show *Low* power variance after all variance
        checks are given a threshold.  Stages with *High* variance are skipped
        (threshold set to ``None``) because their baseline would be unreliable.

        Returns
        -------
        Dict[str, Optional[float]]
            Dictionary mapping stage names to baseline power thresholds
        """
        thresholds: Dict[str, Optional[float]] = {}

        for stage in [COOLING_STAGE_1, COOLING_STAGE_2, HEATING_STAGE_1, HEATING_STAGE_2]:
            # Only calculate threshold if power variance is "Low"
            if self.power_variance.get(stage, {}).get("variance", "Low") != "Low":
                thresholds[stage] = None
                continue

            stage_df = self.cleaned_data.get(stage, pd.DataFrame())

            # Calculate baseline from curated data (empty dataframes result in None)
            if stage_df.empty:
                thresholds[stage] = None
            else:
                thresholds[stage] = round(stage_df["median_energy_cycle"].median(), 0)

        return thresholds
    
    def trigger_ai_issue_classification(self, model_provider: str = "openai", model_name: str = "o3-mini") -> str:
        """
        Trigger AI issue classification

        NOTE: this is not automatically triggered, must be called after object is instantiated
        - Reason: this can be slow and expensive, so want to allow downstream code to intelligently trigger this (i.e. in UI)
        """
        return UnitPowerAiClassification(self).trigger_ai_issue_classification(model_provider, model_name)
```