"""Five-metric synthetic scorer with NAVSIM comfort and progress semantics."""

import numpy as np

from navsim.planning.simulation.planner.pdm_planner.scoring.navtest_comfort_metrics import ego_is_comfortable_navtest
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex


class NavtestAlignedFiveMetricScorer(PDMScorer):
    """Score future-only comfort and zero progress when NC or DAC fails."""

    def _calculate_history_comfort(self) -> None:
        # The output field retains its historical name for CSV compatibility.
        time_point_s = np.arange(self.proposal_sampling.num_poses + 1) * self.proposal_sampling.interval_length
        is_comfortable = ego_is_comfortable_navtest(self._states, time_point_s)
        self._weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT] = is_comfortable.all(axis=-1)

    def _aggregate_pdm_scores(self) -> np.ndarray:
        if not self._config.five_metric_only:
            raise ValueError("NavtestAlignedFiveMetricScorer requires five_metric_only=True")

        safety_gate = self._multiplicative_metric_scores()
        safe_progress = self._progress_raw * safety_gate
        max_safe_progress = np.max(safe_progress)
        if max_safe_progress > self._config.progress_distance_threshold:
            normalized_progress = safe_progress / max_safe_progress
        else:
            normalized_progress = np.ones_like(safe_progress)
            normalized_progress[safety_gate == 0.0] = 0.0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        weights = self._config.weighted_metrics_array
        weighted_score = (self._weighted_metrics * weights[:, None]).sum(axis=0) / weights.sum()
        return safety_gate * weighted_score
