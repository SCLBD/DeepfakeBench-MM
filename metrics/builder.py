import numpy as np
import os
import sys

from prettytable import PrettyTable

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from metrics.accuracy import calculate_binary_accuracy
from metrics.auc import calculate_binary_auc
from metrics.average_precision import calculate_binary_ap
from metrics.demographic_parity import calculate_binary_dp
from metrics.eer import calculate_binary_eer
from metrics.equal_odds import calculate_binary_eo
from metrics.maximum_equalized_odds import calculate_binary_meo
from metrics.overall_accuracy_equality import calculate_binary_oae
from metrics.standard_deviation_auc import calculate_binary_sdauc


class MetricBuilder:
    """Track per-dataset metric histories, their macro-average, and best values."""

    metrics_to_minimize = [
        'loss',
        'sample-level EER',
        'video-level EER',
        'sample-level DP',
        'video-level DP',
        'sample-level EO',
        'video-level EO',
        'sample-level OAE',
        'video-level OAE',
        'sample-level MEO',
        'video-level MEO',
        'video-level SDAUC'
    ]
    metrics_to_maximize = [
        'sample-level AUC',
        'video-level AUC',
        'sample-level AP',
        'video-level AP',
        'sample-level accuracy',
        'video-level accuracy'
    ]

    def __init__(self, metric_name, dataset_list):
        """Initialize one metric for the configured datasets.

        Args:
            metric_name: Supported, case-sensitive metric name, including its level.
            dataset_list: Dataset names; an average entry is appended internally.
        """
        if metric_name not in self.metrics_to_minimize and metric_name not in self.metrics_to_maximize:
            raise NotImplementedError(f'{metric_name} has not been implemented.')
        self.metric_name = metric_name
        self.dataset_list = dataset_list + ['average']

        if self.metric_name in self.metrics_to_maximize:
            self.metric_records = {dataset_name: [] for dataset_name in self.dataset_list}
            self.best_scores = {dataset_name: -float('inf') for dataset_name in self.dataset_list}
        else:
            self.metric_records = {dataset_name: [] for dataset_name in self.dataset_list}
            self.best_scores = {dataset_name: float('inf') for dataset_name in self.dataset_list}

    def _update_average(self):
        """Append the unweighted mean after every dataset has one new score."""
        # Require one new score per dataset before adding the round average.
        for dataset_name in self.dataset_list:
            if dataset_name != 'average':
                assert len(self.metric_records[dataset_name]) == len(self.metric_records['average']) + 1

        latest_scores = [self.metric_records[dataset_name][-1] for dataset_name in self.dataset_list
                         if dataset_name != 'average']
        if float('inf') in latest_scores:
            self.metric_records['average'].append(float('inf'))
        elif -float('inf') in latest_scores:
            self.metric_records['average'].append(-float('inf'))
        else:
            self.metric_records['average'].append(np.array(latest_scores).mean())

    def _is_best(self, best_until_now, current_score):
        """Return whether the current score strictly improves the stored best score."""
        if self.metric_name in self.metrics_to_minimize:
            return current_score < best_until_now
        else:
            return current_score > best_until_now

    def update(self, records, dataset_name):
        """Evaluate one dataset and append its latest score.

        Args:
            records: Prediction records, or a Recorder when metric_name is loss.
                Prediction record requirements are documented by each metric function.
            dataset_name: Configured dataset whose history receives the score.
        """
        if self.metric_name == 'loss':
            current_score = records.average()
            if current_score is None:
                current_score = float('inf')
        else:
            # Performance Score
            if self.metric_name == 'sample-level EER':
                current_score = calculate_binary_eer(records, type='sample-level')
            elif self.metric_name == 'video-level EER':
                current_score = calculate_binary_eer(records, type='video-level')
            elif self.metric_name == 'sample-level AUC':
                current_score = calculate_binary_auc(records, type='sample-level')
            elif self.metric_name == 'video-level AUC':
                current_score = calculate_binary_auc(records, type='video-level')
            elif self.metric_name == 'sample-level AP':
                current_score = calculate_binary_ap(records, type='sample-level')
            elif self.metric_name == 'video-level AP':
                current_score = calculate_binary_ap(records, type='video-level')
            elif self.metric_name == 'sample-level accuracy':
                current_score = calculate_binary_accuracy(records, type='sample-level')
            elif self.metric_name == 'video-level accuracy':
                current_score = calculate_binary_accuracy(records, type='video-level')

            # Fairness Score
            elif self.metric_name == 'sample-level DP':
                current_score = calculate_binary_dp(records, type='sample-level')
            elif self.metric_name == 'video-level DP':
                current_score = calculate_binary_dp(records, type='video-level')
            elif self.metric_name == 'sample-level EO':
                current_score = calculate_binary_eo(records, type='sample-level')
            elif self.metric_name == 'video-level EO':
                current_score = calculate_binary_eo(records, type='video-level')
            elif self.metric_name == 'sample-level OAE':
                current_score = calculate_binary_oae(records, type='sample-level')
            elif self.metric_name == 'video-level OAE':
                current_score = calculate_binary_oae(records, type='video-level')
            elif self.metric_name == 'sample-level MEO':
                current_score = calculate_binary_meo(records, type='sample-level')
            elif self.metric_name == 'video-level MEO':
                current_score = calculate_binary_meo(records, type='video-level')
            elif self.metric_name == 'video-level SDAUC':
                current_score = calculate_binary_sdauc(records)
            else:
                raise NotImplementedError(f'{self.metric_name} has not been implemented.')

        self.metric_records[dataset_name].append(current_score)

    def update_best(self):
        """Update the dataset macro-average and best scores for the result round.

        Returns:
            Names of datasets, including average, whose metric strictly improved.
        """
        self._update_average()
        update_list = []
        for dataset_name in self.dataset_list:
            if self._is_best(self.best_scores[dataset_name], self.metric_records[dataset_name][-1]):
                update_list.append(dataset_name)
                self.best_scores[dataset_name] = self.metric_records[dataset_name][-1]

        return update_list

    def parse_metrics(self, include_latest=False):
        """Format best scores and optionally the latest completed scores.

        Args:
            include_latest: Include a column for each dataset's most recent score.

        Returns:
            PrettyTable containing the requested scores.
        """
        table = PrettyTable()
        table.field_names = ['Dataset', 'Latest', 'Best'] if include_latest else ['Dataset', 'Best']
        for dataset_name in self.dataset_list:
            if include_latest:
                table.add_row([dataset_name,
                               f'{self.metric_records[dataset_name][-1]:.4f}',
                               f'{self.best_scores[dataset_name]:.4f}'])
            else:
                table.add_row([dataset_name,
                               f'{self.best_scores[dataset_name]:.4f}'])

        return table
