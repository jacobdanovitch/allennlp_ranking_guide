# source: https://github.com/allegro/allRank/blob/master/allrank/models/metrics.py
# reference: https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/auc.py

from typing import Optional, List

import numpy as np
import torch

from allennlp.training.metrics.metric import Metric

from allenrank.training.metrics.ranking_metric import (
    RankingMetric, 
    __apply_mask_and_get_true_sorted_by_preds, 
    pad_to_max_length
)

import torchsnooper


@Metric.register("mrr")
class MRR(RankingMetric):
    """
    Computes NDCG.
    """

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        predictions = pad_to_max_length(self._all_predictions, padding_value=self._padding_value)
        labels = pad_to_max_length(self._all_gold_labels, padding_value=self._padding_value)
        
        score = mrr(predictions, labels, padding_indicator=self._padding_value).mean().item()

        if reset:
            self.reset()
        return score

def mrr(y_pred, y_true, ats=None, padding_indicator=-1):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result