from typing import List
import torch

from typing import Optional, List

import numpy as np
import torch

from allennlp.training.metrics.metric import Metric

class RankingMetric(Metric):
    def __init__(
        self,
        padding_value: int = -1
    ) -> None:
        self._padding_value = padding_value
        self.reset()
        
    def __call__(
            self,
            predictions: torch.LongTensor,
            gold_labels: torch.LongTensor,
            mask: torch.LongTensor = None
        ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of real-valued predictions of shape (batch_size, slate_length).
        gold_labels : ``torch.Tensor``, required.
            A tensor of real-valued labels of shape (batch_size, slate_length).
        """
        
        if mask is None:
            mask = torch.ones(gold_labels.size(0), device=gold_labels.device).bool()
        
        #self._all_predictions = self._all_predictions.to(predictions.device)
        #self._all_gold_labels = self._all_gold_labels.to(gold_labels.device)
        
        # self._all_predictions = torch.cat(
        #     [self._all_predictions, torch.masked_select(predictions, mask).type_as(self._all_predictions)], dim=0
        # )
        self._all_predictions.append(torch.masked_select(predictions, mask))
        # self._all_gold_labels = torch.cat(
        #     [self._all_gold_labels, torch.masked_select(gold_labels, mask).type_as(self._all_gold_labels)], dim=0
        # )
        self._all_gold_labels.append(torch.masked_select(gold_labels, mask))
        
    def get_metric(self, reset: bool = False):
        raise NotImplementedError()
    
    def reset(self):
        self._all_predictions = []
        self._all_gold_labels = []

def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=-1):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)

def pad_to_max_length(seq: List[torch.Tensor], padding_value: int = -1):
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)