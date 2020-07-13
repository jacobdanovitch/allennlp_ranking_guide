from typing import Dict, Iterator, List

import torch
import torch.nn as nn

from allennlp.modules.matrix_attention import CosineMatrixAttention
from allennlp.data import TextFieldTensors

from allenrank.modules.relevance.base import RelevanceMatcher
from allenrank.modules.kernels import KernelFunction

import torchsnooper

@RelevanceMatcher.register('knrm')
class KNRM(RelevanceMatcher):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17

    Adapted from: https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/knrm.py
    '''
    def __init__(
        self,
        kernel_function: KernelFunction,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._kernel = kernel_function

    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        candidates_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        candidates_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        kernel_output = self._kernel(
            query_embeddings,
            candidates_embeddings,
            query_mask,
            candidates_mask
        )
        
        score = self.dense(kernel_output)
        score = torch.squeeze(score,1)
        return score