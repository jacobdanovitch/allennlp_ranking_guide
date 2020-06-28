from typing import Dict, Iterator, List

import torch
import torch.nn as nn

from allennlp.modules import Seq2VecEncoder
from allennlp.data import TextFieldTensors

from allenrank.modules.relevance.base import RelevanceMatcher

@RelevanceMatcher.register('bert_cls')
class BertCLS(RelevanceMatcher):
    def __init__(
        self,
        seq2vec_encoder: Seq2VecEncoder # should probably be cls_pooler
    ):

        super().__init__()

        self._seq2vec_encoder = seq2vec_encoder

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(self._seq2vec_encoder.get_output_dim()*4, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        candidates_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        candidates_mask: torch.Tensor = None,
        output_secondary_output: bool = False
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_encoded = self._seq2vec_encoder(query_embeddings, query_mask)
        candidate_encoded = self._seq2vec_encoder(candidates_embeddings, candidates_mask)

        interaction_vector = torch.cat([query_encoded, candidate_encoded, query_encoded-candidate_encoded, query_encoded*candidate_encoded], dim=1)
        dense_out = self.dense(interaction_vector)
        score = torch.squeeze(dense_out,1)

        return score