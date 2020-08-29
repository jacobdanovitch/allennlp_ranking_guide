from typing import Dict, Iterator, List

import torch
import torch.nn as nn

from allennlp.modules import Seq2VecEncoder
from allennlp.data import TextFieldTensors
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler

from allenrank.modules.relevance.base import RelevanceMatcher

@RelevanceMatcher.register('bert_cls')
class BertCLS(RelevanceMatcher):
    def __init__(
        self,
        input_dim: int,
        **kwargs
    ):
        super().__init__(input_dim=input_dim*4, **kwargs)

        # Gets the [CLS] token from BERT
        self._seq2vec_encoder = ClsPooler(embedding_dim=input_dim)

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

        interaction_vector = torch.cat([query_encoded, candidate_encoded, torch.abs(query_encoded-candidate_encoded), query_encoded*candidate_encoded], dim=1)
        dense_out = self.dense(interaction_vector)
        score = torch.squeeze(dense_out,1)

        return score