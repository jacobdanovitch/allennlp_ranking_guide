import torch
from torch import nn

from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors

class RelevanceMatcher(Registrable, nn.Module):
    # https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py
    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        candidates_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        candidates_mask: torch.Tensor = None
    ):
        raise NotImplementedError()