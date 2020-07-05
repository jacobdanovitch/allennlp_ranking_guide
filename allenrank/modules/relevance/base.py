import torch
from torch import nn

from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors

class RelevanceMatcher(Registrable, nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1
    ):
        super().__init__()
        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(input_dim, num_classes, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    # https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py
    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        candidates_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        candidates_mask: torch.Tensor = None
    ):
        raise NotImplementedError()