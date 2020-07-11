from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, Auc, F1Measure, FBetaMeasure, PearsonCorrelation

from allenrank.modules.relevance.base import RelevanceMatcher
from allenrank.training.metrics.multilabel_f1 import MultiLabelF1Measure
from allenrank.training.metrics import NDCG, MRR

import torchsnooper


@Model.register("ranker")
class DocumentRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        relevance_matcher: RelevanceMatcher,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._relevance_matcher = TimeDistributed(relevance_matcher)

        self._dropout = dropout and torch.nn.Dropout(dropout)

        self._label_namespace = label_namespace
        self._namespace = namespace

        self._auc = Auc()
        self._mrr = MRR(padding_value=-1)
        self._ndcg = NDCG(padding_value=-1)
        
        self._loss = torch.nn.MSELoss(reduction='none') # BCEWithLogitsLoss MSELoss # CrossEntropyLoss BCELoss
        initializer(self)

    def forward(  # type: ignore
        self, 
        tokens: TextFieldTensors, # batch * words
        options: TextFieldTensors, # batch * num_options * words
        labels: torch.IntTensor = None # batch * num_options
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).long()

        embedded_options = self._text_field_embedder(options, num_wrapping_dims=1) # options_mask.dim() - 2
        options_mask = get_text_field_mask(options).long()

        output_shape = options_mask.size()[:2] # [batch, num_options] (can't use labels here in case it doesn't exist)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)
            embedded_options = self._dropout(embedded_options)

        """
        This isn't exactly a 'hack', but it's definitely not the most efficient way to do it.
        Our matcher expects a single (query, document) pair, but we have (query, [d_0, ..., d_n]).
        To get around this, we expand the query embeddings to create these pairs, and then
        flatten both into the 3D tensor [batch*num_options, words, dim] expected by the matcher. 
        The expansion does this:

        [
            (q_0, [d_{0,0}, ..., d_{0,n}]), 
            (q_1, [d_{1,0}, ..., d_{1,n}])
        ]
        =>
        [
            [ (q_0, d_{0,0}), ..., (q_0, d_{0,n}) ],
            [ (q_1, d_{1,0}), ..., (q_1, d_{1,n}) ]
        ]

        Which we then flatten along the batch dimension. It would likely be more efficient^* 
        to rewrite the matrix multiplications in the relevance matchers, but this is a more general solution.
        """

        embedded_text = embedded_text.unsqueeze(1).expand(-1, embedded_options.size(1), -1, -1) # [batch, num_options, words, dim]
        mask = mask.unsqueeze(1).expand(-1, embedded_options.size(1), -1)
        
        scores = self._relevance_matcher(embedded_text, embedded_options, mask, options_mask).squeeze() # [batch, ...num_labels]
        probs = torch.sigmoid(scores).view(-1)

        output_dict = {"logits": scores, "probs": probs} # .view(*output_shape)
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if labels is not None:
            _labels = labels
            labels = labels.view(-1)
            candidate_mask = (labels != -1)
            
            loss = self._loss(probs, labels.type_as(scores))
            output_dict["loss"] = loss.masked_fill(~candidate_mask, 0).sum() / candidate_mask.sum()
            
            binary_labels = (labels > 0.5).long()
            
            self._auc(probs, binary_labels, candidate_mask)
            self._mrr(probs.view_as(scores), _labels, candidate_mask.view_as(scores))
            self._ndcg(probs.view_as(scores), _labels, candidate_mask.view_as(scores))

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "auc": self._auc.get_metric(reset),
            "mrr": self._mrr.get_metric(reset),
            "ndcg": self._ndcg.get_metric(reset),
        }
        return metrics

    default_predictor = "document_ranker"


# one_cold? many_hot?
def inverse_one_hot(t: torch.Tensor):
    t = t.view(-1, 1)
    out = torch.cat([t, 1-t], -1)
    assert (out.sum(-1) == 1).all()
    return out

