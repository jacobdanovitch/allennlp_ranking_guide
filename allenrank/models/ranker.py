from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Auc, F1Measure

from allenrank.modules.relevance.base import RelevanceMatcher

import torchsnooper


@Model.register("ranker")
class DocumentRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        relevance_matcher: RelevanceMatcher,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._relevance_matcher = relevance_matcher

        self._dropout = dropout and torch.nn.Dropout(dropout)

        self._label_namespace = label_namespace
        self._namespace = namespace

        self._auc = Auc() # CategoricalAccuracy()
        self._f1 = F1Measure(positive_label=1)
        self._loss = torch.nn.BCEWithLogitsLoss()
        initializer(self)

    @torchsnooper.snoop(watch='labels')
    def forward(  # type: ignore
        self, 
        tokens: TextFieldTensors, # batch * words
        options: TextFieldTensors, # batch * num_options * words
        labels: torch.IntTensor = None # batch * num_options
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).long()

        embedded_options = self._text_field_embedder(options, num_wrapping_dims=1) # mask.dim() - 2
        options_mask = get_text_field_mask(options).long()

        output_shape = options_mask.size()[:2] # [batch, num_options] (can't use labels here in case it doesn't exist)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)
            embedded_options = self._dropout(embedded_options)

        """
        This isn't exactly a 'hack', but it's definitely not the most efficient way to do it.
        Our matcher expects a single query/document pair, but we have query/[d_0, ..., d_n].
        To get around this, we expand the query embeddings to create these pairs, and then
        flatten both into the 3D tensor [batch*num_options, words, dim] as expected by the matcher.

        It would likely be more efficient^* to rewrite the matrix multiplications in the relevant matchers,
        but this is a more general solution.

        ^* @Matt Gardner: Is this actually inefficient? `torch.expand` doesn't use additional memory to make copies.
        """

        embedded_text = embedded_text.unsqueeze(1).expand(-1, embedded_options.size(1), -1, -1).flatten(0, 1) # [batch, num_options, words, dim] =>  [batch*num_options, ...]
        mask = mask.unsqueeze(1).expand(-1, embedded_options.size(1), -1).flatten(0, 1)

        embedded_options = embedded_options.flatten(0, 1)
        options_mask = options_mask.flatten(0, 1)
        
        scores = self._relevance_matcher(embedded_text, embedded_options, mask, options_mask)
        probs = torch.sigmoid(scores)

        output_dict = {"logits": scores.view(*output_shape), "probs": probs.view(*output_shape)}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if labels is not None:
            labels = labels.view(-1)
            loss = self._loss(scores, labels.type_as(scores)) # .long()
            output_dict["loss"] = loss

            # @Matt Gardner: What's the best way to compute metrics here, since it's a ListField (thus some are masked)?
            # Is there something for POS tagging or similar?

            # self._auc(scores, labels)
            # self._f1(scores.unsqueeze(-1), labels)

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
        metrics = {} # "auc": self._auc.get_metric(reset)
        return metrics

    default_predictor = "document_ranker"