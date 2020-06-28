
from typing import Dict, List
import logging

from overrides import overrides
from nltk.tree import Tree

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


@DatasetReader.register("MSMARCODatasetReader")
class MSMARCODatasetReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            raise NotImplementedError()

    @overrides
    def text_to_instance(
        self, tokens: List[str], sentiment: str = None
    ) -> Instance:  # type: ignore
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        raise NotImplementedError()