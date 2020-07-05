
from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.common.checks import ConfigurationError

import pandas as pd

logger = logging.getLogger(__name__)


@DatasetReader.register("mimics")
class MIMICSDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer('bert-base-uncased')
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            
            _options_columns = [f'option_{i}' for i in range(1, 6)] # option_1, ..., option_5
            _label_columns = [f'option_label_{i}' for i in range(1, 6)] # option_label_1, ..., option_label_5

            columns = ['query','question', *_options_columns, *_label_columns]
            df = pd.read_csv(data_file, sep='\t', usecols=columns)

            df['options'] = df[_options_columns].fillna('').values.tolist()
            df['labels'] = df[_label_columns].values.tolist()

            df = df.drop(columns=[*_options_columns, *_label_columns])

            for row in df.to_dict(orient='records'):
                yield self.text_to_instance(**row)

    def _make_textfield(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers=self.token_indexers)

    @overrides
    def text_to_instance(
        self,
        query: str, 
        question: str,
        options: List[str],
        labels: List[str] = None
    ) -> Instance:  # type: ignore
        # query_field = self._make_textfield(query)
        # question_field = self._make_textfield(question)
        token_field = self._make_textfield((query, question))

        options_field = ListField([self._make_textfield(o) for o in options if o != ''])
        # fields = { 'query': query_field, 'question': question_field, 'options': options_field }
        fields = { 'tokens': token_field, 'options': options_field }

        if labels:
            # 0 = no click, [1,2] = click
            # int(l > 0)
            labels = map(int, filter(lambda x: not pd.isnull(x), labels))
            label_list = [LabelField(int(l), skip_indexing=True) for l in labels]
            fields['labels'] = ListField(label_list)
        
        return Instance(fields)