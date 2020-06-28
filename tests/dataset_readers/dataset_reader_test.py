# pylint: disable=no-self-use,invalid-name

import warnings; warnings.simplefilter('ignore')

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from allenrank.dataset_readers import MIMICSDatasetReader

class TestMIMICSDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = MIMICSDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/mimics.tsv'))

        # assert len(instances) == 100
        print(instances[0].fields)

        for inst in instances:
            for label_field in inst.fields['labels'].field_list:
                assert label_field.label in [0, 1], f"Binary labels expected but found: {label_field.label}"
            break