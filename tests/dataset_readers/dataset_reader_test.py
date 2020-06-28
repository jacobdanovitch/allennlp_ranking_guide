# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from allenrank.dataset_readers import MIMICSDatasetReader

class TestMIMICSDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = MIMICSDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/mimics.tsv'))

        assert len(instances) == 100
        print(instances[0].fields)