from typing import Dict, List, Optional, Union
from overrides import overrides
import json
import logging
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from depccg import utils
from depccg.tools.ja.data import convert_ccgbank_to_json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def read_dataset_ccgbank_or_json(file_path: str):
    if utils.is_json(file_path):
        logger.info(f'Reading instances from lines in json file at: {file_path}')
        with open(file_path, 'r') as data_file:
            json_data = json.load(data_file)
    else:
        logger.info(f'Reading trees in auto file at: {file_path}')
        json_data = convert_ccgbank_to_json(file_path)
    logger.info(f'loaded {len(json_data)} instances')
    return json_data


@DatasetReader.register("ja_supertagging_dataset")
class JaSupertaggingDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        json_data = read_dataset_ccgbank_or_json(cached_path(file_path))
        for instance in json_data:
            sentence, labels = instance
            tags, deps = labels
            yield self.text_to_instance(sentence, tags, deps)

    @overrides
    def text_to_instance(self,
                         sentence: str,
                         tags: List[str] = None,
                         deps: List[int] = None,
                         weight: float = 1.0) -> Instance:
        tokens = [Token(token) for token in sentence.split(' ')]
        token_field = TextField(tokens, self._token_indexers)
        metadata = MetadataField({'words': sentence})
        weight = ArrayField(numpy.array([weight], 'f'))
        fields = {
            'words': token_field,
            'metadata': metadata,
            'weight': weight,
        }
        if tags is not None and deps is not None:
            fields['head_tags'] = SequenceLabelField(
                tags, token_field, label_namespace='head_tags')
            fields['head_indices'] = SequenceLabelField(
                deps, token_field, label_namespace='head_indices')
        return Instance(fields)
