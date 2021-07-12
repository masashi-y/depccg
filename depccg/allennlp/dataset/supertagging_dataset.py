from typing import Dict, List, Optional, Union
from overrides import overrides
import json
import logging
import numpy
import random

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from depccg import utils
from depccg.tools.data import convert_auto_to_json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def read_dataset_auto_or_json(file_path: str):
    if utils.is_json(file_path):
        logger.info(
            f'Reading instances from lines in json file at: {file_path}')
        with open(file_path, 'r') as data_file:
            json_data = json.load(data_file)
    else:
        logger.info(f'Reading trees in auto file at: {file_path}')
        json_data = convert_auto_to_json(file_path)
    logger.info(f'loaded {len(json_data)} instances')
    return json_data


@DatasetReader.register("supertagging_dataset")
class SupertaggingDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        token_indexers: Dict[str, TokenIndexer] = None
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        json_data = read_dataset_auto_or_json(cached_path(file_path))
        for instance in json_data:
            sentence, labels = instance
            tags, deps = labels
            yield self.text_to_instance(sentence, tags, deps)

    @overrides
    def text_to_instance(
        self,
        sentence: str,
        tags: List[str] = None,
        deps: List[int] = None,
        weight: float = 1.0
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokens = [
            Token(utils.normalize(token))
            for token in sentence.split(' ')
        ]
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


@DatasetReader.register("tritrain_supertagging_dataset")
class TritrainSupertaggingDatasetReader(SupertaggingDatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        noisy_weight: Optional[float] = 0.4,
        token_indexers: Dict[str, TokenIndexer] = None
    ) -> None:
        super().__init__(lazy)
        self.noisy_weight = noisy_weight
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_paths):
        """
        read ccgbank and tritrain datasets (both in json file).
        :param file_paths: a pair of file paths
        :return:
        """
        ccgbank, tritrain = file_paths['ccgbank'], file_paths['tritrain']

        logger.info(f'Reading instances from CCGBank at: {ccgbank}')
        ccgbank = read_dataset_auto_or_json(cached_path(ccgbank))

        logger.info(f'Reading instances from tri-train dataset at: {tritrain}')
        tritrain = read_dataset_auto_or_json(cached_path(tritrain))

        ccgbank_size = len(ccgbank)
        tritrain_size = len(tritrain)
        max_len = ccgbank_size * 15 + tritrain_size
        # make the size of the whole training data equal to that of ccgbank
        # please use this with iterator.cache_instances False!
        for _ in range(ccgbank_size):
            index = random.randint(0, max_len - 1)
            if index < tritrain_size:
                sentence, labels = tritrain[index]
                weight = self.noisy_weight
            else:
                index = (index - tritrain_size) % ccgbank_size
                sentence, labels = ccgbank[index]
                weight = 1.
            tags, deps = labels

            if any(len(word) == 0 for word in sentence.split(' ')):
                logging.info(f'skipping example: {sentence}')
                continue
            new_instance = self.text_to_instance(sentence, tags, deps, weight)
            yield new_instance


def dataset_times(iter, n: Union[int, float]):
    listed = list(iter)
    if isinstance(n, int):
        for _ in range(n):
            random.shuffle(listed)
            for elem in listed:
                yield elem
    else:
        assert isinstance(n, float)
        for _ in range(int(n * len(listed))):
            i = random.randint(0, len(listed) - 1)
            yield listed[i]


@DatasetReader.register("finetune_supertagging_dataset")
class FinetuneSupertaggingDatasetReader(SupertaggingDatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tritrain_noisy_weight: Optional[float] = 1,
        auxiliary_noisy_weight: Optional[float] = 1,
        ccgbank_ratio: Union[int, float] = 1,
        tritrain_ratio: Union[int, float] = 1,
        auxiliary_ratio: Union[int, float] = 1,
        token_indexers: Dict[str, TokenIndexer] = None
    ) -> None:
        super().__init__(lazy)
        self.tritrain_noisy_weight = tritrain_noisy_weight
        self.auxiliary_noisy_weight = auxiliary_noisy_weight
        self.ccgbank_ratio = ccgbank_ratio
        self.tritrain_ratio = tritrain_ratio
        self.auxiliary_ratio = auxiliary_ratio
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_paths):
        ccgbank, aux_data = file_paths['ccgbank'], file_paths['auxiliary']
        logger.info(f'Reading instances from CCGBank at: {ccgbank}')
        ccgbank = read_dataset_auto_or_json(cached_path(ccgbank))
        logger.info(f'Reading instances from auxiliary dataset at: {aux_data}')
        aux_data = read_dataset_auto_or_json(cached_path(aux_data))

        tritrain = file_paths.get('tritrain', None)
        if tritrain is not None:
            logger.info(
                f'Reading instances from tri-training dataset at: {tritrain}')
            tritrain = read_dataset_auto_or_json(cached_path(tritrain))
        else:
            tritrain = []

        ccgbank = list(dataset_times(ccgbank, self.ccgbank_ratio))
        logger.info(
            f'ccgbank ratio = {self.ccgbank_ratio}: use {len(ccgbank)} sentences from ccgbank'
        )
        tritrain = list(dataset_times(tritrain, self.tritrain_ratio))
        logger.info(
            f'tritrain ratio = {self.tritrain_ratio}: use {len(tritrain)} sentences from ccgbank'
        )
        aux_data = list(dataset_times(aux_data, self.auxiliary_ratio))
        logger.info(
            f'auxiliary ratio = {self.auxiliary_ratio}: use {len(aux_data)} sentences from auxiliary data'
        )
        logger.info(f'noisy weight = {self.tritrain_noisy_weight}')
        logger.info(f'noisy weight = {self.auxiliary_noisy_weight}')

        dataset = (
            [(x, 1.) for x in ccgbank]
            + [(x, self.tritrain_noisy_weight) for x in tritrain]
            + [(x, self.auxiliary_noisy_weight) for x in aux_data]
        )

        dataset_size = len(dataset)
        for _ in range(dataset_size):
            index = random.randint(0, dataset_size - 1)
            (sentence, labels), weight = dataset[index]
            tags, deps = labels
            yield self.text_to_instance(sentence, tags, deps, weight)
