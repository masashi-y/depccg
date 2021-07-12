from typing import Tuple, List
import numpy
import logging
from itertools import islice

from depccg.allennlp.predictor.supertagger_predictor import SupertaggerPredictor
from depccg.allennlp.dataset.ja_supertagging_dataset import JaSupertaggingDatasetReader
from depccg.allennlp.dataset.supertagging_dataset import TritrainSupertaggingDatasetReader
from depccg.allennlp.dataset.supertagging_dataset import SupertaggingDatasetReader
from depccg.allennlp.models.supertagger import Supertagger
from allennlp.models.archival import load_archive

from depccg.types import ScoringResult

logger = logging.getLogger(__name__)


def lazy_groups_of(iterator, group_size):
    return iter(lambda: list(islice(iterator, 0, group_size)), [])


class AllennlpSupertagger(object):
    def __init__(self, predictor):
        self.predictor = predictor
        self.dataset_reader = predictor._dataset_reader

    def predict_doc(
        self,
        splitted,
        batchsize=32,
    ) -> Tuple[List[ScoringResult], List[str]]:

        instances = (
            self.dataset_reader.text_to_instance(' '.join(sentence))
            for sentence in splitted
        )

        categories = None
        scores = []
        for batch in lazy_groups_of(instances, batchsize):
            for json_dict in self.predictor.predict_batch_instance(batch):
                if categories is None:
                    categories = list(json_dict['categories'])
                dep_scores = numpy.array(json_dict['heads']) \
                    .reshape(json_dict['heads_shape']) \
                    .astype(numpy.float32)
                tag_scores = numpy.array(json_dict['head_tags']) \
                    .reshape(json_dict['head_tags_shape']) \
                    .astype(numpy.float32)

                scores.append(ScoringResult(tag_scores, dep_scores))
        return scores, categories


def load_allennlp_tagger(
    model_path: str,
    device: int = -1
) -> AllennlpSupertagger:

    if device >= 0:
        logger.info(f'sending the supertagger to gpu: {device}')
    archive = load_archive(model_path, cuda_device=device)
    predictor = SupertaggerPredictor.from_archive(
        archive, 'supertagger-predictor'
    )
    return AllennlpSupertagger(predictor)
