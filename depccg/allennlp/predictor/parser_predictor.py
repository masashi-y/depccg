from typing import Dict, Any, List

from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN

import depccg.parsing
from depccg.all.predictor.supertagger_predictor import SupertaggerPredictor


@Predictor.register('parser-predictor')
class ParserPredictor(SupertaggerPredictor):
    def __init__(
        self, model: Model,
        dataset_reader: DatasetReader,
    ) -> None:
        super().__init__(model, dataset_reader)

    def _make_json(self, output_dicts: List[Dict[str, Any]]) -> List[JsonDict]:
        output_dicts = super()._make_json(output_dicts)

        results = depccg.parsing.run(
            doc,
            score_result,
            categories,
            root_categories,
            apply_binary_rules,
            apply_unary_rules,
            **kwargs,
        )
        return output_dicts
