from typing import Dict, Any, List

from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN


@Predictor.register('supertagger-predictor')
class SupertaggerPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["sentence"])

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        [result] = self._make_json([outputs])
        return result

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return self._make_json(outputs)

    def _make_json(self, output_dicts: List[Dict[str, Any]]) -> List[JsonDict]:
        all_categories = self._model.vocab.get_index_to_token_vocabulary(
            'head_tags')
        all_categories = [token for _, token in sorted(all_categories.items())]
        categories, paddings = all_categories[2:], all_categories[:2]
        assert all(padding in [DEFAULT_PADDING_TOKEN,
                               DEFAULT_OOV_TOKEN] for padding in paddings)
        for output_dict in output_dicts:
            length = len(output_dict['words'].split(' '))
            output_dict['categories'] = categories
            head_tags = output_dict["head_tags"][:length, :]
            assert head_tags.shape[-1] == len(
                categories), f"{head_tags.shape[-1]}, {len(all_categories)}"
            heads = output_dict["heads"][:length, :length + 1]
            output_dict["predicted_head_tags"] = [
                all_categories[i] for i in output_dict["predicted_head_tags"]
            ]
            output_dict["predicted_heads"] = output_dict["predicted_heads"].tolist()
            output_dict["head_tags"] = head_tags.flatten().astype(
                float).tolist()
            output_dict["heads"] = heads.flatten().astype(float).tolist()
            output_dict["head_tags_shape"] = list(head_tags.shape)
            output_dict["heads_shape"] = list(heads.shape)
            output_dict.pop("mask")
        return output_dicts
