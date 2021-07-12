from typing import Dict, Any, List, Optional

import numpy
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

import depccg.parsing
from depccg.types import ScoringResult, Token
from depccg.allennlp.predictor.supertagger_predictor import SupertaggerPredictor
from depccg.allennlp.utils import read_params
from depccg.cat import Category
from depccg.printer.my_json import json_of


@Predictor.register('parser-predictor')
class ParserPredictor(SupertaggerPredictor):

    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        grammar_json_path: str,
        parsing_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model, dataset_reader)
        (
            self.apply_binary_rules,
            self.apply_unary_rules,
            self.category_dict,
            self.root_categories,
        ) = read_params(grammar_json_path)
        self.parsing_kwargs = parsing_kwargs or {}

    def _make_json(self, output_dicts: List[Dict[str, Any]]) -> List[JsonDict]:

        categories = None
        score_results = []
        doc = []
        for output_dict in super()._make_json(output_dicts):
            if categories is None:
                categories = [
                    Category.parse(category)
                    for category in output_dict['categories']
                ]

            tokens = [
                Token.of_word(word)
                for word in output_dict['words'].split(' ')
            ]
            doc.append(tokens)

            dep_scores = numpy.array(output_dict['heads']) \
                .reshape(output_dict['heads_shape']) \
                .astype(numpy.float32)
            tag_scores = numpy.array(output_dict['head_tags']) \
                .reshape(output_dict['head_tags_shape']) \
                .astype(numpy.float32)
            score_results.append(ScoringResult(tag_scores, dep_scores))

        if self.category_dict is not None:
            doc, score_results = depccg.parsing.apply_category_filters(
                doc,
                score_results,
                categories,
                self.category_dict,
            )

        results = depccg.parsing.run(
            doc,
            score_results,
            categories,
            self.root_categories,
            self.apply_binary_rules,
            self.apply_unary_rules,
            **self.parsing_kwargs,
        )

        for output_dict, trees in zip(output_dicts, results):
            output_dict['trees'] = []
            for tree, log_prob in trees:
                tree_dict = json_of(tree)
                tree_dict['log_prob'] = log_prob
                output_dict['trees'].append(tree_dict)

        return output_dicts
