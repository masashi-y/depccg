from typing import Callable, List, Dict, Optional, Union
# from multiprocessing import Pool
import numpy
import depccg._parsing
from depccg.types import Token, CombinatorResult, ScoringResult
from depccg.tree import ScoredTree
from depccg.cat import Category


def _binarize(indices, length):
    result = numpy.ones(length, dtype=numpy.float32) * 10e-16
    result[indices] = 1
    return numpy.log(result)


def _type_check(doc, score_results):
    many_sentences = (
        isinstance(doc, list)
        and isinstance(doc[0], list)
        and isinstance(doc[0][0], Token)
    )
    many_scores = (
        isinstance(score_results, list)
        and isinstance(score_results[0], ScoringResult)
    )

    if (
        many_sentences != many_scores
        or many_sentences and len(doc) != len(score_results)
    ):
        raise RuntimeError(
            'doc and score_results are not compatible types.')

    if not many_sentences:
        score_results = [score_results]
        doc = [doc]

    return doc, score_results


def run(
    doc: Union[Token, List[List[Token]]],
    score_results: Union[ScoringResult, List[ScoringResult]],
    categories: List[Category],
    root_categories: List[Category],
    binary_fun: Callable[[Category, Category], List[CombinatorResult]],
    unary_fun: Callable[[Category], List[CombinatorResult]],
    category_dict: Optional[Dict[str, List[Category]]] = None,
    unary_penalty: float = 0.1,
    beta: float = 0.00001,
    use_beta: bool = True,
    pruning_size: int = 50,
    nbest: int = 1,
    max_step: int = 10000000,
    max_length: int = 250,
    # processes: int = 2,
) -> List[List[ScoredTree]]:

    doc, score_results = _type_check(doc, score_results)

    category_ids = {
        cat: index for index, cat in enumerate(categories)
    }

    category_dict = {
        word: _binarize(
            [category_ids[cat] for cat in cats],
            score_results[0].tag_scores.shape[1]
        )
        for word, cats in category_dict.items()
    }

    kwargs = {
        'num_tags': score_results[0][0].shape[1],
        'unary_penalty': unary_penalty,
        'beta': beta,
        'use_beta': use_beta,
        'pruning_size': pruning_size,
        'nbest': nbest,
        'max_step': max_step,
        'max_length': max_length
    }

    for tokens, (tag_scores, _) in zip(doc, score_results):
        for index, token in enumerate(tokens):
            if token.word in category_dict:
                tag_scores[index] += category_dict[token.word]

    results = depccg._parsing.run(
        doc,
        score_results,
        categories,
        binary_fun,
        unary_fun,
        root_categories,
        **kwargs,
    )

    return results
