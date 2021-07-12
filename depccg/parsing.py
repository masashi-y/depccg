from typing import Callable, List, Dict, Union, Tuple
from multiprocessing import Pool
import numpy
import depccg._parsing
import time
import math
from depccg.types import Token, CombinatorResult, ScoringResult
from depccg.tree import ScoredTree
from depccg.cat import Category


def _chunks(list_, num_chunks):
    splits = math.ceil(len(list_) / max(num_chunks, 1))
    for i in range(0, len(list_), splits):
        yield list_[i:i + splits]


def _binarize(indices, length):
    result = numpy.ones(length, dtype=numpy.bool)
    result[indices] = 0
    return result


def _type_check(doc, score_results, categories):
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

    num_tags = len(categories)
    for tokens, (tag_scores, dep_scores) in zip(doc, score_results):
        if num_tags != tag_scores.shape[1]:
            raise RuntimeError(
                ("all inputs to depccg.parsing.run must contain scores for"
                 " the equal number of categories as the `categories` list.")
            )
        num_tokens = len(tokens)
        expected_tag_score = (num_tokens, num_tags)
        expected_dep_score = (num_tokens, num_tokens + 1)
        if (
            expected_tag_score != tag_scores.shape
            or expected_dep_score != dep_scores.shape
        ):
            raise RuntimeError(
                ('invalid shape of input matrices:\n'
                 f'Expected P_tag: {expected_tag_score}, P_dep: {expected_dep_score}\n'
                 f'Actual P_tag: {tag_scores.shape}, P_dep: {dep_scores.shape}')
            )

    return doc, score_results


def apply_category_filters(
    doc: Union[Token, List[List[Token]]],
    score_results: Union[ScoringResult, List[ScoringResult]],
    categories: List[Category],
    category_dict: Dict[str, List[Category]],
    large_negative_value: float = -10e+32,
) -> Union[Tuple[List[Token], ScoringResult], Tuple[List[List[Token]], List[ScoringResult]]]:

    doc, score_results = _type_check(doc, score_results, categories)

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

    for tokens, (tag_scores, _) in zip(doc, score_results):
        for index, token in enumerate(tokens):
            if token.word in category_dict:
                tag_scores[index, category_dict[token.word]
                           ] = large_negative_value

    return doc, score_results


def run(
    doc: Union[Token, List[List[Token]]],
    score_results: Union[ScoringResult, List[ScoringResult]],
    categories: List[Category],
    root_categories: List[Category],
    binary_fun: Callable[[Category, Category], List[CombinatorResult]],
    unary_fun: Callable[[Category], List[CombinatorResult]],
    unary_penalty: float = 0.1,
    beta: float = 0.00001,
    use_beta: bool = True,
    pruning_size: int = 50,
    nbest: int = 1,
    max_step: int = 10000000,
    max_length: int = 250,
    processes: int = 2,
    max_chunk_size: int = 20,
) -> List[List[ScoredTree]]:

    doc, score_results = _type_check(doc, score_results, categories)

    args = (
        categories,
        binary_fun,
        unary_fun,
        root_categories
    )

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

    if len(doc) <= max_chunk_size:

        results = depccg._parsing.run(
            doc,
            score_results,
            *args,
            **kwargs,
        )

    else:
        chunks = _chunks(list(zip(doc, score_results)), processes)
        with Pool(processes) as pool:
            tasks = []
            for chunk_index, chunk in enumerate(chunks):
                doc_, score_results_ = zip(*chunk)
                task = pool.apply_async(
                    depccg._parsing.run,
                    args=(list(doc_), list(score_results_)) + args,
                    kwds={**kwargs, 'process_id': chunk_index},
                )
                tasks.append(task)

            while not all(task.ready() for task in tasks):
                time.sleep(1)

            results = [
                result
                for task in tasks
                for result in task.get()
            ]

    return results
