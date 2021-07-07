from typing import List, TypeVar, Tuple, Set, Dict
from depccg.types import Combinator, CombinatorResult
from depccg.py_cat import Category
from depccg.py_tree import Tree

X = TypeVar('X')
Pair = Tuple[X, X]


def apply_rules(
    left: Category,
    right: Category,
    seen_rules: Set[Pair[Category]],
    combinators: List[Combinator],
    cache: Dict[Pair[Category], List[CombinatorResult]],
) -> List[CombinatorResult]:
    cats = (left, right)
    if cats in cache:
        return cache[cats]

    results = []
    if cats in seen_rules:
        for combinator in combinators:
            result = combinator(*cats)
            if result is not None:
                results.append(result)

    cache[cats] = results
    return results