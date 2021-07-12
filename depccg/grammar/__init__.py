from typing import Callable, List, TypeVar, Tuple, Set, Dict
from depccg.types import Combinator, CombinatorResult
from depccg.cat import Category

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


def guess_combinator_by_triplet(
    binary_rules: Callable[[Category, Category], List[Category]],
    target: Category,
    x: Category,
    y: Category,
) -> CombinatorResult:
    for rule in binary_rules(x, y):
        if rule.cat == target:
            return rule
    return CombinatorResult(
        cat=rule.cat,
        op_string="unk",
        op_symbol="<unk>",
        head_is_left=True
    )
