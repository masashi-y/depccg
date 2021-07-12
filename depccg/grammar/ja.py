from typing import Optional, List, Dict, Set, TypeVar, Tuple
from depccg.cat import Category
from depccg.unification import Unification
from depccg.types import Combinator, CombinatorResult

X = TypeVar('X')
Pair = Tuple[X, X]


def _is_modifier(x: Category) -> bool:
    return x.is_functor and x.left == x.right


def forward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b")
    if uni(x, y):
        result = y if _is_modifier(x) else uni['a']
        return CombinatorResult(
            cat=result,
            op_string="fa",
            op_symbol=">",
            head_is_left=False,
        )
    return None


def backward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("b", "a\\b")
    if uni(x, y):
        result = x if _is_modifier(y) else uni['a']
        return CombinatorResult(
            cat=result,
            op_string="ba",
            op_symbol="<",
            head_is_left=False,
        )
    return None


def forward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b/c")
    if uni(x, y):
        result = y if _is_modifier(x) else uni['a'] / uni['c']
        return CombinatorResult(
            cat=result,
            op_string="fc",
            op_symbol=">B",
            head_is_left=False,
        )
    return None


def generalized_backward_composition1(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("b\\c", "a\\b")
    if uni(x, y):
        result = x if _is_modifier(y) else uni['a'] | uni['c']
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_symbol="<B1",
            head_is_left=False,
        )
    return None


def generalized_backward_composition2(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("(b\\c)|d", "a\\b")
    if uni(x, y):
        result = x if _is_modifier(y) else x.functor(
            uni['a'] | uni['c'], uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_symbol="<B2",
            head_is_left=False,
        )
    return None


def generalized_backward_composition3(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("((b\\c)|d)|e", "a\\b")
    if uni(x, y):
        result = x if _is_modifier(y) else x.functor(
            x.left.functor(uni['a'] | uni['c'], uni['d']), uni['e']
        )
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_symbol="<B3",
            head_is_left=False,
        )
    return None


def generalized_backward_composition4(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("(((b\\c)|d)|e)|f", "a\\b")
    if uni(x, y):
        result = x if _is_modifier(y) else x.functor(
            x.left.functor(
                x.left.left.functor(uni['a'] | uni['c'], uni['d']),
                uni['e']
            ),
            uni['f']
        )
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_symbol="<B4",
            head_is_left=False,
        )
    return None


def generalized_forward_composition1(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b\\c")
    if uni(x, y):
        result = y if _is_modifier(x) else uni['a'] / uni['c']
        return CombinatorResult(
            cat=result,
            op_string="fx",
            op_symbol=">Bx1",
            head_is_left=False,
        )
    return None


def generalized_forward_composition2(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "(b\\c)|d")
    if uni(x, y):
        result = y if _is_modifier(x) else y.functor(
            uni['a'] | uni['c'], uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="fx",
            op_symbol=">Bx2",
            head_is_left=False,
        )
    return None


def generalized_forward_composition3(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "((b\\c)|d)|e")
    if uni(x, y):
        result = y if _is_modifier(x) else y.functor(y.left.functor(
            uni['a'] | uni['c'], uni['d']), uni['e']
        )
        return CombinatorResult(
            cat=result,
            op_string="fx",
            op_symbol=">Bx3",
            head_is_left=False,
        )
    return None


_possible_root_categories = [
    Category.parse("NP[case=nc,mod=nm,fin=f]"),
    Category.parse("NP[case=nc,mod=nm,fin=t]"),
    Category.parse("S[mod=nm,form=attr,fin=t]"),
    Category.parse("S[mod=nm,form=base,fin=f]"),
    Category.parse("S[mod=nm,form=base,fin=t]"),
    Category.parse("S[mod=nm,form=cont,fin=f]"),
    Category.parse("S[mod=nm,form=cont,fin=t]"),
    Category.parse("S[mod=nm,form=da,fin=f]"),
    Category.parse("S[mod=nm,form=da,fin=t]"),
    Category.parse("S[mod=nm,form=hyp,fin=t]"),
    Category.parse("S[mod=nm,form=imp,fin=f]"),
    Category.parse("S[mod=nm,form=imp,fin=t]"),
    Category.parse("S[mod=nm,form=r,fin=t]"),
    Category.parse("S[mod=nm,form=s,fin=t]"),
    Category.parse("S[mod=nm,form=stem,fin=f]"),
    Category.parse("S[mod=nm,form=stem,fin=t]")
]


def conjoin(x: Category, y: Category) -> Optional[CombinatorResult]:
    if (
        x in _possible_root_categories
        and y in _possible_root_categories
    ):
        # if x == y and x in _possible_root_categories:
        result = y
        return CombinatorResult(
            cat=result,
            op_string="other",
            op_symbol="SSEQ",
            head_is_left=False,
        )


combinators: List[Combinator] = [
    forward_application,
    backward_application,
    forward_composition,
    generalized_backward_composition1,
    generalized_backward_composition2,
    generalized_backward_composition3,
    generalized_backward_composition4,
    generalized_forward_composition1,
    generalized_forward_composition2,
    generalized_forward_composition3,
    conjoin,
]


def apply_binary_rules(
    x: Category,
    y: Category,
    seen_rules: Optional[Set[Pair[Category]]] = None,
) -> List[CombinatorResult]:
    key = (x, y)
    results = []
    if seen_rules is None or key in seen_rules:
        for combinator in combinators:
            result = combinator(*key)
            if result is not None:
                results.append(result)

    return results


def _unary_rule_symbol(x: Category) -> str:

    features = set(x.arg(0).feature.items())
    if ('mod', 'adn') in features:
        if x.clear_features == 'S':
            return 'ADNext'
        return 'ADNint'
    elif ('mod', 'adv') in features:
        if x.clear_features == 'S\\NP':
            return 'ADV1'
        elif x.clear_features == '(S\\NP)\\NP':
            return 'ADV2'
        return 'ADV0'
    return 'OTHER'


def apply_unary_rules(
    x: Category,
    unary_rules: Dict[Category, List[Category]]
) -> List[CombinatorResult]:
    if x not in unary_rules:
        return []
    results = []
    for result in unary_rules[x]:
        op_string = _unary_rule_symbol(x)
        results.append(
            CombinatorResult(
                cat=result,
                op_string=op_string,
                op_symbol=op_string,
                head_is_left=True,
            )
        )
    return results
