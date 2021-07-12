from typing import Optional, List, TypeVar, Tuple, Set, Dict
from string import ascii_letters
from depccg.cat import Category
from depccg.unification import Unification
from depccg.types import Combinator, CombinatorResult

X = TypeVar('X')
Pair = Tuple[X, X]


def _match(x: Category, y: Category) -> bool:
    if x.is_functor and y.is_functor:
        return _match(x.left, y.left) and _match(x.right, y.right)
    elif x.is_atomic and y.is_atomic:
        return (
            x.base == y.base
            and x.feature.unifies(y.feature)
            and y.feature.unifies(x.feature)
        )
    return False


def _is_modifier(x: Category) -> bool:
    return x.is_functor and x.left == x.right


def _is_punct(x: Category) -> bool:
    if x.is_functor:
        return False

    return (
        not x.base[0] in ascii_letters
        or x.base in ("LRB", "RRB", "LQU", "RQU")
    )


def _is_type_raised(x: Category) -> bool:
    if x.is_atomic:
        return False
    return (
        x.right.is_functor and x.right.left == x.left
    )


def forward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b")
    if uni(x, y):
        result = y if _is_modifier(x) else uni['a']
        return CombinatorResult(
            cat=result,
            op_string="fa",
            op_symbol=">",
            head_is_left=True,
        )
    return None


def backward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    if x == 'S[dcl]' and y == 'S[em]\\S[em]':
        result = x
    else:
        uni = Unification("b", "a\\b")
        if uni(x, y):
            result = x if _is_modifier(y) else uni['a']
        else:
            return None

    return CombinatorResult(
        cat=result,
        op_string="ba",
        op_symbol="<",
        head_is_left=True,
    )


def forward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b/c")
    if uni(x, y):
        result = y if _is_modifier(x) else uni['a'] / uni['c']
        return CombinatorResult(
            cat=result,
            op_string="fc",
            op_symbol=">B",
            head_is_left=True,
        )
    return None


def backward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("b/c", "a\\b")
    if uni(x, y):
        if str(uni["b"]) in ("N", "NP"):
            return None
        result = x if _is_modifier(y) else uni['a'] / uni['c']
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_symbol="<B",
            head_is_left=True,
        )
    return None


def generalized_forward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "(b/c)|d")
    if uni(x, y):
        result = y if _is_modifier(x) else y.functor(
            (uni['a'] / uni['c']), uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="gfc",
            op_symbol=">B",
            head_is_left=True,
        )
    return None


def generalized_backward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("(b/c)|d", "a/b")
    if uni(x, y):
        if str(uni["b"]) in ("N", "NP"):
            return None
        result = x if _is_modifier(y) else x.functor(
            (uni['a'] / uni['c']), uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="gbx",
            op_symbol="<B",
            head_is_left=True,
        )
    return None


def conjunction(x: Category, y: Category) -> Optional[CombinatorResult]:
    if (
        not _is_punct(y)
        and not _is_type_raised(y)
        and x in (",", ";", "conj")
        and not (y ^ "NP\\NP")
    ):
        result = y | y
        return CombinatorResult(
            cat=result,
            op_string="conj",
            op_symbol="<Φ>",
            head_is_left=True,
        )
    return None


def conjunction2(x: Category, y: Category) -> Optional[CombinatorResult]:
    if x == "conj" and y == "NP\\NP":
        result = y
        return CombinatorResult(
            cat=result,
            op_string="conj",
            op_symbol="<Φ>",
            head_is_left=True,
        )
    return None


def remove_punctuation1(x: Category, y: Category) -> Optional[CombinatorResult]:
    if _is_punct(x):
        result = y
        return CombinatorResult(
            cat=result,
            op_string="lp",
            op_symbol="<lp>",
            head_is_left=True,
        )
    return None


def remove_punctuation2(x: Category, y: Category) -> Optional[CombinatorResult]:
    if _is_punct(y):
        result = x
        return CombinatorResult(
            cat=result,
            op_string="rp",
            op_symbol="<rp>",
            head_is_left=True,
        )
    return None


def remove_punctuation_left(x: Category, y: Category) -> Optional[CombinatorResult]:
    if x in ("LQU", "LRB"):
        result = y | y
        return CombinatorResult(
            cat=result,
            op_string="lp",
            op_symbol="<lp>",
            head_is_left=True,
        )
    return None


def comma_vp_to_adv(x: Category, y: Category) -> Optional[CombinatorResult]:
    if x == "," and y in ("S[ng]\\NP", "S[pss]\\NP"):
        result = Category.parse("(S\\NP)\\(S\\NP)")
        return CombinatorResult(
            cat=result,
            op_string="lp",
            op_symbol="<*>",
            head_is_left=True
        )
    return None


def parenthetical_direct_speech(x: Category, y: Category) -> Optional[CombinatorResult]:
    if x == "," and y == "S[dcl]/S[dcl]":
        result = Category.parse("(S\\NP)/(S\\NP)")
        return CombinatorResult(
            cat=result,
            op_string="lp",
            op_symbol="<*>",
            head_is_left=True
        )
    return None


combinators: List[Combinator] = [
    forward_application,
    backward_application,
    forward_composition,
    backward_composition,
    generalized_forward_composition,
    generalized_backward_composition,
    conjunction,
    conjunction2,
    remove_punctuation1,
    remove_punctuation2,
    remove_punctuation_left,
    comma_vp_to_adv,
    parenthetical_direct_speech,
]


def apply_binary_rules(
    x: Category,
    y: Category,
    seen_rules: Optional[Set[Pair[Category]]] = None,
) -> List[CombinatorResult]:
    key = (x.clear_features('nb'), y.clear_features('nb'))
    seen_key = (
        x.clear_features('X', 'nb'), y.clear_features('X', 'nb')
    )
    results = []
    if seen_rules is None or seen_key in seen_rules:
        for combinator in combinators:
            result = combinator(*key)
            if result is not None:
                results.append(result)

    return results


def apply_unary_rules(
    x: Category,
    unary_rules: Dict[Category, List[Category]]
) -> List[CombinatorResult]:
    if x not in unary_rules:
        return []
    results = []
    for result in unary_rules[x]:
        type_raised = (
            x.is_atomic
            and x.base in ('NP', 'PP')
            and _is_type_raised(result)
        )

        results.append(
            CombinatorResult(
                cat=result,
                op_string='tr' if type_raised else 'lex',
                op_symbol='<un>',
                head_is_left=True,
            )
        )
    return results
