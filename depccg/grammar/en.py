from typing import Optional, List
from string import ascii_letters
from depccg.py_cat import Category
from depccg.unification import Unification
from depccg.types import Combinator, CombinatorResult


def forward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b")
    if uni(x, y):
        result = uni['a']
        return CombinatorResult(
            cat=result,
            op_string="fa",
            op_string=">",
            head_is_left=True,
        )
    return None


def backward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    if x == 'S[dcl]' and y == 'S[em]\\S[em]':
        result = x
    else:
        uni = Unification("b", "a\\b")
        if uni(x, y):
            result = uni['a']
        else:
            return None

    return CombinatorResult(
        cat=result,
        op_string="ba",
        op_string="<",
        head_is_left=True,
    )


def forward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b/c")
    if uni(x, y):
        result = uni['a'] / uni['c']
        return CombinatorResult(
            cat=result,
            op_string="fc",
            op_string=">B",
            head_is_left=True,
        )
    return None


def backward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("b/c", "a\\b")
    if uni(x, y):
        result = uni['a'] / uni['c']
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_string="<B",
            head_is_left=True,
        )
    return None


def generalized_forward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "(b/c)|d")
    if uni(x, y):
        result = y.functor((uni['a'] / uni['c']), uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="gfc",
            op_string=">B",
            head_is_left=True,
        )
    return None


def generalized_backward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("(b/c)|d", "a/b")
    if uni(x, y):
        result = x.functor((uni['a'] / uni['c']), uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="gbx",
            op_string="<B",
            head_is_left=True,
        )
    return None


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


def conjunction(x: Category, y: Category) -> Optional[CombinatorResult]:
    if (
        not _is_punct(y)
        and not _is_type_raised(y)
        and x in (",", ";", "conj")
        and y != "NP\\NP"
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
            op_symbol="<rp>",
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
            op_symbol="<rp>",
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
