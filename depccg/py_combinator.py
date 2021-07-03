from typing import Optional, NamedTuple
from depccg.py_cat import Category
from depccg.unification import Unification


class CombinatorResult(NamedTuple):
    cat: Category
    op_string: str
    op_symbol: str
    head_is_left: bool


def forward_application(x: Category, y: Category) -> Optional[Category]:
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


def backward_application(x: Category, y: Category) -> Optional[Category]:
    return None


def forward_composition(x: Category, y: Category) -> Optional[Category]:
    uni = Unification("a/b", "b/c")
    if uni(x, y):
        return uni['a'] / uni['c']
    return None


def backward_composition(x: Category, y: Category) -> Optional[Category]:
    uni = Unification("a/b", "(b/c)|d")
    if uni(x, y):
        y.functor((uni['a'] / uni['d']), uni['e'])
    return None


def generalized_forward_composition(x: Category, y: Category) -> Optional[Category]:
    return None


def generalized_backward_composition(x: Category, y: Category) -> Optional[Category]:
    return None


def is_punct(x: Category) -> bool:
    if x.is_functor:
        return False
    type_ = x.wo_feature().str()
    if ('A' <= type_[0] <= 'Z') or ('a' <= type_[0] <= 'z'):
        return False
    return type_ in ("LRB", "RRB", "LQU", "RQU")


def coordination(x: Category, y: Category) -> Optional[Category]:
    if x == "conj" and y == "NP\\NP":
        return Category("NP")
    if is_punct(y) and x in (",", ";", "conj"):
        return y | y
    return None


def remove_punctuation(x: Category, y: Category) -> Optional[Category]:
    if is_punct(x):
        return y
    if is_punct(y):
        return x
    return None


def comma_vp_to_adv(x: Category, y: Category) -> Optional[Category]:
    if x == "," and y in ("S[ng]\\NP", "S[pss]\\NP"):
        return Category("(S\\NP)\\(S\\NP)")
    return None


def parenthetical_direct_speech(x: Category, y: Category) -> Optional[Category]:
    if x == "," and y == "S[dcl]/S[dcl]":
        return Category("(S\\NP)/(S\\NP)")
    return None
