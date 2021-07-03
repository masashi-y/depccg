from typing import Optional, List
from depccg.py_cat import Category
from depccg.unification import Unification
from depccg.types import Combinator, CombinatorResult


JA_VARIABLES = {'X1', 'X2', 'X3'}


def forward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b", JA_VARIABLES)
    if uni(x, y):
        result = uni['a']
        return CombinatorResult(
            cat=result,
            op_string="fa",
            op_string=">",
            head_is_left=False,
        )
    return None


def backward_application(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("b", "a\\b", JA_VARIABLES)
    if uni(x, y):
        result = uni['a']
        return CombinatorResult(
            cat=result,
            op_string="ba",
            op_string="<",
            head_is_left=False,
        )
    return None


def forward_composition(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b/c", JA_VARIABLES)
    if uni(x, y):
        result = uni['a'] / uni['c']
        return CombinatorResult(
            cat=result,
            op_string="fc",
            op_string=">B",
            head_is_left=False,
        )
    return None


def generalized_backward_composition1(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("b\\c", "a\\b", JA_VARIABLES)
    if uni(x, y):
        result = uni['a'] | uni['c']
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_string="<B1",
            head_is_left=False,
        )
    return None


def generalized_backward_composition2(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("(b\\c)|d", "a\\b", JA_VARIABLES)
    if uni(x, y):
        result = x.functor(uni['a'] | uni['c'], uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_string="<B2",
            head_is_left=False,
        )
    return None


def generalized_backward_composition3(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("((b\\c)|d)|e", "a\\b", JA_VARIABLES)
    if uni(x, y):
        result = x.functor(
            x.left.functor(uni['a'] | uni['c'], uni['d']), uni['e']
        )
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_string="<B3",
            head_is_left=False,
        )
    return None


def generalized_backward_composition4(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("(((b\\c)|d)|e)|f", "a\\b", JA_VARIABLES)
    if uni(x, y):
        result = x.functor(
            x.left.functor(
                x.left.left.functor(uni['a'] | uni['c'], uni['d']),
                uni['e']
            ),
            uni['f']
        )
        return CombinatorResult(
            cat=result,
            op_string="bx",
            op_string="<B4",
            head_is_left=False,
        )
    return None


def generalized_forward_composition1(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "b\\c", JA_VARIABLES)
    if uni(x, y):
        result = uni['a'] | uni['c']
        return CombinatorResult(
            cat=result,
            op_string="fx",
            op_string=">Bx1",
            head_is_left=False,
        )
    return None


def generalized_forward_composition2(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "(b\\c)|d", JA_VARIABLES)
    if uni(x, y):
        result = y.functor(uni['a'] | uni['c'], uni['d'])
        return CombinatorResult(
            cat=result,
            op_string="fx",
            op_string=">Bx2",
            head_is_left=False,
        )
    return None


def generalized_forward_composition3(x: Category, y: Category) -> Optional[CombinatorResult]:
    uni = Unification("a/b", "((b\\c)|d)|e", JA_VARIABLES)
    if uni(x, y):
        result = y.functor(y.left.functor(
            uni['a'] | uni['c'], uni['d']), uni['e']
        )
        return CombinatorResult(
            cat=result,
            op_string="fx",
            op_string=">Bx3",
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
        x in _possible_root_categories and x == y and not x.is_functor
    ):
        result = x
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
