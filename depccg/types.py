from typing import Optional, NamedTuple, Callable
from depccg.py_cat import Category


class CombinatorResult(NamedTuple):
    cat: Category
    op_string: str
    op_symbol: str
    head_is_left: bool


Combinator = Callable[[Category, Category], Optional[CombinatorResult]]
