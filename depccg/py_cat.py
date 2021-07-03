from typing import Optional, Callable

import re

cat_split = r'([\[\]\(\)/\\])'
punctuations = [',', '.', ';', ':', 'LRB', 'RRB', 'conj', '*START*', '*END*']


# class Feature(OrderedDict):
#     DEFAULT_KEY = '**DEFAULT_KEY**'
#
#     def __init__(self, x):
#         if isinstance(x, str):
#             self[Feature.DEFAULT_KEY] = x
#         else:
#             assert isinstance(x, dict)
#             self.update(x)
#
#     @property
#     def simple(self) -> bool:
#         return (
#             len(self) == 1 and Feature.DEFAULT_KEY in self
#         )
#
#     def __str__(self):
#         if self.simple:
#             return self[Feature.DEFAULT_KEY]
#         else:
#             return None

class Category(object):
    def __init__(self, text: str):
        return self.parse(text)

    @property
    def is_functor(self):
        return not self.is_atomic

    @property
    def is_atomic(self):
        return not self.is_functor

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def parse(cls, text: str) -> 'Category':
        items = re.sub(cat_split, r' \1 ', text)
        buf = list(reversed([i for i in items.split(' ') if i != '']))
        stack = []

        while len(buf):
            item = buf.pop()
            if item in punctuations:
                stack.append(Atom(item))
            elif item == '(':
                pass
            elif item == ')':
                y = stack.pop()
                f = stack.pop()
                x = stack.pop()
                stack.append(f(x, y))
            elif item == '/':
                stack.append(lambda x, y: Functor(x, '/', y))
            elif item == '\\':
                stack.append(lambda x, y: Functor(x, '\\', y))
            elif item == '|':
                stack.append(lambda x, y: Functor(x, '|', y))
            else:
                if len(buf) >= 3 and buf[-1] == '[':
                    buf.pop()
                    feature = buf.pop()
                    assert buf.pop() == ']'
                    stack.append(Atom(item, feature))
                else:
                    stack.append(Atom(item))

        if len(stack) == 1:
            return stack[0]
        x, f, y = stack
        return f(x, y)


class Atom(Category):
    def __init__(self, base: str, feature: Optional[str] = None) -> None:
        self.base = base
        self.feature = feature

    @property
    def is_atomic(self):
        return True

    def __str__(self) -> str:
        if self.feature is None:
            return self.base
        return f'{self.base}[{self.feature}]'


class Functor(Category):
    def __init__(self, left: Category, slash: str, right: Category) -> None:
        self.left = left
        self.slash = slash
        self.right = right

    def __str__(self) -> str:
        def _str(cat):
            if isinstance(cat, Functor):
                return f'({cat})'
            return str(cat)
        return _str(self.left) + self.slash + _str(self.right)

    @property
    def functor(self) -> Callable[[Category, Category], Category]:
        return lambda x, y: Functor(x, self.slash, y)

    @property
    def is_functor(self):
        return True
