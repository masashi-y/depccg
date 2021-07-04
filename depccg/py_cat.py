from typing import Optional, Callable, List, Tuple, TypeVar
from dataclasses import dataclass
import re

X = TypeVar('X')
Pair = Tuple[X, X]

cat_split = re.compile(r'([\[\]\(\)/\\])')
punctuations = [',', '.', ';', ':', 'LRB', 'RRB', 'conj', '*START*', '*END*']


class Feature(object):
    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def parse(cls, text: str) -> 'Feature':
        if '=' in text and ',' in text:
            return TernaryFeature(*[tuple(kv.split('=')) for kv in text.split(',')])
        return UnaryFeature(text)


@dataclass(frozen=True)
class UnaryFeature(Feature):
    value: str

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class TernaryFeature(Feature):
    kv1: Pair[str]
    kv2: Pair[str]
    kv3: Pair[str]

    def items(self) -> List[Pair[str]]:
        return [self.kv1, self.kv2, self.kv3]

    def __str__(self) -> str:
        return ','.join(f'{k}={v}' for k, v in self.items())


class Category(object):
    @property
    def is_functor(self):
        return not self.is_atomic

    @property
    def is_atomic(self):
        return not self.is_functor

    def __repr__(self) -> str:
        return str(self)

    def __truediv__(self, other: 'Category') -> 'Category':
        return Functor(self, '/', other)

    def __or__(self, other: 'Category') -> 'Category':
        return Functor(self, '\\', other)

    @classmethod
    def parse(cls, text: str) -> 'Category':
        items = cat_split.sub(r' \1 ', text)
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
                if len(stack) == 0:
                    return y
                f = stack.pop()
                x = stack.pop()
                stack.append(Functor(x, f, y))
            elif item in ('/', '\\', '|'):
                stack.append(item)
            else:
                if len(buf) >= 3 and buf[-1] == '[':
                    buf.pop()
                    feature = Feature.parse(buf.pop())
                    assert buf.pop() == ']'
                    stack.append(Atom(item, feature))
                else:
                    stack.append(Atom(item))

        if len(stack) == 1:
            return stack[0]
        x, f, y = stack
        return Functor(x, f, y)


@dataclass(frozen=True)
class Atom(Category):
    base: str
    feature: Optional[Feature] = None

    def __str__(self) -> str:
        if self.feature is None:
            return self.base
        return f'{self.base}[{self.feature}]'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str(self) == other
        elif not isinstance(other, Atom):
            return False
        return (
            self.base == other.base
            and self.feature == other.feature
        )

    @property
    def is_atomic(self):
        return True


@dataclass(frozen=True)
class Functor(Category):
    left: Category
    slash: str
    right: Category

    def __str__(self) -> str:
        def _str(cat):
            if isinstance(cat, Functor):
                return f'({cat})'
            return str(cat)
        return _str(self.left) + self.slash + _str(self.right)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str(self) == other
        elif not isinstance(other, Functor):
            return False
        return (
            self.left == other.left
            and self.slash == other.slash
            and self.right == other.right
        )

    @property
    def functor(self) -> Callable[[Category, Category], Category]:
        return lambda x, y: Functor(x, self.slash, y)

    @property
    def is_functor(self):
        return True
