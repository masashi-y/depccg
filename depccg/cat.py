from typing import Optional, Callable, Tuple, TypeVar, Iterator
from dataclasses import dataclass
import re

X = TypeVar('X')
Pair = Tuple[X, X]

cat_split = re.compile(r'([\[\]\(\)/\\|])')
punctuations = [',', '.', ';', ':', 'LRB', 'RRB', 'conj', '*START*', '*END*']


class Feature(object):
    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def parse(cls, text: str) -> 'Feature':
        if '=' in text and ',' in text:
            return TernaryFeature(*[tuple(kv.split('=')) for kv in text.split(',')])
        return UnaryFeature(text)


@dataclass(frozen=True, repr=False)
class UnaryFeature(Feature):
    """Common feature type widely used in many CCGBanks.
    This assumes None or "X" values as representing a variable feature.
    As commonly done in the parsing literature, the 'nb' variable is treated
    sometimes as not existing, i.e., NP[conj] and NP[nb] can match.
    """

    value: Optional[str] = None

    def __str__(self) -> str:
        return self.value if self.value is not None else ''

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self == Feature.parse(other)
        elif not isinstance(other, UnaryFeature):
            return False
        return self.value == other.value

    def unifies(self, other: 'UnaryFeature') -> bool:
        return (
            self.is_variable
            or self.is_ignorable
            or self == other
        )

    @property
    def is_variable(self) -> bool:
        return self.value == "X"

    @property
    def is_ignorable(self) -> bool:
        return self.value is None or self.value == "nb"


@dataclass(frozen=True, repr=False)
class TernaryFeature(Feature):
    """Feature type used in the Japanese version of CCGBank.
    This assumes a feature with values (X1, X2, X3) as representing a variable.
    """

    kv1: Pair[str]
    kv2: Pair[str]
    kv3: Pair[str]

    def items(self) -> Iterator[Pair[str]]:
        return (self.kv1, self.kv2, self.kv3)

    def values(self) -> Iterator[str]:
        return (v for _, v in self.items())

    def keys(self) -> Iterator[str]:
        return (k for k, _ in self.items())

    def __str__(self) -> str:
        return ','.join(f'{k}={v}' for k, v in self.items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self == Feature.parse(other)
        elif not isinstance(other, TernaryFeature):
            return False
        return (
            self.kv1 == other.kv1
            and self.kv2 == other.kv2
            and self.kv3 == other.kv3
        )

    def unifies(self, other: 'UnaryFeature') -> bool:
        if self == other:
            return True

        if list(self.keys()) != list(other.keys()):
            return False

        return all(
            v1 == v2 or v1.startswith('X')
            for v1, v2 in zip(self.values(), other.values())
        )

    @property
    def is_variable(self) -> bool:
        return any(v.startswith('X') for v in self.values())


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
        tokens = cat_split.sub(r' \1 ', text)
        buffer = list(reversed([i for i in tokens.split(' ') if i != '']))
        stack = []

        while len(buffer):
            item = buffer.pop()
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
                if len(buffer) >= 3 and buffer[-1] == '[':
                    buffer.pop()
                    feature = Feature.parse(buffer.pop())
                    assert buffer.pop() == ']'
                    stack.append(Atom(item, feature))
                else:
                    stack.append(Atom(item))

        if len(stack) == 1:
            return stack[0]
        x, f, y = stack
        return Functor(x, f, y)


@dataclass(frozen=True, repr=False)
class Atom(Category):
    base: str
    feature: Feature = UnaryFeature()

    def __str__(self) -> str:
        feature = str(self.feature)
        if len(feature) == 0:
            return self.base
        return f'{self.base}[{feature}]'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str(self) == other
        elif not isinstance(other, Atom):
            return False
        return (
            self.base == other.base
            and self.feature == other.feature
        )

    def __xor__(self, other: object) -> bool:
        if not isinstance(other, Atom):
            return False
        return self.base == other.base

    @property
    def is_atomic(self):
        return True

    @property
    def nargs(self) -> int:
        return 0

    def arg(self, index: int) -> Optional[Category]:
        if index == 0:
            return self
        return None

    def clear_features(self, *args) -> 'Atom':
        if self.feature in args:
            return Atom(self.base)
        return self


@dataclass(frozen=True, repr=False)
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

    def __xor__(self, other: object) -> bool:
        if not isinstance(other, Functor):
            return False
        return (
            self.left ^ other.left
            and self.slash == other.slash
            and self.right ^ other.right
        )

    @property
    def functor(self) -> Callable[[Category, Category], Category]:
        return lambda x, y: Functor(x, self.slash, y)

    @property
    def is_functor(self):
        return True

    @property
    def nargs(self) -> int:
        return 1 + self.left.nargs

    def arg(self, index: int) -> Optional[Category]:
        if self.nargs == index:
            return self
        else:
            return self.left.arg(index)

    def clear_features(self, *args) -> 'Functor':
        return self.functor(
            self.left.clear_features(*args),
            self.right.clear_features(*args)
        )
