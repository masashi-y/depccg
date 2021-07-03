from typing import Optional, Dict, Set
from depccg.py_cat import Category, Atom


VARIABLES = {'X', 'Y', 'Z'}


class Unification(object):
    """This performs unification on category variables.
    Usage:
    >>> uni = Unification("a/b", "b")
    >>> x = Category.parse("S[X]/NP[X]")
    >>> y = Category.parse("NP[mod]")
    >>> uni(x, y)
    True
    >>> uni["a"]
    S[mod]

    Args:
        x: a string pattern (e.g., "a/b") to match against the first argument (x above).
        y: a string pattern ("b") to match against the first argument (y above).
    """

    def __init__(self, x: str, y: str, variables: Optional[Set[str]] = None) -> None:
        self.cats = {}
        self.mapping = {}
        self.x = Category.parse(x)
        self.y = Category.parse(y)
        self.x_features = {}
        self.y_features = {}
        self.variables = variables or VARIABLES
        self.success = False

    def __call__(self, x: Category, y: Category) -> bool:

        def scan_deep(s: Category, v: str, index: int, results: Dict[str, str]):
            if s.is_functor:
                index = scan_deep(s.left, v, index, results)
                index = scan_deep(s.right, v, index, results)
                return index
            results[v + str(index)] = s.feature
            return index + 1

        def scan(s: Category, t: Category, results: Dict[str, str]) -> bool:
            if (
                s.is_functor and t.is_functor and (
                    s.slash == t.slash or '|' in (s.slash, t.slash))
            ):
                return (
                    scan(s.left, t.left, results) and scan(
                        s.right, t.right, results)
                )
            elif s.is_atomic and t.is_functor:
                self.cats[s.base] = t
                scan_deep(t, s.base, 0, results)
                return True
            elif s.is_atomic and t.is_atomic:
                self.cats[s.base] = t
                if t.feature is not None:
                    results[s.base] = t.feature
                return True
            return False

        self.success = (
            scan(self.x, x, self.x_features) and scan(
                self.y, y, self.y_features)
        )

        if not self.success:
            return False

        # meta variables such as a, b, c, d, etc.
        meta_vars = set(self.x_features.keys()) & set(self.y_features.keys())

        if len(meta_vars) == 0:
            return True

        for var in meta_vars:
            if self.ignore(self.x_features[var]) and not self.ignore(self.y_features[var]):
                self.mapping[self.x_features[var]] = self.y_features[var]
            elif not self.ignore(self.x_features[var]) and self.ignore(self.y_features[var]):
                self.mapping[self.y_features[var]] = self.x_features[var]
            elif self.x_features[var] != self.y_features[var]:
                return False
        return True

    def ignore(self, x: str) -> bool:
        return x in self.variables

    def __getitem__(self, key: str) -> Category:

        def rec(x: Category) -> Category:
            if x.is_functor:
                return x.functor(rec(x.left), rec(x.right))
            else:
                if x.feature in self.variables:
                    return Atom(x.base, self.mapping[x.feature])
                else:
                    return x

        assert self.success
        return rec(self.cats[key])
