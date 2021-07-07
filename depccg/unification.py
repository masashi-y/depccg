from typing import Dict, Union
from depccg.cat import Category, Atom, Feature


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
        meta_x: a string pattern (e.g., "a/b") to match against the first argument (x above).
        meta_y: a string pattern ("b") to match against the first argument (y above).
    """

    def __init__(
        self,
        meta_x: Union[str, Category],
        meta_y: Union[str, Category],
    ) -> None:
        self.cats: Dict[str, Category] = {}
        # mapping of variable feature to its instantiation
        self.mapping: Dict[Feature, Feature] = {}
        self.meta_x = (
            Category.parse(meta_x) if isinstance(meta_x, str) else meta_x
        )
        self.meta_y = (
            Category.parse(meta_y) if isinstance(meta_y, str) else meta_y
        )
        # meta variables to feature values
        self.x_features: Dict[str, Feature] = {}
        self.y_features: Dict[str, Feature] = {}
        self.success = False
        self.done = False

    def __call__(self, x: Category, y: Category) -> bool:
        if self.done:
            raise RuntimeError(
                "cannot use the same Unification object more than once."
            )
        self.done = True

        def scan_deep(s: Category, v: str, index: int, results: Dict[str, Feature]):
            if s.is_functor:
                index = scan_deep(s.left, v, index, results)
                index = scan_deep(s.right, v, index, results)
                return index
            results[f'{v}{index}'] = s.feature
            return index + 1

        def scan(s: Category, t: Category, results: Dict[str, Feature]) -> bool:
            # collect categories corresponding to meta variables
            if s.is_atomic:
                if s.base in self.cats and not (t ^ self.cats[s.base]):
                    return False
                self.cats[s.base] = t

            if (
                s.is_functor and t.is_functor and (
                    s.slash == t.slash or '|' in (s.slash, t.slash))
            ):
                return (
                    scan(s.left, t.left, results) and scan(
                        s.right, t.right, results)
                )
            elif s.is_atomic and t.is_functor:
                scan_deep(t, s.base, 0, results)
                return True
            elif s.is_atomic and t.is_atomic:
                if t.feature is not None:
                    results[s.base] = t.feature
                return True
            return False  # s.is_functor and t.is_atomic

        self.success = (
            scan(self.meta_x, x, self.x_features) and scan(
                self.meta_y, y, self.y_features)
        )

        if not self.success:
            return False

        # meta variables such as a, b, c, d, etc.
        meta_vars = set(self.x_features.keys()) & set(self.y_features.keys())

        if len(meta_vars) == 0:
            return True

        for var in meta_vars:
            x_feature = self.x_features[var]
            y_feature = self.y_features[var]

            # these pairs can match: (NP[nb], NP[conj]), (NP, NP[conj]), (NP[X], NP[conj])
            if x_feature.unifies(y_feature):
                # if (NP[X], NP[conj]), further memorize the matching `X := conj`
                # if there're two of `X := conj` and `X := nb`, choose one arbitrarily
                if x_feature.is_variable:
                    self.mapping[x_feature] = y_feature

            elif y_feature.unifies(x_feature):
                if y_feature.is_variable:
                    self.mapping[y_feature] = x_feature
            else:
                self.success = False
                return False

        return True

    def __getitem__(self, key: str) -> Category:

        def rec(x: Category) -> Category:
            if x.is_functor:
                return x.functor(rec(x.left), rec(x.right))
            else:
                if x.feature in self.mapping:
                    return Atom(x.base, self.mapping[x.feature])
                else:
                    return x

        assert self.success, \
            ("the unification has not been successful. "
             "Unification.__getitem__ is not callable in that case.")

        if key not in self.cats:
            raise KeyError(f'meta category `{key}` has not been observed.')
        return rec(self.cats[key])
