import pytest

from depccg.cat import Category
from depccg.grammar import en

observed_binary_rules = [
    tuple(Category.parse(category) for category in text.strip().split(' '))
    for text in open('tests/grammar/rules.txt')
]


@pytest.mark.parametrize("x, y, expect", observed_binary_rules)
def test_binary_rule(x, y, expect):
    assert expect in [result.cat for result in en.apply_binary_rules(x, y)]
