import pytest

from depccg.cat import Category
from depccg.grammar import ja

observed_binary_rules = [
    tuple(Category.parse(category) for category in text.strip().split(' '))
    for text in open('tests/grammar/rules.ja.txt')
]


@pytest.mark.parametrize("expect, x, y", observed_binary_rules)
def test_binary_rule(expect, x, y):
    assert expect in [result.cat for result in ja.apply_binary_rules(x, y)]
