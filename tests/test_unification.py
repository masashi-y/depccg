from depccg.cat import Category
from depccg.unification import Unification
import pytest


def test_basic():
    uni = Unification("(((a/b)/c)/d)/e", "f")
    x = Category.parse("(((a/b)/c)/d)/e")
    y = Category.parse("f")
    assert uni(x, y)
    assert uni["a"] == "a"
    assert uni["b"] == "b"
    assert uni["c"] == "c"
    assert uni["d"] == "d"
    assert uni["e"] == "e"
    assert uni["f"] == "f"
    with pytest.raises(RuntimeError, match="cannot use the same *"):
        uni(x, y)


def test_deep():
    uni = Unification("a/b", "c")
    x = Category.parse("(((a/b)/c)/d)/e")
    y = Category.parse("f")
    assert uni(x, y)
    assert uni["a"] == "((a/b)/c)/d"
    assert uni["b"] == "e"
    assert uni["c"] == "f"


def test_english():
    uni = Unification("a/b", "b")
    x = Category.parse("S[X]/NP[X]")
    y = Category.parse("NP[mod]")
    assert uni(x, y)
    assert uni["a"] == Category.parse('S[mod]')


def test_japanese():
    uni = Unification("(a\\b)/c", "c")

    x = Category.parse(
        "(S[mod=nm,form=base,fin=f]\\S[mod=nm,form=base,fin=f])/S[mod=nm,form=base,fin=f]")
    y = Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni(x, y)
    assert uni["a"] == Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni["b"] == Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni["c"] == Category.parse("S[mod=nm,form=base,fin=f]")

    # three variables
    uni = Unification("(a\\b)/c", "c")
    x = Category.parse(
        "(S[mod=X1,form=X2,fin=X3]\\S[mod=X1,form=X2,fin=X3])/S[mod=X1,form=X2,fin=X3]")
    y = Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni(x, y)
    assert uni["a"] == Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni["b"] == Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni["c"] == Category.parse("S[mod=nm,form=base,fin=f]")

    # only two variables
    uni = Unification("(a\\b)/c", "c")
    x = Category.parse(
        "(S[mod=X1,form=X2,fin=f]\\S[mod=X1,form=X2,fin=f])/S[mod=X1,form=X2,fin=f]")
    y = Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni(x, y)
    assert uni["a"] == Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni["b"] == Category.parse("S[mod=nm,form=base,fin=f]")
    assert uni["c"] == Category.parse("S[mod=nm,form=base,fin=f]")
