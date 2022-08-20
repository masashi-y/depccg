import pytest
import re
from depccg.cat import Category, Atom, Functor, TernaryFeature, UnaryFeature


en_categories = [
    (Category.parse(text.strip()), text.strip())
    for text in open('tests/cats.txt')
]

ja_categories = [
    (Category.parse(text.strip()), text.strip())
    for text in open('tests/cats.ja.txt')
]


@pytest.mark.parametrize("result, expect", en_categories)
def test_parse_many_en(result, expect):
    assert str(result) == expect


@pytest.mark.parametrize("result, expect", en_categories)
def test_parse_many_ja(result, expect):
    assert str(result) == expect


def test_parse():
    assert Category.parse("NP") == Atom("NP")
    assert Category.parse("(NP)") == Atom("NP")
    assert Category.parse("S/NP") == Functor(Atom("S"), '/', Atom("NP"))
    assert Category.parse("(S/NP)") == Functor(Atom("S"), '/', Atom("NP"))
    assert Category.parse(
        "S[dcl]/NP") == Functor(Atom("S", UnaryFeature("dcl")), '/', Atom("NP"))

    # japanese categories
    assert Category.parse("NP[case=nc,mod=nm,fin=t]") == Atom(
        "NP", TernaryFeature(("case", "nc"), ("mod", "nm"), ("fin", "t")))
    assert Category.parse("S[mod=nm,form=attr,fin=t]") == Atom(
        "S", TernaryFeature(("mod", "nm"), ("form", "attr"), ("fin", "t")))

    # punctuations
    assert Category.parse(',') == Atom(',')
    assert Category.parse('.') == Atom('.')
    assert Category.parse(';') == Atom(';')
    assert Category.parse(':') == Atom(':')
    assert Category.parse('LRB') == Atom('LRB')
    assert Category.parse('RRB') == Atom('RRB')
    assert Category.parse('conj') == Atom('conj')
    assert Category.parse('*START*') == Atom('*START*')
    assert Category.parse('*END*') == Atom('*END*')


def test_binop():
    S = Category.parse("S")
    NP = Category.parse("NP")

    assert S / NP == Category.parse("S/NP")
    assert S | NP == Category.parse("S\\NP")
    assert S | (S / NP) == Category.parse("S\\(S/NP)")
    assert (S | NP) / NP == Category.parse("(S\\NP)/NP")

    assert S / NP == "S/NP"
    assert S | NP == "S\\NP"
    assert S | (S / NP) == "S\\(S/NP)"
    assert (S | NP) / NP == "(S\\NP)/NP"

    assert "S/NP" == S / NP
    assert "S\\NP" == S | NP
    assert "S\\(S/NP)" == S | (S / NP)
    assert "(S\\NP)/NP" == (S | NP) / NP


def test_parse_marked_ja_categries():

    DEPENDENCY = re.compile(r'{.+?}')

    def parse_with_dependency(text, is_leaf=False):
        if is_leaf:
            text = text[:text.find('_')]
        text = DEPENDENCY.sub('', text)
        return Category.parse(text)

    assert (
        parse_with_dependency("S[mod=nm,form=base,fin=t]{I1}") == Category.parse(
            "S[mod=nm,form=base,fin=t]")
    )
    assert (
        parse_with_dependency("(NP[case=nc,mod=X1,fin=X2]{I1}/NP[case=nc,mod=X1,fin=X2]{I1}){I2}") == Category.parse(
            "NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]")
    )
    assert (
        parse_with_dependency("(((NP[case=X1,mod=X2,fin=f]{I1}/NP[case=X1,mod=X2,fin=f]{I1}){I2})\\NP[case=nc,mod=nm,fin=f]{I2}){I3}_none", is_leaf=True) == Category.parse(
            "(NP[case=X1,mod=X2,fin=f]/NP[case=X1,mod=X2,fin=f])\\NP[case=nc,mod=nm,fin=f]")
    )
    assert (
        parse_with_dependency("(NP[case=X1,mod=X2,fin=f]{I1}/NP[case=X1,mod=X2,fin=f]{I1}){I2}_none", is_leaf=True) == Category.parse(
            "NP[case=X1,mod=X2,fin=f]/NP[case=X1,mod=X2,fin=f]")
    )
    assert (
        parse_with_dependency("(((S[mod=adn,form=base,fin=f]{I1}\\NP[case=ga,mod=nm,fin=f]{I2}){I1})\\NP[case=ni,mod=nm,fin=f]{I3}){I1}_I1(I2,_,I3,_)", is_leaf=True) == Category.parse(
            "(S[mod=adn,form=base,fin=f]\\NP[case=ga,mod=nm,fin=f])\\NP[case=ni,mod=nm,fin=f]")
    )
    assert (
        parse_with_dependency("(((S[mod=adn,form=base,fin=f]{I1}\\NP[case=ga,mod=nm,fin=f]{I2}){I1})\\NP[case=o,mod=nm,fin=f]{I3}){I1}_I1(I2,I3,_,_)", is_leaf=True) == Category.parse(
            "(S[mod=adn,form=base,fin=f]\\NP[case=ga,mod=nm,fin=f])\\NP[case=o,mod=nm,fin=f]")
    )
    assert (
        parse_with_dependency("(((S[mod=X1,form=X2,fin=f]{I1}/S[mod=X1,form=X2,fin=f]{I1}){I2})\\NP[case=nc,mod=nm,fin=f]{I3}){I2}_none", is_leaf=True) == Category.parse(
            "(S[mod=X1,form=X2,fin=f]/S[mod=X1,form=X2,fin=f])\\NP[case=nc,mod=nm,fin=f]")
    )
    assert (
        parse_with_dependency("(S[mod=nm,form=da,fin=f]{I1}\\NP[case=ga,mod=nm,fin=f]{I2}){I1}_I1(I2,_,_,_)", is_leaf=True) == Category.parse(
            "S[mod=nm,form=da,fin=f]\\NP[case=ga,mod=nm,fin=f]")
    )
