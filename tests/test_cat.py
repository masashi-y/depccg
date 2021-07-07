from depccg.py_cat import Category, Atom, Functor, Feature, TernaryFeature, UnaryFeature


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