from depccg.py_cat import Category, Atom, Functor, Feature


def test_parse():
    assert Category.parse("NP") == Atom("NP")
    assert Category.parse("(NP)") == Atom("NP")
    assert Category.parse("S/NP") == Functor(Atom("S"), '/', Atom("NP"))
    assert Category.parse("(S/NP)") == Functor(Atom("S"), '/', Atom("NP"))
    assert Category.parse(
        "S[dcl]/NP") == Functor(Atom("S", Feature("dcl")), '/', Atom("NP"))

    assert Category.parse("NP[case=nc,mod=nm,fin=t]") == Atom(
        "NP", Feature([("case", "nc"), ("mod", "nm"), ("fin", "t")]))
    assert Category.parse("S[mod=nm,form=attr,fin=t]") == Atom(
        "S", Feature([("mod", "nm"), ("form", "attr"), ("fin", "t")]))


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
