import pytest
from depccg.cat import Category
from depccg.types import Token
from depccg.tree import Tree, ScoredTree
from depccg.printer import to_string
from depccg.instance_models import SEMANTIC_TEMPLATES
from depccg.lang import set_global_language_to


@pytest.fixture()
def en_tree():
    return Tree.make_binary(
        Category.parse("S[dcl]"),
        Tree.make_binary(
            Category.parse("S[dcl]"),
            Tree.make_binary(
                Category.parse('NP'),
                Tree.make_terminal(
                    Token(word="This", pos="DT", entity="O",
                          lemma="this", chunk="XX"),
                    Category.parse("NP[nb]/N"),
                ),
                Tree.make_terminal(
                    Token(word="paper", pos="NN", entity="O",
                          lemma="paper", chunk="XX"),
                    Category.parse("N"),
                ),
                'fa',
                '>'
            ),
            Tree.make_binary(
                Category.parse("S[dcl]\\NP"),
                Tree.make_terminal(
                    Token(word="discusses", pos="VBZ",
                          entity="O", lemma="discuss", chunk="XX"),
                    Category.parse("(S[dcl]\\NP)/NP"),
                ),
                Tree.make_binary(
                    Category.parse("NP"),
                    Tree.make_unary(
                        Category.parse("NP"),
                        Tree.make_binary(
                            Category.parse("N"),
                            Tree.make_terminal(
                                Token(word="crucial", pos="JJ", entity="O",
                                      lemma="crucial", chunk="XX"),
                                Category.parse("N/N"),
                            ),
                            Tree.make_terminal(
                                Token(word="aspects", pos="NNS", entity="O",
                                      lemma="aspect", chunk="XX"),
                                Category.parse("N"),
                            ),
                            "fa",
                            ">"
                        ),
                    ),
                    Tree.make_binary(
                        Category.parse("NP\\NP"),
                        Tree.make_terminal(
                            Token(word="of", pos="IN", entity="O",
                                  lemma="of", chunk="XX"),
                            Category.parse("(NP\\NP)/NP"),
                        ),
                        Tree.make_binary(
                            Category.parse("NP"),
                            Tree.make_terminal(
                                Token(word="this", pos="DT", entity="O",
                                      lemma="this", chunk="XX"),
                                Category.parse("NP[nb]/N"),
                            ),
                            Tree.make_binary(
                                Category.parse('N'),
                                Tree.make_terminal(
                                    Token(word="new", pos="JJ", entity="O",
                                          lemma="new", chunk="XX"),
                                    Category.parse("N/N"),
                                ),
                                Tree.make_binary(
                                    Category.parse('N'),
                                    Tree.make_terminal(
                                        Token(word="annotation", pos="NN", entity="O",
                                              lemma="annotation", chunk="XX"),
                                        Category.parse("N/N"),
                                    ),
                                    Tree.make_terminal(
                                        Token(word="scheme", pos="NN", entity="O",
                                              lemma="scheme", chunk="XX"),
                                        Category.parse("N"),
                                    ),
                                    'fa',
                                    '>'
                                ),
                                'fa',
                                '>'
                            ),
                            'fa',
                            '>'
                        ),
                        'fa',
                        '>'
                    ),
                    "ba",
                    "<"
                ),
                'fa',
                '>'
            ),
            "ba",
            "<"
        ),
        Tree.make_terminal(
            Token(word=".", pos=".", entity="O", lemma=".", chunk="XX"),
            Category.parse("."),
        ),
        "rp",
        "<rp>"
    )


@pytest.fixture()
def scored_en_tree(en_tree):
    return ScoredTree(en_tree, -0.05)


@pytest.fixture()
def ja_tree():
    return Tree.make_binary(
        Category.parse("S[mod=nm,form=base,fin=t]"),
        Tree.make_binary(
            Category.parse("S[mod=nm,form=base,fin=f]"),
            Tree.make_binary(
                Category.parse(
                    "S[mod=X1,form=X2,fin=f]/S[mod=X1,form=X2,fin=f]"),
                Tree.make_binary(
                    Category.parse("NP[case=nc,mod=nm,fin=f]"),
                    Tree.make_terminal(
                        Token(word='メロス', surf='メロス', pos='名詞', pos1='一般', pos2='*', pos3='*',
                              nflectionForm='*', inflectionType='*', reading='*', base='メロス'),
                        Category.parse("NP[case=nc,mod=nm,fin=f]")),
                    Tree.make_terminal(
                        Token(word='に', surf='に', pos='助詞', pos1='格助詞', pos2='一般', pos3='*',
                              inflectionForm='*', inflectionType='*', reading='ニ', base='に'),
                        Category.parse("NP[case=nc,mod=nm,fin=f]\\NP[case=nc,mod=nm,fin=f]")),
                    "<",
                    "<",
                ),
                Tree.make_terminal(
                    Token(word='は', surf='は', pos='助詞', pos1='係助詞', pos2='*', pos3='*',
                          nflectionForm='*', inflectionType='*', reading='ハ', base='は'),
                    Category.parse("(S[mod=X1,form=X2,fin=f]/S[mod=X1,form=X2,fin=f])\\NP[case=nc,mod=nm,fin=f]")),
                "<",
                "<",
            ),
            Tree.make_binary(
                Category.parse("S[mod=nm,form=base,fin=f]"),
                Tree.make_binary(
                    Category.parse("NP[case=ga,mod=nm,fin=f]"),
                    Tree.make_terminal(
                        Token(word='政治', surf='政治', pos='名詞', pos1='一般', pos2='*', pos3='*',
                              inflectionForm='*', inflectionType='*', reading='セイジ', base='政治'),
                        Category.parse("NP[case=nc,mod=nm,fin=f]")),
                    Tree.make_terminal(
                        Token(word='が', surf='が', pos='助詞', pos1='格助詞', pos2='一般', pos3='*',
                              nflectionForm='*', inflectionType='*', reading='ガ', base='が'),
                        Category.parse("NP[case=ga,mod=nm,fin=f]\\NP[case=nc,mod=nm,fin=f]")),
                    "<",
                    "<",
                ),
                Tree.make_binary(
                    Category.parse(
                        "S[mod=nm,form=base,fin=f]\\NP[case=ga,mod=nm,fin=f]"),
                    Tree.make_terminal(
                        Token(word='わから', surf='わから', pos='動詞', pos1='自立', pos2='*', pos3='*',
                              inflectionForm='未然形', inflectionType='五段・ラ行', reading='ワカラ', base='わかる'),
                        Category.parse("S[mod=nm,form=neg,fin=f]\\NP[case=ga,mod=nm,fin=f]")),
                    Tree.make_terminal(
                        Token(word='ぬ', surf='ぬ', pos='助動詞', pos1='*', pos2='*', pos3='*',
                              inflectionForm='基本形', inflectionType='特殊・ヌ', reading='ヌ', base='ぬ'),
                        Category.parse("S[mod=nm,form=base,fin=f]\\S[mod=nm,form=neg,fin=f]")),
                    "<B1",
                    "<B1",
                ),
                "<",
                "<",
            ),
            ">",
            ">",
        ),
        Tree.make_terminal(
            Token(word='。', surf='。', pos='記号', pos1='句点', pos2='*', pos3='*',
                  inflectionForm='*', inflectionType='*', reading='。', base='。'),
            Category.parse("S[mod=nm,form=base,fin=t]\\S[mod=nm,form=base,fin=f]")),
        "<",
        "<",
    )


@pytest.fixture()
def scored_ja_tree(ja_tree):
    return ScoredTree(ja_tree, -0.05)


def test_en_auto(scored_en_tree):

    expected = (
        'ID=1, log probability=-0.05000000\n'
        '(<T S[dcl] 0 2> (<T S[dcl] 0 2> (<T NP 0 2> (<L NP[nb]/N DT DT This NP[nb]/N>)'
        ' (<L N NN NN paper N>) ) (<T S[dcl]\\NP 0 2> (<L (S[dcl]\\NP)/NP VBZ VBZ discusses'
        ' (S[dcl]\\NP)/NP>) (<T NP 0 2> (<T NP 0 1> (<T N 0 2> (<L N/N JJ JJ crucial N/N>)'
        ' (<L N NNS NNS aspects N>) ) ) (<T NP\\NP 0 2> (<L (NP\\NP)/NP IN IN of (NP\\NP)/NP>)'
        ' (<T NP 0 2> (<L NP[nb]/N DT DT this NP[nb]/N>) (<T N 0 2> (<L N/N JJ JJ new N/N>)'
        ' (<T N 0 2> (<L N/N NN NN annotation N/N>) (<L N NN NN scheme N>) ) ) ) ) ) ) ) (<L . . . . .>) )\n'
    )

    assert to_string([scored_en_tree], format='auto') == expected


def test_en_auto_extended(scored_en_tree):
    expected = (
        'ID=1, log probability=-0.05000000\n'
        '(<T S[dcl] rp 0 2> (<T S[dcl] ba 0 2> (<T NP fa 0 2> (<L NP[nb]/N This this DT O XX NP[nb]/N>)'
        ' (<L N paper paper NN O XX N>) ) (<T S[dcl]\\NP fa 0 2> (<L (S[dcl]\\NP)/NP discusses discuss'
        ' VBZ O XX (S[dcl]\\NP)/NP>) (<T NP ba 0 2> (<T NP lex 0 1> (<T N fa 0 2> (<L N/N crucial crucial'
        ' JJ O XX N/N>) (<L N aspects aspect NNS O XX N>) ) ) (<T NP\\NP fa 0 2> (<L (NP\\NP)/NP of of IN O'
        ' XX (NP\\NP)/NP>) (<T NP fa 0 2> (<L NP[nb]/N this this DT O XX NP[nb]/N>) (<T N fa 0 2> (<L N/N'
        ' new new JJ O XX N/N>) (<T N fa 0 2> (<L N/N annotation annotation NN O XX N/N>) (<L N scheme'
        ' scheme NN O XX N>) ) ) ) ) ) ) ) (<L . . . . O XX .>) )\n'
    )
    assert to_string([scored_en_tree], format='auto_extended') == expected


def test_en_xml(scored_en_tree):
    expected = (
        '<candc>\n'
        '  <ccg sentence="1" id="1">\n'
        '    <rule type="rp" cat="S[dcl]">\n'
        '      <rule type="ba" cat="S[dcl]">\n'
        '        <rule type="fa" cat="NP">\n'
        '          <lf start="0" span="1" cat="NP[nb]/N" word="This" pos="DT" entity="O" lemma="this" chunk="XX"/>\n'
        '          <lf start="1" span="1" cat="N" word="paper" pos="NN" entity="O" lemma="paper" chunk="XX"/>\n'
        '        </rule>\n'
        '        <rule type="fa" cat="S[dcl]\\NP">\n'
        '          <lf start="2" span="1" cat="(S[dcl]\\NP)/NP" word="discusses" pos="VBZ" entity="O" lemma="discuss" chunk="XX"/>\n'
        '          <rule type="ba" cat="NP">\n'
        '            <rule type="lex" cat="NP">\n'
        '              <rule type="fa" cat="N">\n'
        '                <lf start="3" span="1" cat="N/N" word="crucial" pos="JJ" entity="O" lemma="crucial" chunk="XX"/>\n'
        '                <lf start="4" span="1" cat="N" word="aspects" pos="NNS" entity="O" lemma="aspect" chunk="XX"/>\n'
        '              </rule>\n'
        '            </rule>\n'
        '            <rule type="fa" cat="NP\\NP">\n'
        '              <lf start="5" span="1" cat="(NP\\NP)/NP" word="of" pos="IN" entity="O" lemma="of" chunk="XX"/>\n'
        '              <rule type="fa" cat="NP">\n'
        '                <lf start="6" span="1" cat="NP[nb]/N" word="this" pos="DT" entity="O" lemma="this" chunk="XX"/>\n'
        '                <rule type="fa" cat="N">\n'
        '                  <lf start="7" span="1" cat="N/N" word="new" pos="JJ" entity="O" lemma="new" chunk="XX"/>\n'
        '                  <rule type="fa" cat="N">\n'
        '                    <lf start="8" span="1" cat="N/N" word="annotation" pos="NN" entity="O" lemma="annotation" chunk="XX"/>\n'
        '                    <lf start="9" span="1" cat="N" word="scheme" pos="NN" entity="O" lemma="scheme" chunk="XX"/>\n'
        '                  </rule>\n'
        '                </rule>\n'
        '              </rule>\n'
        '            </rule>\n'
        '          </rule>\n'
        '        </rule>\n'
        '      </rule>\n'
        '      <lf start="10" span="1" cat="." word="." pos="." entity="O" lemma="." chunk="XX"/>\n'
        '    </rule>\n'
        '  </ccg>\n'
        '</candc>\n'
    )
    assert to_string([scored_en_tree], format='xml') == expected


def test_en_deriv(scored_en_tree):

    expected = (
        'ID=1, log probability=-0.05000000\n'
        ' NP[nb]/N    N    (S[dcl]\\NP)/NP    N/N       N     (NP\\NP)/NP  NP[nb]/N  N/N     N/N        N     .\n'
        '   This    paper    discusses     crucial  aspects      of        this    new  annotation  scheme  .\n'
        '----------------->\n'
        '       NP\n'
        '                                 ------------------>\n'
        '                                         N\n'
        '                                 ------------------<un>\n'
        '                                         NP\n'
        '                                                                              -------------------->\n'
        '                                                                                       N\n'
        '                                                                         ------------------------->\n'
        '                                                                                     N\n'
        '                                                               ----------------------------------->\n'
        '                                                                               NP\n'
        '                                                   ----------------------------------------------->\n'
        '                                                                        NP\\NP\n'
        '                                 -----------------------------------------------------------------<\n'
        '                                                                NP\n'
        '                 --------------------------------------------------------------------------------->\n'
        '                                                     S[dcl]\\NP\n'
        '--------------------------------------------------------------------------------------------------<\n'
        '                                              S[dcl]\n'
        '-----------------------------------------------------------------------------------------------------<rp>\n'
        '                                               S[dcl]\n\n'
    )

    assert to_string([scored_en_tree], format='deriv') == expected


def test_en_prolog(scored_en_tree):

    expected = (
        ":- op(601, xfx, (/)).\n"
        ":- op(601, xfx, (\\)).\n"
        ":- multifile ccg/2, id/2.\n"
        ":- discontiguous ccg/2, id/2.\n"
        "\n"
        "ccg(1,\n"
        " rp(s:dcl,\n"
        "  ba(s:dcl,\n"
        "   fa(np,\n"
        "    t((np:nb/n), 'This', 'this', 'DT', 'XX', 'O'),\n"
        "    t(n, 'paper', 'paper', 'NN', 'XX', 'O')),\n"
        "   fa((s:dcl\\np),\n"
        "    t(((s:dcl\\np)/np), 'discusses', 'discuss', 'VBZ', 'XX', 'O'),\n"
        "    ba(np,\n"
        "     lx(np, n,\n"
        "      fa(n,\n"
        "       t((n/n), 'crucial', 'crucial', 'JJ', 'XX', 'O'),\n"
        "       t(n, 'aspects', 'aspect', 'NNS', 'XX', 'O'))),\n"
        "     fa((np\\np),\n"
        "      t(((np\\np)/np), 'of', 'of', 'IN', 'XX', 'O'),\n"
        "      fa(np,\n"
        "       t((np:nb/n), 'this', 'this', 'DT', 'XX', 'O'),\n"
        "       fa(n,\n"
        "        t((n/n), 'new', 'new', 'JJ', 'XX', 'O'),\n"
        "        fa(n,\n"
        "         t((n/n), 'annotation', 'annotation', 'NN', 'XX', 'O'),\n"
        "         t(n, 'scheme', 'scheme', 'NN', 'XX', 'O')))))))),\n"
        "  t(period, '.', '.', '.', 'XX', 'O'))).\n\n"
    )

    assert to_string([scored_en_tree], format='prolog') == expected


def test_en_jigg_xml(scored_en_tree):

    expected = (
        '<root>\n'
        '  <document>\n'
        '    <sentences>\n'
        '      <sentence>\n'
        '        <tokens>\n'
        '          <token start="0" cat="NP[nb]/N" id="s0_0" pos="DT" entity="O" chunk="XX" surf="This" base="this"/>\n'
        '          <token start="1" cat="N" id="s0_1" pos="NN" entity="O" chunk="XX" surf="paper" base="paper"/>\n'
        '          <token start="2" cat="(S[dcl]\\NP)/NP" id="s0_2" pos="VBZ" entity="O" chunk="XX" surf="discusses" base="discuss"/>\n'
        '          <token start="3" cat="N/N" id="s0_3" pos="JJ" entity="O" chunk="XX" surf="crucial" base="crucial"/>\n'
        '          <token start="4" cat="N" id="s0_4" pos="NNS" entity="O" chunk="XX" surf="aspects" base="aspect"/>\n'
        '          <token start="5" cat="(NP\\NP)/NP" id="s0_5" pos="IN" entity="O" chunk="XX" surf="of" base="of"/>\n'
        '          <token start="6" cat="NP[nb]/N" id="s0_6" pos="DT" entity="O" chunk="XX" surf="this" base="this"/>\n'
        '          <token start="7" cat="N/N" id="s0_7" pos="JJ" entity="O" chunk="XX" surf="new" base="new"/>\n'
        '          <token start="8" cat="N/N" id="s0_8" pos="NN" entity="O" chunk="XX" surf="annotation" base="annotation"/>\n'
        '          <token start="9" cat="N" id="s0_9" pos="NN" entity="O" chunk="XX" surf="scheme" base="scheme"/>\n'
        '          <token start="10" cat="." id="s0_10" pos="." entity="O" chunk="XX" surf="." base="."/>\n'
        '        </tokens>\n'
        '        <ccg id="s0_ccg0" root="s0_sp0" score="-0.05">\n'
        '          <span category="S[dcl=true]" id="s0_sp0" child="s0_sp1 s0_sp21" rule="rp" begin="0" end="11" root="true"/>\n'
        '          <span category="S[dcl=true]" id="s0_sp1" child="s0_sp2 s0_sp5" rule="ba" begin="0" end="10"/>\n'
        '          <span category="NP" id="s0_sp2" child="s0_sp3 s0_sp4" rule="fa" begin="0" end="2"/>\n'
        '          <span category="NP[nb=true]/N" id="s0_sp3" terminal="s0_0" begin="0" end="1"/>\n'
        '          <span category="N" id="s0_sp4" terminal="s0_1" begin="1" end="2"/>\n'
        '          <span category="S[dcl=true]\\NP" id="s0_sp5" child="s0_sp6 s0_sp7" rule="fa" begin="2" end="10"/>\n'
        '          <span category="(S[dcl=true]\\NP)/NP" id="s0_sp6" terminal="s0_2" begin="2" end="3"/>\n'
        '          <span category="NP" id="s0_sp7" child="s0_sp8 s0_sp12" rule="ba" begin="3" end="10"/>\n'
        '          <span category="NP" id="s0_sp8" child="s0_sp9" rule="lex" begin="3" end="5"/>\n'
        '          <span category="N" id="s0_sp9" child="s0_sp10 s0_sp11" rule="fa" begin="3" end="5"/>\n'
        '          <span category="N/N" id="s0_sp10" terminal="s0_3" begin="3" end="4"/>\n'
        '          <span category="N" id="s0_sp11" terminal="s0_4" begin="4" end="5"/>\n'
        '          <span category="NP\\NP" id="s0_sp12" child="s0_sp13 s0_sp14" rule="fa" begin="5" end="10"/>\n'
        '          <span category="(NP\\NP)/NP" id="s0_sp13" terminal="s0_5" begin="5" end="6"/>\n'
        '          <span category="NP" id="s0_sp14" child="s0_sp15 s0_sp16" rule="fa" begin="6" end="10"/>\n'
        '          <span category="NP[nb=true]/N" id="s0_sp15" terminal="s0_6" begin="6" end="7"/>\n'
        '          <span category="N" id="s0_sp16" child="s0_sp17 s0_sp18" rule="fa" begin="7" end="10"/>\n'
        '          <span category="N/N" id="s0_sp17" terminal="s0_7" begin="7" end="8"/>\n'
        '          <span category="N" id="s0_sp18" child="s0_sp19 s0_sp20" rule="fa" begin="8" end="10"/>\n'
        '          <span category="N/N" id="s0_sp19" terminal="s0_8" begin="8" end="9"/>\n'
        '          <span category="N" id="s0_sp20" terminal="s0_9" begin="9" end="10"/>\n'
        '          <span category="." id="s0_sp21" terminal="s0_10" begin="10" end="11"/>\n'
        '        </ccg>\n'
        '      </sentence>\n'
        '    </sentences>\n'
        '  </document>\n'
        '</root>\n'
    )

    assert to_string([scored_en_tree], format='jigg_xml') == expected


def test_en_jigg_xml_ccg2lambda(scored_en_tree):

    expected = (
        '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n'
        '<root>\n'
        '  <document>\n'
        '    <sentences>\n'
        '      <sentence>\n'
        '        <tokens>\n'
        '          <token start="0" cat="NP[nb]/N" id="s0_0" pos="DT" entity="O" chunk="XX" surf="This" base="this"/>\n'
        '          <token start="1" cat="N" id="s0_1" pos="NN" entity="O" chunk="XX" surf="paper" base="paper"/>\n'
        '          <token start="2" cat="(S[dcl]\\NP)/NP" id="s0_2" pos="VBZ" entity="O" chunk="XX" surf="discusses" base="discuss"/>\n'
        '          <token start="3" cat="N/N" id="s0_3" pos="JJ" entity="O" chunk="XX" surf="crucial" base="crucial"/>\n'
        '          <token start="4" cat="N" id="s0_4" pos="NNS" entity="O" chunk="XX" surf="aspects" base="aspect"/>\n'
        '          <token start="5" cat="(NP\\NP)/NP" id="s0_5" pos="IN" entity="O" chunk="XX" surf="of" base="of"/>\n'
        '          <token start="6" cat="NP[nb]/N" id="s0_6" pos="DT" entity="O" chunk="XX" surf="this" base="this"/>\n'
        '          <token start="7" cat="N/N" id="s0_7" pos="JJ" entity="O" chunk="XX" surf="new" base="new"/>\n'
        '          <token start="8" cat="N/N" id="s0_8" pos="NN" entity="O" chunk="XX" surf="annotation" base="annotation"/>\n'
        '          <token start="9" cat="N" id="s0_9" pos="NN" entity="O" chunk="XX" surf="scheme" base="scheme"/>\n'
        '          <token start="10" cat="." id="s0_10" pos="." entity="O" chunk="XX" surf="." base="."/>\n'
        '        </tokens>\n'
        '        <ccg id="s0_ccg0" root="s0_sp0" score="-0.05">\n'
        '          <span category="S[dcl=true]" id="s0_sp0" child="s0_sp1 s0_sp21" rule="rp" begin="0" end="11" root="true"/>\n'
        '          <span category="S[dcl=true]" id="s0_sp1" child="s0_sp2 s0_sp5" rule="ba" begin="0" end="10"/>\n'
        '          <span category="NP" id="s0_sp2" child="s0_sp3 s0_sp4" rule="fa" begin="0" end="2"/>\n'
        '          <span category="NP[nb=true]/N" id="s0_sp3" terminal="s0_0" begin="0" end="1"/>\n'
        '          <span category="N" id="s0_sp4" terminal="s0_1" begin="1" end="2"/>\n'
        '          <span category="S[dcl=true]\\NP" id="s0_sp5" child="s0_sp6 s0_sp7" rule="fa" begin="2" end="10"/>\n'
        '          <span category="(S[dcl=true]\\NP)/NP" id="s0_sp6" terminal="s0_2" begin="2" end="3"/>\n'
        '          <span category="NP" id="s0_sp7" child="s0_sp8 s0_sp12" rule="ba" begin="3" end="10"/>\n'
        '          <span category="NP" id="s0_sp8" child="s0_sp9" rule="lex" begin="3" end="5"/>\n'
        '          <span category="N" id="s0_sp9" child="s0_sp10 s0_sp11" rule="fa" begin="3" end="5"/>\n'
        '          <span category="N/N" id="s0_sp10" terminal="s0_3" begin="3" end="4"/>\n'
        '          <span category="N" id="s0_sp11" terminal="s0_4" begin="4" end="5"/>\n'
        '          <span category="NP\\NP" id="s0_sp12" child="s0_sp13 s0_sp14" rule="fa" begin="5" end="10"/>\n'
        '          <span category="(NP\\NP)/NP" id="s0_sp13" terminal="s0_5" begin="5" end="6"/>\n'
        '          <span category="NP" id="s0_sp14" child="s0_sp15 s0_sp16" rule="fa" begin="6" end="10"/>\n'
        '          <span category="NP[nb=true]/N" id="s0_sp15" terminal="s0_6" begin="6" end="7"/>\n'
        '          <span category="N" id="s0_sp16" child="s0_sp17 s0_sp18" rule="fa" begin="7" end="10"/>\n'
        '          <span category="N/N" id="s0_sp17" terminal="s0_7" begin="7" end="8"/>\n'
        '          <span category="N" id="s0_sp18" child="s0_sp19 s0_sp20" rule="fa" begin="8" end="10"/>\n'
        '          <span category="N/N" id="s0_sp19" terminal="s0_8" begin="8" end="9"/>\n'
        '          <span category="N" id="s0_sp20" terminal="s0_9" begin="9" end="10"/>\n'
        '          <span category="." id="s0_sp21" terminal="s0_10" begin="10" end="11"/>\n'
        '        </ccg>\n'
        '        <semantics status="success" ccg_id="s0_ccg0" root="s0_sp0">\n'
        '          <span id="s0_sp0" child="s0_sp1 s0_sp21" sem="exists x.(_paper(x) &amp; True &amp; exists z2.(_aspect(z2) &amp; _crucial(z2) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (z2 = z1)) &amp; True &amp; exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = z2) &amp; True)))"/>\n'
        '          <span id="s0_sp1" child="s0_sp2 s0_sp5" sem="\\K.exists x.(_paper(x) &amp; True &amp; exists z2.(_aspect(z2) &amp; _crucial(z2) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (z2 = z1)) &amp; True &amp; exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = z2) &amp; K(e))))"/>\n'
        '          <span id="s0_sp2" child="s0_sp3 s0_sp4" sem="\\F2 F3.exists x.(_paper(x) &amp; F2(x) &amp; F3(x))"/>\n'
        '          <span id="s0_sp3" sem="\\F1 F2 F3.exists x.(F1(x) &amp; F2(x) &amp; F3(x))"/>\n'
        '          <span id="s0_sp4" sem="\\x._paper(x)" type="_paper : Entity -&gt; Prop"/>\n'
        '          <span id="s0_sp5" child="s0_sp6 s0_sp7" sem="\\Q2 K.Q2(\\x.True,\\x.exists z2.(_aspect(z2) &amp; _crucial(z2) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (z2 = z1)) &amp; True &amp; exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = z2) &amp; K(e))))"/>\n'
        '          <span id="s0_sp6" sem="\\Q1 Q2 K.Q2(\\x.True,\\x.Q1(\\y.True,\\y.exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = y) &amp; K(e))))" type="_discuss : Event -&gt; Prop"/>\n'
        '          <span id="s0_sp7" child="s0_sp8 s0_sp12" sem="\\F1 F2.exists x.(_aspect(x) &amp; _crucial(x) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (x = z1)) &amp; F1(x) &amp; F2(x))"/>\n'
        '          <span id="s0_sp8" child="s0_sp9" sem="\\F1 F2.exists x.(_aspect(x) &amp; _crucial(x) &amp; F1(x) &amp; F2(x))"/>\n'
        '          <span id="s0_sp9" child="s0_sp10 s0_sp11" sem="\\x.(_aspect(x) &amp; _crucial(x))"/>\n'
        '          <span id="s0_sp10" sem="\\F x.(F(x) &amp; _crucial(x))" type="_crucial : Entity -&gt; Prop"/>\n'
        '          <span id="s0_sp11" sem="\\x._aspect(x)" type="_aspect : Entity -&gt; Prop"/>\n'
        '          <span id="s0_sp12" child="s0_sp13 s0_sp14" sem="\\Q2 F1 F2.Q2(\\x.(exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (x = z1)) &amp; F1(x)),F2)"/>\n'
        '          <span id="s0_sp13" sem="\\Q1 Q2 F1 F2.Q2(\\x.(Q1(\\w.True,\\y.(x = y)) &amp; F1(x)),F2)"/>\n'
        '          <span id="s0_sp14" child="s0_sp15 s0_sp16" sem="\\F2 F3.exists x.(_scheme(x) &amp; _annotation(x) &amp; _new(x) &amp; F2(x) &amp; F3(x))"/>\n'
        '          <span id="s0_sp15" sem="\\F1 F2 F3.exists x.(F1(x) &amp; F2(x) &amp; F3(x))"/>\n'
        '          <span id="s0_sp16" child="s0_sp17 s0_sp18" sem="\\x.(_scheme(x) &amp; _annotation(x) &amp; _new(x))"/>\n'
        '          <span id="s0_sp17" sem="\\F x.(F(x) &amp; _new(x))" type="_new : Entity -&gt; Prop"/>\n'
        '          <span id="s0_sp18" child="s0_sp19 s0_sp20" sem="\\x.(_scheme(x) &amp; _annotation(x))"/>\n'
        '          <span id="s0_sp19" sem="\\F x.(F(x) &amp; _annotation(x))" type="_annotation : Entity -&gt; Prop"/>\n'
        '          <span id="s0_sp20" sem="\\x._scheme(x)" type="_scheme : Entity -&gt; Prop"/>\n'
        '          <span id="s0_sp21" sem="\\X.X"/>\n'
        '        </semantics>\n'
        '      </sentence>\n'
        '    </sentences>\n'
        '  </document>\n'
        '</root>\n'
    )

    assert (
        to_string(
            [scored_en_tree],
            format='jigg_xml_ccg2lambda',
            semantic_templates=SEMANTIC_TEMPLATES['en'],
        ) == expected
    )


def test_en_conll(scored_en_tree):

    expected = (
        '# ID=1\n'
        '# log probability=-0.05000000\n'
        '1	This	this	DT	DT	_	0	NP[nb]/N	_	(<T S[dcl] 0 2> (<T S[dcl] 0 2> (<T NP 0 2> (<L NP[nb]/N DT DT This NP[nb]/N>)\n'
        '2	paper	paper	NN	NN	_	1	N	_	(<L N NN NN paper N>) )\n'
        '3	discusses	discuss	VBZ	VBZ	_	1	(S[dcl]\\NP)/NP	_	(<T S[dcl]\\NP 0 2> (<L (S[dcl]\\NP)/NP VBZ VBZ discusses (S[dcl]\\NP)/NP>)\n'
        '4	crucial	crucial	JJ	JJ	_	3	N/N	_	(<T NP 0 2> (<T NP 0 1> (<T N 0 2> (<L N/N JJ JJ crucial N/N>)\n'
        '5	aspects	aspect	NNS	NNS	_	4	N	_	(<L N NNS NNS aspects N>) ) )\n'
        '6	of	of	IN	IN	_	4	(NP\\NP)/NP	_	(<T NP\\NP 0 2> (<L (NP\\NP)/NP IN IN of (NP\\NP)/NP>)\n'
        '7	this	this	DT	DT	_	6	NP[nb]/N	_	(<T NP 0 2> (<L NP[nb]/N DT DT this NP[nb]/N>)\n'
        '8	new	new	JJ	JJ	_	7	N/N	_	(<T N 0 2> (<L N/N JJ JJ new N/N>)\n'
        '9	annotation	annotation	NN	NN	_	8	N/N	_	(<T N 0 2> (<L N/N NN NN annotation N/N>)\n'
        '10	scheme	scheme	NN	NN	_	9	N	_	(<L N NN NN scheme N>) ) ) ) ) ) ) )\n'
        '11	.	.	.	.	_	1	.	_	(<L . . . . .>) )\n'
    )

    assert to_string([scored_en_tree], format='conll') == expected


def test_en_json(scored_en_tree):

    expected = (
        '{\n'
        '    "1": [\n'
        '        {\n'
        '            "type": "rp",\n'
        '            "cat": "S[dcl]",\n'
        '            "children": [\n'
        '                {\n'
        '                    "type": "ba",\n'
        '                    "cat": "S[dcl]",\n'
        '                    "children": [\n'
        '                        {\n'
        '                            "type": "fa",\n'
        '                            "cat": "NP",\n'
        '                            "children": [\n'
        '                                {\n'
        '                                    "word": "This",\n'
        '                                    "pos": "DT",\n'
        '                                    "entity": "O",\n'
        '                                    "lemma": "this",\n'
        '                                    "chunk": "XX",\n'
        '                                    "cat": "NP[nb]/N"\n'
        '                                },\n'
        '                                {\n'
        '                                    "word": "paper",\n'
        '                                    "pos": "NN",\n'
        '                                    "entity": "O",\n'
        '                                    "lemma": "paper",\n'
        '                                    "chunk": "XX",\n'
        '                                    "cat": "N"\n'
        '                                }\n'
        '                            ]\n'
        '                        },\n'
        '                        {\n'
        '                            "type": "fa",\n'
        '                            "cat": "S[dcl]\\\\NP",\n'
        '                            "children": [\n'
        '                                {\n'
        '                                    "word": "discusses",\n'
        '                                    "pos": "VBZ",\n'
        '                                    "entity": "O",\n'
        '                                    "lemma": "discuss",\n'
        '                                    "chunk": "XX",\n'
        '                                    "cat": "(S[dcl]\\\\NP)/NP"\n'
        '                                },\n'
        '                                {\n'
        '                                    "type": "ba",\n'
        '                                    "cat": "NP",\n'
        '                                    "children": [\n'
        '                                        {\n'
        '                                            "type": "lex",\n'
        '                                            "cat": "NP",\n'
        '                                            "children": [\n'
        '                                                {\n'
        '                                                    "type": "fa",\n'
        '                                                    "cat": "N",\n'
        '                                                    "children": [\n'
        '                                                        {\n'
        '                                                            "word": "crucial",\n'
        '                                                            "pos": "JJ",\n'
        '                                                            "entity": "O",\n'
        '                                                            "lemma": "crucial",\n'
        '                                                            "chunk": "XX",\n'
        '                                                            "cat": "N/N"\n'
        '                                                        },\n'
        '                                                        {\n'
        '                                                            "word": "aspects",\n'
        '                                                            "pos": "NNS",\n'
        '                                                            "entity": "O",\n'
        '                                                            "lemma": "aspect",\n'
        '                                                            "chunk": "XX",\n'
        '                                                            "cat": "N"\n'
        '                                                        }\n'
        '                                                    ]\n'
        '                                                }\n'
        '                                            ]\n'
        '                                        },\n'
        '                                        {\n'
        '                                            "type": "fa",\n'
        '                                            "cat": "NP\\\\NP",\n'
        '                                            "children": [\n'
        '                                                {\n'
        '                                                    "word": "of",\n'
        '                                                    "pos": "IN",\n'
        '                                                    "entity": "O",\n'
        '                                                    "lemma": "of",\n'
        '                                                    "chunk": "XX",\n'
        '                                                    "cat": "(NP\\\\NP)/NP"\n'
        '                                                },\n'
        '                                                {\n'
        '                                                    "type": "fa",\n'
        '                                                    "cat": "NP",\n'
        '                                                    "children": [\n'
        '                                                        {\n'
        '                                                            "word": "this",\n'
        '                                                            "pos": "DT",\n'
        '                                                            "entity": "O",\n'
        '                                                            "lemma": "this",\n'
        '                                                            "chunk": "XX",\n'
        '                                                            "cat": "NP[nb]/N"\n'
        '                                                        },\n'
        '                                                        {\n'
        '                                                            "type": "fa",\n'
        '                                                            "cat": "N",\n'
        '                                                            "children": [\n'
        '                                                                {\n'
        '                                                                    "word": "new",\n'
        '                                                                    "pos": "JJ",\n'
        '                                                                    "entity": "O",\n'
        '                                                                    "lemma": "new",\n'
        '                                                                    "chunk": "XX",\n'
        '                                                                    "cat": "N/N"\n'
        '                                                                },\n'
        '                                                                {\n'
        '                                                                    "type": "fa",\n'
        '                                                                    "cat": "N",\n'
        '                                                                    "children": [\n'
        '                                                                        {\n'
        '                                                                            "word": "annotation",\n'
        '                                                                            "pos": "NN",\n'
        '                                                                            "entity": "O",\n'
        '                                                                            "lemma": "annotation",\n'
        '                                                                            "chunk": "XX",\n'
        '                                                                            "cat": "N/N"\n'
        '                                                                        },\n'
        '                                                                        {\n'
        '                                                                            "word": "scheme",\n'
        '                                                                            "pos": "NN",\n'
        '                                                                            "entity": "O",\n'
        '                                                                            "lemma": "scheme",\n'
        '                                                                            "chunk": "XX",\n'
        '                                                                            "cat": "N"\n'
        '                                                                        }\n'
        '                                                                    ]\n'
        '                                                                }\n'
        '                                                            ]\n'
        '                                                        }\n'
        '                                                    ]\n'
        '                                                }\n'
        '                                            ]\n'
        '                                        }\n'
        '                                    ]\n'
        '                                }\n'
        '                            ]\n'
        '                        }\n'
        '                    ]\n'
        '                },\n'
        '                {\n'
        '                    "word": ".",\n'
        '                    "pos": ".",\n'
        '                    "entity": "O",\n'
        '                    "lemma": ".",\n'
        '                    "chunk": "XX",\n'
        '                    "cat": "."\n'
        '                }\n'
        '            ],\n'
        '            "log_prob": -0.05\n'
        '        }\n'
        '    ]\n'
        '}'
    )

    assert to_string([scored_en_tree], format='json') == expected


def test_en_ptb(scored_en_tree):

    expected = (
        'ID=1, log probability=-0.05000000\n'
        '(ROOT (S[dcl] (S[dcl] (NP (NP[nb]/N This) (N paper)) (S[dcl]\\NP ((S[dcl]\\NP)/NP discusses)'
        ' (NP (NP (N (N/N crucial) (N aspects))) (NP\\NP ((NP\\NP)/NP of) (NP (NP[nb]/N this) (N (N/N new)'
        ' (N (N/N annotation) (N scheme)))))))) (. .)))\n'
    )

    assert to_string([scored_en_tree], format='ptb') == expected


def test_ja_ja(scored_ja_tree):

    expected = (
        'ID=1, log probability=-0.05000000\n'
        '{< S[mod=nm,form=base,fin=t] {> S[mod=nm,form=base,fin=f] {< S[mod=X1,form=X2,fin=f]/S[mod=X1,form=X2,fin=f]'
        ' {< NP[case=nc,mod=nm,fin=f] {NP[case=nc,mod=nm,fin=f] メロス/メロス/名詞-一般/_}'
        ' {NP[case=nc,mod=nm,fin=f]\\NP[case=nc,mod=nm,fin=f] に/に/助詞-格助詞-一般/_}}'
        ' {(S[mod=X1,form=X2,fin=f]/S[mod=X1,form=X2,fin=f])\\NP[case=nc,mod=nm,fin=f] は/は/助詞-係助詞/_}}'
        ' {< S[mod=nm,form=base,fin=f] {< NP[case=ga,mod=nm,fin=f] {NP[case=nc,mod=nm,fin=f] 政治/政治/名詞-一般/_}'
        ' {NP[case=ga,mod=nm,fin=f]\\NP[case=nc,mod=nm,fin=f] が/が/助詞-格助詞-一般/_}}'
        ' {<B1 S[mod=nm,form=base,fin=f]\\NP[case=ga,mod=nm,fin=f] {S[mod=nm,form=neg,fin=f]\\NP[case=ga,mod=nm,fin=f]'
        ' わから/わから/動詞-自立/未然形-五段・ラ行} {S[mod=nm,form=base,fin=f]\\S[mod=nm,form=neg,fin=f]'
        ' ぬ/ぬ/助動詞/基本形-特殊・ヌ}}}} {S[mod=nm,form=base,fin=t]\\S[mod=nm,form=base,fin=f] 。/。/記号-句点/_}}\n'
    )

    assert to_string([scored_ja_tree], format='ja') == expected


def test_ja_prolog(scored_ja_tree):

    expected = (
        ":- op(601, xfx, (/)).\n"
        ":- op(601, xfx, (\\)).\n"
        ":- multifile ccg/2, id/2.\n"
        ":- discontiguous ccg/2, id/2.\n"
        "\n"
        "ccg(1,\n"
        " ba(s,\n"
        "  fa(s,\n"
        "   ba((s/s),\n"
        "    ba(np:nc,\n"
        "     t(np:nc, 'メロス', 'メロス', '名詞/一般/*/*', '*', '*'),\n"
        "     t((np:nc\\np:nc), 'に', 'に', '助詞/格助詞/一般/*', '*', '*')),\n"
        "    t(((s/s)\\np:nc), 'は', 'は', '助詞/係助詞/*/*', '*', '*')),\n"
        "   ba(s,\n"
        "    ba(np:ga,\n"
        "     t(np:nc, '政治', '政治', '名詞/一般/*/*', '*', '*'),\n"
        "     t((np:ga\\np:nc), 'が', 'が', '助詞/格助詞/一般/*', '*', '*')),\n"
        "    bc1((s\\np:ga),\n"
        "     t((s\\np:ga), 'わから', 'わかる', '動詞/自立/*/*', '未然形', '五段・ラ行'),\n"
        "     t((s\\s), 'ぬ', 'ぬ', '助動詞/*/*/*', '基本形', '特殊・ヌ')))),\n"
        "  t((s\\s), '。', '。', '記号/句点/*/*', '*', '*'))).\n\n"
    )
    set_global_language_to('ja')
    assert to_string([scored_ja_tree], format='prolog') == expected
