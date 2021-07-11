
from depccg.cat import Category
from depccg.types import Token
from depccg.tree import Tree

Tree.make_terminal(Category.parse("NP[nb]/N"), Token(word="This" pos="DT" entity="O" lemma="this" chunk="XX"))
Tree.make_terminal(Category.parse("N"), Token(word="paper" pos="NN" entity="O" lemma="paper" chunk="XX"))
Tree.make_terminal(Category.parse("(S[dcl]\NP)/NP"), Token(word="discusses" pos="VBZ" entity="O" lemma="discuss" chunk="XX"))
Tree.make_terminal(Category.parse("N/N"), Token(word="crucial" pos="JJ" entity="O" lemma="crucial" chunk="XX"))
Tree.make_terminal(Category.parse("N"), Token(word="aspects" pos="NNS" entity="O" lemma="aspect" chunk="XX"))
Tree.make_terminal(Category.parse("(NP\NP)/NP"), Token(word="of" pos="IN" entity="O" lemma="of" chunk="XX"))
Tree.make_terminal(Category.parse("NP[nb]/N"), Token(word="this" pos="DT" entity="O" lemma="this" chunk="XX"))
Tree.make_terminal(Category.parse("N/N"), Token(word="new" pos="JJ" entity="O" lemma="new" chunk="XX"))
Tree.make_terminal(Category.parse("N/N"), Token(word="annotation" pos="NN" entity="O" lemma="annotation" chunk="XX"))
Tree.make_terminal(Category.parse("N"), Token(word="scheme" pos="NN" entity="O" lemma="scheme" chunk="XX"))
Tree.make_terminal(Category.parse("."), Token(word="." pos="." entity="O" lemma="." chunk="XX")



(r'ID=1, log probability=-0.10101445019245148'
 r'(<T S[dcl] 0 2> (<T S[dcl] 0 2> (<T NP 0 2> (<L NP[nb]/N DT DT This NP[nb]/N>)'
 r' (<L N NN NN paper N>) ) (<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/NP VBZ VBZ discusses'
 r' (S[dcl]\NP)/NP>) (<T NP 0 2> (<T NP 0 1> (<T N 0 2> (<L N/N JJ JJ crucial N/N>)'
 r' (<L N NNS NNS aspects N>) ) ) (<T NP\NP 0 2> (<L (NP\NP)/NP IN IN of (NP\NP)/NP>)'
 r' (<T NP 0 2> (<L NP[nb]/N DT DT this NP[nb]/N>) (<T N 0 2> (<L N/N JJ JJ new N/N>)'
 r' (<T N 0 2> (<L N/N NN NN annotation N/N>) (<L N NN NN scheme N>) ) ) ) ) ) ) ) (<L . . . . .>) ))'

(r'ID=1, log probability=-0.10101445019245148'
 r'(<T S[dcl] rp 0 2> (<T S[dcl] ba 0 2> (<T NP fa 0 2> (<L NP[nb]/N This this DT O XX NP[nb]/N>)'
 r' (<L N paper paper NN O XX N>) ) (<T S[dcl]\NP fa 0 2> (<L (S[dcl]\NP)/NP discusses discuss'
 r' VBZ O XX (S[dcl]\NP)/NP>) (<T NP ba 0 2> (<T NP lex 0 1> (<T N fa 0 2> (<L N/N crucial crucial'
 r' JJ O XX N/N>) (<L N aspects aspect NNS O XX N>) ) ) (<T NP\NP fa 0 2> (<L (NP\NP)/NP of of IN O'
 r' XX (NP\NP)/NP>) (<T NP fa 0 2> (<L NP[nb]/N this this DT O XX NP[nb]/N>) (<T N fa 0 2> (<L N/N'
 r' new new JJ O XX N/N>) (<T N fa 0 2> (<L N/N annotation annotation NN O XX N/N>) (<L N scheme'
 r' scheme NN O XX N>) ) ) ) ) ) ) ) (<L . . . . O XX .>) )')

(r'<candc>'
 r'  <ccg sentence="1" id="1">'
 r'    <rule type="rp" cat="S[dcl]">'
 r'      <rule type="ba" cat="S[dcl]">'
 r'        <rule type="fa" cat="NP">'
 r'          <lf start="0" span="1" cat="NP[nb]/N" word="This" pos="DT" entity="O" lemma="this" chunk="XX"/>'
 r'          <lf start="1" span="1" cat="N" word="paper" pos="NN" entity="O" lemma="paper" chunk="XX"/>'
 r'        </rule>'
 r'        <rule type="fa" cat="S[dcl]\NP">'
 r'          <lf start="2" span="1" cat="(S[dcl]\NP)/NP" word="discusses" pos="VBZ" entity="O" lemma="discuss" chunk="XX"/>'
 r'          <rule type="ba" cat="NP">'
 r'            <rule type="lex" cat="NP">'
 r'              <rule type="fa" cat="N">'
 r'                <lf start="3" span="1" cat="N/N" word="crucial" pos="JJ" entity="O" lemma="crucial" chunk="XX"/>'
 r'                <lf start="4" span="1" cat="N" word="aspects" pos="NNS" entity="O" lemma="aspect" chunk="XX"/>'
 r'              </rule>'
 r'            </rule>'
 r'            <rule type="fa" cat="NP\NP">'
 r'              <lf start="5" span="1" cat="(NP\NP)/NP" word="of" pos="IN" entity="O" lemma="of" chunk="XX"/>'
 r'              <rule type="fa" cat="NP">'
 r'                <lf start="6" span="1" cat="NP[nb]/N" word="this" pos="DT" entity="O" lemma="this" chunk="XX"/>'
 r'                <rule type="fa" cat="N">'
 r'                  <lf start="7" span="1" cat="N/N" word="new" pos="JJ" entity="O" lemma="new" chunk="XX"/>'
 r'                  <rule type="fa" cat="N">'
 r'                    <lf start="8" span="1" cat="N/N" word="annotation" pos="NN" entity="O" lemma="annotation" chunk="XX"/>'
 r'                    <lf start="9" span="1" cat="N" word="scheme" pos="NN" entity="O" lemma="scheme" chunk="XX"/>'
 r'                  </rule>'
 r'                </rule>'
 r'              </rule>'
 r'            </rule>'
 r'          </rule>'
 r'        </rule>'
 r'      </rule>'
 r'      <lf start="10" span="1" cat="." word="." pos="." entity="O" lemma="." chunk="XX"/>'
 r'    </rule>'
 r'  </ccg>'
 r'</candc>')

(r'ID=1, log probability=-0.10101445019245148'
 r' NP[nb]/N    N    (S[dcl]\NP)/NP    N/N       N     (NP\NP)/NP  NP[nb]/N  N/N     N/N        N     .'
 r'   This    paper    discusses     crucial  aspects      of        this    new  annotation  scheme  .'
 r'----------------->'
 r'       NP'
 r'                                 ------------------>'
 r'                                         N'
 r'                                 ------------------<un>'
 r'                                         NP'
 r'                                                                              -------------------->'
 r'                                                                                       N'
 r'                                                                         ------------------------->'
 r'                                                                                     N'
 r'                                                               ----------------------------------->'
 r'                                                                               NP'
 r'                                                   ----------------------------------------------->'
 r'                                                                        NP\NP'
 r'                                 -----------------------------------------------------------------<'
 r'                                                                NP'
 r'                 --------------------------------------------------------------------------------->'
 r'                                                     S[dcl]\NP'
 r'--------------------------------------------------------------------------------------------------<'
 r'                                              S[dcl]'
 r'-----------------------------------------------------------------------------------------------------<rp>'
 r'                                               S[dcl]')


(r":- op(601, xfx, (/))."
 r":- op(601, xfx, (\))."
 r":- multifile ccg/2, id/2."
 r":- discontiguous ccg/2, id/2."
 r""
 r"ccg(1,"
 r" rp(s:dcl,"
 r"  ba(s:dcl,"
 r"   fa(np,"
 r"    t((np:nb/n), 'This', 'this', 'DT', 'XX', 'O'),"
 r"    t(n, 'paper', 'paper', 'NN', 'XX', 'O')),"
 r"   fa((s:dcl\np),"
 r"    t(((s:dcl\np)/np), 'discusses', 'discuss', 'VBZ', 'XX', 'O'),"
 r"    ba(np,"
 r"     lx(np, n,"
 r"      fa(n,"
 r"       t((n/n), 'crucial', 'crucial', 'JJ', 'XX', 'O'),"
 r"       t(n, 'aspects', 'aspect', 'NNS', 'XX', 'O'))),"
 r"     fa((np\np),"
 r"      t(((np\np)/np), 'of', 'of', 'IN', 'XX', 'O'),"
 r"      fa(np,"
 r"       t((np:nb/n), 'this', 'this', 'DT', 'XX', 'O'),"
 r"       fa(n,"
 r"        t((n/n), 'new', 'new', 'JJ', 'XX', 'O'),"
 r"        fa(n,"
 r"         t((n/n), 'annotation', 'annotation', 'NN', 'XX', 'O'),"
 r"         t(n, 'scheme', 'scheme', 'NN', 'XX', 'O')))))))),"
 r"  t(period, '.', '.', '.', 'XX', 'O'))).")

(r'<root>'
 r'  <document>'
 r'    <sentences>'
 r'      <sentence>'
 r'        <tokens>'
 r'          <token start="0" cat="NP[nb]/N" id="s0_0" pos="DT" entity="O" chunk="XX" surf="This" base="this"/>'
 r'          <token start="1" cat="N" id="s0_1" pos="NN" entity="O" chunk="XX" surf="paper" base="paper"/>'
 r'          <token start="2" cat="(S[dcl]\NP)/NP" id="s0_2" pos="VBZ" entity="O" chunk="XX" surf="discusses" base="discuss"/>'
 r'          <token start="3" cat="N/N" id="s0_3" pos="JJ" entity="O" chunk="XX" surf="crucial" base="crucial"/>'
 r'          <token start="4" cat="N" id="s0_4" pos="NNS" entity="O" chunk="XX" surf="aspects" base="aspect"/>'
 r'          <token start="5" cat="(NP\NP)/NP" id="s0_5" pos="IN" entity="O" chunk="XX" surf="of" base="of"/>'
 r'          <token start="6" cat="NP[nb]/N" id="s0_6" pos="DT" entity="O" chunk="XX" surf="this" base="this"/>'
 r'          <token start="7" cat="N/N" id="s0_7" pos="JJ" entity="O" chunk="XX" surf="new" base="new"/>'
 r'          <token start="8" cat="N/N" id="s0_8" pos="NN" entity="O" chunk="XX" surf="annotation" base="annotation"/>'
 r'          <token start="9" cat="N" id="s0_9" pos="NN" entity="O" chunk="XX" surf="scheme" base="scheme"/>'
 r'          <token start="10" cat="." id="s0_10" pos="." entity="O" chunk="XX" surf="." base="."/>'
 r'        </tokens>'
 r'        <ccg id="s0_ccg0" root="s0_sp0" score="-0.10101445019245148">'
 r'          <span category="S[dcl=true]" id="s0_sp0" child="s0_sp1 s0_sp21" rule="rp" begin="0" end="11" root="true"/>'
 r'          <span category="S[dcl=true]" id="s0_sp1" child="s0_sp2 s0_sp5" rule="ba" begin="0" end="10"/>'
 r'          <span category="NP" id="s0_sp2" child="s0_sp3 s0_sp4" rule="fa" begin="0" end="2"/>'
 r'          <span category="NP[nb=true]/N" id="s0_sp3" terminal="s0_0" begin="0" end="1"/>'
 r'          <span category="N" id="s0_sp4" terminal="s0_1" begin="1" end="2"/>'
 r'          <span category="S[dcl=true]\NP" id="s0_sp5" child="s0_sp6 s0_sp7" rule="fa" begin="2" end="10"/>'
 r'          <span category="(S[dcl=true]\NP)/NP" id="s0_sp6" terminal="s0_2" begin="2" end="3"/>'
 r'          <span category="NP" id="s0_sp7" child="s0_sp8 s0_sp12" rule="ba" begin="3" end="10"/>'
 r'          <span category="NP" id="s0_sp8" child="s0_sp9" rule="lex" begin="3" end="5"/>'
 r'          <span category="N" id="s0_sp9" child="s0_sp10 s0_sp11" rule="fa" begin="3" end="5"/>'
 r'          <span category="N/N" id="s0_sp10" terminal="s0_3" begin="3" end="4"/>'
 r'          <span category="N" id="s0_sp11" terminal="s0_4" begin="4" end="5"/>'
 r'          <span category="NP\NP" id="s0_sp12" child="s0_sp13 s0_sp14" rule="fa" begin="5" end="10"/>'
 r'          <span category="(NP\NP)/NP" id="s0_sp13" terminal="s0_5" begin="5" end="6"/>'
 r'          <span category="NP" id="s0_sp14" child="s0_sp15 s0_sp16" rule="fa" begin="6" end="10"/>'
 r'          <span category="NP[nb=true]/N" id="s0_sp15" terminal="s0_6" begin="6" end="7"/>'
 r'          <span category="N" id="s0_sp16" child="s0_sp17 s0_sp18" rule="fa" begin="7" end="10"/>'
 r'          <span category="N/N" id="s0_sp17" terminal="s0_7" begin="7" end="8"/>'
 r'          <span category="N" id="s0_sp18" child="s0_sp19 s0_sp20" rule="fa" begin="8" end="10"/>'
 r'          <span category="N/N" id="s0_sp19" terminal="s0_8" begin="8" end="9"/>'
 r'          <span category="N" id="s0_sp20" terminal="s0_9" begin="9" end="10"/>'
 r'          <span category="." id="s0_sp21" terminal="s0_10" begin="10" end="11"/>'
 r'        </ccg>'
 r'      </sentence>'
 r'    </sentences>'
 r'  </document>'
 r'</root>')


(r'<?xml version=\'1.0\' encoding=\'utf-8\'?>'
 r'<root>'
 r'  <document>'
 r'    <sentences>'
 r'      <sentence>'
 r'        <tokens>'
 r'          <token start="0" cat="NP[nb]/N" id="s0_0" pos="DT" entity="O" chunk="XX" surf="This" base="this"/>'
 r'          <token start="1" cat="N" id="s0_1" pos="NN" entity="O" chunk="XX" surf="paper" base="paper"/>'
 r'          <token start="2" cat="(S[dcl]\NP)/NP" id="s0_2" pos="VBZ" entity="O" chunk="XX" surf="discusses" base="discuss"/>'
 r'          <token start="3" cat="N/N" id="s0_3" pos="JJ" entity="O" chunk="XX" surf="crucial" base="crucial"/>'
 r'          <token start="4" cat="N" id="s0_4" pos="NNS" entity="O" chunk="XX" surf="aspects" base="aspect"/>'
 r'          <token start="5" cat="(NP\NP)/NP" id="s0_5" pos="IN" entity="O" chunk="XX" surf="of" base="of"/>'
 r'          <token start="6" cat="NP[nb]/N" id="s0_6" pos="DT" entity="O" chunk="XX" surf="this" base="this"/>'
 r'          <token start="7" cat="N/N" id="s0_7" pos="JJ" entity="O" chunk="XX" surf="new" base="new"/>'
 r'          <token start="8" cat="N/N" id="s0_8" pos="NN" entity="O" chunk="XX" surf="annotation" base="annotation"/>'
 r'          <token start="9" cat="N" id="s0_9" pos="NN" entity="O" chunk="XX" surf="scheme" base="scheme"/>'
 r'          <token start="10" cat="." id="s0_10" pos="." entity="O" chunk="XX" surf="." base="."/>'
 r'        </tokens>'
 r'        <ccg id="s0_ccg0" root="s0_sp0" score="-0.10101445019245148">'
 r'          <span category="S[dcl=true]" id="s0_sp0" child="s0_sp1 s0_sp21" rule="rp" begin="0" end="11" root="true"/>'
 r'          <span category="S[dcl=true]" id="s0_sp1" child="s0_sp2 s0_sp5" rule="ba" begin="0" end="10"/>'
 r'          <span category="NP" id="s0_sp2" child="s0_sp3 s0_sp4" rule="fa" begin="0" end="2"/>'
 r'          <span category="NP[nb=true]/N" id="s0_sp3" terminal="s0_0" begin="0" end="1"/>'
 r'          <span category="N" id="s0_sp4" terminal="s0_1" begin="1" end="2"/>'
 r'          <span category="S[dcl=true]\NP" id="s0_sp5" child="s0_sp6 s0_sp7" rule="fa" begin="2" end="10"/>'
 r'          <span category="(S[dcl=true]\NP)/NP" id="s0_sp6" terminal="s0_2" begin="2" end="3"/>'
 r'          <span category="NP" id="s0_sp7" child="s0_sp8 s0_sp12" rule="ba" begin="3" end="10"/>'
 r'          <span category="NP" id="s0_sp8" child="s0_sp9" rule="lex" begin="3" end="5"/>'
 r'          <span category="N" id="s0_sp9" child="s0_sp10 s0_sp11" rule="fa" begin="3" end="5"/>'
 r'          <span category="N/N" id="s0_sp10" terminal="s0_3" begin="3" end="4"/>'
 r'          <span category="N" id="s0_sp11" terminal="s0_4" begin="4" end="5"/>'
 r'          <span category="NP\NP" id="s0_sp12" child="s0_sp13 s0_sp14" rule="fa" begin="5" end="10"/>'
 r'          <span category="(NP\NP)/NP" id="s0_sp13" terminal="s0_5" begin="5" end="6"/>'
 r'          <span category="NP" id="s0_sp14" child="s0_sp15 s0_sp16" rule="fa" begin="6" end="10"/>'
 r'          <span category="NP[nb=true]/N" id="s0_sp15" terminal="s0_6" begin="6" end="7"/>'
 r'          <span category="N" id="s0_sp16" child="s0_sp17 s0_sp18" rule="fa" begin="7" end="10"/>'
 r'          <span category="N/N" id="s0_sp17" terminal="s0_7" begin="7" end="8"/>'
 r'          <span category="N" id="s0_sp18" child="s0_sp19 s0_sp20" rule="fa" begin="8" end="10"/>'
 r'          <span category="N/N" id="s0_sp19" terminal="s0_8" begin="8" end="9"/>'
 r'          <span category="N" id="s0_sp20" terminal="s0_9" begin="9" end="10"/>'
 r'          <span category="." id="s0_sp21" terminal="s0_10" begin="10" end="11"/>'
 r'        </ccg>'
 r'        <semantics status="success" ccg_id="s0_ccg0" root="s0_sp0">'
 r'          <span id="s0_sp0" child="s0_sp1 s0_sp21" sem="exists x.(_paper(x) &amp; True &amp; exists z2.(_aspect(z2) &amp; _crucial(z2) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (z2 = z1)) &amp; True &amp; exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = z2) &amp; True)))"/>'
 r'          <span id="s0_sp1" child="s0_sp2 s0_sp5" sem="\K.exists x.(_paper(x) &amp; True &amp; exists z2.(_aspect(z2) &amp; _crucial(z2) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (z2 = z1)) &amp; True &amp; exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = z2) &amp; K(e))))"/>'
 r'          <span id="s0_sp2" child="s0_sp3 s0_sp4" sem="\F2 F3.exists x.(_paper(x) &amp; F2(x) &amp; F3(x))"/>'
 r'          <span id="s0_sp3" sem="\F1 F2 F3.exists x.(F1(x) &amp; F2(x) &amp; F3(x))"/>'
 r'          <span id="s0_sp4" sem="\x._paper(x)" type="_paper : Entity -&gt; Prop"/>'
 r'          <span id="s0_sp5" child="s0_sp6 s0_sp7" sem="\Q2 K.Q2(\x.True,\x.exists z2.(_aspect(z2) &amp; _crucial(z2) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (z2 = z1)) &amp; True &amp; exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = z2) &amp; K(e))))"/>'
 r'          <span id="s0_sp6" sem="\Q1 Q2 K.Q2(\x.True,\x.Q1(\y.True,\y.exists e.(_discuss(e) &amp; (Subj(e) = x) &amp; (Acc(e) = y) &amp; K(e))))" type="_discuss : Event -&gt; Prop"/>'
 r'          <span id="s0_sp7" child="s0_sp8 s0_sp12" sem="\F1 F2.exists x.(_aspect(x) &amp; _crucial(x) &amp; exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (x = z1)) &amp; F1(x) &amp; F2(x))"/>'
 r'          <span id="s0_sp8" child="s0_sp9" sem="\F1 F2.exists x.(_aspect(x) &amp; _crucial(x) &amp; F1(x) &amp; F2(x))"/>'
 r'          <span id="s0_sp9" child="s0_sp10 s0_sp11" sem="\x.(_aspect(x) &amp; _crucial(x))"/>'
 r'          <span id="s0_sp10" sem="\F x.(F(x) &amp; _crucial(x))" type="_crucial : Entity -&gt; Prop"/>'
 r'          <span id="s0_sp11" sem="\x._aspect(x)" type="_aspect : Entity -&gt; Prop"/>'
 r'          <span id="s0_sp12" child="s0_sp13 s0_sp14" sem="\Q2 F1 F2.Q2(\x.(exists z1.(_scheme(z1) &amp; _annotation(z1) &amp; _new(z1) &amp; True &amp; (x = z1)) &amp; F1(x)),F2)"/>'
 r'          <span id="s0_sp13" sem="\Q1 Q2 F1 F2.Q2(\x.(Q1(\w.True,\y.(x = y)) &amp; F1(x)),F2)"/>'
 r'          <span id="s0_sp14" child="s0_sp15 s0_sp16" sem="\F2 F3.exists x.(_scheme(x) &amp; _annotation(x) &amp; _new(x) &amp; F2(x) &amp; F3(x))"/>'
 r'          <span id="s0_sp15" sem="\F1 F2 F3.exists x.(F1(x) &amp; F2(x) &amp; F3(x))"/>'
 r'          <span id="s0_sp16" child="s0_sp17 s0_sp18" sem="\x.(_scheme(x) &amp; _annotation(x) &amp; _new(x))"/>'
 r'          <span id="s0_sp17" sem="\F x.(F(x) &amp; _new(x))" type="_new : Entity -&gt; Prop"/>'
 r'          <span id="s0_sp18" child="s0_sp19 s0_sp20" sem="\x.(_scheme(x) &amp; _annotation(x))"/>'
 r'          <span id="s0_sp19" sem="\F x.(F(x) &amp; _annotation(x))" type="_annotation : Entity -&gt; Prop"/>'
 r'          <span id="s0_sp20" sem="\x._scheme(x)" type="_scheme : Entity -&gt; Prop"/>'
 r'          <span id="s0_sp21" sem="\X.X"/>'
 r'        </semantics>'
 r'      </sentence>'
 r'    </sentences>'
 r'  </document>'
 r'</root>')



(r'# ID=0'
 r'# log probability=-1.0101e-01'
 r'1	This	this	DT	DT	_	0	NP[nb]/N	_	(<T S[dcl] 0 2> (<T S[dcl] 0 2> (<T NP 0 2> (<L NP[nb]/N DT DT This NP[nb]/N>)'
 r'2	paper	paper	NN	NN	_	1	N	_	(<L N NN NN paper N>) )'
 r'3	discusses	discuss	VBZ	VBZ	_	1	(S[dcl]\NP)/NP	_	(<T S[dcl]\NP 0 2> (<L (S[dcl]\NP)/NP VBZ VBZ discusses (S[dcl]\NP)/NP>)'
 r'4	crucial	crucial	JJ	JJ	_	3	N/N	_	(<T NP 0 2> (<T NP 0 1> (<T N 0 2> (<L N/N JJ JJ crucial N/N>)'
 r'5	aspects	aspect	NNS	NNS	_	4	N	_	(<L N NNS NNS aspects N>) ) )'
 r'6	of	of	IN	IN	_	4	(NP\NP)/NP	_	(<T NP\NP 0 2> (<L (NP\NP)/NP IN IN of (NP\NP)/NP>)'
 r'7	this	this	DT	DT	_	6	NP[nb]/N	_	(<T NP 0 2> (<L NP[nb]/N DT DT this NP[nb]/N>)'
 r'8	new	new	JJ	JJ	_	7	N/N	_	(<T N 0 2> (<L N/N JJ JJ new N/N>)'
 r'9	annotation	annotation	NN	NN	_	8	N/N	_	(<T N 0 2> (<L N/N NN NN annotation N/N>)'
 r'10	scheme	scheme	NN	NN	_	9	N	_	(<L N NN NN scheme N>) ) ) ) ) ) ) )'
 r'11	.	.	.	.	_	1	.	_	(<L . . . . .>) )')


('{"type": "rp", "cat": "S[dcl]", "children": [{"type": "ba", "cat": "S[dcl]", "children":'
' [{"type": "fa", "cat": "NP", "children": [{"word": "This", "pos": "DT", "entity": "O",'
' "lemma": "this", "chunk": "XX", "cat": "NP[nb]/N"}, {"word": "paper", "pos": "NN", "entity":'
' "O", "lemma": "paper", "chunk": "XX", "cat": "N"}]}, {"type": "fa", "cat": "S[dcl]\\NP",'
' "children": [{"word": "discusses", "pos": "VBZ", "entity": "O", "lemma": "discuss", "chunk":'
' "XX", "cat": "(S[dcl]\\NP)/NP"}, {"type": "ba", "cat": "NP", "children": [{"type": "lex",'
' "cat": "NP", "children": [{"type": "fa", "cat": "N", "children": [{"word": "crucial", "pos": "JJ",'
' "entity": "O", "lemma": "crucial", "chunk": "XX", "cat": "N/N"}, {"word": "aspects", "pos": "NNS",'
' "entity": "O", "lemma": "aspect", "chunk": "XX", "cat": "N"}]}]}, {"type": "fa", "cat": "NP\\NP",'
' "children": [{"word": "of", "pos": "IN", "entity": "O", "lemma": "of", "chunk": "XX", "cat":'
' "(NP\\NP)/NP"}, {"type": "fa", "cat": "NP", "children": [{"word": "this", "pos": "DT",'
' "entity": "O", "lemma": "this", "chunk": "XX", "cat": "NP[nb]/N"}, {"type": "fa", "cat": "N",'
' "children": [{"word": "new", "pos": "JJ", "entity": "O", "lemma": "new", "chunk": "XX", "cat": "N/N"},'
' {"type": "fa", "cat": "N", "children": [{"word": "annotation", "pos": "NN", "entity": "O",'
' "lemma": "annotation", "chunk": "XX", "cat": "N/N"}, {"word": "scheme", "pos": "NN", "entity": "O",'
' "lemma": "scheme", "chunk": "XX", "cat": "N"}]}]}]}]}]}]}]}, {"word": ".", "pos": ".", "entity": "O",'
' "lemma": ".", "chunk": "XX", "cat": "."}], "id": 1, "prob": -0.10101445019245148}')

(r'ID=1, log probability=-0.10101445019245148\n'
r'(ROOT (S[dcl] (S[dcl] (NP (NP[nb]/N This) (N paper)) (S[dcl]\NP ((S[dcl]\NP)/NP discusses)'
r' (NP (NP (N (N/N crucial) (N aspects))) (NP\NP ((NP\NP)/NP of) (NP (NP[nb]/N this) (N (N/N new)'
r' (N (N/N annotation) (N scheme)))))))) (. .)))')

(r'ID=1, log probability=-1.488217830657959\n'
r'{< S[mod=nm,form=base,fin=t] {> S[mod=nm,form=base,fin=f] {< S[mod=X1,form=X2,fin=f]/S[mod=X1,form=X2,fin=f]'
r' {< NP[case=nc,mod=nm,fin=f] {NP[case=nc,mod=nm,fin=f] メロス/メロス/名詞-一般/_}'
r' {NP[case=nc,mod=nm,fin=f]\NP[case=nc,mod=nm,fin=f] に/に/助詞-格助詞-一般/_}}'
r' {(S[mod=X1,form=X2,fin=f]/S[mod=X1,form=X2,fin=f])\NP[case=nc,mod=nm,fin=f] は/は/助詞-係助詞/_}}'
r' {< S[mod=nm,form=base,fin=f] {< NP[case=ga,mod=nm,fin=f] {NP[case=nc,mod=nm,fin=f] 政治/政治/名詞-一般/_}'
r' {NP[case=ga,mod=nm,fin=f]\NP[case=nc,mod=nm,fin=f] が/が/助詞-格助詞-一般/_}}'
r' {<B1 S[mod=nm,form=base,fin=f]\NP[case=ga,mod=nm,fin=f] {S[mod=nm,form=neg,fin=f]\NP[case=ga,mod=nm,fin=f]'
r' わから/わから/動詞-自立/未然形-五段・ラ行} {S[mod=nm,form=base,fin=f]\S[mod=nm,form=neg,fin=f]'
r' ぬ/ぬ/助動詞/基本形-特殊・ヌ}}}} {S[mod=nm,form=base,fin=t]\S[mod=nm,form=base,fin=f] 。/。/記号-句点/_}}')

(r":- op(601, xfx, (/))."
r":- op(601, xfx, (\))."
r":- multifile ccg/2, id/2."
r":- discontiguous ccg/2, id/2."
r""
r"ccg(1,"
r" ba(s,"
r"  fa(s,"
r"   ba((s/s),"
r"    ba(np:nc,"
r"     t(np:nc, 'メロス', 'メロス', '名詞/一般/*/*', '*', '*'),"
r"     t((np:nc\np:nc), 'に', 'に', '助詞/格助詞/一般/*', '*', '*')),"
r"    t(((s/s)\np:nc), 'は', 'は', '助詞/係助詞/*/*', '*', '*')),"
r"   ba(s,"
r"    ba(np:ga,"
r"     t(np:nc, '政治', '政治', '名詞/一般/*/*', '*', '*'),"
r"     t((np:ga\np:nc), 'が', 'が', '助詞/格助詞/一般/*', '*', '*')),"
r"    bc1((s\np:ga),"
r"     t((s\np:ga), 'わから', 'わかる', '動詞/自立/*/*', '未然形', '五段・ラ行'),"
r"     t((s\s), 'ぬ', 'ぬ', '助動詞/*/*/*', '基本形', '特殊・ヌ')))),"
r"  t((s\s), '。', '。', '記号/句点/*/*', '*', '*'))).")