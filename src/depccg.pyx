
from __future__ import print_function, unicode_literals
cimport numpy as np
import numpy as np
import sys, re
from libc.stdlib cimport malloc, free
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref

if sys.version_info.major == 3:
    unicode = str


#######################################################
###################### EXTERNs ########################
#######################################################

cdef extern from "<iostream>" namespace "std":
    #TODO: does not resolve template???
    cdef cppclass ostream:
        ostream& operator<< (const Node* val)
        ostream& operator<< (const Derivation& deriv)
        ostream& operator<< (const JaCCG& deriv)
        ostream& operator<< (const PyXML& deriv)
        ostream& operator<< (const CoNLL& deriv)
    ostream cout

cdef extern from "feat.h" namespace "myccg" nogil:
    ctypedef const Feature* Feat
    cdef cppclass Feature:
        # static Feat Parse(const std::string& string);
        string ToStr() const
        bint IsEmpty() const
        bint Matches(Feat other) const
        bint ContainsWildcard() const
        string SubstituteWildcard(const string& string) const
        bint ContainsKeyValue(const string& key, const string& value) const
        Feat ToMultiValue() const


cdef extern from "cat.h" namespace "myccg" nogil:
    cdef cppclass Slash:
        bint IsForward() const
        bint IsBackward() const
        string ToStr() const

    ctypedef const Category* Cat
    cdef cppclass Category:
        @staticmethod
        Cat Parse(const string& cat)
        string ToStr()
        string ToStrWithoutFeat()

        # ctypedef const string& Str
        Cat StripFeat() const
        # Cat StripFeat(Str f1) const
        # Cat StripFeat(Str f1, Str f2) const
        # Cat StripFeat(Str f1, Str f2, Str f3) const
        # Cat StripFeat(Str f1, Str f2, Str f3, Str f4) const

        const string& GetType() const
        Feat GetFeat() const
        Cat GetLeft() const
        Cat GetRight() const

        # template<int i> Cat GetLeft() const;
        # template<int i> Cat GetRight() const;
        # template<int i> bool HasFunctorAtLeft() const;
        # template<int i> bool HasFunctorAtRight() const;
        Slash GetSlash() const

        const string WithBrackets() const
        bint IsModifier() const
        bint IsModifierWithoutFeat() const
        bint IsTypeRaised() const
        bint IsTypeRaisedWithoutFeat() const
        bint IsForwardTypeRaised() const
        bint IsBackwardTypeRaised() const
        bint IsFunctor() const
        bint IsPunct() const
        bint IsNorNP() const
        int NArgs() const
        # Feat GetSubstitution(Cat other) const = 0;
        bint Matches(Cat other) const
        Cat Arg(int argn) const
        Cat LeftMostArg() const
        bint IsFunctionInto(Cat cat) const
        Cat ToMultiValue() const
        # Cat Substitute(Feat feat) const


cdef extern from "combinator.h" namespace "myccg" nogil:
    cdef cppclass Combinator:
        const string ToStr() const
    ctypedef const Combinator* Op


cdef extern from "tree.h" namespace "myccg" nogil:
    cdef cppclass Leaf:
        Leaf(const string& word, Cat cat, int position)

    cdef cppclass Tree:
        Op GetRule() const

    cdef cppclass Node:
        Cat GetCategory() const
        const int GetLength() const
        shared_ptr[const Node] GetLeftChild() const
        shared_ptr[const Node] GetRightChild() const
        bint IsLeaf() const
        # RuleType GetRuleType()
        # RuleType GetRuleType()
        const Leaf* GetHeadLeaf() const
        int GetStartOfSpan() const
        string GetWord() const
        int GetHeadId() const
        int GetDependencyLength() const
        bint HeadIsLeft() const
        bint IsUnary() const
        int NumDescendants() const
        int RightNumDescendants() const
        int LeftNumDescendants() const
        ostream& operator<<(ostream& ost, const Node* node)

    ctypedef shared_ptr[const Node] NodeType
    ctypedef shared_ptr[const Tree] TreeType
    ctypedef shared_ptr[const Leaf] LeafType

    cdef cppclass AUTO:
        AUTO(const Node* tree)
        AUTO(NodeType tree)
        string Get()

    cdef cppclass Derivation:
        Derivation(const Node* tree, bint feat)
        Derivation(NodeType tree, bint feat)
        ostream& operator<<(ostream& ost, const Derivation& deriv)
        string Get()

    cdef cppclass JaCCG:
        JaCCG(const Node* tree)
        JaCCG(NodeType tree)
        ostream& operator<<(ostream& ost, const JaCCG& deriv)
        string Get()

    cdef cppclass PyXML:
        PyXML(const Node* tree, bint feat)
        PyXML(NodeType tree, bint feat)
        ostream& operator<<(ostream& ost, const PyXML& xml)
        string Get()

    cdef cppclass CoNLL:
        CoNLL(const Node* tree)
        CoNLL(NodeType tree)
        ostream& operator<<(ostream& ost, const CoNLL& xml)
        string Get()


cdef extern from "chainer_tagger.h" namespace "myccg" nogil:
    cdef cppclass Tagger:
        Tagger(const string& model) except +


cdef extern from "grammar.h" namespace "myccg" nogil:
    cdef cppclass En:
        @staticmethod
        string ResolveCombinatorName(const Node*)

    cdef const unordered_set[Cat] en_possible_root_cats     "myccg::En::possible_root_cats"
    cdef const vector[Op]         en_headfirst_binary_rules "myccg::En::headfirst_binary_rules"
    cdef const vector[Op]         en_binary_rules           "myccg::En::binary_rules"
    cdef const vector[Op]         en_dep_binary_rules       "myccg::En::dep_binary_rules"

    cdef cppclass Ja:
        @staticmethod
        string ResolveCombinatorName(const Node*)

    cdef const unordered_set[Cat] ja_possible_root_cats     "myccg::Ja::possible_root_cats"
    cdef const vector[Op]         ja_binary_rules           "myccg::Ja::binary_rules"
    cdef const vector[Op]         ja_headfinal_binary_rules "myccg::Ja::headfinal_binary_rules"
    cdef const vector[Op]         ja_cg_binary_rules "myccg::Ja::cg_binary_rules"


cdef extern from "logger.h" namespace "myccg" nogil:
    enum LogLevel:
        Debug
        Info
        Warn
        Error

    cdef cppclass ParserLogger:
        void InitStatistics(int num_sents)
        void ShowStatistics()
        void RecordTimeStartRunning()
        void RecordTimeEndOfTagging()
        void RecordTimeEndOfParsing()
        void Report()
        void CompleteOne()
        void CompleteOne(int id, int agenda_size)

cdef extern from "chart.h" namespace "myccg" nogil:
    ctypedef pair[NodeType, float] ScoredNode

cdef extern from "parser_tools.h" namespace "myccg" nogil:
    cdef cppclass AgendaItem:
        pass

    bint NormalComparator(const AgendaItem& left, const AgendaItem& right)
    bint JapaneseComparator(const AgendaItem& left, const AgendaItem& right)
    bint LongerDependencyComparator(const AgendaItem& left, const AgendaItem& right)
    ctypedef bint (*Comparator)(const AgendaItem&, const AgendaItem&)


cdef extern from "parser.h" namespace "myccg" nogil:
    cdef cppclass Parser:
        vector[ScoredNode] Parse(int id, const string& sent, float* scores)
        vector[ScoredNode] Parse(int id, const string& sent, float* tag_scores, float* dep_scores)
        vector[vector[ScoredNode]] Parse(const vector[string]& doc)
        vector[vector[ScoredNode]] Parse(const vector[string]& doc, float** scores)
        vector[vector[ScoredNode]] Parse(const vector[string]& doc, float** tag_scores, float** dep_scores)
        void LoadSeenRules()
        void LoadCategoryDict()
        void SetComparator(Comparator comp)
        void SetBeta(float beta)
        void SetUseBeta(bint use_beta)
        void SetPruningSize(int prune)
        ParserLogger& GetLogger()

    cdef cppclass AStarParser[Lang]:
        AStarParser(
                Tagger* tagger,
                const string& model,
                const unordered_set[Cat]& possible_root_cats,
                Comparator comparator,
                vector[Op] binary_rules,
                unsigned nbest,
                float beta,
                int pruning_size,
                LogLevel loglevel) except +

cdef extern from "dep.h" namespace "myccg" nogil:
    cdef cppclass DepAStarParser[Lang]:
        ctypedef AStarParser[Lang] Base

        DepAStarParser(
                    Tagger* tagger,
                    const string& model,
                    const unordered_set[Cat]& possible_root_cats,
                    Comparator comparator,
                    vector[Op] binary_rules,
                    unsigned nbest,
                    float beta,
                    int pruning_size,
                    LogLevel loglevel) except +


## TODO: ugly code
cdef ResolveCmobinatorName(const Node* tree, bytes lang):
    cdef string res;
    if lang == b"en":
        res = En.ResolveCombinatorName(tree);
    elif lang == b"ja":
        res = Ja.ResolveCombinatorName(tree);
    else:
        res = b"error: " + lang
    return res.decode("utf-8")


def __show_mathml(tree):
    if not tree.is_leaf:
        return """\
<mrow>
  <mfrac linethickness='2px'>
    <mrow>{}</mrow>
    <mstyle mathcolor='Red'>{}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>{}</mtext>
</mrow>""".format(
                "".join(map(__show_mathml, tree.children)),
                __show_mathml_cat(str(tree.cat)),
                tree.op_string)
    else:
        return """\
<mrow>
  <mfrac linethickness='2px'>
    <mtext mathsize='1.0' mathcolor='Black'>{}</mtext>
    <mstyle mathcolor='Red'>{}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>lex</mtext>
</mrow>""".format(
                tree.word,
                __show_mathml_cat(str(tree.cat)))


def __show_mathml_cat(cat):
    cats_feats = re.findall(r'([\w\\/()]+)(\[.+?\])*', cat)
    mathml_str = ''
    for cat, feat in cats_feats:
        cat_mathml = """\
<mi mathvariant='italic'
  mathsize='1.0' mathcolor='Red'>{}</mi>
    """.format(cat)

        if feat != '':
            mathml_str += """\
<msub>{}
  <mrow>
  <mi mathvariant='italic'
    mathsize='0.8' mathcolor='Purple'>{}</mi>
  </mrow>
</msub>""".format(cat_mathml, feat)

        else:
            mathml_str += cat_mathml
    return mathml_str


def to_mathml(trees, file=sys.stdout):
    def __show(tree):
        res = ""
        for t in tree:
            if isinstance(t, tuple):
                t, prob = t
                res += "<p>Log prob={:.5e}</p>".format(prob)
            res += "<math xmlns='http://www.w3.org/1998/Math/MathML'>{}</math>".format(
                __show_mathml(t))
        return res

    string = ""
    for i, tree in enumerate(trees):
        words = tree[0][0].word if isinstance(tree[0], tuple) else tree[0].word
        string += "<p>ID={}: {}</p>{}".format(
                i, words, __show(tree))

    print("""\
<!doctype html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <style>
    body {{
      font-size: 1em;
    }}
  </style>
  <script type="text/javascript"
     src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
</head>
<body>{}
</body></html>""".format(string), file=file)


#######################################################
####################### Category ######################
#######################################################

cdef class PyCat:
    cdef Cat cat_

    def __cinit__(self):
        pass

    def __str__(self):
        cdef string res = self.cat_.ToStr()
        return res.decode("utf-8")

    def __repr__(self):
        cdef string res = self.cat_.ToStr()
        return res.decode("utf-8")

    @staticmethod
    def parse(cat):
        if not isinstance(cat, bytes):
            cat = cat.encode("utf-8")
        c = PyCat()
        c.cat_ = Category.Parse(cat)
        return c

    @staticmethod
    cdef PyCat from_ptr(Cat cat):
        c = PyCat()
        c.cat_ = cat
        return c

    property multi_valued:
        def __get__(self):
            return PyCat.from_ptr(self.cat_.ToMultiValue())

    property without_feat:
        def __get__(self):
            cdef string res = self.cat_.ToStrWithoutFeat()
            return res.decode("utf-8")

        # const string& GetType() const
    property left:
        def __get__(self):
            assert self.is_functor, \
                "Error {} is not functor type.".format(str(self))
            return PyCat.from_ptr(self.cat_.GetLeft())

    property right:
        def __get__(self):
            assert self.is_functor, \
                "Error {} is not functor type.".format(str(self))
            return PyCat.from_ptr(self.cat_.GetRight())

    property is_modifier:
        def __get__(self):
            return self.cat_.IsModifier()

    property is_modifier_without_feat:
        def __get__(self):
            return self.cat_.IsModifierWithoutFeat()

    property is_type_raised:
        def __get__(self):
            return self.cat_.IsTypeRaised()

    property is_type_raised_without_feat:
        def __get__(self):
            return self.cat_.IsTypeRaisedWithoutFeat()

    property is_forward_type_raised:
        def __get__(self):
            return self.cat_.IsForwardTypeRaised()

    property is_backward_type_raised:
        def __get__(self):
            return self.cat_.IsBackwardTypeRaised()

    property is_functor:
        def __get__(self):
            return self.cat_.IsFunctor()

    property is_punct:
        def __get__(self):
            return self.cat_.IsPunct()

    property is_NorNP:
        def __get__(self):
            return self.cat_.IsNorNP()

    def is_function_into(self, cat):
        return self._is_function_into(cat)

    cdef bint _is_function_into(self, PyCat cat):
        cdef Cat ccat = cat.cat_
        return self.cat_.IsFunctionInto(ccat)

    property n_args:
        def __get__(self):
            return self.cat_.NArgs()

    def matches(self, other):
        return self._matches(other)

    cdef bint _matches(self, PyCat other):
        return self.cat_.Matches(other.cat_)

    def arg(self, i):
        return PyCat.from_ptr(self.cat_.Arg(i))

    property slash:
        def __get__(self):
            assert self.is_functor, \
                "Error {} is not functor type.".format(str(self))
            cdef string res = self.cat_.GetSlash().ToStr()
            return res.decode("utf-8")


#######################################################
###################### Parse Tree #####################
#######################################################

cdef class Parse:
    cdef NodeType node
    cdef public bint suppress_feat
    cdef bytes lang

    @staticmethod
    cdef Parse from_ptr(NodeType node, lang):
        p = Parse()
        p.node = node
        p.lang = lang
        return p

    def __cinit__(self):
        self.suppress_feat = False

    property cat:
        def __get__(self):
            return PyCat.from_ptr(deref(self.node).GetCategory())

    property op_string:
        def __get__(self):
            assert not self.is_leaf, \
                "This node is leaf and does not have combinator!"
            cdef const Node* c_node = &deref(self.node)
            return ResolveCmobinatorName(c_node, self.lang)
            # cdef string res = (<const Tree*>c_node).GetRule().ToStr()
            # return res.decode("utf-8")

    def __len__(self):
        return deref(self.node).GetLength()

    property children:
        def __get__(self):
            res = [self.left_child]
            if not self.is_unary:
                res.append(self.right_child)
            return res

    property left_child:
        def __get__(self):
            assert not self.is_leaf, \
                "This node is leaf and does not have any child!"
            return Parse.from_ptr(<NodeType>deref(self.node).GetLeftChild(), self.lang)

    property right_child:
        def __get__(self):
            assert not self.is_leaf, \
                "This node is leaf and does not have any child!"
            assert not self.is_unary, \
                "This node does not have right child!"
            return Parse.from_ptr(<NodeType>deref(self.node).GetRightChild(), self.lang)

    property is_leaf:
        def __get__(self):
            return deref(self.node).IsLeaf()

    # property head_leaf:
    #     def __get__(self):
    #         return Parse.from_ptr(deref(self.node).GetHeadLeaf())
    #
    property start_of_span:
        def __get__(self):
            return deref(self.node).GetStartOfSpan()

    property word:
        def __get__(self):
            cdef string res = deref(self.node).GetWord()
            return res.decode("utf-8")

    property head_id:
        def __get__(self):
            return deref(self.node).GetHeadId()

    property dependency_length:
        def __get__(self):
            return deref(self.node).GetDependencyLength()

    property head_is_left:
        def __get__(self):
            return deref(self.node).HeadIsLeft()

    property is_unary:
        def __get__(self):
            return deref(self.node).IsUnary()

    property num_descendants:
        def __get__(self):
            return deref(self.node).NumDescendants()

    property right_num_descendants:
        def __get__(self):
            return deref(self.node).RightNumDescendants()

    property left_num_descendants:
        def __get__(self):
            return deref(self.node).LeftNumDescendants()

    def __str__(self):
        return self.auto

    def __repr__(self):
        return self.auto

    property auto:
        def __get__(self):
            cdef string res = AUTO(self.node).Get()
            return res.decode("utf-8")

    property deriv:
        def __get__(self):
            cdef string res = Derivation(self.node, not self.suppress_feat).Get()
            return res.decode("utf-8")

    property xml:
        def __get__(self):
            cdef string res = PyXML(self.node, not self.suppress_feat).Get()
            return res.decode("utf-8")

    property ja:
        def __get__(self):
            cdef string res = JaCCG(self.node).Get()
            return res.decode("utf-8")

    property conll:
        def __get__(self):
            cdef string res = CoNLL(self.node).Get()
            return res.decode("utf-8")

#######################################################
################### English Parser ####################
#######################################################

import os
import json
import chainer
# from py.ja_lstm_parser import JaLSTMParser
# from py.lstm_parser import LSTMParser
# from py.tagger import EmbeddingTagger
# from py.japanese_tagger import JaCCGEmbeddingTagger
# from py.ja_lstm_tagger import JaLSTMTagger
# from py.lstm_tagger import LSTMTagger
# from py.lstm_tagger_ph import PeepHoleLSTMTagger
# from py.ja_lstm_parser_ph import PeepHoleJaLSTMParser
from py.precomputed_parser import PrecomputedParser
from py.lstm_parser_bi_fast import FastBiaffineLSTMParser
from py.ja_lstm_parser_bi import BiaffineJaLSTMParser
from libc.string cimport memcpy

cdef class PyAStarParser:
    cdef Tagger* tagger_
    cdef Parser* parser_
    cdef object path
    cdef object py_tagger
    cdef object use_seen_rules
    cdef object use_cat_dict
    cdef object use_beta
    cdef object nbest
    cdef object beta
    cdef object pruning_size
    cdef object batchsize
    cdef object loglevel
    cdef bytes  lang

    def __init__(self, path,
                  use_seen_rules=True,
                  use_cat_dict=True,
                  use_beta=True,
                  nbest=1,
                  beta=0.00001,
                  pruning_size=50,
                  batchsize=16,
                  loglevel=3,
                  type_check=False):

        self.path           = path.encode("utf-8")
        self.use_seen_rules = use_seen_rules
        self.use_cat_dict   = use_cat_dict
        self.use_beta       = use_beta
        self.nbest          = nbest
        self.beta           = beta
        self.pruning_size   = pruning_size
        self.batchsize      = batchsize
        self.loglevel       = loglevel
        self.lang           = b"en"

        self.tagger_ = new Tagger(self.path)
        self.parser_ = self.load_parser()

        if use_seen_rules:
            self.parser_.LoadSeenRules()
        if use_cat_dict:
            self.parser_.LoadCategoryDict()
        if not use_beta:
            self.parser_.SetUseBeta(False)

        with open(os.path.join(path, "tagger_defs.txt")) as f:
            self.py_tagger = eval(json.load(f)["model"])(path)
        model = os.path.join(path, "tagger_model")
        if os.path.exists(model):
            chainer.serializers.load_npz(model, self.py_tagger)
        else:
            print("not loading parser model", file=sys.stderr)

        # disable chainer's type chacking for efficiency
        if not type_check:
            os.environ["CHAINER_TYPE_CHECK"] = "0"

    cdef Parser* load_parser(self) except *:
        return <Parser*>new DepAStarParser[En](
                        self.tagger_,
                        self.path,
                        en_possible_root_cats,
                        NormalComparator,
                        en_headfirst_binary_rules,
                        self.nbest,
                        self.beta,
                        self.pruning_size,
                        self.loglevel)

    def parse(self, sent):
        if not isinstance(sent, list):
            assert isinstance(sent, unicode)
            splitted = sent.split(" ")
            sent = sent.encode("utf-8")
        else:
            splitted = sent
            sent = " ".join(sent).encode("utf-8")

        [mat] = self.py_tagger.predict([splitted])
        if isinstance(mat, (tuple, list)):
            return self._parse_tag_and_dep(sent, mat[0], mat[1])
        else:
            return self._parse_tag(sent, mat)

    def parse_doc(self, sents, probs=None):
        cdef list res
        cdef ParserLogger* logger = &self.parser_.GetLogger()

        if not isinstance(sents[0], list):
            assert isinstance(sents[0],  unicode)
            splitted = [s.split(" ") for s in sents]
            sents = [s.encode("utf-8") for s in sents]
        else:
            assert isinstance(sents[0][0],  unicode)
            splitted = sents
            sents = [" ".join(s).encode("utf-8") for s in sents]

        logger.InitStatistics(len(sents))
        logger.RecordTimeStartRunning()
        if probs is None:
            probs = self.py_tagger.predict_doc(splitted, batchsize=self.batchsize)
        logger.RecordTimeEndOfTagging()
        res = self._parse_doc_tag_and_dep(sents, probs)
        logger.RecordTimeEndOfParsing()
        logger.Report()
        return res

    cdef Parse _parse_tag(self, bytes sent, np.ndarray[float, ndim=2, mode="c"] mat):
        cdef string csent = sent
        cdef vector[ScoredNode] res = self.parser_.Parse(0, csent, &mat[0, 0])
        return Parse.from_ptr(res[0].first, self.lang)

    cdef Parse _parse_tag_and_dep(self, bytes sent,
                                  np.ndarray[float, ndim=2, mode="c"] tag,
                                  np.ndarray[float, ndim=2, mode="c"] dep):
        cdef vector[ScoredNode] res = self.parser_.Parse(0, sent, &tag[0, 0], &dep[0, 0])
        return Parse.from_ptr(res[0].first, self.lang)

    cdef list _parse_doc_tag_and_dep(self, list sents, list probs):
        cdef int doc_size = len(sents), i, j
        cdef np.ndarray[float, ndim=2, mode="c"] cat_scores, dep_scores
        cdef vector[string] csents = sents
        cdef float **tags = <float**>malloc(doc_size * sizeof(float*))
        cdef float **deps = <float**>malloc(doc_size * sizeof(float*))
        for i, _, (cat_scores, dep_scores) in probs:
            tags[i] = &cat_scores[0, 0]
            deps[i] = &dep_scores[0, 0]
        if self.loglevel < 3: print("start parsing", file=sys.stderr)
        cdef vector[vector[ScoredNode]] cres = self.parser_.Parse(csents, tags, deps)
        cdef list tmp, res = []
        cdef Parse parse
        for i in range(len(sents)):
            tmp = []
            for j in range(min(self.nbest, cres[i].size())):
                tmp.append((Parse.from_ptr(cres[i][j].first, self.lang),
                                cres[i][j].second))
            res.append(tmp)
        free(tags)
        free(deps)
        return res

#######################################################
################## Japanese Parser ####################
#######################################################

cdef class PyJaAStarParser(PyAStarParser):

    def __init__(self, path,
                  use_seen_rules=True,
                  use_cat_dict=False,
                  use_beta=False,
                  nbest=1,
                  beta=0.00001,
                  pruning_size=50,
                  batchsize=16,
                  loglevel=3,
                  type_check=False):

        super(PyJaAStarParser, self).__init__(path,
                  use_seen_rules, use_cat_dict, use_beta,
                  nbest, beta, pruning_size, batchsize,
                  loglevel, type_check)

        self.lang = b"ja"

    # cdef Parser* load_parser(self):
    cdef Parser* load_parser(self) except *:
        return <Parser*>new DepAStarParser[Ja](
                        self.tagger_,
                        self.path,
                        ja_possible_root_cats,
                        NormalComparator,
                        ja_headfinal_binary_rules,
                        # ja_cg_binary_rules,
                        self.nbest,
                        self.beta,
                        self.pruning_size,
                        self.loglevel)
