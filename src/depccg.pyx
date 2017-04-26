
from __future__ import print_function, unicode_literals
cimport numpy as np
import numpy as np
import sys
from libc.stdlib cimport malloc, free
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
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
        ostream& operator<< (const XML& deriv)
        ostream& operator<< (const CoNLL& deriv)
    ostream cout


cdef extern from "cat.h" namespace "myccg" nogil:
    ctypedef const Category* Cat
    cdef cppclass Category:
        @staticmethod
        Cat Parse(const string& cat)
        string ToStr()
        string ToStrWithoutFeat()

        # ctypedef const string& Str
        # Cat StripFeat() const
        # Cat StripFeat(Str f1) const
        # Cat StripFeat(Str f1, Str f2) const
        # Cat StripFeat(Str f1, Str f2, Str f3) const
        # Cat StripFeat(Str f1, Str f2, Str f3, Str f4) const

        const string& GetType() const
        # Feat GetFeat() const = 0;
        Cat GetLeft() const
        Cat GetRight() const

        # template<int i> Cat GetLeft() const;
        # template<int i> Cat GetRight() const;
        # template<int i> bool HasFunctorAtLeft() const;
        # template<int i> bool HasFunctorAtRight() const;
        # virtual Slash GetSlash() const = 0;

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


cdef extern from "tree.h" namespace "myccg" nogil:
    cdef cppclass Leaf:
        Leaf(const string& word, Cat cat, int position)

    cdef cppclass Tree:
        pass

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

    cdef cppclass XML:
        XML(const Node* tree, bint feat)
        XML(NodeType tree, bint feat)
        ostream& operator<<(ostream& ost, const XML& xml)
        string Get()

    cdef cppclass CoNLL:
        CoNLL(const Node* tree)
        CoNLL(NodeType tree)
        ostream& operator<<(ostream& ost, const CoNLL& xml)
        string Get()


cdef extern from "combinator.h" namespace "myccg" nogil:
    cdef cppclass Combinator:
        pass
    ctypedef const Combinator* Op


cdef extern from "chainer_tagger.h" namespace "myccg" nogil:
    cdef cppclass Tagger:
        Tagger(const string& model)


cdef extern from "grammar.h" namespace "myccg" nogil:
    cdef cppclass En:
        pass
    cdef const unordered_set[Cat] en_possible_root_cats     "myccg::En::possible_root_cats"
    cdef const vector[Op]         en_headfirst_binary_rules "myccg::En::headfirst_binary_rules"
    cdef const vector[Op]         en_binary_rules           "myccg::En::binary_rules"
    cdef const vector[Op]         en_dep_binary_rules       "myccg::En::dep_binary_rules"

    cdef cppclass Ja:
        pass
    cdef const unordered_set[Cat] ja_possible_root_cats     "myccg::Ja::possible_root_cats"
    cdef const vector[Op]         ja_binary_rules           "myccg::Ja::binary_rules"
    cdef const vector[Op]         ja_headfinal_binary_rules "myccg::Ja::headfinal_binary_rules"


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


cdef extern from "parser_tools.h" namespace "myccg" nogil:
    cdef cppclass AgendaItem:
        pass

    bint NormalComparator(const AgendaItem& left, const AgendaItem& right)
    bint JapaneseComparator(const AgendaItem& left, const AgendaItem& right)
    bint LongerDependencyComparator(const AgendaItem& left, const AgendaItem& right)
    ctypedef bint (*Comparator)(const AgendaItem&, const AgendaItem&)


cdef extern from "parser.h" namespace "myccg" nogil:
    cdef cppclass Parser:
        NodeType Parse(int id, const string& sent, float* scores)
        NodeType Parse(int id, const string& sent, float* tag_scores, float* dep_scores)
        vector[NodeType] Parse(const vector[string]& doc)
        vector[NodeType] Parse(const vector[string]& doc, float** scores)
        vector[NodeType] Parse(const vector[string]& doc, float** tag_scores, float** dep_scores)
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
                    float beta,
                    int pruning_size,
                    LogLevel loglevel) except +


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

    property without_feat:
        def __get__(self):
            return self.cat_.ToStrWithoutFeat()

        # const string& GetType() const
    property left:
        def __get__(self):
            assert self.is_functor
            return PyCat.from_ptr(self.cat_.GetLeft())

    property right:
        def __get__(self):
            assert self.is_functor
            return PyCat.from_ptr(self.cat_.GetRight())

        # const string WithBrackets()

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


#######################################################
###################### Parse Tree #####################
#######################################################

cdef class Parse:
    cdef NodeType node
    cdef public bint suppress_feat

    @staticmethod
    cdef Parse from_ptr(NodeType node):
        p = Parse()
        p.node = node
        return p

    def __cinit__(self):
        self.suppress_feat = False
        # self.node.reset(<const Node*>new const Leaf("fail", Category.Parse("NP"), 0))

    property cat:
        def __get__(self):
            return PyCat.from_ptr(deref(self.node).GetCategory())

    def __len__(self):
        return deref(self.node).GetLength()

    property left_child:
        def __get__(self):
            return Parse.from_ptr(<NodeType>deref(self.node).GetLeftChild())

    property right_child:
        def __get__(self):
            return Parse.from_ptr(<NodeType>deref(self.node).GetRightChild())

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
            return deref(self.node).GetWord()

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
            cdef string res = XML(self.node, not self.suppress_feat).Get()
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
from py.ja_lstm_parser import JaLSTMParser
from py.lstm_parser import LSTMParser
from py.tagger import EmbeddingTagger
from py.japanese_tagger import JaCCGEmbeddingTagger
from py.ja_lstm_tagger import JaLSTMTagger
from py.lstm_tagger import LSTMTagger
from py.lstm_tagger_ph import PeepHoleLSTMTagger
from py.ja_lstm_parser_ph import PeepHoleJaLSTMParser
from py.lstm_parser_bi import BiaffineLSTMParser
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
    cdef object beta
    cdef object pruning_size
    cdef object batchsize
    cdef object loglevel

    def __init__(self, path,
                  use_seen_rules=True,
                  use_cat_dict=True,
                  use_beta=True,
                  beta=0.00001,
                  pruning_size=50,
                  batchsize=16,
                  loglevel=3,
                  type_check=False):

        self.path           = path.encode("utf-8")
        self.use_seen_rules = use_seen_rules
        self.use_cat_dict   = use_cat_dict
        self.use_beta       = use_beta
        self.beta           = beta
        self.pruning_size   = pruning_size
        self.batchsize      = batchsize
        self.loglevel       = loglevel

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

    cdef Parser* load_parser(self):
        return <Parser*>new DepAStarParser[En](
                        self.tagger_,
                        self.path,
                        en_possible_root_cats,
                        NormalComparator,
                        en_headfirst_binary_rules,
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

    def parse_doc(self, sents):
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
        probs = self.py_tagger.predict_doc(splitted, batchsize=self.batchsize)
        logger.RecordTimeEndOfTagging()
        res = self._parse_doc_tag_and_dep(sents, probs)
        logger.RecordTimeEndOfParsing()
        logger.Report()
        return res

    cdef Parse _parse_tag(self, bytes sent, np.ndarray[float, ndim=2, mode="c"] mat):
        cdef string csent = sent
        cdef NodeType res = self.parser_.Parse(0, csent, &mat[0, 0])
        return Parse.from_ptr(res)

    cdef Parse _parse_tag_and_dep(self, bytes sent,
                                  np.ndarray[float, ndim=2, mode="c"] tag,
                                  np.ndarray[float, ndim=2, mode="c"] dep):
        cdef NodeType res = self.parser_.Parse(0, sent, &tag[0, 0], &dep[0, 0])
        return Parse.from_ptr(res)

    cdef list _parse_doc_tag_and_dep(self, list sents, list probs):
        cdef int doc_size = len(sents), i
        cdef np.ndarray[float, ndim=2, mode="c"] cat_scores, dep_scores
        cdef vector[string] csents = sents
        cdef float **tags = <float**>malloc(doc_size * sizeof(float*))
        cdef float **deps = <float**>malloc(doc_size * sizeof(float*))
        for i, _, (cat_scores, dep_scores) in probs:
            tags[i] = &cat_scores[0, 0]
            deps[i] = &dep_scores[0, 0]
        if self.loglevel < 3: print("start parsing", sys.stderr)
        cdef vector[NodeType] cres = self.parser_.Parse(csents, tags, deps)
        cdef list res = []
        cdef Parse parse
        for i in range(len(sents)):
            parse = Parse.from_ptr(cres[i])
            res.append(parse)
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
                  beta=0.00001,
                  pruning_size=50,
                  batchsize=16,
                  loglevel=3,
                  type_check=False):

        super(PyJaAStarParser, self).__init__(path,
                  use_seen_rules, use_cat_dict, use_beta,
                  beta, pruning_size, batchsize,
                  loglevel, type_check)

    cdef Parser* load_parser(self):
        return <Parser*>new DepAStarParser[Ja](
                        self.tagger_,
                        self.path,
                        ja_possible_root_cats,
                        NormalComparator,
                        ja_headfinal_binary_rules,
                        self.beta,
                        self.pruning_size,
                        self.loglevel)
