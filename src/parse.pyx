
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set

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

cdef extern from "tree.h" namespace "myccg" nogil:
    cdef cppclass Node:
        ostream& operator<<(ostream& ost, const Node* node)

    cdef cppclass Leaf:
        Leaf(const string& word, Cat cat, int position)

    cdef cppclass Tree:
        pass

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
        # const unordered_set[Cat] possible_root_cats
        # const vector[Op] binary_rules
        # const vector[Op] dep_binary_rules
        # const vector[Op] headfirst_binary_rules

    # cdef cppclass Ja:
        # pass
        # const unordered_set[Cat] possible_root_cats
        # const vector[Op] binary_rules
        # const vector[Op] headfinal_binary_rules

cdef extern from "grammar.h" namespace "myccg::En" nogil:
    const unordered_set[Cat] possible_root_cats
    const vector[Op] binary_rules
    const vector[Op] dep_binary_rules
    const vector[Op] headfirst_binary_rules

# cdef extern from "grammar.h" namespace "myccg::Ja" nogil:
#     const unordered_set[Cat] possible_root_cats
#     const vector[Op] binary_rules
#     const vector[Op] headfinal_binary_rules

cdef extern from "logger.h" namespace "myccg" nogil:
    enum LogLevel:
        Debug
        Info
        Warn
        Error

cdef extern from "parser_tools.h" namespace "myccg" nogil:
    cdef cppclass AgendaItem:
        pass

    bint NormalComparator(const AgendaItem& left, const AgendaItem& right)
    bint JapaneseComparator(const AgendaItem& left, const AgendaItem& right)
    bint LongerDependencyComparator(const AgendaItem& left, const AgendaItem& right)
    ctypedef bint (*Comparator)(const AgendaItem&, const AgendaItem&)


cdef extern from "parser.h" namespace "myccg" nogil:
    cdef cppclass Parser:
        NodeType Parse(int id, const string& sent, float* tag_scores)
        NodeType Parse(int id, const string& sent, float* tag_scores, float* dep_scores)
        void LoadSeenRules()
        void LoadCategoryDict()
        void SetComparator(Comparator comp)
        void SetBeta(float beta)
        void SetUseBeta(bint use_beta)
        void SetPruningSize(int prune)

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
        NodeType Parse(int id, const string& sent, float* tag_scores)

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

        vector[NodeType] ParseDoc(const vector[string]& doc, float** tag_scores, float** dep_scores)
        NodeType Parse(int id, const string& sent, float* tag_scores)
        NodeType Parse(int id, const string& sent, float* tag_scores, float* dep_scores)


cdef class Parse:
    cdef NodeType node
    cdef public bint suppress_feat

    cdef Parse from_ptr(self, NodeType node):
        self.node = node
        return self

    def __cinit__(self):
        self.suppress_feat = False
        # self.node.reset(<const Node*>new const Leaf("fail", Category.Parse("NP"), 0))

    def __str__(self):
        return self.auto

    def __repr__(self):
        return self.auto

    property auto:
        def __get__(self):
            return AUTO(self.node).Get()

    property deriv:
        def __get__(self):
            return Derivation(self.node, self.suppress_feat).Get()

    property xml:
        def __get__(self):
            return XML(self.node, self.suppress_feat).Get()

    property ja:
        def __get__(self):
            return JaCCG(self.node).Get()

    property conll:
        def __get__(self):
            return CoNLL(self.node).Get()


cdef class PyAStarParser:
    cdef Tagger* tagger_
    cdef DepAStarParser[En]* parser_

    def __cinit__(self, path, beta=0.00001, pruning_size=50):
        self.tagger_ = new Tagger(path)
        self.parser_ = new DepAStarParser[En](
                        self.tagger_,
                        path,
                        possible_root_cats,
                        NormalComparator,
                        headfirst_binary_rules,
                        beta,
                        pruning_size,
                        Error)
        cdef Parser* p = <Parser*>self.parser_
        p.LoadSeenRules()
        p.LoadCategoryDict()
        # SetComparator(Comparator comp)
        # SetBeta(float beta)
        # SetUseBeta(bint use_beta)
        # SetPruningSize(int prune)

    def parse(self, sent, mat):
        if isinstance(mat, (tuple, list)):
            return self._parse_tag_and_dep(sent, mat[0], mat[1])
        else:
            return self._parse_tag(sent, mat)

    cdef Parse _parse_tag(self, str sent, np.ndarray[np.float32_t, ndim=2] mat):
        cdef np.ndarray[np.float32_t, ndim=1] flatten = mat.flatten()
        cdef NodeType res = self.parser_.Parse(0, sent, <float*>flatten.data)
        cdef Parse parse = Parse()
        return parse.from_ptr(res)

    cdef Parse _parse_tag_and_dep(self, str sent,
                                  np.ndarray[np.float32_t, ndim=2] tag,
                                  np.ndarray[np.float32_t, ndim=2] dep):
        cdef np.ndarray[np.float32_t, ndim=1] flat_tag = tag.flatten()
        cdef np.ndarray[np.float32_t, ndim=1] flat_dep = dep.flatten()
        cdef NodeType res = self.parser_.Parse(
                0, sent, <float*>flat_tag.data, <float*>flat_dep.data)
        cdef Parse parse = Parse()
        return parse.from_ptr(res)

    def parse_doc(self, sents, probs):
        return self._parse_doc_tag_and_dep(sents, probs)

    cdef list _parse_doc_tag_and_dep(self, list sents, list probs):
        cdef int doc_size = len(sents), i
        cdef np.ndarray[np.float32_t, ndim=2] cat_scores, dep_scores
        cdef np.ndarray[np.float32_t, ndim=1] cat_flat_scores, dep_flat_scores
        cdef vector[string] csents = sents
        cdef float **tags = <float**>malloc(doc_size * sizeof(float*))
        cdef float **deps = <float**>malloc(doc_size * sizeof(float*))
        for i, _, (cat_scores, dep_scores) in probs:
            cat_flat_scores = cat_scores.flatten()
            dep_flat_scores = dep_scores.flatten()
            tags[i] = <float*>cat_flat_scores.data
            deps[i] = <float*>dep_flat_scores.data
        cdef vector[NodeType] cres = self.parser_.ParseDoc(sents, tags, deps)
        cdef list res = []
        cdef Parse parse
        for i in range(len(sents)):
            parse = Parse()
            parse.from_ptr(cres[i])
            res.append(parse)
        free(tags)
        free(deps)
        return res

        # cdef Parse parse = Parse()
        # return parse.from_ptr(res)
