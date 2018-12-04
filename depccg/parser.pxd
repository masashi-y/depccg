from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from .cat cimport Category, Cat, CatPair
from .tree cimport Tree, ScoredNode
from .combinator cimport Op


cdef extern from "depccg.h" namespace "myccg" nogil:
    cdef cppclass RuleCache

    cdef cppclass AgendaItem

    ctypedef vector[RuleCache]& (*ApplyBinaryRules)(
            unordered_map[CatPair, vector[RuleCache]]&,
            const vector[Op]&, const unordered_set[CatPair]&, Cat, Cat)

    ctypedef vector[Cat] (*ApplyUnaryRules)(
            const unordered_map[Cat, vector[Cat]]&, NodeType)

    ApplyUnaryRules EnApplyUnaryRules

    ApplyUnaryRules JaApplyUnaryRules

    ApplyBinaryRules EnGetRules

    ApplyBinaryRules JaGetRules

    vector[ScoredNode] ParseSentence(
            unsigned id,
            const string& sent,
            float* tag_scores,
            float* dep_scores,
            const unordered_map[string, unordered_set[Cat]]& category_dict,
            const vector[Cat]& tag_list,
            float beta,
            bool use_beta,
            unsigned pruning_size,
            unsigned nbest,
            const unordered_set[Cat]& possible_root_cats,
            const unordered_map[Cat, vector[Cat]]& unary_rules,
            const vector[Op]& binary_rules,
            unordered_map[CatPair, vector[RuleCache]]& cache,
            const unordered_set[CatPair]& seen_rules,
            ApplyBinaryRules apply_binary_rules,
            ApplyUnaryRules apply_unary_rules,
            unsigned max_length) nogil

    vector[vector[ScoredNode]] ParseSentences(
            vector[string]& sents,
            float** tag_scores,
            float** dep_scores,
            const unordered_map[string, unordered_set[Cat]]& category_dict,
            const vector[Cat]& tag_list,
            float beta,
            bint use_beta,
            unsigned pruning_size,
            unsigned nbest,
            const unordered_set[Cat]& possible_root_cats,
            const unordered_map[Cat, vector[Cat]]& unary_rules,
            const vector[Op]& binary_rules,
            unordered_map[CatPair, vector[RuleCache]]& cache,
            const unordered_set[CatPair]& seen_rules,
            ApplyBinaryRules apply_binary_rules,
            ApplyUnaryRules apply_unary_rules,
            unsigned max_length)


