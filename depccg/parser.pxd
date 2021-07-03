from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
# from libcpp.functional cimport function
from .cat cimport Category, Cat, CatPair
from .tree cimport Tree, ScoredNode, NodeType
from .combinator cimport Op


cdef extern from "depccg.h" namespace "myccg" nogil:
    cdef cppclass RuleCache

    cdef cppclass AgendaItem

    cdef cppclass PartialConstraints:
        PartialConstraints()
        PartialConstraints(const unordered_map[Cat, unordered_set[Cat]]& unary_rules)
        void Add(Cat, unsigned, unsigned)
        void Add(unsigned, unsigned)

    ctypedef vector[NodeType] (*ApplyBinaryRules)(
            const unordered_set[CatPair]&, NodeType, NodeType, unsigned, unsigned)

    ctypedef unordered_set[Cat] (*ApplyUnaryRules)(
            const unordered_map[Cat, unordered_set[Cat]]&, NodeType)

    ApplyUnaryRules DefaultApplyUnaryRules

    ApplyBinaryRules MakeDefaultApplyBinaryRules(const vector[Op]&)

    ApplyUnaryRules EnApplyUnaryRules

    ApplyBinaryRules MakeEnApplyBinaryRules(const vector[Op]&)

    vector[ScoredNode] ParseSentence(
            unsigned id,
            const string& sent,
            float* tag_scores,
            float* dep_scores,
            const unordered_map[string, unordered_set[Cat]]& category_dict,
            const vector[Cat]& tag_list,
            float unary_penalty,
            float beta,
            bool use_beta,
            unsigned pruning_size,
            unsigned nbest,
            const unordered_set[Cat]& possible_root_cats,
            const unordered_map[Cat, unordered_set[Cat]]& unary_rules,
            const unordered_set[CatPair]& seen_rules,
            ApplyBinaryRules apply_binary_rules,
            ApplyUnaryRules apply_unary_rules,
            PartialConstraints constraints,
            unsigned max_length,
            unsigned max_steps) nogil

    vector[vector[ScoredNode]] ParseSentences(
            vector[string]& sents,
            float** tag_scores,
            float** dep_scores,
            const unordered_map[string, unordered_set[Cat]]& category_dict,
            const vector[Cat]& tag_list,
            float unary_penalty,
            float beta,
            bool use_beta,
            unsigned pruning_size,
            unsigned nbest,
            const unordered_set[Cat]& possible_root_cats,
            const unordered_map[Cat, unordered_set[Cat]]& unary_rules,
            const unordered_set[CatPair]& seen_rules,
            const vector[ApplyBinaryRules]& apply_binary_rules,
            ApplyUnaryRules apply_unary_rules,
            vector[PartialConstraints]& constraints,
            unsigned max_length,
            unsigned max_steps,
            bool silent)