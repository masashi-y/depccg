from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from .cat cimport Cat, Slash, Category
from libcpp cimport bool

cdef extern from "combinator.h" namespace "myccg" nogil:
    cdef enum RuleType:
        FA      = 0,
        BA      = 1,
        FC      = 2,
        BC      = 3,
        GFC     = 4,
        GBC     = 5,
        FX      = 6,
        BX      = 7,
        CONJ    = 8,
        CONJ2   = 9,
        RP      = 10,
        LP      = 11,
        NOISE   = 12,
        UNARY   = 13,
        LEXICON = 14,
        NONE    = 15,
        SSEQ    = 16,
        F_MOD   = 17,
        B_MOD   = 18,
        FWD_TYPERAISE = 19,
        BWD_TYPERAISE = 20,
        COORD = 21

    cdef cppclass CCombinator:
        string ToStr() const
        bool CanApply(Cat left, Cat right)
        Cat Apply(Cat left, Cat right) const
        bool HeadIsLeft(Cat left, Cat right) const

    ctypedef const CCombinator* Op

    cdef cppclass UnaryRule:
        UnaryRule()

    cdef cppclass Conjunction2:
        Conjunction2()

    cdef cppclass Coordinate:
        Coordinate()

    cdef cppclass Conjunction:
        Conjunction()

    cdef cppclass CommaAndVerbPhraseToAdverb:
        CommaAndVerbPhraseToAdverb()

    cdef cppclass ParentheticalDirectSpeech:
        ParentheticalDirectSpeech()

    cdef cppclass RemovePunctuation:
        RemovePunctuation(bool punct_is_left)

    cdef cppclass RemovePunctuationLeft:
        RemovePunctuationLeft()

    cdef cppclass ForwardApplication:
        ForwardApplication()

    cdef cppclass BackwardApplication:
        BackwardApplication()

    cdef cppclass ForwardComposition "myccg::GeneralizedForwardComposition<0, myccg::FC>":
        ForwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass GeneralizedForwardComposition "myccg::GeneralizedForwardComposition<1, myccg::GFC>":
        GeneralizedForwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass BackwardComposition "myccg::GeneralizedBackwardComposition<0, myccg::BC>":
        BackwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass GeneralizedBackwardComposition "myccg::GeneralizedBackwardComposition<1, myccg::GBC>":
        GeneralizedBackwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass HeadFirstCombinator:
        HeadFirstCombinator(Op)

    cdef cppclass HeadFinalCombinator:
        HeadFinalCombinator(Op)

    cdef cppclass ENBackwardApplication:
        ENBackwardApplication()

    cdef cppclass ENForwardApplication:
        ENForwardApplication()

    cdef cppclass Conjoin:
        Conjoin()

    cdef cppclass JAForwardApplication:
        JAForwardApplication()

    cdef cppclass JABackwardApplication:
        JABackwardApplication()

    cdef cppclass JAGeneralizedForwardComposition0 "myccg::JAGeneralizedForwardComposition<0, myccg::FX>":
        JAGeneralizedForwardComposition0(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass JAGeneralizedForwardComposition1 "myccg::JAGeneralizedForwardComposition<1, myccg::FX>":
        JAGeneralizedForwardComposition1(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass JAGeneralizedForwardComposition2 "myccg::JAGeneralizedForwardComposition<2, myccg::FX>":
        JAGeneralizedForwardComposition2(const Slash&, const Slash&, const Slash&, const string& string)


    cdef cppclass JAGeneralizedBackwardComposition0 "myccg::JAGeneralizedBackwardComposition<0, myccg::BC>":
        JAGeneralizedBackwardComposition0(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass JAGeneralizedBackwardComposition1 "myccg::JAGeneralizedBackwardComposition<1, myccg::BC>":
        JAGeneralizedBackwardComposition1(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass JAGeneralizedBackwardComposition2 "myccg::JAGeneralizedBackwardComposition<2, myccg::BC>":
        JAGeneralizedBackwardComposition2(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass JAGeneralizedBackwardComposition3 "myccg::JAGeneralizedBackwardComposition<3, myccg::BC>":
        JAGeneralizedBackwardComposition3(const Slash&, const Slash&, const Slash&, const string& string)


    cdef cppclass RemoveDisfluency:
        RemoveDisfluency()

    cdef cppclass UnknownCombinator:
        UnknownCombinator()


cdef class Combinator:
    cdef Op op_

    @staticmethod
    cdef Combinator from_ptr(Op op)

    cdef bool _can_apply(self, Category left, Category right)

    cdef Category _apply(self, Category left, Category right)

    cdef bool _head_is_left(self, Category left, Category right)

cdef vector[Op] combinator_list_to_vector(list combinators)
