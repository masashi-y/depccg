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

    cdef cppclass CUnaryRule "myccg::UnaryRule":
        CUnaryRule()

    cdef cppclass CConjunction2 "myccg::Conjunction2":
        CConjunction2()

    cdef cppclass CCoordinate "myccg::Coordinate":
        CCoordinate()

    cdef cppclass CConjunction "myccg::Conjunction":
        CConjunction()

    cdef cppclass CCommaAndVerbPhraseToAdverb "myccg::CommaAndVerbPhraseToAdverb":
        CCommaAndVerbPhraseToAdverb()

    cdef cppclass CParentheticalDirectSpeech "myccg::ParentheticalDirectSpeech":
        CParentheticalDirectSpeech()

    cdef cppclass CRemovePunctuation "myccg::RemovePunctuation":
        CRemovePunctuation(bool punct_is_left)

    cdef cppclass CRemovePunctuationLeft "myccg::RemovePunctuationLeft":
        CRemovePunctuationLeft()

    cdef cppclass CForwardApplication "myccg::ForwardApplication":
        CForwardApplication()

    cdef cppclass CBackwardApplication "myccg::BackwardApplication":
        CBackwardApplication()

    cdef cppclass CForwardComposition "myccg::GeneralizedForwardComposition<0, myccg::FC>":
        CForwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass CGeneralizedForwardComposition "myccg::GeneralizedForwardComposition<1, myccg::GFC>":
        CGeneralizedForwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass CBackwardComposition "myccg::GeneralizedBackwardComposition<0, myccg::BC>":
        CBackwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass CGeneralizedBackwardComposition "myccg::GeneralizedBackwardComposition<1, myccg::GBC>":
        CGeneralizedBackwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass CHeadFirstCombinator "myccg::HeadFirstCombinator":
        CHeadFirstCombinator(Op)

    cdef cppclass CHeadFinalCombinator "myccg::HeadFinalCombinator":
        CHeadFinalCombinator(Op)

    cdef cppclass CENBackwardApplication "myccg::ENBackwardApplication":
        CENBackwardApplication()

    cdef cppclass CENForwardApplication "myccg::ENForwardApplication":
        CENForwardApplication()

    cdef cppclass CConjoin "myccg::Conjoin":
        CConjoin()

    cdef cppclass CJAForwardApplication "myccg::JAForwardApplication":
        CJAForwardApplication()

    cdef cppclass CJABackwardApplication "myccg::JABackwardApplication":
        CJABackwardApplication()

    cdef cppclass CJAGeneralizedForwardComposition0 "myccg::JAGeneralizedForwardComposition<0, myccg::FX>":
        CJAGeneralizedForwardComposition0(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass CJAGeneralizedForwardComposition1 "myccg::JAGeneralizedForwardComposition<1, myccg::FX>":
        CJAGeneralizedForwardComposition1(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass CJAGeneralizedForwardComposition2 "myccg::JAGeneralizedForwardComposition<2, myccg::FX>":
        CJAGeneralizedForwardComposition2(const Slash&, const Slash&, const Slash&, const string& string)


    cdef cppclass CJAGeneralizedBackwardComposition0 "myccg::JAGeneralizedBackwardComposition<0, myccg::BC>":
        CJAGeneralizedBackwardComposition0(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass CJAGeneralizedBackwardComposition1 "myccg::JAGeneralizedBackwardComposition<1, myccg::BC>":
        CJAGeneralizedBackwardComposition1(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass CJAGeneralizedBackwardComposition2 "myccg::JAGeneralizedBackwardComposition<2, myccg::BC>":
        CJAGeneralizedBackwardComposition2(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass CJAGeneralizedBackwardComposition3 "myccg::JAGeneralizedBackwardComposition<3, myccg::BC>":
        CJAGeneralizedBackwardComposition3(const Slash&, const Slash&, const Slash&, const string& string)

    cdef cppclass CSpecialCombinator "myccg::SpecialCombinator":
        CSpecialCombinator(Cat left, Cat right, Cat result, bool head_is_left)

    cdef cppclass CRemoveDisfluency "myccg::RemoveDisfluency":
        CRemoveDisfluency()

    cdef cppclass CUnknownCombinator "myccg::UnknownCombinator":
        CUnknownCombinator()


cdef class Combinator:
    cdef Op op_

    cdef bool _can_apply(self, Category left, Category right)

    cdef Category _apply(self, Category left, Category right)

    cdef bool _head_is_left(self, Category left, Category right)

cdef vector[Op] combinator_list_to_vector(list combinators)
