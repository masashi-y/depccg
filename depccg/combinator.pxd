from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from .cat cimport Cat, Slash
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

    cdef cppclass Combinator:
        const string ToStr() const
    ctypedef const Combinator* Op

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

    cdef cppclass GeneralizedForwardComposition[int, Order, RuleType]:
        GeneralizedForwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass GeneralizedBackwardComposition[int, Order, RuleType]:
        GeneralizedBackwardComposition(const Slash& left, const Slash& right, const Slash& result)

    cdef cppclass HeadFirstCombinator[T]:
        HeadFirstCombinator(T)

    cdef cppclass HeadFinalCombinator[T]:
        HeadFinalCombinator(T)

    cdef HeadFirstCombinator[T]* HeadFirst[T](T)

    cdef HeadFinalCombinator[T]* HeadFinal[T](T)

    cdef cppclass ENBackwardApplication:
        ENBackwardApplication()

    cdef cppclass ENForwardApplication:
        ENForwardApplication()

    cdef const vector[Op] en_binary_rules

    cdef const vector[Op] ja_binary_rules
