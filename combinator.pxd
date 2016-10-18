
from cat cimport Slash, Cat
cimport cat
from structs cimport FA, BA, FC, BX, GFC, GBX, CONJ, RP, LP, NOISE, UNARY, LEXICON, NONE

cdef class Combinator(object):
    cdef readonly int rule_type

cdef class UnaryRule(Combinator):
    pass

cdef class Conjunction(Combinator):
    pass

cdef class RemovePunctuation(Combinator):
    cdef readonly bint punct_is_left


cdef class RemovePunctuationLeft(Combinator):
    pass

cdef class SpecialCombinator(Combinator):
    cdef readonly Cat left
    cdef readonly Cat right
    cdef readonly Cat result
    cdef readonly bint head_is_left


cdef class ForwardApplication(Combinator):
    pass

cdef class BackwardApplication(Combinator):
    pass

cdef class ForwardComposition(Combinator):
    cdef readonly Slash left_slash
    cdef readonly Slash right_slash
    cdef readonly Slash result_slash


cdef class BackwardComposition(Combinator):
    cdef readonly Slash left_slash
    cdef readonly Slash right_slash
    cdef readonly Slash result_slash


cdef class GeneralizedForwardComposition(Combinator):
    cdef readonly Slash left_slash
    cdef readonly Slash right_slash
    cdef readonly Slash result_slash


cdef class GeneralizedBackwardComposition(Combinator):
    cdef readonly Slash left_slash
    cdef readonly Slash right_slash
    cdef readonly Slash result_slash


cdef list standard_combinators

cdef Combinator unary_rule
