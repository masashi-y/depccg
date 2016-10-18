# -*- coding: utf-8 -*-

cdef enum Slash_E:
    FwdApp = 0
    BwdApp = 1
    EitherApp = 2

cdef class Slash(object):
    cdef readonly int _slash


cdef class Cat(object):
    cdef str string
    cdef readonly int id


cdef class Functor(Cat):
    cdef readonly Cat left
    cdef readonly Slash slash
    cdef readonly Cat right
    cdef readonly object semantics


cdef class Atomic(Cat):
    cdef readonly str type
    cdef readonly object feat
    cdef readonly object semantics

cdef Cat parse(str cat)

cdef Cat parse_uncached(str cat)

cdef Cat make(Cat left, Slash op, Cat right)
