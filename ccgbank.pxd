
from cat cimport Cat
from combinator cimport Combinator
cimport combinator
from structs cimport LEXICON, NONE, UNARY

cdef class AutoReader(object):
    cdef list lines

cdef class AutoLineReader(object):
    cdef str line
    cdef int index

cdef class Node(object):
    cdef readonly Cat cat
    cdef readonly int rule_type

cdef class Leaf(Node):
    cdef readonly str word
    cdef readonly int pos

cdef class Tree(Node):
    cdef readonly list children
    cdef readonly bint left_is_head
    cdef readonly Combinator op


