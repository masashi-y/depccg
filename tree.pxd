
from cat import Cat
from combinator import Combinator
import combinator
from structs cimport LEXICON, NONE, UNARY

cdef class Node(object):
    cdef readonly object cat
    cdef readonly int rule_type

cdef class Leaf(Node):
    cdef readonly unicode word
    cdef readonly int pos

cdef class Tree(Node):
    cdef readonly list children
    cdef readonly bint left_is_head
    cdef readonly object op
