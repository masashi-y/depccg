from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from .cat cimport Cat

cdef extern from "combinator.h" namespace "myccg" nogil:
    cdef cppclass Combinator:
        const string ToStr() const
    ctypedef const Combinator* Op


cdef extern from "grammar.h" namespace "myccg" nogil:
    cdef const unordered_set[Cat] en_possible_root_cats     "myccg::En::possible_root_cats"
    cdef const vector[Op]         en_headfirst_binary_rules "myccg::En::headfirst_binary_rules"

    cdef const unordered_set[Cat] ja_possible_root_cats     "myccg::Ja::possible_root_cats"
    cdef const vector[Op]         ja_headfinal_binary_rules "myccg::Ja::headfinal_binary_rules"
