from libcpp.string cimport string

cdef extern from "feat.h" namespace "myccg" nogil:
    cdef cppclass Slash

cdef extern from "cat.h" namespace "myccg" nogil:
    ctypedef const Category* Cat
    cdef cppclass Category:
        @staticmethod
        Cat Parse(const string& cat)



cdef class PyCat:

    @staticmethod
    cdef PyCat from_ptr(Cat cat)

