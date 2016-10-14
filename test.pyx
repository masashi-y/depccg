
from libc.stdlib cimport malloc

ctypedef struct S:
    void *data

cdef S *test():
    cdef S* s = <S *>malloc(sizeof(S))
    cdef dict d = dict()
    s.data = <void *>d
    return s

def funcaa():
    cdef S* s = test()
    print <dict>(s.data)
