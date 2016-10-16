
cdef extern from "c/pqueue.h":
    ctypedef struct PQueue_s:
        size_t size
        size_t capacity
        void **data
        int (*compare)(const void *d1, const void *d2)

    ctypedef PQueue_s PQueue

    PQueue *pqueue_new(int (*compare)(const void *d1, const void *d2))

    void pqueue_delete(PQueue *q)

    void pqueue_enqueue(PQueue *q, const void *data)

    void* pqueue_dequeue(PQueue *q)

