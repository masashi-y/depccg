
cdef extern from "c/pqueue.h":
    ctypedef struct PQueue:
        size_t size
        size_t capacity
        void **data
        int (*cmp)(const void *d1, const void *d2)

    PQueue *pqueue_new(int (*cmp)(const void *d1, const void *d2))

    void pqueue_delete(PQueue *q)

    void pqueue_enqueue(PQueue *q, const void *data)

    void *pqueue_dequeue(PQueue *q)

