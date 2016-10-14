from libc.stdlib cimport malloc, realloc, free, calloc

ctypedef int PRIORITY_T

ctypedef struct node_t:
    PRIORITY_T priority
    void* data

ctypedef struct heap_t:
    node_t* nodes
    int len
    int size

cdef void push(heap_t* h, PRIORITY_T priority, void* data):
    if h.len + 1 >= h.size:
        h.size = h.size * 2 if h.size != 0 else 4
        h.nodes = <node_t*>realloc(h.nodes, h.size * sizeof(node_t))

    cdef int i = h.len + 1
    cdef int j = i / 2
    while i > 1 and h.nodes[j].priority > priority:
        h.nodes[i] = h.nodes[j]
        i = j
        j = j / 2

    h.nodes[i].priority = priority
    h.nodes[i].data = data
    h.len += 1

cdef int isempty(heap_t* h):
    return 1 if h.len == 0 else 0

cdef void* pop(heap_t* h):
    cdef int i, j, k
    if h.len == 0:
        return NULL
    cdef void* data = h.nodes[1].data
    h.nodes[1] = h.nodes[h.len]
    h.len -= 1
    i = 1
    while True:
        k = i
        j = 2 * i
        if j <= h.len and h.nodes[j].priority < h.nodes[k].priority:
            k = j
        if j + 1 <= h.len and h.nodes[j + 1].priority < h.nodes[k].priority:
            k = j + 1
        if k == i:
            break
        h.nodes[i] = h.nodes[k]
        i = k
    h.nodes[i] = h.nodes[h.len + 1]
    return data

def test(inp):
    cdef heap_t* h = <heap_t*>calloc(1, sizeof(heap_t))
    cdef int i
    cdef PRIORITY_T p
    for i, (p, x) in enumerate(inp):
        push(h, p, <void*>x)

    while not isempty(h):
        print <object>pop(h)
