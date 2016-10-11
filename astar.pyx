
from cat import Cat
from libc.stdlib cimport malloc, free
cimport numpy as np

ctypedef np.float32_t ARRAY_T

cdef struct ChartItem:
    float score

cpdef load_unary(str filename):
    cdef:
        list res = []
        list items
        str line
        int comment

    for line in open(filename):
        comment = line.find("#")
        if comment > -1:
            line = line[:comment]
        line = line.strip()
        if len(line) == 0:
            continue
        items = line.split()
        assert len(items) == 2
        res.append((Cat.parse(items[0]), Cat.parse(items[1])))
    return res

cdef ChartItem* init_chart(int size):
    cdef int i, j
    cdef ChartItem *chart = <ChartItem *>malloc(sizeof(ChartItem) * size * size)
    for i in range(size):
        for j in range(size):
            chart[i * size + j] = ChartItem(0.0)
    return chart

cdef float* compute_outsize_probs(np.ndarray[ARRAY_T, ndim=2] supertags):
    cdef int sent_size = supertags.shape[0]
    cdef float* res = <float *>malloc(sizeof(float) * (sent_size + 1) * (sent_size + 1))
    cdef float* from_left = <float *>malloc(sizeof(float) * (sent_size + 1))
    cdef float* from_right = <float *>malloc(sizeof(float) * (sent_size + 1))
    cdef int i, j

    from_left[0] = .0
    from_right[sent_size] = .0

    for i in xrange(sent_size - 1):
        j = sent_size - i
        from_left[i + 1] = from_left[i] + <float>supertags[i, 0]
        from_right[j - 1] = from_right[j] + <float>supertags[j - 1, 0]

    for i in xrange(sent_size + 1):
        for j in xrange(i, sent_size + 1):
            res[i * sent_size + j] = from_left[i] + from_right[j]

    free(from_left)
    free(from_right)
    return res

cdef class AStarParser(object):

    cpdef parse(self, np.ndarray[ARRAY_T, ndim=2] supertags):
        """
        Inputs:
            supertags (np.ndarray[ARRAY_T, ndim=2]): (words, supertags)
        """
        cdef int sent_size = supertags.shape[0]
        cdef int tag_size = supertags.shape[1]
        cdef ChartItem* chart = init_chart(sent_size)
        cdef float* outside_probs = compute_outsize_probs(supertags)
        cdef int i, j
        for i in xrange(sent_size):
            for j in xrange(tag_size):
                chart[i * sent_size + j] = ChartItem(outside_probs[i * sent_size + j])


def test():
    cdef ChartItem* chart = init_chart(100)
    print chart[0].score
    free(chart)
