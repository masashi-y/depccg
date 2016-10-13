
from cat import Cat
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue
from ccgbank import Tree, Leaf

cdef extern from"math.h":
    float logf(float)

ctypedef np.float32_t FLOAT_T
ctypedef np.int32_t INT_T

cdef void* void_ptr

cdef struct ChartItem:
    float score
    void* tree

# cdef class ChartItem:
#     cdef float score
#     def __cinit__(self, score):
#         self.score = score

ctypedef ChartItem* ChartItem_ptr
ctypedef vector[vector[ChartItem_ptr]] SUPERTAGS

cdef ChartItem* init_chart(int size):
    cdef int i, j
    cdef ChartItem *chart = <ChartItem *>malloc(sizeof(ChartItem) * size * size)
    for i in range(size):
        for j in range(size):
            chart[i * size + j] = ChartItem(0.0)
    return chart

cdef float* compute_outsize_probs(list supertags):
    cdef int sent_size = len(supertags)
    cdef float* res = <float *>malloc(sizeof(float) * (sent_size + 1) * (sent_size + 1))
    cdef float* from_left = <float *>malloc(sizeof(float) * (sent_size + 1))
    cdef float* from_right = <float *>malloc(sizeof(float) * (sent_size + 1))
    cdef int i, j

    from_left[0] = .0
    from_right[sent_size] = .0

    for i in xrange(sent_size - 1):
        j = sent_size - i
        from_left[i + 1]  = from_left[i] + <float>supertags[i][0].score
        from_right[j - 1] = from_right[j] + <float>supertags[j - 1][0].score

    for i in xrange(sent_size + 1):
        for j in xrange(i, sent_size + 1):
            res[i * sent_size + j] = from_left[i] + from_right[j]

    free(from_left)
    free(from_right)
    return res


cdef class AStarParser(object):
    cdef object tagger
    cdef int tag_size
    cdef list cats

    def __init__(self, tagger):
        self.tagger = tagger
        self.tag_size = len(tagger.targets)
        self.cats = map(Cat.parse, tagger.cats)

    def test(self):
        cdef list sent = "this is test".split(" ")
        cdef SUPERTAGS sup
        for _ in range(3):
            sup.push_back(vector[ChartItem_ptr](0))
        self.assign_supertags(sent, &sup)
        for i in range(3):
            for j in range(10):
                print sup[i][j].score
                print <object>(sup[i][j]).tree


    cdef void assign_supertags(self, list tokens, SUPERTAGS* out):
        """
        Inputs:
            tokens (list[str])
        """
        cdef np.ndarray[FLOAT_T, ndim=2] scores = \
                self.tagger.predict(tokens)
        cdef np.ndarray[long, ndim=2] index = \
                np.argsort(scores, 1)
        cdef np.ndarray[FLOAT_T, ndim=1] totals = \
                np.sum(scores, 1)
        cdef:
            int i, j, k
            int ntokens = len(tokens)
            int ntargets = self.tag_size
        cdef float threshold, score, log_prob
        cdef ChartItem* res
        cdef object cat, leaf

        for i in xrange(ntokens):
            # TODO: threshold cut with beta
            threshold = 0.0
            for j in xrange(ntargets - 1, -1, -1):
                k = index[i, j]
                score = <float>scores[i, k]
                if score < threshold:
                    break
                cat = self.cats[k]
                leaf = Leaf(tokens[i], cat, None)
                res = <ChartItem *>malloc(sizeof(ChartItem))
                log_prob = logf(score / <float>totals[i])
                res.score = log_prob
                res.tree = <void*>leaf
                out[0][i].push_back(res)

    cdef _parse(self, list supertags):
        """
        Inputs:
            supertags (np.ndarray[FLOAT_T, ndim=2]): (words, supertags)
        """
        cdef int sent_size = len(supertags)
        cdef int tag_size = self.tag_size
        cdef int i, j
        cdef ChartItem* chart = init_chart(sent_size)
        cdef float* outside_probs = compute_outsize_probs(supertags)
        # cdef object chart = [None] * tag_size

        for i in xrange(sent_size):
            tag_size = len(supertags[i])
            for j in xrange(tag_size):
                chart[i * sent_size + j] = ChartItem(outside_probs[i * sent_size + j])


