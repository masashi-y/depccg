
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import chainer
import os


cdef public void tag(const char* model_path, const char* c_str, int length, float* out):
    from tagger import EmbeddingTagger
    cdef list py_str = c_str[:length].decode("utf-8").encode("utf-8").split(" ")
    cdef object tagger = EmbeddingTagger(model_path)
    cdef str model = os.path.join(model_path, "tagger_model")
    chainer.serializers.load_npz(model, tagger)
    cdef np.ndarray[np.float32_t, ndim=1] res = tagger.predict(py_str).flatten()
    cdef int i, j
    # cdef int nword = res.shape[0]
    # cdef int ntag = res.shape[1]

    memcpy(out, res.data, res.shape[0] * sizeof(float))
    # for i in xrange(nword):
        # out[i] = res[i]
        # for j in xrange(ntag):


cdef public void tag_doc(const char* model_path, const char** c_strs, int* lengths, int doc_size, float** outs):
    from tagger import EmbeddingTagger
    cdef object tagger = EmbeddingTagger(model_path)
    cdef str model = os.path.join(model_path, "tagger_model")
    chainer.serializers.load_npz(model, tagger)

    cdef int i
    cdef object _
    cdef np.ndarray[np.float32_t, ndim=2] scores
    cdef np.ndarray[np.float32_t, ndim=1] flat_scores
    cdef list res
    cdef list inputs = []
    for i in xrange(doc_size):
        inputs.append(
                c_strs[i][:lengths[i]].decode("utf-8").encode("utf-8").split(" "))

    res = tagger.predict_doc(inputs)
    for i, _, scores in res:
        flat_scores = scores.flatten()
        memcpy(outs[i], flat_scores.data, flat_scores.shape[0] * sizeof(float))

