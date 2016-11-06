
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
import chainer
import os


cdef public void tag(const char* model_path, const char* c_str, int length, float* out):
    from tagger import EmbeddingTagger
    cdef list py_str = c_str[:length].encode("utf-8").split(" ")
    cdef object tagger = EmbeddingTagger(model_path)
    cdef str model = os.path.join(model_path, "tagger_model")
    chainer.serializers.load_npz(model, tagger)
    cdef np.ndarray[np.float32_t, ndim=2] res = tagger.predict(py_str)
    cdef int i, j
    cdef int nword = res.shape[0]
    cdef int ntag = res.shape[1]

    for i in xrange(nword):
        for j in xrange(ntag):
            out[i * ntag + j] = res[i, j]

