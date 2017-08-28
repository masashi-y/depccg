
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

import chainer
import os
import sys
import json
import pickle

from py.ja_lstm_parser import JaLSTMParser
from py.lstm_parser import LSTMParser
from py.tagger import EmbeddingTagger
from py.japanese_tagger import JaCCGEmbeddingTagger
from py.ja_lstm_tagger import JaLSTMTagger
from py.lstm_tagger import LSTMTagger
from py.lstm_tagger_ph import PeepHoleLSTMTagger
from py.ja_lstm_parser_ph import PeepHoleJaLSTMParser
from py.lstm_parser_bi import BiaffineLSTMParser
from py.precomputed_parser import PrecomputedParser
from py.lstm_parser_bi_fast import FastBiaffineLSTMParser
from py.ja_lstm_parser_bi import BiaffineJaLSTMParser

##############################################################
################# DEPRECATED NOT MAINTAINED ##################
##############################################################

cdef public void tag(const char* model_path, const char* c_str, int length, float* out):

    cdef list py_str = c_str[:length].decode("utf-8").split(" ")
    cdef object tagger = EmbeddingTagger(model_path)
    cdef str model = os.path.join(model_path, "tagger_model")
    chainer.serializers.load_npz(model, tagger)
    cdef np.ndarray[np.float32_t, ndim=1] res = tagger.predict(py_str).flatten()
    cdef int i, j

    memcpy(out, res.data, res.shape[0] * sizeof(float))


cdef public void tag_doc(const char* model_path, const char** c_strs, int* lengths, int doc_size, float** outs):
    cdef object tagger
    cdef str model = os.path.join(model_path, "tagger_model")
    with open(os.path.join(model_path, "tagger_defs.txt")) as f:
        tagger = eval(json.load(f)["model"])(model_path)
    if os.path.exists(model):
        chainer.serializers.load_npz(model, tagger)
    else:
        print >> sys.stderr, "not loading parser model"

    cdef int i
    cdef object _
    cdef object scores
    cdef np.ndarray[np.float32_t, ndim=1] flat_scores
    cdef list res
    cdef list inputs = []
    for i in xrange(doc_size):
        inputs.append(
                c_strs[i][:lengths[i]].decode("utf-8").split(" "))
    res = tagger.predict_doc(inputs)
    for i, _, scores in res:
        if isinstance(scores, tuple):
            scores = scores[0]
        flat_scores = scores.flatten()
        memcpy(outs[i], flat_scores.data, flat_scores.shape[0] * sizeof(float))


cdef public void tag_and_parse_doc(const char* model_path, const char** c_strs, int* lengths, int doc_size, float** tags, float** deps):
    cdef object tagger
    cdef str model = os.path.join(model_path, "tagger_model")

    with open(os.path.join(model_path, "tagger_defs.txt")) as f:
        tagger = eval(json.load(f)["model"])(model_path)
    if os.path.exists(model):
        chainer.serializers.load_npz(model, tagger)
    else:
        print >> sys.stderr, "not loading parser model"

    cdef int i
    cdef object _
    cdef np.ndarray[np.float32_t, ndim=2] cat_scores, dep_scores
    cdef np.ndarray[np.float32_t, ndim=1] cat_flat_scores, dep_flat_scores
    cdef list res
    cdef list inputs = []
    for i in xrange(doc_size):
        inputs.append(
                c_strs[i][:lengths[i]].decode("utf-8").split(" "))

    res = tagger.predict_doc(inputs)

    for i, _, (cat_scores, dep_scores) in res:
        cat_flat_scores = cat_scores.flatten()
        dep_flat_scores = dep_scores.flatten()
        memcpy(tags[i], cat_flat_scores.data, cat_flat_scores.shape[0] * sizeof(float))
        memcpy(deps[i], dep_flat_scores.data, dep_flat_scores.shape[0] * sizeof(float))

