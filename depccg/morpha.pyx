from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free


cdef extern from "c/morpha.h":
    void morph_initialise(const char* filename)
    int morph_analyse(char* buffer, const char *str, int tag)


cdef unsigned MAX_MORPHA_LEN = 32;


cdef class MorphaStemmer:
    def __cinit__(self, str filename):
        cdef string c_filename = filename.encode('utf-8')
        morph_initialise(c_filename.c_str())

    cdef bool use_morpha(self, str word, str pos):
        if len(word) >= MAX_MORPHA_LEN:
            return False

        if pos == "NNP" or pos == "NNPS":
            return False

        if any(not (chr.isalpha() or chr == '-') for chr in word):
            return False

        return True

    cdef string analyze_one(self, string &word, string &pos, char *out):
        cdef char under_bar = b'_'
        cdef string inp = word
        inp += under_bar
        inp += pos
        morph_analyse(out, inp.c_str(), True)

    def analyze(self, list words, list poss):
        assert len(words) == len(poss)
        assert all(isinstance(word, str) for word in words)
        assert all(isinstance(pos, str) for pos in poss)
        cdef str word, pos
        cdef bytes out
        cdef string c_word, c_pos
        cdef char *buffer = <char *> malloc(MAX_MORPHA_LEN * sizeof(char))
        res = []
        for word, pos in zip(words, poss):
            if self.use_morpha(word, pos):
                c_word = word.encode('utf-8')
                c_pos = pos.encode('utf-8')
                self.analyze_one(c_word, c_pos, buffer)
                out = <bytes> buffer
                res.append(out.decode('utf-8'))
            else:
                res.append(word)
        free(buffer)
        return res


