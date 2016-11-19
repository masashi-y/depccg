
import numpy as np

from preshed.maps cimport PreshMap
import cat
import re

cpdef str drop_brackets(str cat):
    if cat.startswith('(') and \
        cat.endswith(')') and \
        find_closing_bracket(cat, 0) == len(cat)-1:
        return cat[1:-1]
    else:
        return cat


cpdef int find_closing_bracket(str source, int start) except -1:
    cdef int open_brackets = 0
    for i, c in enumerate(source):
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1

        if open_brackets == 0:
            return i

    raise Exception("Mismatched brackets in string: " + source)


cpdef int find_non_nested_char(str haystack, str needles):
    cdef int open_brackets = 0

    for i, c in enumerate(haystack):
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1
        elif open_brackets == 0:
            for n in needles:
                if n == c: return i
    return -1


cpdef list get_context_by_window(
        list items, int window_size, object lpad, object rpad):
    cdef list res = []
    cdef list context
    cdef int i, j
    cdef object item
    for i, item in enumerate(items):
        context = []
        if window_size - i > 0:
            for j in xrange(window_size - i):
                context.append(lpad)
            for j in xrange(i):
                context.append(items[j])
        else:
            for j in xrange(i - window_size, i):
                context.append(items[j])
        context.append(item)
        if i + window_size >= len(items):
            for j in xrange(i + 1, len(items)):
                context.append(items[j])
            for j in xrange(i + window_size - len(items) + 1):
                context.append(rpad)
        else:
            for j in xrange(i + 1, i + window_size + 1):
                context.append(items[j])
        assert len(context) == window_size * 2 + 1

        res.append(context)
    return res


cpdef np.ndarray[FLOAT_T, ndim=2] read_pretrained_embeddings(str filepath):
    cdef object io
    cdef int i, dim
    cdef int nvocab = 0
    cdef str line
    cdef np.ndarray[FLOAT_T, ndim=2] res

    io = open(filepath)
    dim = len(io.readline().split())
    io.seek(0)
    for _ in io:
        nvocab += 1
    io.seek(0)
    res = np.empty((nvocab, dim), dtype=np.float32)
    for i, line in enumerate(io):
        line = line.strip()
        if len(line) == 0: continue
        res[i] = line.split()
    io.close()
    return res


cpdef dict read_model_defs(str filepath):
    """
    input file is made up of lines, "ITEM FREQUENCY".
    """
    cdef dict res = {}
    cdef int i
    cdef str line, word, _

    for i, line in enumerate(open(filepath)):
        word, _ = line.strip().split(" ")
        res[word] = i
    return res


cpdef dict load_unary(str filename):
    cdef dict res = {}
    cdef str line
    cdef int comment
    cdef list items
    cdef object inp, out

    for line in open(filename):
        comment = line.find("#")
        if comment > -1:
            line = line[:comment]
        line = line.strip()
        if len(line) == 0:
            continue
        items = line.split()
        assert len(items) == 2
        inp = cat.parse(items[0])
        out = cat.parse(items[1])
        if res.has_key(inp):
            res[inp].append(out)
        else:
            res[inp] = [out]
    return res

feat = re.compile("\[nb\]|\[X\]")
cdef PreshMap load_seen_rules(str filename):
    cdef PreshMap res = PreshMap()
    cdef str line
    cdef int comment
    cdef list items
    cdef object cat1, cat2

    for line in open(filename):
        comment = line.find("#")
        if comment > -1:
            line = line[:comment]
        line = line.strip()
        if len(line) == 0:
            continue
        items = line.split()
        assert len(items) == 2
        cat1 = cat.parse(feat.sub("", items[0]))
        cat2 = cat.parse(feat.sub("", items[1]))
        res[hash_int_int(cat1.id, cat2.id)] = 1
    return res


