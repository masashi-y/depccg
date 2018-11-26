
import numpy as np
import re

def drop_brackets(cat):
    if cat.startswith('(') and \
        cat.endswith(')') and \
        find_closing_bracket(cat, 0) == len(cat)-1:
        return cat[1:-1]
    else:
        return cat


def find_closing_bracket(source, start):
    open_brackets = 0
    for i, c in enumerate(source):
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1

        if open_brackets == 0:
            return i

    raise Exception("Mismatched brackets in string: " + source)


def find_non_nested_char(haystack, needles):
    open_brackets = 0
    # for i, c in enumerate(haystack):
    for i in range(len(haystack) -1, -1, -1):
        c = haystack[i]
        if c == '(':
            open_brackets += 1
        elif c == ')':
            open_brackets -= 1
        elif open_brackets == 0:
            for n in needles:
                if n == c: return i
    return -1


def get_context_by_window(items, window_size, lpad, rpad):
    res = []
    for i, item in enumerate(items):
        context = []
        if window_size - i > 0:
            for j in range(window_size - i):
                context.append(lpad)
            for j in range(i):
                context.append(items[j])
        else:
            for j in range(i - window_size, i):
                context.append(items[j])
        context.append(item)
        if i + window_size >= len(items):
            for j in range(i + 1, len(items)):
                context.append(items[j])
            for j in range(i + window_size - len(items) + 1):
                context.append(rpad)
        else:
            for j in range(i + 1, i + window_size + 1):
                context.append(items[j])
        assert len(context) == window_size * 2 + 1

        res.append(context)
    return res


def read_pretrained_embeddings(filepath):
    nvocab = 0
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


import codecs
def read_model_defs(filepath):
    res = {}
    for i, line in enumerate(codecs.open(filepath, encoding="utf-8")):
        word, _ = line.strip().split(" ")
        res[word] = i
    return res


