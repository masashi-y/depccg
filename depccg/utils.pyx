from typing import Dict
import numpy as np
cimport numpy as np
import logging
from .cat cimport Category
from libcpp.pair cimport pair


logger = logging.getLogger(__name__)


def read_pretrained_embeddings(filepath: str) -> np.ndarray:
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


def read_model_defs(filepath: str) -> Dict[str, int]:
    res = {}
    for i, line in enumerate(open(filepath, encoding='utf-8')):
        word, _ = line.strip().split(' ')
        res[word] = i
    return res


def remove_comment(line):
    comment = line.find('#')
    if comment != -1:
        line = line[:comment]
    return line.strip()


cdef vector[Cat] cat_list_to_vector(list cats):
    cdef vector[Cat] results
    cdef Category cat
    for cat in cats:
        results.push_back(cat.cat_)
    return results


cdef unordered_set[Cat] cat_list_to_unordered_set(list cats):
    cdef unordered_set[Cat] results
    cdef Category cat
    for cat in cats:
        results.insert(cat.cat_)
    return results


cdef unordered_map[string, vector[bool]] convert_cat_dict(dict cat_dict, list cat_list):
    cdef unordered_map[string, vector[bool]] results
    cdef vector[bool] tmp
    cdef str py_word
    cdef string c_word
    cdef list cats
    cat_to_index = {str(cat): i for i, cat in enumerate(cat_list)}
    for py_word, cats in cat_dict.items():
        c_word = py_word.encode('utf-8')
        tmp = vector[bool](len(cat_list), False)
        for cat in cats:
            tmp[cat_to_index[str(cat)]] = True
        results[c_word] = tmp
    logger.info(f'convert_cat_dict: {results.size()}')
    return results


cdef unordered_map[Cat, vector[Cat]] convert_unary_rules(list unary_rules):
    cdef unordered_map[Cat, vector[Cat]] results
    cdef vector[Cat] tmp
    cdef Category cat1, cat2
    for cat1, cat2 in unary_rules:
        if results.count(cat1.cat_) == 0:
            results[cat1.cat_] = vector[Cat]()
        results[cat1.cat_].push_back(cat2.cat_)
    return results


cpdef read_unary_rules(filename):
    results = []
    for line in open(filename, encoding='utf-8'):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        cat1, cat2 = line.split()
        cat1 = Category.parse(cat1)
        cat2 = Category.parse(cat2)
        results.append((cat1, cat2))
    logger.info(f'load {len(results)} unary rules')
    return results


cpdef read_cat_dict(filename):
    results = {}
    for line in open(filename, encoding='utf-8'):
        word, *cats = line.strip().split()
        results[word] = [Category.parse(cat) for cat in cats]
    logger.info(f'load {len(results)} cat dictionary entries')
    return results


cpdef read_cat_list(filename):
    results = []
    for line in open(filename, encoding='utf-8'):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        cat = line.split()[0]
        results.append(Category.parse(cat))
    logger.info(f'load {len(results)} categories')
    return results


cpdef read_seen_rules(filename, preprocess):
    cdef list results = []
    cdef Category cat1, cat2
    for line in open(filename, encoding='utf-8'):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        tmp1, tmp2 = line.split()
        cat1 = preprocess(Category.parse(tmp1))
        cat2 = preprocess(Category.parse(tmp2))
        results.append((cat1, cat2))
    logger.info(f'load {len(results)} seen rules')
    return results


cdef unordered_set[CatPair] convert_seen_rules(seen_rule_list):
    cdef unordered_set[CatPair] results
    cdef Category cat1, cat2
    for cat1, cat2 in seen_rule_list:
        results.insert(CatPair(cat1.cat_, cat2.cat_))
    return results


cdef unordered_set[Cat] read_possible_root_categories(list cats):
    cdef unordered_set[Cat] res
    cdef Category tmp
    for cat in cats:
        tmp = Category.parse(cat)
        res.insert(tmp.cat_)
    return res


def maybe_split_and_join(string):
    if isinstance(string, list):
        split = string
        join = string.split(' ')
    else:
        assert isinstance(string, str)
        split = string.split(' ')
        join = string
    return split, join
