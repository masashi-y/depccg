import numpy as np
cimport numpy as np
import logging
from .cat cimport Category
from .combinator import SpecialCombinator
from libcpp.pair cimport pair


from depccg.py_utils import (
    is_json,
    normalize,
    denormalize,
    read_pretrained_embeddings,
    read_model_defs,
    remove_comment,
    read_partial_tree,
    maybe_split_and_join,
    read_weights
)

logger = logging.getLogger(__name__)


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


cdef unordered_map[string, unordered_set[Cat]] convert_cat_dict(dict cat_dict):
    cdef unordered_map[string, unordered_set[Cat]] results
    cdef str py_word
    cdef string c_word
    cdef list cats
    for py_word, cats in cat_dict.items():
        c_word = py_word.encode('utf-8')
        results[c_word] = cat_list_to_unordered_set(cats)
    return results


cdef unordered_map[Cat, unordered_set[Cat]] convert_unary_rules(list unary_rules):
    cdef unordered_map[Cat, unordered_set[Cat]] results
    cdef unordered_set[Cat] tmp
    cdef Category cat1, cat2
    for cat1, cat2 in unary_rules:
        if results.count(cat1.cat_) == 0:
            results[cat1.cat_] = unordered_set[Cat]()
        results[cat1.cat_].insert(cat2.cat_)
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


cpdef read_binary_rules(filename):
    results = []
    for line in open(filename, encoding='utf-8'):
        line = remove_comment(line.strip())
        if len(line) == 0:
            continue
        head_is_left, left, right, result = line.split()
        left = Category(left)
        right = Category(right)
        result = Category(result)
        combinator = SpecialCombinator(left, right, result, head_is_left)
        results.append(combinator)
    logger.info(f'load {len(results)} binary rules')
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