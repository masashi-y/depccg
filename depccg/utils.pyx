from typing import Dict
import numpy as np
cimport numpy as np
import logging
import json
from .cat cimport Category
from libcpp.pair cimport pair


logger = logging.getLogger(__name__)


def is_json(file_path: str):
    try:
        with open(file_path, 'r') as data_file:
            json.load(data_file)
            return True
    except json.JSONDecodeError:
        return False


def normalize(word):
    if word == "-LRB-":
        return "("
    elif word == "-RRB-":
        return ")"
    elif word == "-LCB-":
        return "{"
    elif word == "-RCB-":
        return "}"
    elif word == "-LSB-":
        return "["
    elif word == "-RSB-":
        return "]"
    else:
        return word


def denormalize(word):
    if word == "(":
        return "-LRB-"
    elif word == ")":
        return "-RRB-"
    elif word == "{":
        return "-LCB-"
    elif word == "}":
        return "-RCB-"
    elif word == "[":
        return "-LSB-"
    elif word == "]":
        return "-RSB-"
    word = word.replace(">", "-RAB-")
    word = word.replace("<", "-LAB-")
    return word


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


def read_partial_tree(string):
    stack = []
    spans = []
    words = []
    buf = list(reversed(string.split()))
    counter = 0
    while buf:
        item = buf.pop()
        if item.startswith('<'):
            cat = item[1:]
            cat = None if cat == 'X' else Category.parse(cat)
            stack.append(cat)
            stack.append(counter)
        elif item == '>':
            start = stack.pop()
            cat = stack.pop()
            spans.append((cat, start, counter - start))
        else:
            items = item.split('|')
            if len(items) == 1:
                words.append(items[0])
            elif len(items) == 2:
                assert len(cat) > 0 and len(word) > 0, \
                    'failed to parse partially annotated sentence.'
                cat, word = items
                words.append(word)
                spans.append((Category.parse(cat), counter))
            counter += 1
    assert len(stack) == 0, 'failed to parse partially annotated sentence.'
    return words, spans


def maybe_split_and_join(string):
    if isinstance(string, list):
        split = string
        join = ' '.join(string)
    else:
        assert isinstance(string, str)
        split = string.split(' ')
        join = string
    return split, join


def read_weights(filename, file_type='json'):
    assert file_type == 'json'
    categories = None
    probs = []
    constraints = []
    for line in open(filename):
        json_dict = json.loads(line.strip())
        if categories is None:
            categories = [Category.parse(cat) for cat in json_dict['categories']]

        sent = ' '.join(denormalize(word) for word in json_dict['words'].split(' '))
        dep = np.array(json_dict['heads']).reshape(json_dict['heads_shape']).astype(np.float32)
        tag = np.array(json_dict['head_tags']).reshape(json_dict['head_tags_shape']).astype(np.float32)
        probs.append((tag, dep))
    return probs, categories

