from typing import List
from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
cimport numpy as np
import copy

from tqdm import tqdm
from depccg.tree import Tree, ScoredTree
from depccg.cat import Category
from depccg.types import ScoringResult

cdef extern from "<limits>":
    cdef unsigned UINT_MAX


cdef extern from "depccg/parsing.h" namespace "parsing":
    cdef struct cell_item:
        bint fin
        unsigned cat
        cell_item *left
        cell_item *right
        float in_score
        float out_score
        unsigned start_of_span
        unsigned span_length
        unsigned head_id
        unsigned rule_id

        float score()


cdef extern from "depccg/parsing.h":
    cdef struct combinator_result:
        unsigned cat_id
        unsigned rule_id
        bint head_is_left
        string op_string
        string op_symbol

    ctypedef unordered_map[pair[unsigned, unsigned], vector[combinator_result]] cache_type

    ctypedef int (*scaffold_type)(void *callback_func, unsigned x, unsigned y, vector[combinator_result] *results) except -1

    ctypedef unsigned (*finalizer_type)(cell_item *, unsigned *, cache_type *cache, void *)

    cdef struct config:
        unsigned num_tags
        float unary_penalty
        float beta
        bint use_beta
        unsigned pruning_size
        unsigned nbest
        unsigned max_step

    cdef unsigned parse_sentence(
        float *tag_scores,
        float *dep_scores,
        unsigned length,
        const unordered_set[unsigned] &possible_root_cats,
        void *binary_callback,
        void *unary_callback,
        finalizer_type finalizer_callback,
        scaffold_type scaffold,
        void *finalizer_args,
        cache_type *cache,
        config *config) except +


cdef int scaffold(
    void *callback_func,
    unsigned x,
    unsigned y,
    vector[combinator_result] *results,
) except -1:
    cdef list py_results
    cdef cat_id, rule_id
    cdef bint head_is_left
    cdef combinator_result c_result

    py_results = (<object>callback_func)(<object>x, <object>y)
    for cat_id, rule_id, result in py_results:
        c_result.cat_id = cat_id
        c_result.rule_id = rule_id
        c_result.head_is_left = result.head_is_left
        c_result.op_string = result.op_string.encode('utf-8')
        c_result.op_symbol = result.op_symbol.encode('utf-8')
        results.push_back(c_result)
    return 0;


cdef init_config(config *c_config, dict kwargs):
    c_config.num_tags = kwargs['num_tags']
    c_config.unary_penalty = kwargs.pop('unary_penalty', 0.1)
    c_config.beta = kwargs.pop('beta', 0.00001)
    c_config.use_beta = kwargs.pop('use_beta', True)
    c_config.pruning_size = kwargs.pop('pruning_size', 50)
    c_config.nbest = kwargs.pop('nbest', 1)
    c_config.max_step = kwargs.pop('max_step', 10000000)


cdef unsigned retrieve_tree(
    cell_item *item,
    unsigned *token_id,
    cache_type *cache,
    void *c_kwargs,
):
    """
    retrieve (n-best) parsing results from cell_item.
    they are placed in kwargs['stack']
    """
    cdef pair[unsigned, unsigned] key
    cdef object kwargs = <object>c_kwargs

    if item.fin:
        kwargs['scores'].append(item.score())
        return retrieve_tree(item.left, token_id, cache, c_kwargs)

    cat = kwargs['categories'][item.cat]
    stack = kwargs['stack']
    if item.left == NULL and item.right == NULL:
        stack.append(
            Tree.make_terminal(kwargs['tokens'][token_id[0]], cat)
        )
        token_id[0] += 1
        return item.cat
    else:
        child_cat = retrieve_tree(item.left, token_id, cache, c_kwargs)
        kwargs = <object>c_kwargs
        if item.right == NULL:
            child = stack.pop()
            key.first = child_cat
            key.second = -1
            combinator_result = cache[0][key][item.rule_id]
            stack.append(
                Tree.make_unary(
                    cat,
                    child,
                    combinator_result.op_string.decode('utf-8'),
                    combinator_result.op_symbol.decode('utf-8'),
                )
            )
            return item.cat
        right_child_cat = retrieve_tree(item.right, token_id, cache, c_kwargs)
        key.first = child_cat
        key.second = right_child_cat
        kwargs = <object>c_kwargs
        right = stack.pop()
        left = stack.pop()
        combinator_result = cache[0][key][item.rule_id]
        stack.append(
            Tree.make_binary(
                cat,
                left,
                right,
                combinator_result.op_string.decode('utf-8'),
                combinator_result.op_symbol.decode('utf-8'),
            )
        )
        return item.cat


def run(
    list doc,
    list scoring_results,
    list categories,
    apply_binary_rules,
    apply_unary_rules,
    object possible_root_cats,
    process_id=0,
    **kwargs
) -> List[ScoredTree]:

    def failed():
        return [
            ScoredTree(
                tree=Tree.make_terminal("FAILED", Category.parse("NP")),
                score=-float('inf')
            )
        ]

    if len(set(categories)) != len(categories):
        raise RuntimeError(
            'argument `categories` cannot contain duplicate elements.'
        )

    categories_ = copy.copy(categories)

    category_ids = {
        category: index
        for index, category in enumerate(categories_)
    }

    def maybe_add_and_get(cat):
        if cat not in category_ids:
            categories_.append(cat)
            category_ids[cat] = len(category_ids)
        if len(categories_) >= UINT_MAX:
            raise RuntimeError('too many categories')
        return category_ids[cat]

    def binary_callback(x_id, y_id):
        x, y = categories_[x_id], categories_[y_id]

        results = []
        for rule_id, result in enumerate(apply_binary_rules(x, y)):
            cat_id = maybe_add_and_get(result.cat)
            results.append((cat_id, rule_id, result))
        return results

    def unary_callback(x_id, _):
        x = categories_[x_id]

        results = []
        for rule_id, result in enumerate(apply_unary_rules(x)):
            cat_id = maybe_add_and_get(result.cat)
            results.append((cat_id, rule_id, result))
        return results

    cdef list tokens
    cdef np.ndarray[float, ndim=2, mode='c'] tag_scores
    cdef np.ndarray[float, ndim=2, mode='c'] dep_scores
    cdef float *c_tag_scores
    cdef float *c_dep_scores
    cdef unsigned length, status
    cdef unordered_set[unsigned] c_possible_root_cat
    cdef cache_type c_cache
    cdef config c_config
    init_config(&c_config, kwargs)

    cdef unsigned cat_id
    for cat in possible_root_cats:
        cat_id = maybe_add_and_get(cat)
        c_possible_root_cat.insert(cat_id)

    all_results = []
    iter_ = tqdm(
        list(zip(doc, scoring_results)),
        desc=f'#{process_id:>2} ',
        position=process_id
    )
    for tokens, (tag_scores, dep_scores) in iter_:
        c_tag_scores = <float*>tag_scores.data
        c_dep_scores = <float*>dep_scores.data
        length = len(tokens)

        if (
            'max_length' in kwargs
            and len(tokens) > kwargs['max_length']
        ):
            all_results.append(failed())
            continue

        results = []
        scores = []
        finalizer_args = {
            'categories': categories_,
            'tokens': tokens,
            'stack': results,
            'scores': scores,
        }

        status = parse_sentence(
            c_tag_scores,
            c_dep_scores,
            length,
            c_possible_root_cat,
            <void*>binary_callback,
            <void*>unary_callback,
            retrieve_tree,
            scaffold,
            <void*>finalizer_args,
            &c_cache,
            &c_config,
        )

        if status > 0:
            all_results.append(failed())
            continue

        if len(results) != len(scores):
            raise RuntimeError(
                'unexpected behavior occured during parsing'
            )

        all_results.append(
            [
                ScoredTree(tree=tree, score=score)
                for tree, score in zip(results, scores)
            ]
        )

    return all_results
