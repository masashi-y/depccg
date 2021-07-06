from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
# cimport numpy as np

from depccg.grammar.en import apply_rules


cdef extern from "cpp/parsing.h" nogil:
    cdef struct combinator_result:
        unsigned cat_id
        unsigned rule_id
        bint head_is_left

    ctypedef unsigned (*scaffold_type)(void *callback_func, unsigned x, unsigned y, vector[combinator_result] *results);
# 
#     cdef struct config:
#         unsigned num_tags
#         float unary_penalty
#         float beta
#         bint use_beta
#         unsigned pruning_size
#         unsigned nbest
#         unsigned max_step
# 
#     cdef unsigned parse_sentence(
#         float *tag_scores,
#         float *dep_scores,
#         unsigned length,
#         const unordered_set[unsigned] &possible_root_cats,
#         void *binary_callback,
#         void *unary_callback,
#         void *finalize,
#         scaffold_type scaffold,
#         config *config)


cdef unsigned scaffold(
    void *callback_func,
    unsigned x,
    unsigned y,
    vector[combinator_result] *results
):
    cdef list py_result = (<object>callback_func)(<object>x, <object>y)
    cdef int cat_id, rule_id
    cdef bint head_is_left
    cdef combinator_result c_result
    for cat_id, rule_id, head_is_left in py_result:
        c_result.cat_id = cat_id
        c_result.rule_id = rule_id
        c_result.head_is_left = head_is_left
        results.push_back(c_result)


# cdef init_config(config *c_config, dict kwargs):
    # pass
    # c_config.num_tags = kwargs['num_tags']
    # c_config.unary_penalty = kwargs.pop('unary_penalty', 0.1)
    # c_config.beta = kwargs.pop('beta', 0.00001)
    # c_config.use_beta = kwargs.pop('use_beta', True)
    # c_config.pruning_size = kwargs.pop('pruning_size', 50)
    # c_config.nbest = kwargs.pop('nbest', 1)
    # c_config.max_step = kwargs.pop('max_step', 10000000)


def run(
    tokens,
    # np.ndarray[float, ndim=2, mode='c'] cat_scores,
    # np.ndarray[float, ndim=2, mode='c'] dep_scores,
    cache,
    categories,
    **kwargs
):

    if (
        'max_length' in kwargs
        and len(tokens) > kwargs.pop('max_length')
    ):
        return None

    assert len(set(categories)) == len(categories)

    category_ids = {
        category: index
        for index, category in enumerate(categories)
    }

    def maybe_add_and_get(cat):
        if cat not in categories:
            categories.append(cat)
            category_ids[cat] = len(category_ids)
        return category_ids[cat]

    def binary_callback(x_id, y_id):
        x, y = categories[x_id], categories[y_id]
        if (x, y) in cache:
            combinator_results = cache[x, y]
        else:
            combinator_results = apply_rules(x, y)
            cache[x, y] = combinator_results

        results = []
        for rule_id, result in enumerate(combinator_results):
            cat_id = maybe_add_and_get(result.cat)
            results.append((cat_id, rule_id, result.head_is_left))
        return results

    def unary_callback(x_id, _):
        x = categories[x_id]
        if x in cache:
            combinator_results = cache[x]
        else:
            combinator_results = apply_rules(x)  # TODO
            cache[x] = combinator_results

        results = []
        for rule_id, result in enumerate(combinator_results):
            cat_id = maybe_add_and_get(result.cat)
            results.append((cat_id, rule_id, result.head_is_left))
        return results

    # def finalize_callback():
    #     pass

    # cdef float *c_cat_scores = <float*>cat_scores.data
    # cdef float *c_dep_scores = <float*>dep_scores.data
    # cdef unsigned length = len(tokens)
    # cdef unordered_set[unsigned] c_possible_root_cat
    # cdef config c_config
    # init_config(&c_config, kwargs)

    # cdef unsigned result = parse_sentence(
    #     c_cat_scores,
    #     c_dep_scores,
    #     length,
    #     c_possible_root_cat,
    #     <void*>binary_callback,
    #     <void*>unary_callback,
    #     <void*>unary_callback,
    #     scaffold,
    #     &c_config,
    # )
    # return result
