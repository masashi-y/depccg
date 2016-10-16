# -*- coding: utf-8 -*-

cimport cython
# from libcpp.vector cimport vector
from cymem.cymem cimport Pool
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from utils cimport load_unary, load_seen_rules
from pqueue cimport PQueue, pqueue_new, pqueue_delete
from pqueue cimport pqueue_enqueue, pqueue_dequeue
cimport numpy as np
import os
import numpy as np
import chainer
from ccgbank import Tree, Leaf
from combinator import standard_combinators as binary_rules
from combinator import unary_rule
from combinator import RuleType, Combinator
from tagger import EmbeddingTagger
from cat import Cat
from preshed.maps cimport PreshMap


cdef extern from "<math.h>":
    float logf(float)

# reference count gets 0 when casting to void* ?
refs = []

ctypedef np.float32_t FLOAT_T
ctypedef long LONG_T

ctypedef struct AgendaItem:
    void *parse
    float in_prob
    float out_prob
    float cost
    int start_of_span
    int span_len

cdef AgendaItem* agendaitem_new(Pool mem, object parse,
        float in_prob, float out_prob, int start_of_span, int span_len):
    refs.append(parse)
    cdef AgendaItem* item = <AgendaItem *>mem.alloc(1, sizeof(AgendaItem))
    item.parse = <void *>parse
    item.in_prob = in_prob
    item.out_prob = out_prob
    item.cost = in_prob + out_prob
    item.start_of_span = start_of_span
    item.span_len = span_len
    return item

cdef int agendaitem_compare(const void* a1, const void* a2):
    return <int>((<AgendaItem*>a1).cost - (<AgendaItem*>a2).cost)

ctypedef pair[void*, float] cell_item

ctypedef struct ChartCell:
    unordered_map[int, cell_item]* items
    float best_cost
    void* best


cdef ChartCell *chartcell_new(Pool mem):
    cdef ChartCell* cell = <ChartCell *>mem.alloc(1, sizeof(ChartCell))
    cell.items = new unordered_map[int, cell_item]()
    cell.best_cost = 1000000
    cell.best = NULL
    return cell


cdef int chartcell_update(ChartCell* cell, object parse, float cost):
    cdef unordered_map[int, cell_item]* items = cell.items
    cdef int key = parse.cat.id
    cdef void* parse_ptr = <void*>parse
    if items.count(key):
        if cost < cell.best_cost:
            items[0][key] = cell_item(parse_ptr, cost)
            cell.best_cost = cost
            cell.best = parse_ptr
            return 1
        else:
            return 0
    else:
        items[0][key] = cell_item(parse_ptr, cost)
        cell.best_cost = cost
        cell.best = parse_ptr
        return 1


cdef class AStarParser(object):
    cdef object tagger
    cdef int tag_size
    cdef list cats
    cdef dict unary_rules
    cdef dict rule_cache
    cdef dict seen_rules
    cdef list possible_root_cats
    cdef list binary_rules

    def __init__(self, model_path):
        self.tagger = EmbeddingTagger(model_path)
        chainer.serializers.load_npz(os.path.join(
                            model_path, "tagger_model"), self.tagger)
        self.tag_size = len(self.tagger.targets)
        self.cats = map(Cat.parse, self.tagger.cats)
        self.unary_rules = load_unary(os.path.join(
                            model_path, "unary_rules.txt"))
        self.binary_rules = binary_rules
        self.rule_cache = {}
        self.seen_rules = load_seen_rules(os.path.join(
                            model_path, "seen_rules.txt"))
        self.possible_root_cats = \
            map(Cat.parse,
                    ["S[dcl]", "S[wq]", "S[q]", "S[qem]", "NP"])

    def parse(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split(" ")
        res = self._parse(tokens)
        return res

    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef void initialize(self, Pool mem,
            list tokens, PQueue* agenda, float* out_probs, float beta=0.00001):
        """
        Inputs:
            tokens (list[str])
        """
        cdef float threshold, log_prob
        cdef object leaf
        cdef str token
        cdef int i, j, k
        cdef int s_len = len(tokens)
        cdef:
            np.ndarray[FLOAT_T, ndim=2] scores = \
                np.exp(self.tagger.predict(tokens))
            np.ndarray[LONG_T, ndim=2]  index = \
                np.argsort(scores, 1)
            np.ndarray[FLOAT_T, ndim=1]  totals = \
                np.sum(scores, 1)
            np.ndarray[FLOAT_T, ndim=2]  log_probs = \
                -np.log(scores / totals.reshape((s_len, 1)))
            np.ndarray[FLOAT_T, ndim=1]  best_log_probs = \
                np.min(log_probs)

        compute_outsize_probs(best_log_probs, out_probs)

        for i, token in enumerate(tokens):
            threshold = beta * scores[i, index[i, -1]]
            for j in xrange(self.tag_size - 1, -1, -1):
                k = index[i, j]
                if scores[i, k] <= threshold:
                    break
                leaf = Leaf(token, self.cats[k], None)
                item = agendaitem_new(mem, leaf, log_probs[i, k], out_probs[i * s_len + (i + 1)], i, 1)
                pqueue_enqueue(agenda, item)


    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef object _parse(self, list tokens):

        cdef int s_len = len(tokens)
        cdef Pool mem = Pool()
        cdef PQueue* agenda = pqueue_new(agendaitem_compare)
        cdef float* out_probs = <float*>mem.alloc(
                            (s_len + 1) * (s_len + 1), sizeof(float))
        cdef int span_len, start_of_span, i, head_is_left
        cdef float prob, in_prob, out_prob
        cdef object unary, parse, subtree, out, left, right, rule
        cdef AgendaItem* item
        cdef AgendaItem* new_item
        cdef ChartCell* cell
        cdef ChartCell* other

        self.initialize(mem, tokens, agenda, out_probs)

        cdef ChartCell** chart = \
                <ChartCell **>mem.alloc(s_len * s_len, sizeof(ChartCell*))
        for i in xrange(s_len * s_len):
            chart[i] = chartcell_new(mem)

        while chart[s_len - 1].items.size() == 0 and agenda.size > 0:

            item = <AgendaItem *>pqueue_dequeue(agenda)
            parse = <object>item.parse
            cell = chart[item.start_of_span * s_len + (item.span_len - 1)]

            if chartcell_update(cell, parse, item.in_prob):

                # unary rules
                if item.span_len != s_len:
                    for unary in self.unary_rules.get(parse.cat, []):
                        subtree = Tree(unary, True, [parse], unary_rule)
                        out_prob = out_probs[item.start_of_span * s_len +
                                item.start_of_span + item.span_len]
                        new_item = agendaitem_new(mem, subtree,
                                            item.in_prob,
                                            out_prob,
                                            item.start_of_span,
                                            item.span_len)
                        pqueue_enqueue(agenda, new_item)

                # binary rule `parse` being left argument
                for span_len in range(
                        item.span_len + 1, 1 + s_len - item.start_of_span):
                    other = chart[(item.start_of_span + item.span_len) * s_len +
                                                (span_len - item.span_len - 1)]
                    if other.items.size() != 0:
                        # for right, prob in chartcell_iter(other):
                        for it in other.items[0]:
                            right = <object>it.second.first
                            prob = it.second.second
                            if not self.seen_rules.has_key((parse.cat, right.cat)): continue
                            for rule, out, head_is_left in self.get_rules(parse.cat, right.cat):
                                if is_normal_form(rule.rule_type, parse, right) and \
                                        self.acceptable_root_or_subtree(out, span_len, s_len):
                                    subtree = Tree(out, head_is_left, [parse, right], rule)
                                    in_prob = item.in_prob + prob
                                    out_prob = out_probs[item.span_len * s_len + item.start_of_span + span_len]
                                    new_item = agendaitem_new(mem, subtree, in_prob,
                                            out_prob, item.start_of_span, span_len)
                                    pqueue_enqueue(agenda, new_item)

                # binary rule `parse` being right argument
                for start_of_span in range(0, item.start_of_span):
                    span_len = item.start_of_span + item.span_len - start_of_span
                    other = chart[start_of_span * s_len +
                                        (span_len - item.span_len - 1)]
                    if other.items.size() != 0:
                        # for left, prob in chartcell_iter(other):
                        for it in other.items[0]:
                            left = <object>it.second.first
                            prob = it.second.second
                            if not self.seen_rules.has_key((left.cat, parse.cat)): continue
                            for rule, out, head_is_left in self.get_rules(left.cat, parse.cat):
                                if is_normal_form(rule.rule_type, left, parse) and \
                                        self.acceptable_root_or_subtree(out, span_len, s_len):
                                    subtree = Tree(out, head_is_left, [left, parse], rule)
                                    in_prob = prob + item.in_prob
                                    out_prob = out_probs[start_of_span * s_len + start_of_span + span_len]
                                    new_item = agendaitem_new(mem, subtree, in_prob,
                                            out_prob, start_of_span, span_len)
                                    pqueue_enqueue(agenda, new_item)

        parse = <object>(chart[s_len - 1].best)
        return parse


    cdef list get_rules(self, object left, object right):
        cdef list res
        if not self.rule_cache.has_key((left, right)):
            res = Combinator.get_rules(left, right, self.binary_rules)
            self.rule_cache[(left, right)] = res
            return res
        else:
            return self.rule_cache[(left, right)]

    cdef bint acceptable_root_or_subtree(self, object out, int span_len, int s_len):
        if span_len == s_len and \
                not out in self.possible_root_cats:
            return False
        return True


cdef bint is_normal_form(int rule_type, object left, object right):
    if (left.rule_type == RuleType.FC or \
            left.rule_type == RuleType.GFC) and \
        (rule_type == RuleType.FA or \
            rule_type == RuleType.FC or \
            rule_type == RuleType.GFC):
        return False
    if (right.rule_type == RuleType.BX or \
            left.rule_type == RuleType.GBX) and \
        (rule_type == RuleType.BA or \
            rule_type == RuleType.BX or \
            left.rule_type == RuleType.GBX):
        return False
    if left.rule_type == RuleType.UNARY and \
            rule_type == RuleType.FA and \
            right.cat.is_forward_type_raised:
        return False
    if right.rule_type == RuleType.UNARY and \
            rule_type == RuleType.BA and \
            right.cat.is_backward_type_raised:
        return False
    return True


cdef void compute_outsize_probs(np.ndarray[FLOAT_T, ndim=1] probs, float* out):
    cdef int s_len = probs.shape[0]
    cdef int i, j
    cdef:
        np.ndarray[FLOAT_T, ndim=1] from_left = \
            np.zeros((s_len + 1,), 'f')
        np.ndarray[FLOAT_T, ndim=1] from_right = \
            np.zeros((s_len + 1,), 'f')

    for i in xrange(s_len - 1):
        j = s_len - i
        from_left[i + 1]  = from_left[i] + probs[i]
        from_right[j - 1] = from_right[j] + probs[j - 1]

    for i in xrange(s_len + 1):
        for j in xrange(i, s_len + 1):
            out[i * s_len + j] = from_left[i] + from_right[j]


