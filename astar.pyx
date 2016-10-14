# -*- coding: utf-8 -*-

from libc.stdlib cimport malloc, free
from utils cimport compute_outsize_probs, load_unary, load_seen_rules
from pqueue cimport PQueue, pqueue_new, pqueue_dequeue
from pqueue cimport pqueue_delete, pqueue_enqueue
cimport numpy as np
import os
import math
import numpy as np
import chainer
from ccgbank import Tree, Leaf
from combinator import standard_combinators as binary_rules
from combinator import unary_rule
from combinator import RuleType, Combinator
from tagger import EmbeddingTagger
from cat import Cat

ctypedef np.float32_t FLOAT_T
ctypedef long LONG_T

ctypedef struct AgendaItem:
    void *parse
    float in_prob
    float out_prob
    float cost
    int start_of_span
    int span_len

cdef AgendaItem* agendaitem_new(object parse, float in_prob, float out_prob,
        int start_of_span, int span_len):
    cdef AgendaItem* item = <AgendaItem *>malloc(sizeof(AgendaItem))
    item.parse = <void *>parse
    item.in_prob = in_prob
    item.out_prob = out_prob
    item.cost = in_prob + out_prob
    item.start_of_span = start_of_span
    item.span_len = span_len
    return item

cdef void agendaitem_delete(AgendaItem *item):
    item.parse = NULL
    free(item)

cdef int agendaitem_compare(const void* a1, const void* a2):
    return <int>((<AgendaItem*>a1).cost - (<AgendaItem*>a2).cost)

ctypedef struct ChartCell:
    void *items
    int size
    float best_cost
    void *best

cdef ChartCell *chartcell_new():
    cdef ChartCell* cell = <ChartCell *>malloc(sizeof(ChartCell))
    cdef d = dict()
    cell.items = <void *>d
    cell.size = 0
    cell.best_cost = 1000000
    cell.best = NULL
    return cell

cdef int chartcell_update(ChartCell* cell, object parse, float cost):
    cdef dict items = <dict>cell.items
    if items.has_key(parse.cat):
        if cost > items[parse.cat][1]:
            items[parse.cat] = parse, cost
            cell.size += 1
            if cost > cell.best_cost:
                cell.best_cost = cost
                cell.best = <void *>parse
            return 1
        else:
            return 0
    else:
        items[parse.cat] = parse, cost
        cell.size += 1
        if cost > cell.best_cost:
            cell.best_cost = cost
            cell.best = <void *>parse
        return 1

cdef list chartcell_iter(ChartCell* cell):
    cdef dict items = <dict>cell.items
    return items.values()

cdef void chartcell_delete(ChartCell* cell):
    cell.items = NULL
    cell.best = NULL
    free(cell)


# cdef class ChartCell:
#     cdef object items
#     cdef float best_prob
#     cdef object best
#
#     def __init__(self):
#         self.items = {}
#         self.best_prob = float('inf')
#         self.best = None
#
#     cpdef update(self, object parse, float prob):
#         if self.items.has_key(parse.cat):
#             return False
#         else:
#             self.items[parse.cat] = parse, prob
#             # if prob > self.best_prob:
#             self.best_prob = prob
#             self.best = parse
#             return True
#
#     def __iter__(self):
#         cdef object parse
#         cdef float prob
#
#         for parse, prob in self.items.values():
#             yield parse, prob
#
#     @property
#     def best_item(self):
#         if self.best is None:
#             return None
#         res, _ = self.items[self.best.cat]
#         return res
#
#     @property
#     def isempty(self):
#         return len(self.items) == 0


cdef class AStarParser(object):
    cdef object tagger
    cdef int tag_size
    cdef list cats
    cdef dict unary_rules
    cdef dict rule_cache
    cdef dict seen_rules
    cdef list possible_root_cats

    def __init__(self, model_path):
        self.tagger = EmbeddingTagger(model_path)
        chainer.serializers.load_npz(os.path.join(
                            model_path, "tagger_model"), self.tagger)
        self.tag_size = len(self.tagger.targets)
        self.cats = map(Cat.parse, self.tagger.cats)
        self.unary_rules = load_unary(os.path.join(
                            model_path, "unary_rules.txt"))
        self.rule_cache = {}
        self.seen_rules = load_seen_rules(os.path.join(
                            model_path, "seen_rules.txt"))
        self.possible_root_cats = \
            map(Cat.parse,
                    ["S[dcl]", "S[wq]", "S[q]", "S[qem]", "NP"])

    def parse(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split(" ")
        supertags = self.assign_supertags(tokens)
        res = self._parse(supertags)
        return res

    cpdef list assign_supertags(self, list tokens, float beta=0.00001):
        """
        Inputs:
            tokens (list[str])
        """
        # TODO: threshold cut with beta
        cdef float threshold = 0.0
        cdef float score, prob, log_prob
        cdef object leaf, token
        cdef int i
        cdef np.ndarray[FLOAT_T, ndim=2] scores = \
                np.exp(self.tagger.predict(tokens))
        cdef np.ndarray[LONG_T, ndim=2]  index = \
                np.argsort(scores, 1)
        cdef np.ndarray[FLOAT_T, ndim=1] totals = \
                np.sum(scores, 1)

        cdef list res = [[] for _ in tokens]
        for i, token in enumerate(tokens):
            threshold = beta * scores[i, index[i, -1]]
            for j in xrange(self.tag_size - 1, -1, -1):
                k = index[i, j]
                score = scores[i, k]
                if score <= threshold:
                    break
                leaf = Leaf(token, self.cats[k], None)
                prob = score / totals[i]
                log_prob = -math.log(prob)
                res[i].append((leaf, log_prob))
        return res

    cdef object _parse(self, list supertags):

        cdef int s_len = len(supertags)
        cdef float prob, out_prob
        cdef int span_len, start_of_span, i
        cdef object unary, leaf, parse
        cdef AgendaItem* item
        cdef ChartCell *cell
        cdef ChartCell *other
        cdef list stags

        cdef np.ndarray[FLOAT_T, ndim=2] out_probs = \
                            compute_outsize_probs(supertags)

        cdef PQueue* agenda = pqueue_new(agendaitem_compare)

        for i, stags in enumerate(supertags):
            for leaf, in_prob in stags:
                out_prob = out_probs[i, i+1]
                item = agendaitem_new(leaf, in_prob, out_prob, i, 1)
                pqueue_enqueue(agenda, item)

        cdef ChartCell** chart = <ChartCell **>malloc(
                                s_len * s_len * sizeof(ChartCell*))

        while chart[s_len - 1].size == 0 and agenda.size > 0:

            item = <AgendaItem *>pqueue_dequeue(agenda)
            parse = <object>item.parse
            cell = chart[item.start_of_span * (item.span_len - 1)]

            if chartcell_update(cell, parse, item.in_prob):

                if item.span_len != s_len:
                    for unary in self.unary_rules.get(parse.cat, []):
                        subtree = Tree(unary, True, [parse], unary_rule)
                        out_prob = out_probs[item.start_of_span,
                                item.start_of_span + item.span_len]
                        item = agendaitem_new(subtree,
                                            item.in_prob,
                                            out_prob,
                                            item.start_of_span,
                                            item.span_len)
                        pqueue_enqueue(agenda, item)

                for span_len in range(
                        item.span_len + 1, 1 + s_len - item.start_of_span):
                    other = chart[(item.start_of_span + item.span_len) *
                                                (span_len - item.span_len - 1)]
                    if other.size != 0:
                        for right, right_prob in chartcell_iter(other):
                            self.update_agenda(agenda,
                                               s_len,
                                               item.start_of_span,
                                               span_len,
                                               parse, right,
                                               item.in_prob,
                                               right_prob,
                                               out_probs[item.span_len,
                                                   item.start_of_span + span_len])

                for start_of_span in range(0, item.start_of_span):
                    span_len = item.start_of_span + item.span_len - start_of_span
                    other = chart[start_of_span *
                                        (span_len - item.span_len - 1)]
                    if other.size != 0:
                        for left, left_prob in chartcell_iter(other):
                            self.update_agenda(agenda,
                                               s_len,
                                               start_of_span,
                                               span_len,
                                               left, parse,
                                               left_prob,
                                               item.in_prob,
                                               out_probs[start_of_span,
                                                   start_of_span + span_len])


        return <object>(chart[s_len - 1].best)

    cdef void update_agenda(self, PQueue* agenda, int s_len,
            int start_of_span, int span_len,
                object left, object right, float left_prob,
                float right_prob, float out_prob):
        cdef object rule, out, rule_type, subtree
        cdef int head_is_left
        cdef float in_prob
        cdef AgendaItem *item

        if not self.seen_rules.has_key((left.cat, right.cat)):
            for rule, out, head_is_left in \
                    self.get_rules(left.cat, right.cat, binary_rules):
                rule_type = rule.rule_type
                if (left.rule_type == RuleType.FC or \
                        left.rule_type == RuleType.GFC) and \
                    (rule_type == RuleType.FA or \
                        rule_type == RuleType.FC or \
                        rule_type == RuleType.GFC):
                    continue
                elif (right.rule_type == RuleType.BX or \
                        left.rule_type == RuleType.GBX) and \
                    (rule_type == RuleType.BA or \
                        rule_type == RuleType.BX or \
                        left.rule_type == RuleType.GBX):
                    continue
                elif left.rule_type == RuleType.UNARY and \
                        rule_type == RuleType.FA and \
                        right.cat.is_forward_type_raised:
                    continue
                elif right.rule_type == RuleType.UNARY and \
                        rule_type == RuleType.BA and \
                        right.cat.is_backward_type_raised:
                    continue
                elif span_len == s_len and \
                        not out in self.possible_root_cats:
                    continue
                else:
                    subtree = Tree(
                            out, head_is_left, [left, right], rule)
                    in_prob = left_prob + right_prob
                    item = agendaitem_new(subtree, in_prob,
                            out_prob, start_of_span, span_len)
                    pqueue_enqueue(agenda, item)

    cdef list get_rules(self, object left, object right, list rules):
        cdef list res
        if not self.rule_cache.has_key((left, right)):
            res = Combinator.get_rules(left, right, rules)
            self.rule_cache[(left, right)] = res
            return res
        else:
            return self.rule_cache[(left, right)]

