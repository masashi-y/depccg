# -*- coding: utf-8 -*-

cimport cython
from cpython cimport Py_INCREF
from preshed.maps cimport PreshMap
from cymem.cymem cimport Pool
from utils cimport load_unary, load_seen_rules, hash_int_int
from pqueue cimport PQueue, pqueue_new, pqueue_delete, \
                    pqueue_enqueue, pqueue_dequeue
from preshed.maps cimport map_init, key_t, \
        MapStruct, map_set, map_get, map_iter
from tree cimport Tree, Leaf, Node
from combinator import standard_combinators, unary_rule, Combinator
from structs cimport FC, GFC, FA, BX, GBX, BA, UNARY
from tagger import EmbeddingTagger
import cat
import os
cimport numpy as np
import numpy as np
import chainer
import multiprocessing

failure_node = Tree(cat.parse("NP"), True, [Leaf("FAILED", cat.parse("NP"), 0)], unary_rule)

cdef inline void* to_void_ptr(object pyob):
    Py_INCREF(pyob) # reference count gets 0 when casting to void* ?
    return <void*>pyob

cdef extern from "<math.h>":
    float logf(float)
    float fabsf(float)

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
    cdef AgendaItem* item = <AgendaItem *>mem.alloc(1, sizeof(AgendaItem))
    item.parse = to_void_ptr(parse)
    item.in_prob = in_prob
    item.out_prob = out_prob
    item.cost = in_prob + out_prob
    item.start_of_span = start_of_span
    item.span_len = span_len
    return item

cdef int agendaitem_compare(const void* a1, const void* a2):
    cdef AgendaItem* item1 = <AgendaItem*>a1
    cdef AgendaItem* item2 = <AgendaItem*>a2
    cdef object p1, p2
    cdef float res = item2.cost - item1.cost
    if res != 0 and fabsf(res) < 0.0000001:
        res = 0
    if res == 0:
        p1 = <object>item1.parse
        p2 = <object>item2.parse
        return p1.deplen - p2.deplen
    else:
        return <int>res

ctypedef struct ChartCell:
    MapStruct* items
    float best_cost
    void* best

ctypedef struct CellItem:
    void* item
    float cost

cdef ChartCell *chartcell_new(Pool mem):
    cdef ChartCell* cell = <ChartCell *>mem.alloc(1, sizeof(ChartCell))
    cell.items = <MapStruct*>mem.alloc(1, sizeof(MapStruct))
    map_init(mem, cell.items, 16)
    cell.best_cost = 1000000
    cell.best = NULL
    return cell


cdef bint chartcell_update(Pool mem, ChartCell* cell, object parse, float cost):
    cdef MapStruct* items = cell.items
    cdef key_t key = parse.cat.id
    cdef void* parse_ptr = <void*>parse
    cdef CellItem* item
    if map_get(items, key) != NULL and cost >= cell.best_cost:
        return False
    else:
        item = <CellItem*>mem.alloc(1, sizeof(CellItem))
        item.item = parse_ptr
        item.cost = cost
        map_set(mem, items, key, <void*>item)
        if cell.best_cost > cost:
            cell.best_cost = cost
            cell.best = parse_ptr
        return True

cdef class AStarParser(object):
    cdef object tagger
    cdef int tag_size
    cdef list cats
    cdef dict unary_rules
    cdef PreshMap rule_cache
    cdef PreshMap seen_rules
    cdef list possible_root_cats
    cdef list binary_rules

    def __cinit__(self, model_path):
        self.tagger = EmbeddingTagger(model_path)
        chainer.serializers.load_npz(os.path.join(
                            model_path, "tagger_model"), self.tagger)
        self.tag_size = len(self.tagger.targets)
        self.cats = map(cat.parse, self.tagger.cats)
        self.unary_rules = load_unary(os.path.join(
                            model_path, "unary_rules.txt"))
        self.binary_rules = standard_combinators
        self.rule_cache = PreshMap()
        self.seen_rules = load_seen_rules(os.path.join(
                            model_path, "seen_rules.txt"))
        self.possible_root_cats = \
            map(cat.parse,
                    ["S[dcl]", "S[wq]", "S[q]", "S[qem]", "NP"])

    def parse(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split(" ")
        cdef np.ndarray[FLOAT_T, ndim=2] scores = \
            self.tagger.predict(tokens)
        res = self._parse(tokens, scores)
        return res

    def worker(self, in_queue, out_queue):
        cdef object res
        while True:
            task = in_queue.get()
            if task is None:
                in_queue.task_done()
                break
            i, tokens, scores = task
            res = self._parse(tokens, scores)
            in_queue.task_done()
            #TODO: cython object is not picklable
            out_queue.put((i, str(res)))
        return

    def parse_doc(self, list doc):
        cdef int n_process = multiprocessing.cpu_count()
        cdef object data_queue = multiprocessing.JoinableQueue()
        cdef object res_queue = multiprocessing.Queue()
        cdef object p, tokens, parse
        cdef int i, _
        cdef dict res = {}
        cdef list scored_doc = \
                self.tagger.predict_doc(doc, batchsize=100)

        for _ in range(n_process):
            p = multiprocessing.Process(
                    target=self.worker, args=(data_queue, res_queue))
            p.start()

        scored_doc = sorted(scored_doc, key=lambda x: len(x[1]), reverse=True)
        for i, tokens, scores in scored_doc:
            data_queue.put((i, tokens, scores))
        for _ in range(n_process):
            data_queue.put(None)
        data_queue.join()
        for _ in range(len(doc)):
            i, parse = res_queue.get()
            res[i] = parse
        return zip(*sorted(res.items(), key=lambda x: x[0]))[1]


    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef object _parse(self, list tokens,
            np.ndarray[FLOAT_T, ndim=2] scores, beta=0.00001):
        cdef int s_len = len(tokens)
        cdef Pool mem = Pool()
        cdef PQueue* agenda = pqueue_new(agendaitem_compare)
        cdef int span_len, start_of_span, i, j, k, head_is_left
        cdef float prob, in_prob, out_prob, threshold
        cdef object parse, subtree, left, right
        cdef object rule
        cdef object unary, out
        cdef AgendaItem *item, *new_item
        cdef ChartCell *cell, *other
        cdef key_t key
        cdef void* value
        cdef CellItem* cell_item
        cdef object leaf
        cdef str token

        # Initialization
        scores = np.exp(scores)
        cdef:
            np.ndarray[LONG_T, ndim=2]  index = \
                np.argsort(scores, 1)
            np.ndarray[FLOAT_T, ndim=1]  totals = \
                np.sum(scores, 1)
            np.ndarray[FLOAT_T, ndim=2]  log_probs = \
                -np.log(scores / totals.reshape((s_len, 1)))
            np.ndarray[FLOAT_T, ndim=1]  best_log_probs = \
                np.min(log_probs, 1)

        # Compute outside probabiliities
        cdef float* out_probs = <float*>mem.alloc(
                            (s_len + 1) * (s_len + 1), sizeof(float))
        compute_outsize_probs(best_log_probs, out_probs)

        # Initialize agenda
        for i, token in enumerate(tokens):
            threshold = beta * scores[i, index[i, -1]]
            for j in xrange(self.tag_size - 1, self.tag_size - 50, -1):
                k = index[i, j]
                if scores[i, k] <= threshold:
                    break
                leaf = Leaf(token, self.cats[k], i)
                item = agendaitem_new(mem, leaf, \
                        log_probs[i, k], out_probs[i * s_len + (i + 1)], i, 1)
                pqueue_enqueue(agenda, item)
        print agenda.size / float(s_len)

        # Initialize chart
        cdef ChartCell** chart = \
                <ChartCell **>mem.alloc(s_len * s_len, sizeof(ChartCell*))
        for i in xrange(s_len * s_len):
            chart[i] = chartcell_new(mem)

        # Start chart parsing
        step = 0
        while chart[s_len - 1].items.filled == 0 and agenda.size > 0:
            step += 1
            print step

            item = <AgendaItem *>pqueue_dequeue(agenda)
            parse = <object>item.parse
            cell = chart[item.start_of_span * s_len + (item.span_len - 1)]

            if chartcell_update(mem, cell, parse, item.in_prob):

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
                for span_len in xrange(
                        item.span_len + 1, 1 + s_len - item.start_of_span):
                    other = chart[(item.start_of_span + item.span_len) * s_len +
                                                (span_len - item.span_len - 1)]
                    i = 0
                    while map_iter(other.items, &i, &key, &value):
                        cell_item = <CellItem*>value
                        right = <object>cell_item.item
                        prob = cell_item.cost
                        if not self.is_seen(parse.cat, right.cat): continue
                        for rule, out, head_is_left in self.get_rules(parse.cat, right.cat):
                            if is_normal_form(rule.rule_type, parse, right) and \
                                    self.acceptable_root_or_subtree(out, span_len, s_len):
                                subtree = Tree(out, head_is_left, [parse, right], rule)
                                in_prob = item.in_prob + prob
                                out_prob = out_probs[item.start_of_span * s_len + item.start_of_span + span_len]
                                new_item = agendaitem_new(mem, subtree, in_prob,
                                        out_prob, item.start_of_span, span_len)
                                pqueue_enqueue(agenda, new_item)

                # binary rule `parse` being right argument
                for start_of_span in xrange(0, item.start_of_span):
                    span_len = item.start_of_span + item.span_len - start_of_span
                    other = chart[start_of_span * s_len +
                                        (span_len - item.span_len - 1)]
                    i = 0
                    while map_iter(other.items, &i, &key, &value):
                        cell_item = <CellItem*>value
                        left = <object>cell_item.item
                        prob = cell_item.cost
                        if not self.is_seen(left.cat, parse.cat): continue
                        for rule, out, head_is_left in self.get_rules(left.cat, parse.cat):
                            if is_normal_form(rule.rule_type, left, parse) and \
                                    self.acceptable_root_or_subtree(out, span_len, s_len):
                                subtree = Tree(out, head_is_left, [left, parse], rule)
                                in_prob = prob + item.in_prob
                                out_prob = out_probs[start_of_span * s_len + start_of_span + span_len]
                                new_item = agendaitem_new(mem, subtree, in_prob,
                                        out_prob, start_of_span, span_len)
                                pqueue_enqueue(agenda, new_item)

        if chart[s_len - 1].items.filled == 0:
            return failure_node
        parse = <object>(chart[s_len - 1].best)
        return parse

    cdef inline bint is_seen(self, object cat1, object cat2):
        return hash_int_int(cat1.id, cat2.id) in self.seen_rules

    cdef list get_rules(self, object left, object right):
        cdef list res
        cdef int hash_ = hash_int_int(left.id, right.id)
        if not hash_ in self.rule_cache:
            res = Combinator.get_rules(left, right, self.binary_rules)
            self.rule_cache.set(hash_, to_void_ptr(res))
            return res
        else:
            return <list>self.rule_cache.get(hash_)

    cdef bint acceptable_root_or_subtree(self, object out, int span_len, int s_len):
        if span_len == s_len and \
                not out in self.possible_root_cats:
            return False
        return True


cdef bint is_normal_form(int rule_type, object left, object right):
    if (left.rule_type == FC or \
            left.rule_type == GFC) and \
        (rule_type == FA or \
            rule_type == FC or \
            rule_type == GFC):
        return False
    if (right.rule_type == BX or \
            left.rule_type == GBX) and \
        (rule_type == BA or \
            rule_type == BX or \
            left.rule_type == GBX):
        return False
    if left.rule_type == UNARY and \
            rule_type == FA and \
            left.cat.is_forward_type_raised:
        return False
    if right.rule_type == UNARY and \
            rule_type == BA and \
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


