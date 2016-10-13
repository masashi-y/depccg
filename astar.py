# -*- coding: utf-8 -*-

import os
import math
import heapq
import numpy as np
import chainer
from utils import compute_outsize_probs, load_unary, load_seen_rules
from ccgbank import Tree, Leaf
from combinator import standard_combinators as binary_rules
from combinator import unary_rule
from combinator import RuleType, Combinator
from tagger import EmbeddingTagger
from cat import Cat

class AgendaItem(object):
    def __init__(self, parse, in_prob, out_prob, start_of_span, span_len):
        self.parse         = parse
        self.in_prob       = in_prob
        self.out_prob      = out_prob
        self.start_of_span = start_of_span
        self.span_len      = span_len


class ChartCell(object):
    def __init__(self):
        self.items = {}
        self.best_prob = float('inf')
        self.best = None

    def update(self, parse, prob):
        if self.items.has_key(parse.cat):
            return False
        else:
            self.items[parse.cat] = parse, prob
            # if prob > self.best_prob:
            self.best_prob = prob
            self.best = parse
            return True

    def __iter__(self):
        for parse, prob in self.items.values():
            yield parse, prob

    @property
    def best_item(self):
        if self.best is None:
            return None
        return self.items[self.best.cat]

    @property
    def isempty(self):
        return len(self.items) == 0


class AStarParser(object):
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

    def assign_supertags(self, tokens, beta=0.00001):
        """
        Inputs:
            tokens (list[str])
        """
        # TODO: threshold cut with beta
        threshold = 0.0
        scores = np.exp(self.tagger.predict(tokens))
        index = np.argsort(scores, 1)
        totals = np.sum(scores, 1)

        res = [[] for _ in tokens]
        for i, token in enumerate(tokens):
            threshold = beta * scores[i, index[i, -1]]
            for j in xrange(self.tag_size - 1, -1, -1):
                k = index[i, j]
                score = scores[i, k]
                if score <= threshold:
                    break
                leaf = Leaf(token, self.cats[k], None)
                prob = score / totals[i]
                log_prob = -math.log(score)
                res[i].append((leaf, log_prob))
        return res

    def _parse(self, supertags):
        def gen_agenda_item(index, leaf, in_prob):
            out_prob = out_probs[index, index+1]
            heuristic = in_prob + out_prob
            item =  AgendaItem(leaf, in_prob, out_prob, index, 1)
            return heuristic, item

        def update_agenda(start_of_span, span_len,
                    left, right, left_prob, right_prob, out_prob):
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
                    elif span_len == sent_size and \
                            not out in self.possible_root_cats:
                        continue
                    else:
                        subtree = Tree(
                                out, head_is_left, [left, right], rule)
                        in_prob = left_prob + right_prob
                        new_item = AgendaItem(subtree, in_prob,
                                out_prob, start_of_span, span_len)
                        agenda.append((in_prob + out_prob, new_item))

        sent_size = len(supertags)
        out_probs = compute_outsize_probs(supertags)
        agenda = [gen_agenda_item(i, leaf, prob) \
                    for i, stags in enumerate(supertags) \
                        for (leaf, prob) in stags]
        agerda = heapq.heapify(agenda)
        chart = [[ChartCell() for _ in range(sent_size)] \
                    for _ in range(sent_size)]

        while chart[0][sent_size - 1].isempty:
            if len(agenda) == 0: break

            prob, item = heapq.heappop(agenda)
            cell = chart[item.start_of_span][item.span_len - 1]

            if cell.update(item.parse, item.in_prob):

                if item.span_len != sent_size:
                    for unary in self.unary_rules.get(item.parse.cat, []):
                        subtree = Tree(unary, True, [item.parse], unary_rule)
                        out_prob = out_probs[item.start_of_span,
                                item.start_of_span + item.span_len]
                        new_item = AgendaItem(subtree,
                                            item.in_prob,
                                            out_prob,
                                            item.start_of_span,
                                            item.span_len)
                        heapq.heappush(agenda,
                                (item.in_prob + out_prob, new_item))

                for span_len in range(
                        item.span_len + 1, 1 + sent_size - item.start_of_span):
                    left = item.parse
                    right_cell = chart[item.start_of_span + item.span_len]\
                                                [span_len - item.span_len - 1]
                    if not right_cell.isempty:
                        for right, right_prob in right_cell:
                            update_agenda(item.start_of_span,
                                      span_len,
                                      left, right,
                                      item.in_prob,
                                      right_prob,
                                      out_probs[item.span_len,
                                          item.start_of_span + span_len])

                for start_of_span in range(0, item.start_of_span):
                    span_len = item.start_of_span + item.span_len - start_of_span
                    right = item.parse
                    left_cell = chart[start_of_span]\
                                        [span_len - item.span_len - 1]
                    if not left_cell.isempty:
                        for left, left_prob in left_cell:
                            update_agenda(start_of_span,
                                    span_len,
                                    left, right,
                                    left_prob,
                                    item.in_prob,
                                    out_probs[start_of_span,
                                            start_of_span + span_len])

                heapq.heapify(agenda)

        return chart[0][sent_size - 1].best_item

    def get_rules(self, left, right, rules):
        if not self.rule_cache.has_key((left, right)):
            res = Combinator.get_rules(left, right, rules)
            self.rule_cache[(left, right)] = res
            return res
        else:
            return self.rule_cache[(left, right)]

