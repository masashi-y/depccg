
import math
import heapq
from utils import load_unary
from ccgbank import Tree, Leaf
from combinator import standard_combinators as binary_rules
from combinator import RuleType
import combinator


def compute_outsize_probs(supertags):
    sent_size = len(supertags)
    res = [ [.0 for _ in range(sent_size + 1)] for _ in range(sent_size + 1)]
    from_left = [.0 for _ in range(sent_size + 1)]
    from_right = [.0 for _ in range(sent_size + 1)]

    for i in xrange(sent_size - 1):
        j = sent_size - i
        from_left[i + 1]  = from_left[i] + supertags[i][0][1]
        from_right[j - 1] = from_right[j] + supertags[j - 1][0][1]

    for i in xrange(sent_size + 1):
        for j in xrange(i, sent_size + 1):
            res[i, j] = from_left[i] + from_right[j]

    return res


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
            if prob > self.best_prob:
                self.best_prob = prob
                self.best = parse
            return True

    def __iter__(self):
        for parse, prob in self.items.values():
            yield parse, prob

    def best_item(self):
        if self.best is None:
            return None
        return self.items[self.best]

    @property
    def isempty(self):
        return len(self.items) == 0


class SeenRules(object):
    def __init__(self, filepath):
        pass


class AStarParser(object):
    def __init__(self, tagger, unary_rule_path):
        self.tagger = tagger
        self.tag_size = len(tagger.targets)
        self.cats = map(Cat.parse, tagger.cats)
        self.unary_rules = load_unary(unary_rule_path)
        self.rule_cache = {}

    def parse(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split(" ")
        supertags = self.assign_supertags(tokens)
        res = _parse(supertags)
        return res

    def assign_supertags(self, tokens):
        """
        Inputs:
            tokens (list[str])
        """
        # TODO: threshold cut with beta
        threshold = 0.0
        scores = self.tagger.predict(tokens)
        index = np.argsort(scores, 1)
        totals = np.sum(scores, 1)

        res = [[] for _ in tokens]
        for i, token in enumerate(tokens):
            for j in xrange(ntargets - 1, -1, -1):
                k = index[i, j]
                score = scores[i, k]
                if score < threshold:
                    break
                cat = self.cats[k]
                leaf = Leaf(tokens[i], cat, None)
                log_prob = math.log(score / totals[i])
                res[i].append((leaf, log_prob))
        return res

    def _parse(self, supertags):
        def gen_agenda_item(index, leaf, cost):
            out_prob = out_probs[index][index+1]
            heuristic = cost + out_prob
            item =  AgendaItem(leaf, cost, out_prob, index, 1)
            return heuristic, item

        def update_agenda(start_of_span, span_len,
                    left, right, left_prob, right_prob, out_prob):
            if self.seen_rules.not_seen(left.cat, right.cat):
                for rule_type, res, head_is_left in \
                        self.get_rules(left, right, binary_rules):
                    if (left.rule_type == RuleType.FC or \
                            left.rule_type == RuleType.GFC) and \
                        (rule_type == RuleType.FA or \
                            rule_type == RuleType.FC or \
                            rule_type == RuleType.FGC):
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
                            not res in self.possible_root_cats:
                        continue
                    else:
                        subtree = Tree(res, head_is_left, [left, right])
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

        while chart[0][sent_size - 1].isempty():
            if len(agenda) == 0: break

            prob, item = heapq.heappop(agenda)
            cell = chart[item.start_of_span][item.span_len - 1]

            if cell.update(item.parse, item.in_prob):

                if item.span_len != sent_size:
                    for unary in self.unary_rules.get(item.parse.cat, []):
                        subtree = Tree(unary, True, [item.parse])
                        out_prob = out_probs[item.start_of_span]\
                                [item.start_of_span + item.span_len]
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
                                      out_probs[item.span_len]\
                                          [item.start_of_span + span_len])

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
                                    out_probs[start_of_span]\
                                            [start_of_span + span_len])

            heapq.heapify(agenda)
        if chart[0][sent_size - 1].isempty:
            return None
        else:
            return chart[0][sent_size - 1]

    def get_rules(self, left, right):
        if not self.rule_cache.has_key(left):
            self.rule_cache[left] = []

        if not self.rule_cache.has_key(right):
            res = combinator.get_rules(left, right, binary_rules)
            self.rule_cache[right] = res
            return res
        else:
            return self.rule_cache[right]

