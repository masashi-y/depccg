# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
from py.combinator import Combinator, RuleType
import py.combinator
from xml.etree.ElementTree import Element, SubElement, ElementTree, tostring

def is_normal_form(rule_type, left, right):
    if (left.rule_type == FC or \
            left.rule_type == GFC) and \
        (rule_type == FA or \
            rule_type == FC or \
            rule_type == GFC):
        return False
    if (right.rule_type == BC or \
            right.rule_type == GBC) and \
        (rule_type == BA or \
            rule_type == BC or \
            left.rule_type == GBC):
        return False
    if left.rule_type == UNARY and \
            rule_type == FA and \
            left.cat.is_forward_type_raised:
        return False
    if right.rule_type == UNARY and \
            rule_type == BA and \
            right.cat.is_backward_type_raised:
        return False

    if (left.rule_type == FC or left.rule_type == GFC) \
            and (rule_type == FA or rule_type == FC):
        return False
    if (right.rule_type == BC or right.rule_type == GBC) \
            and (rule_type == BA or rule_type == BC):
        return False
    return True

def count_normal_form(trees):
    res = 0
    total = 0
    def rec(tree):
        if isinstance(tree, Tree):
            total += 1
            res += int(tree.is_normal)
            for child in tree.children:
                res(child)

    for tree in trees:
        rec(tree)

    print("normal form:", res, "total:", total)

class Node(object):
    def __init__(self, cat, rule_type):
        self.cat = cat
        self.rule_type = rule_type


class Leaf(Node):
    # (<L N/N NNP NNP Pierre N_73/N_73>)
    def __init__(self, word, cat, position, tag=None):
        super(Leaf, self).__init__(cat, RuleType.LEXICON)
        self.word = word
        self.pos  = position
        self.tag = tag

    def __str__(self):
        pos = "POS" # dummy
        if self.word in ["{", "("]:
            word = "-LRB-"
        elif self.word in ["}", ")"]:
            word = "-RRB-"
        else:
            word = self.word.encode("utf-8")
        return "(<L {0} {1} {1} {2} {0}>)".format(
                self.cat, pos, word)

    def _to_xml(self, parent):
        word = self.word #.encode("utf-8")
        SubElement(parent, "lf",
                {"word": word
                ,"cat": str(self.cat.without_feat)
                ,"start": str(self.pos)
                ,"span": "1"
                ,"lemma": word
                ,"pos": "POS"
                ,"chunk": "CHUNK"
                ,"entity": "O"})

    @property
    def is_normal(self):
        return False

    @property
    def headid(self):
        return self.pos

    @property
    def deplen(self):
        return 0


class Tree(Node):
    # (<T N 1 2> (<L N/N JJ JJ nonexecutive N_43/N_43>) (<L N NN NN director N>) )
    def __init__(self, cat, left_is_head, children, rule=None):
        rule_type = RuleType.NONE if not isinstance(rule, Combinator) \
                                    else rule.rule_type
        super(Tree, self).__init__(cat, rule_type)
        self.children     = children
        self.left_is_head = left_is_head
        self.op = rule

    def __str__(self):
        left_is_head = 0 if self.left_is_head else 1
        children = [str(c) for c in self.children]
        return "(<T {0} {1} {2}> {3} )".format(
                self.cat, left_is_head, len(children), " ".join(children))

    def _to_xml(self, parent=None):
        if parent is None:
            parent = Element("ccg")
        rule = SubElement(parent, "rule",
                {"type": str(self.op)
                ,"cat": str(self.cat.without_feat)})
        for child in self.children:
            child._to_xml(rule)
        return parent

    @property
    def is_normal(self):
        children = self.children
        if len(children) == 1:
            return True
        return is_normal_form(
                self.rule_type, children[0].cat, children[1].cat)

    def resolve_op(self, ops):
        if len(self.children) == 1:
            self.rule_type = RuleType.UNARY
            self.op = py.combinator.UnaryRule()
        else:
            left, right = self.children
            for op in ops:
                if op.can_apply(left.cat, right.cat) and \
                    op.apply(left.cat, right.cat).strip_feat() == self.cat.strip_feat():
                    self.rule_type = op.rule_type
                    new_head = op.head_is_left(left.cat, right.cat)
                    if self.left_is_head != new_head:
                        print("head error!!!: old: {}, new: {}" .format(self.head_is_left, new_head))
                        print(self.show_derivation())
                        raise RuntimeError()
                    self.left_is_head = new_head
                    self.op = op
                    return
            print(left.cat, right.cat, "-->", self.cat, "\n")

            print(self.show_derivation())
            raise RuntimeError()
                # self.rule_type = NONE
                # self.op = Combinator()

    @property
    def headid(self):
        children = self.children
        if len(children) == 1:
            return children[0].headid
        elif len(children) == 2:
            return children[0].headid if self.left_is_head else children[1].headid
        else:
            raise RuntimeError("Number of children of Tree must be 1 or 2.")

    @property
    def deplen(self):
        children = self.children
        if len(children) == 1:
            return children[0].deplen
        elif len(children) == 2:
            return (children[1].headid - children[0].headid) + \
                    children[0].deplen + children[1].deplen
        else:
            raise RuntimeError("Number of children of Tree must be 1 or 2.")

    def show_derivation(self):
        catstr  = ""
        wordstr = ""
        for leaf in get_leaves(self):
            str_cat   = str(leaf.cat)
            str_word  = leaf.word
            nextlen   = 2 + max(len(str_word), len(str_cat))
            lcatlen   = (nextlen - len(str_cat)) // 2
            rcatlen   = lcatlen + (nextlen - len(str_cat)) % 2
            catstr   += " " * lcatlen + str_cat + " " * rcatlen
            lwordlen  = (nextlen - len(str_word)) // 2
            rwordlen  = lwordlen + (nextlen - len(str_word)) % 2
            wordstr  += " " * lwordlen + str_word + " " * rwordlen

        def rec(lwidth, node):
            rwidth = lwidth

            if isinstance(node, Leaf):
                return max(rwidth, 2 + lwidth + len(str(node.cat)),
                        2 + lwidth + len(node.word))

            if isinstance(node, Tree):
                for child in node.children:
                    rwidth = max(rwidth, rec(rwidth, child))

                op = "" if node.op is None else str(node.op)
                print(lwidth * " " + (rwidth - lwidth) * "-" + str(op))
                str_res = str(node.cat)
                if (len(node.children) > 1):
                    str_res += " ->" if node.left_is_head else " <-"
                respadlen = (rwidth - lwidth - len(str_res)) // 2 + lwidth
                print(respadlen * " " + str_res)
                return rwidth

        print(catstr.rstrip())
        print(wordstr.rstrip())
        rec(0, self)

def to_xml(trees, out):
    candc = Element("candc")
    for tree in trees:
        if isinstance(tree, Tree):
            candc.append(tree._to_xml())
    with open(out, "w") as f:
        ElementTree(candc).write(f)


def resolve_op(tree, ops):
    try:
        tree.resolve_op(ops)
        for child in tree.children:
            if isinstance(child, Tree):
                resolve_op(child, ops)
        return True
    except:
        return False

def get_leaves(tree):
    res = []
    def rec(tree):
        for child in tree.children:
            if isinstance(child, Tree):
                rec(child)
            elif isinstance(child, Leaf):
                res.append(child)
            else:
                raise RuntimeError()

    if isinstance(tree, Tree):
        rec(tree)
    else:
        res.append(tree)
    return res


