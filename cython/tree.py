# -*- coding: utf-8 -*-
from combinator import Combinator, RuleType
from xml.etree.ElementTree import Element, SubElement, ElementTree, tostring

class Node(object):
    def __init__(self, cat, rule_type):
        self.cat = cat
        self.rule_type = rule_type


class Leaf(Node):
    # (<L N/N NNP NNP Pierre N_73/N_73>)
    def __init__(self, word, cat, position):
        super(Leaf, self).__init__(cat, RuleType.LEXICON)
        self.word = word
        self.pos  = position

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

    def to_xml(self, parent):
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

    def to_xml(self, parent=None):
        if parent is None:
            ccg = Element("ccg")
            for child in self.children:
                child.to_xml(ccg)
            return ccg
        else:
            rule = SubElement(parent, "rule",
                    {"type": str(self.op)
                    ,"cat": str(self.cat.without_feat)})
            for child in self.children:
                child.to_xml(rule)


    def resolve_op(self, ops):
        if len(self.children) == 1:
            self.rule_type = RuleType.UNARY
        else:
            left, right = self.children
            for op in ops:
                if op.can_apply(left.cat, right.cat) and \
                    op.apply(left.cat, right.cat) == self.cat:
                    self.rule_type = op
                    break
            if self.rule_type is None:
                self.rule_type = NONE

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
                respadlen = (rwidth - lwidth - len(str_res)) // 2 + lwidth
                print(respadlen * " " + str_res)
                return rwidth

        print(catstr.rstrip())
        print(wordstr.rstrip())
        rec(0, self)

def to_xml(trees, out):
    candc = Element("candc")
    for tree in trees:
        candc.append(tree.to_xml())
    with open(out, "w") as f:
        ElementTree(candc).write(f)


def resolve_op(tree, ops):
    tree.resolve_op(ops)
    for child in tree.children:
        if isinstance(child, Tree):
            resolve_op(child, ops)

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


