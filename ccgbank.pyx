# -*- coding: utf-8 -*-

cimport cat
import cat
cimport combinator
from combinator cimport Combinator


cdef class AutoReader(object):
    def __init__(self, str filename):
        self.lines = open(filename).readlines()

    def readall(self, bint suppress_error=False):
        # Inputs:
        # suppress_error (bool): Some CCGbank annotations are not supported and
        #     can raise RuntimeError in Tree.parse. Setting this option True
        #     suppresss the error and ignores the sentence with that annotation.
        cdef dict res = {}
        cdef str line, key
        cdef Tree tree

        for line in self.lines:
            line = line.strip()
            if len(line) == 0: continue
            if line.startswith("ID"):
                key = line
            else:
                try:
                    tree = AutoLineReader(line).parse_tree()
                    res[key] = tree
                except RuntimeError as e:
                    if suppress_error:
                        continue
                    else:
                        raise e
        return res.values()


cdef class AutoLineReader(object):
    def __init__(self, line):
        self.line = line
        self.index = 0

    def next(self):
        cdef int end
        cdef str res
        end = self.line.find(" ", self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, str text, int offset=0):
        if self.line[self.index + offset] != text:
            raise RuntimeError("AutoLineReader.check catches parse error")

    def peek(self):
        return self.line[self.index]

    @property
    def next_node(self):
        if self.line[self.index+2] == "L":
            return self.parse_leaf
        elif self.line[self.index+2] == "T":
            return self.parse_tree
        else:
            raise RuntimeError()

    def parse_leaf(self):
        cdef str _, word, end
        cdef Cat cate
        self.check("(")
        self.check("<", 1)
        self.check("L", 2)
        _    = self.next()
        cate = cat.parse(self.next())
        _    = self.next() # POS tag
        _    = self.next()
        word = self.next()
        end  = self.next()
        return Leaf(word, cate, 0)

    def parse_tree(self):
        cdef str _, end
        cdef Cat cate
        cdef bint left_is_head
        cdef list children

        self.check("(")
        self.check("<", 1)
        self.check("T", 2)
        self.next()
        cate = cat.parse(self.next())
        left_is_head = self.next() == "0"
        _ = self.next()
        children = []
        while self.peek() != ")":
            children.append(self.next_node())
        end = self.next()
        return Tree(cate, left_is_head, children, combinator.UnaryRule())


cdef class Node(object):
    def __init__(self, Cat cat, int rule_type):
        self.cat = cat
        self.rule_type = rule_type


cdef class Leaf(Node):
    # (<L N/N NNP NNP Pierre N_73/N_73>)
    def __init__(self, str word, Cat cat, int position):
        super(Leaf, self).__init__(cat, LEXICON)
        self.word = word
        self.pos  = position

    def __str__(self):
        cdef str pos = "POS" # dummy
        cdef str word
        if self.word in ["{", "("]:
            word = "-LRB-"
        elif self.word in ["}", ")"]:
            word = "-RRB-"
        else:
            word = self.word
        return "(<L {0} {1} {1} {2} {0}>)".format(
                self.cat, pos, word)

    @property
    def headid(self):
        return self.pos

    @property
    def deplen(self):
        return 0


cdef class Tree(Node):
    # (<T N 1 2> (<L N/N JJ JJ nonexecutive N_43/N_43>) (<L N NN NN director N>) )
    def __init__(self, cat, left_is_head, children, rule=None):
        rule_type = NONE if not isinstance(rule, Combinator) \
                                    else rule.rule_type
        super(Tree, self).__init__(cat, rule_type)
        self.children     = children
        self.left_is_head = left_is_head
        self.op = rule

    def __str__(self):
        cdef int left_is_head = 0 if self.left_is_head else 1
        cdef list children = [str(c) for c in self.children]
        return "(<T {0} {1} {2}> {3} )".format(
                self.cat, left_is_head, len(children), " ".join(children))

    def resolve_op(self, ops):
        if len(self.children) == 1:
            self.rule_type = UNARY
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
        cdef list children = self.children
        if len(children) == 1:
            return children[0].headid
        elif len(children) == 2:
            return children[0].headid if self.left_is_head else children[1].headid
        else:
            raise RuntimeError("Number of children of Tree must be 1 or 2.")

    @property
    def deplen(self):
        cdef list children = self.children
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


def resolve_op(Tree tree, ops):
    tree.resolve_op(ops)
    for child in tree.children:
        if isinstance(child, Tree):
            resolve_op(child, ops)

def get_leaves(Tree tree):
    cdef list res = []
    def rec(Tree tree):
        cdef Node child

        for child in tree.children:
            if isinstance(child, Tree):
                rec(child)
            elif isinstance(child, Leaf):
                res.append(child)
            else:
                raise RuntimeError()
    rec(tree)
    return res


