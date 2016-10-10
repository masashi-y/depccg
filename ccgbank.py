# -*- coding: utf-8 -*-

from cat import Cat

class AutoReader(object):
    def __init__(self, filename):
        self.lines = open(filename).readlines()

    def readall(self):
        res = {}
        for line in self.lines:
            line = line.strip()
            if len(line) == 0: continue
            if line.startswith("ID"):
                key = line
            else:
                tree = Tree.parse(AutoLineReader(line))
                res[key] = tree
        return res.values()

class AutoLineReader(object):
    def __init__(self, line):
        self.line = line
        self.index = 0

    def next(self):
        end = self.line.find(" ", self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, text, offset=0):
        if self.line[self.index + offset] != text:
            raise RuntimeError("AutoLineReader.check catches parse error")

    def peek(self):
        return self.line[self.index]

    @property
    def next_node_type(self):
        if self.line[self.index+2] == "L":
            return Leaf
        elif self.line[self.index+2] == "T":
            return Tree
        else:
            raise RuntimeError()


class Leaf(object):
    """
    (<L N/N NNP NNP Pierre N_73/N_73>)
    """
    def __init__(self, word, cat, pos):
        self.word = word
        self.cat  = cat
        self.pos  = pos

    def __str__(self):
        return "(<L {0} {1} {1} {2} {0}>)".format(
                self.cat, self.pos, self.word)

    @staticmethod
    def parse(reader):
        reader.check("(")
        reader.check("<", 1)
        reader.check("L", 2)
        _    = reader.next()
        cat  = Cat.parse(reader.next())
        pos  = reader.next()
        _    = reader.next()
        word = reader.next()
        end  = reader.next()
        return Leaf(word, cat, pos)


class Tree(object):
    """
    (<T N 1 2> (<L N/N JJ JJ nonexecutive N_43/N_43>) (<L N NN NN director N>) )
    """
    def __init__(self, cat, left_is_head, children):
        self.cat          = cat
        self.children     = children
        self.left_is_head = left_is_head
        self.op = None

    def __str__(self):
        left_is_head = 0 if self.left_is_head else 1
        children = [str(c) for c in self.children]
        return "(<T {0} {1} {2}> {3} )".format(
                self.cat, left_is_head, len(children), " ".join(children))

    @staticmethod
    def parse(reader):
        reader.check("(")
        reader.check("<", 1)
        reader.check("T", 2)
        reader.next()
        cat = Cat.parse(reader.next())
        left_is_head = reader.next() == "0"
        _ = reader.next()
        children = []
        while reader.peek() != ")":
            children.append(reader.next_node_type.parse(reader))
        end = reader.next()
        return Tree(cat, left_is_head, children)

    def resolve_op(self, ops):
        if len(self.children) == 1:
            self.op = "<Un>"
        else:
            left, right = self.children
            for op in ops:
                if op.can_apply(left.cat, right.cat) and \
                    op.apply(left.cat, right.cat) == self.cat:
                    self.op = op
                    break
            if self.op is None:
                self.op = "<UNK>"

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
    rec(tree)
    return res


def show_derivation(tree):
    catstr  = ""
    wordstr = ""
    for leaf in get_leaves(tree):
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
    rec(0, tree)
