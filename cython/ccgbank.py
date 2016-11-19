# -*- coding: utf-8 -*-

import cat
from tree import Tree, Leaf
from combinator import Combinator
import combinator


class AutoReader(object):
    def __init__(self, filename):
        self.lines = open(filename).readlines()

    def readall(self, suppress_error=False):
        # Inputs:
        # suppress_error (bool): Some CCGbank annotations are not supported and
        #     can raise RuntimeError in Tree.parse. Setting this option True
        #     suppresss the error and ignores the sentence with that annotation.
        res = []
        for line in self.lines:
            line = line.strip()
            if len(line) == 0: continue
            if line.startswith("ID"):
                key = line
            else:
                try:
                    tree = AutoLineReader(line).parse()
                    res.append(tree)
                except RuntimeError as e:
                    if suppress_error:
                        continue
                    else:
                        raise e
        return res


class AutoLineReader(object):
    def __init__(self, line):
        self.line = line.encode("utf-8")
        self.index = 0

    def next(self):
        end = self.line.find(" ", self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, text, offset=0):
        if self.line[self.index + offset] != text:
            print self.line
            raise RuntimeError("AutoLineReader.check catches parse error")

    def peek(self):
        return self.line[self.index]

    def parse(self):
        return self.next_node()

    @property
    def next_node(self):
        if self.line[self.index+2] == "L":
            return self.parse_leaf
        elif self.line[self.index+2] == "T":
            return self.parse_tree
        else:
            raise RuntimeError()

    def parse_leaf(self):
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


