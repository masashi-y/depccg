
from ccgbank import Tree, Leaf
import cat
from combinator import *
from tree import get_leaves

combinators = ["<", ">", "ADNext", "ADNint", "ADV0",
               "ADV1", "ADV2", ">B", "<B1", "<B2", "<B3",
               "<B4", ">Bx1", ">Bx2", ">Bx3", "SSEQ"]

F = cat.Slash.Fwd()
B = cat.Slash.Bwd()

combinators = {
        "<": BackwardApplication(),
        ">": ForwardApplication(),
        "ADNext": unary_rule,
        "ADNint": unary_rule,
        "ADV0": unary_rule,
        "ADV1": unary_rule,
        "ADV2": unary_rule,
        ">B": GeneralizedForwardComposition(0, RuleType.FC, F, F, F),
        "<B1": GeneralizedBackwardComposition(0, RuleType.BC, B, B, B),
        "<B2": GeneralizedBackwardComposition(1, RuleType.BC, B, B, B),
        "<B3": GeneralizedBackwardComposition(2, RuleType.BC, B, B, B),
        "<B4": GeneralizedBackwardComposition(3, RuleType.BC, B, B, B),
        ">Bx1": GeneralizedForwardComposition(0, RuleType.FX, F, B, B),
        ">Bx2": GeneralizedForwardComposition(1, RuleType.FX, F, B, B),
        ">Bx3": GeneralizedForwardComposition(2, RuleType.FX, F, B, B),
        "SSEQ": SSEQ()
        }



class JaCCGReader(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def readall(self):
        res = []
        for line in open(self.filepath):
            line = line.strip().decode("utf-8")
            if len(line) == 0: continue
            res.append(JaCCGLineReader(line).parse())
        return res


class JaCCGLineReader(object):
    def __init__(self, line):
        self.line = line
        self.index = 0
        self.word_id = -1

    def next(self, target):
        end = self.line.find(target, self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, text, offset=0):
        if self.line[self.index + offset] != text:
            raise RuntimeError("AutoLineReader.check catches parse error")

    def peek(self):
        return self.line[self.index]

    def parse(self):
        return self.next_node()

    @property
    def next_node(self):
        end = self.line.find(" ", self.index)
        if self.line[self.index+1:end] in combinators:
            return self.parse_tree
        else:
            return self.parse_leaf

    def parse_leaf(self):
        self.word_id += 1
        self.check("{")
        cate = self.next(" ")[1:].encode("utf-8")
        cate = cate[:cate.find("_")]
        cate = cat.parse(cate)
        word = self.next("}")[:-1].split("/")[0]
        return Leaf(word, cate, self.word_id)

    def parse_tree(self):
        self.check("{")
        op = self.next(" ")
        op = combinators[op[1:]]
        cate = cat.parse(self.next(" ").encode("utf-8"))
        self.check("{")
        children = []
        while self.peek() != "}":
            children.append(self.next_node())
            if self.peek() == " ":
                self.next(" ")
        self.next("}")
        left_is_head = True if len(children) == 1 else \
                op.head_is_left(children[0].cat, children[1].cat)
        return Tree(cate, left_is_head, children, op)

def test():
    sents = \
        [line.strip().decode("utf-8") for line in open("test.ccgbank")]
    tree = JaCCGLineReader("{< NP {(S\\NP){I2}_none test} {(S\\NP){I2}_none test}}".decode("utf-8")).parse()
    for sent in sents:
        if len(sent) == 0:
            continue
        tree = JaCCGLineReader(sent).parse()
        if len(get_leaves(tree)) < 10:
            # print tree
            if not isinstance(tree, Leaf):
                tree.show_derivation()

# test()

