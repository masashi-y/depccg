
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
        return res

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

