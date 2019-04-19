
import re

from depccg.combinator import ja_default_binary_rules, unary_rule
from depccg.cat import Category
from depccg.tree import Tree
from depccg.token import Token


combinators = {sign: rule for rule, sign in zip(
    ja_default_binary_rules,
    ['SSEQ', '>', '<', '>B', '<B1', '<B2', '<B3', '<B4', '>Bx1', '>Bx2', '>Bx3'])
}

for sign in ['ADNext', 'ADNint', 'ADV0', 'ADV1', 'ADV2']:
    combinators[sign] = unary_rule()

DEPENDENCY = re.compile(r'{.+?}')

def read_ccgbank(filepath):
    for i, line in enumerate(open(filepath)):
        line = line.strip()
        if len(line) == 0:
            continue
        tree, tokens = _JaCCGLineReader(line).parse()
        yield str(i), tokens, tree


class _JaCCGLineReader(object):
    def __init__(self, line):
        self.lang = 'ja'
        self.line = line
        self.index = 0
        self.word_id = -1
        self.tokens = []

    def next(self, target):
        end = self.line.find(target, self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, text, offset=0):
        if self.line[self.index + offset] != text:
            raise RuntimeError('AutoLineReader.check catches parse error')

    def peek(self):
        return self.line[self.index]

    def parse(self):
        res = self.next_node()
        return res, self.tokens

    @property
    def next_node(self):
        end = self.line.find(' ', self.index)
        if self.line[self.index+1:end] in combinators:
            return self.parse_tree
        else:
            return self.parse_leaf

    def parse_leaf(self):
        self.word_id += 1
        self.check('{')
        cat = self.next(' ')[1:]
        cat = cat[:cat.find('_')]
        cat = DEPENDENCY.sub('', cat)
        cat = Category.parse(cat)
        surf, base, pos1, pos2 = self.next('}')[:-1].split('/')
        token = Token(surf=surf, base=base, pos1=pos1, pos2=pos2)
        self.tokens.append(token)
        return Tree.make_terminal(surf, cat, self.word_id, self.lang)

    def parse_tree(self):
        self.check('{')
        op = self.next(' ')
        op = combinators[op[1:]]
        cat = DEPENDENCY.sub('', self.next(' '))
        cat = Category.parse(cat)
        self.check('{')
        children = []
        while self.peek() != '}':
            children.append(self.next_node())
            if self.peek() == ' ':
                self.next(' ')
        self.next('}')
        if len(children) == 1:
            return Tree.make_unary(cat, children[0], self.lang)
        else:
            assert len(children) == 2, f'failed to parse, invalid number of children: {self.line}'
            left, right = children
            left_is_head = op.head_is_left(left.cat, right.cat)
            return Tree.make_binary(cat, left_is_head, left, right, op, self.lang)
