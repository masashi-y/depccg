from typing import Iterator, List, Tuple
import re

from depccg.cat import Category
from depccg.tree import Tree
from depccg.types import Token
from depccg.tools.reader import ReaderResult


combinators = {
    'SSEQ', '>', '<', '>B', '<B1', '<B2', '<B3',
    '<B4', '>Bx1', '>Bx2', '>Bx3',
    'ADNext', 'ADNint', 'ADV0', 'ADV1', 'ADV2'
}

DEPENDENCY = re.compile(r'{.+?}')


def read_ccgbank(filepath: str) -> Iterator[ReaderResult]:
    """read Japanase CCGBank file.

    Args:
        filename (str): file name string

    Yields:
        Iterator[ReaderResult]: iterator object containing parse results
    """

    for i, line in enumerate(open(filepath)):
        line = line.strip()
        if len(line) == 0:
            continue
        tree, tokens = _JaCCGLineReader(line).parse()
        yield ReaderResult(str(i), tokens, tree)


class _JaCCGLineReader(object):
    def __init__(self, line: str) -> None:
        self.line = line
        self.index = 0
        self.word_id = -1
        self.tokens = []

    def next(self, target: str) -> str:
        end = self.line.find(target, self.index)
        result = self.line[self.index:end]
        self.index = end + 1
        return result

    def check(self, text: str, offset: int = 0) -> None:
        if self.line[self.index + offset] != text:
            raise RuntimeError('AutoLineReader.check catches parse error')

    def peek(self) -> str:
        return self.line[self.index]

    def parse(self) -> Tuple[Tree, List[Token]]:
        result = self.next_node()
        return result, self.tokens

    @property
    def next_node(self):
        end = self.line.find(' ', self.index)
        if self.line[self.index + 1:end] in combinators:
            return self.parse_tree
        else:
            return self.parse_leaf

    def parse_leaf(self) -> Tree:
        self.word_id += 1
        self.check('{')
        cat = self.next(' ')[1:]
        cat = cat[:cat.find('_')]
        cat = DEPENDENCY.sub('', cat)
        cat = Category.parse(cat)
        surf, base, pos1, pos2 = self.next('}')[:-1].split('/')
        token = Token(surf=surf, base=base, pos1=pos1, pos2=pos2)
        self.tokens.append(token)
        return Tree.make_terminal(surf, cat)

    def parse_tree(self) -> Tree:
        self.check('{')
        op_string = self.next(' ')
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
            return Tree.make_unary(cat, children[0], op_string, op_string)
        else:
            assert len(
                children) == 2, f'failed to parse, invalid number of children: {self.line}'
            left, right = children
            return Tree.make_binary(cat, left, right, op_string, op_string)
