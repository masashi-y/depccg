from typing import NamedTuple, List, Iterator, Union
from depccg.py_cat import Category
# from depccg.combinator import UNKNOWN_COMBINATOR, guess_combinator_by_triplet
from depccg.types import Token
from depccg.lang import GLOBAL_LANG_NAME
# from depccg.printer.auto import auto_of


class _AutoLineReader(object):
    def __init__(self, line):
        self.line = line
        self.index = 0
        self.word_id = -1
        self.binary_rules = BINARY_RULES[GLOBAL_LANG_NAME]
        self.tokens = []

    def next(self):
        end = self.line.find(' ', self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, text, offset=0):
        if self.line[self.index + offset] != text:
            raise RuntimeError(f'failed to parse: {self.line}')

    def peek(self):
        return self.line[self.index]

    def parse(self):
        tree = self.next_node()
        return tree, self.tokens

    @property
    def next_node(self):
        if self.line[self.index + 2] == 'L':
            return self.parse_leaf
        elif self.line[self.index + 2] == 'T':
            return self.parse_tree
        else:
            raise RuntimeError(f'failed to parse: {self.line}')

    def parse_leaf(self):
        self.word_id += 1
        self.check('(')
        self.check('<', 1)
        self.check('L', 2)
        self.next()
        cat = Category.parse(self.next())
        tag1 = self.next()  # modified POS tag
        tag2 = self.next()  # original POS
        word = self.next().replace('\\', '')
        token = Token(word=word, pos=tag1, tag1=tag1, tag2=tag2)
        self.tokens.append(token)
        if word == '-LRB-':
            word = "("
        elif word == '-RRB-':
            word = ')'
        self.next()
        return Tree.make_terminal(word, cat)

    def parse_tree(self):
        self.check('(')
        self.check('<', 1)
        self.check('T', 2)
        self.next()
        cat = Category.parse(self.next())
        head_is_left = self.next() == '0'
        self.next()
        children = []
        while self.peek() != ')':
            children.append(self.next_node())
        self.next()
        if len(children) == 2:
            left, right = children
            op = guess_combinator_by_triplet(
                self.binary_rules, cat, left.cat, right.cat)
            op = op or UNKNOWN_COMBINATOR
            return Tree.make_binary(
                cat, left, right, op, head_is_left=head_is_left)
        elif len(children) == 1:
            return Tree.make_unary(cat, children[0])
        else:
            raise RuntimeError(f'failed to parse: {self.line}')


class Tree(object):

    def __init__(
        self,
        cat: Category,
        children: Union[List['Tree'], List[Token]],
        op_string: str,
        op_symbol: str,
        head_is_left: bool = True,
    ) -> None:
        assert len({type(child) for child in children}) == 1, \
            "children must contain elements of a unique type"
        assert not isinstance(children[0], Tree) or len(children) in (1, 2), \
            "a tree cannot contain more than two children"
        assert not isinstance(children[0], Token) or len(children) == 1, \
            "a leaf node cannot contain more than one token object"

        self.cat = cat
        self.children = children
        self.op_string = op_string
        self.op_symbol = op_symbol
        self.head_is_left = head_is_left

    @staticmethod
    def make_terminal(
        word: Union[str, Token],
        cat: Category,
        op_string: str = 'lex',
        op_symbol: str = '<lex>',
    ) -> 'Tree':

        if isinstance(word, Token):
            token = word
        else:
            token = Token(word=word)

        return Tree(cat, [token], op_string, op_symbol)

    @staticmethod
    def make_binary(
        cat: Category,
        left: 'Tree',
        right: 'Tree',
        op_string: str,
        op_symbol: str,
        head_is_left: bool = True,
    ) -> 'Tree':
        return Tree(cat, [left, right], op_string, op_symbol, head_is_left)

    @staticmethod
    def make_unary(
        cat: Category,
        child: 'Tree',
        op_string: str = 'lex',
        op_symbol: str = '<un>'
    ) -> 'Tree':
        return Tree(cat, [child], op_string, op_symbol)

    @staticmethod
    def of_auto(line: str) -> 'Tree':
        return _AutoLineReader(line).parse()

    @staticmethod
    def of_nltk_tree(tree) -> 'Tree':

        def rec(node):
            cat = Category.parse(node.label())
            if isinstance(node[0], str):
                word = node[0]
                return Tree.make_terminal(word, cat)
            else:
                children = [rec(child) for child in node]
                if len(children) == 1:
                    return Tree.make_unary(cat, children[0])
                else:
                    assert len(children) == 2
                    left, right = children
                    op = guess_combinator_by_triplet(cat, left.cat, right.cat)
                    op = op or UNKNOWN_COMBINATOR
                    return Tree.make_binary(cat, left, right, op)

        return rec(tree)

    # property op_symbol:
    #     """standard CCG style string representing a combinator. e.g. >, <, >B"""

    #     def __get__(self):
    #         assert not self.is_leaf, "This node is leaf and does not have combinator!"
    #         cdef const CTree * c_node = <const CTree*> & deref(self.node_)
    #         # tentatively put this here
    #         if self.lang == b'ja' and self.is_unary:
    #             child_features = self.child.cat.arg(0).features.items()
    #             if ('mod', 'adn') in child_features:
    #                 if self.child.cat.base == 'S':
    #                     return 'ADNext'
    #                 else:
    #                     return 'ADNint'
    #             elif ('mod', 'adv') in child_features:
    #                 if self.cat.base == '(S\\NP)/(S\\NP)':
    #                     return 'ADV1'
    #                 else:
    #                     return 'ADV0'
    #             # else:
    #             #     raise RuntimeError('this tree is not supported in `ja` format')
    #         return c_node.GetRule().ToStr().decode('utf-8')

    def __len__(self):
        return len(self.leaves)

    @property
    def leaves(self) -> List['Tree']:

        def rec(node):
            if node.is_leaf:
                result.append(node)
            else:
                for child in node.children:
                    rec(child)

        result = []
        rec(self)
        return result

    @property
    def child(self):
        assert self.is_unary, "This node is not unary node! Please use `Tree.children`"
        return self.left_child

    @property
    def left_child(self):
        assert not self.is_leaf, "This node is leaf and does not have any child!"
        return self.children[0]

    @property
    def right_child(self):
        assert not self.is_leaf, "This node is leaf and does not have any child!"
        assert not self.is_unary, "This node does not have right child!"
        return self.children[1]

    @property
    def is_leaf(self):
        return (
            self.is_unary and isinstance(self.children[0], Token)
        )

    @property
    def word(self, token_key='word'):
        return ' '.join(leaf.children[0][token_key] for leaf in self.leaves)

    @property
    def is_unary(self) -> bool:
        return len(self.children) == 1

    # def __str__(self):
    #     return auto_of(self)

    # def __repr__(self):
    #     return auto_of(self)

    def nltk_tree(self):
        from nltk.tree import Tree

        def rec(node):
            if node.is_leaf:
                cat = node.cat
                children = [node.word]
            else:
                cat = node.cat
                children = [rec(child) for child in node.children]
            return Tree(str(cat), children)

        return rec(self)


class ScoredTree(NamedTuple):
    tree: Tree
    score: float


class ParseResult(NamedTuple):
    sentence_index: int
    tree_index: int
    tree: Tree
    tokens: List[Token]
    score: float


def iter_parse_results(
    nbest_trees: List[List[ScoredTree]],
    tagged_doc: List[List[Token]]
) -> Iterator[ParseResult]:

    for sentence_index, (trees, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
        for tree_index, (tree, log_prob) in enumerate(trees, 1):
            yield ParseResult(
                sentence_index,
                tree_index,
                tree,
                tokens,
                log_prob,
            )
