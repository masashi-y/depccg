from depccg.lang import get_global_language
from typing import NamedTuple, List, Iterator, Union
from depccg.cat import Category
from depccg.grammar import guess_combinator_by_triplet, en, ja
from depccg.types import Token


BINARY_RULES = {
    'en': en.apply_binary_rules,
    'ja': ja.apply_binary_rules,
}


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
                    rule = guess_combinator_by_triplet(
                        BINARY_RULES[get_global_language()],
                        cat, left.cat, right.cat
                    )
                    return Tree.make_binary(
                        cat, left, right, rule.op_string, rule.op_symbol, rule.head_is_left
                    )

        return rec(tree)

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
    def tokens(self) -> List[Token]:
        return [leaf.children[0] for leaf in self.leaves]

    @property
    def token(self) -> Token:
        assert self.is_leaf, "Tree.token must be called on leaf objects"
        return self.children[0]

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
        return ' '.join(token[token_key] for token in self.tokens)

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
