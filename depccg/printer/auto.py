from typing import List, Optional
from depccg.py_tree import Tree
from depccg.types import Token
from depccg.py_utils import normalize, denormalize


def auto_of(tree: Tree, tokens: Optional[List[Token]] = None) -> str:
    """tree string in auto format commonly used in English CCGBank.

    Args:
        tree (Tree): tree object
        tokens (Optional[List[Token]], optional): list of token objects. Defaults to None.

    Returns:
        str: tree string in the auto format
    """

    def rec(node):
        if node.is_leaf:
            cat = node.cat
            word = denormalize(node.word)
            pos = poss.pop(0)
            return f'(<L {cat} {pos} {pos} {word} {cat}>)'
        else:
            cat = node.cat
            children = ' '.join(rec(child) for child in node.children)
            num_children = len(node.children)
            head_is_left = 0 if node.head_is_left else 1
            return f'(<T {cat} {head_is_left} {num_children}> {children} )'

    if tokens:
        poss = [token.pos for token in tokens]
    else:
        poss = ['POS' for _ in range(len(tree))]

    return rec(tree)


def auto_flattened_of(tree: Tree, tokens: Optional[List[Token]] = None) -> str:
    """tree string in flattened version of auto format.
    This is mainly used for evaluation only.

    Args:
        tree (Tree): tree object
        tokens (Optional[List[Token]], optional): list of token objects. Defaults to None.

    Returns:
        str: tree string in the flattened auto format
    """

    def rec(node):
        if node.is_leaf:
            cat = node.cat
            word = normalize(node.word).replace('/', '\\/')
            pos = poss.pop(0)
            return f'(<L *** {cat} {pos} {word}>\n)'
        else:
            cat = node.cat
            children = '\n'.join(rec(child) for child in node.children)
            num_children = len(node.children)
            head_is_left = 0 if node.head_is_left else 1
            return f'(<T *** {cat} * {head_is_left} {num_children}>\n{children}\n)'

    if tokens:
        poss = [token.pos for token in tokens]
    else:
        poss = ['POS' for _ in range(len(tree))]

    return f'###\n{rec(tree)}\n'


def auto_extended_of(tree: Tree, tokens: Optional[List[Token]] = None) -> str:
    """tree string in extended version of auto format, used by C&C.

    Args:
        tree (Tree): tree object
        tokens (Optional[List[Token]], optional): list of token objects. Defaults to None.

    Returns:
        str: tree string in the extended auto format
    """

    def rec(node):
        if node.is_leaf:
            cat = node.cat
            word = denormalize(node.word)
            token = tokens.pop(0)
            lemma = token.get('lemma', 'XX')
            pos = token.get('pos', 'XX')
            entity = token.get('entity', 'XX')
            chunk = token.get('chunk', 'XX')
            return f'(<L {cat} {word} {lemma} {pos} {entity} {chunk} {cat}>)'
        else:
            cat = node.cat
            children = ' '.join(rec(child) for child in node.children)
            num_children = len(node.children)
            head_is_left = 0 if node.head_is_left else 1
            rule = node.op_string
            return f'(<T {cat} {rule} {head_is_left} {num_children}> {children} )'

    if tokens is None:
        tokens = [Token.of_word(word) for word in tree.word.split(' ')]

    return rec(tree)
