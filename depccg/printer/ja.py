from typing import List, Optional

from depccg.tree import Tree
from depccg.tokens import Token
from depccg.py_utils import normalize


def ja_of(tree: Tree, tokens: Optional[List[Token]] = None) -> str:
    """tree string in the Japanese CCGBank's format

    Args:
        tree (Tree): tree object
        tokens (Optional[List[Token]], optional): list of token objects. Defaults to None.

    Returns:
        str: tree string in Japanese CCGBank's format
    """

    def rec(node):
        if node.is_leaf:
            cat = node.cat
            word = normalize(node.word)
            token = tokens.pop(0)

            poss = [
                token.get(pos, '*')
                for pos in ('pos', 'pos1', 'pos2', 'pos3')
            ]

            poss = [pos for pos in poss if pos != '*']
            pos = '-'.join(poss) if len(poss) else '_'

            inflections = [
                token.get(i, '*')
                for i in ('inflectionForm', 'inflectionType')
            ]

            inflections = [i for i in inflections if i != '*']
            inflection = '-'.join(inflections) if len(inflections) else '_'

            return f'{{{cat} {word}/{word}/{pos}/{inflection}}}'
        else:
            children = ' '.join(rec(child) for child in node.children)
            return f'{{{node.op_symbol} {node.cat} {children}}}'

    if tokens is None:
        tokens = [Token.from_word(word) for word in tree.word.split(' ')]

    return rec(tree)
