
from typing import List, Optional, Dict, Any
from depccg.py_tree import Tree
from depccg.types import Token
from depccg.py_cat import Category


def _json_of_category(category: Category) -> Dict[str, Any]:

    def rec(node):
        if node.is_functor:
            return {
                'slash': node.slash,
                'left': rec(node.left),
                'right': rec(node.right)
            }
        else:
            feature = node.features
            return {
                'base': node.base,
                'feature': feature if len(feature) > 0 else None
            }

    return rec(category)


def json_of(
    tree: Tree,
    tokens: Optional[List[Token]] = None,
    full: bool = False
) -> Dict[str, Any]:
    """a tree in Python dict object.

    Args:
        tree (Tree): tree object
        tokens (Optional[List[Token]], optional): list of token objects. Defaults to None.
        full (bool): whether to decomopose categories into its components, i.e., 
            {
                'slash': '/',
                'left': {'base': 'S', 'feature': 'adj'},
                'right': {'base': 'NP', 'feature': None},
            },
            or just as a string "S[adj]/NP".

    Returns:
        str: tree string in the CoNLL format
    """

    def rec(node: Tree) -> Dict[str, Any]:

        if node.is_leaf:
            token = tokens.pop(0)
            res = dict(token)
            res['cat'] = _json_of_category(node.cat) if full else str(node.cat)
            return res
        else:
            return {
                'type': node.op_string,
                'cat': _json_of_category(node.cat) if full else str(node.cat),
                'children': [rec(child) for child in node.children]
            }

    if tokens is None:
        tokens = [Token.of_word(word) for word in tree.word.split(' ')]

    tokens = list(tokens)
    return rec(tree)
