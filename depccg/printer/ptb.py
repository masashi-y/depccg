from depccg.tree import Tree


def ptb_of(tree: Tree) -> Tree:
    """PTB-style string of a CCG tree

    Args:
        tree (Tree): tree object

    Returns:
        str: tree string in the PTB style
    """

    def rec(node):
        if node.is_leaf:
            cat = node.cat
            word = node.word
            return f'({cat} {word})'
        else:
            cat = node.cat
            children = ' '.join(rec(child) for child in node.children)
            return f'({cat} {children})'

    return f'(ROOT {rec(tree)})'
