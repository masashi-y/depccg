from typing import List
from depccg.utils import denormalize
from depccg.tree import Tree


def _resolve_dependencies(tree: Tree) -> List[int]:
    results = []

    def rec(node: Tree) -> int:
        if node.is_leaf:
            index = len(results)
            results.append(-1)
            return index
        else:
            if node.is_unary:
                return rec(node.child)
            else:
                left_head = rec(node.left_child)
                right_head = rec(node.right_child)
                if node.head_is_left:
                    results[right_head] = left_head
                    return left_head
                else:
                    results[left_head] = right_head
                    return right_head

    rec(tree)

    assert len(
        [dependency for dependency in results if dependency == -1]
    ) == 1

    return results


def conll_of(tree: Tree) -> str:
    """CoNLL-like format string where dependency relations are constructed by
    nodes' head_is_left property.

    Args:
        tree (Tree): tree object
        tokens (Optional[List[Token]], optional): list of token objects. Defaults to None.

    Returns:
        str: tree string in the CoNLL format
    """

    stack = []
    counter = 1

    def rec(node):
        nonlocal stack, counter
        if node.is_leaf:
            cat = node.cat
            word = denormalize(node.word)
            token = node.token
            lemma = token.get('lemma', '_')
            pos = token.get('pos', '_')
            stack.append(f'(<L {cat} {pos} {pos} {word} {cat}>)')
            subtree = ' '.join(stack)

            line = '\t'.join(
                (
                    str(counter),
                    word,
                    lemma,
                    pos,
                    pos,
                    '_',
                    str(dependencies[counter - 1] + 1),
                    str(cat),
                    '_',
                    subtree
                )
            )

            stack = []
            counter += 1
            return line
        else:
            cat = node.cat
            num_children = len(node.children)
            head_is_left = 0 if node.head_is_left else 1
            stack.append(f'(<T {cat} {head_is_left} {num_children}>')
            children = '\n'.join(rec(child) for child in node.children) + ' )'
            return children

    dependencies = _resolve_dependencies(tree)
    return rec(tree)
