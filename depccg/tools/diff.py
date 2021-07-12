
import argparse
import logging
from .reader import read_trees_guess_extension
from depccg.printer.html import (
    _mathml_subtree, _MATHML_SUBTREE_TERMINAL, _mathml_cat, _MATHML_SUBTREE_NONTERMINAL, _MATHML_MAIN
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def diff(tree1, tree2):
    def rec(node1, node2):
        nodes_are_different = False
        if node1.cat != node2.cat:
            nodes_are_different = True
        elif len(node1.children) != len(node2.children):
            nodes_are_different = True
        elif any(child1.word != child2.word
                 for child1, child2 in zip(node1.children, node2.children)):
            nodes_are_different = True

        if nodes_are_different:
            return (nodes_are_different,
                    _mathml_subtree(node1, bgcolor='peachpuff'),
                    _mathml_subtree(node2, bgcolor='peachpuff'))
        elif node1.is_leaf:
            assert node2.is_leaf
            node1_str = _MATHML_SUBTREE_TERMINAL.format(
                node1.word, _mathml_cat(str(node1.cat)))
            node2_str = _MATHML_SUBTREE_TERMINAL.format(
                node2.word, _mathml_cat(str(node2.cat)))
            return nodes_are_different, node1_str, node2_str
        else:
            children_are_different, node1_children, node2_children = \
                zip(*[rec(child1, child2)
                    for child1, child2 in zip(node1.children, node2.children)])
            node1_children = ''.join(node1_children)
            node2_children = ''.join(node2_children)
            node1_str = _MATHML_SUBTREE_NONTERMINAL.format(
                node1_children, _mathml_cat(str(node1.cat)), node1.op_string, '')
            node2_str = _MATHML_SUBTREE_NONTERMINAL.format(
                node2_children, _mathml_cat(str(node2.cat)), node2.op_string, '')
            nodes_are_different = any(children_are_different)
            return nodes_are_different, node1_str, node2_str

    return rec(tree1, tree2)


def to_diffs(tree_pairs, max_output_num, sampling):
    diffs = [(i, name1, name2, tree1, tree2) + diff(tree1, tree2)
             for i, ((name1, tree1), (name2, tree2)) in enumerate(tree_pairs)]
    diffs = [(i, name1, name2, tree1, tree2, tree1_str, tree2_str)
             for (i, name1, name2, tree1, tree2, trees_are_different, tree1_str, tree2_str) in diffs
             if trees_are_different]
    logger.info(f'The number of input tree pairs: {len(tree_pairs)}')
    logger.info(f'The number of different tree pairs: {len(diffs)}')
    if len(diffs) > max_output_num:
        logger.info(
            f'The number of different tree pairs exceeds --max-output-num: {max_output_num}.')
        logger.info(f'sample the subset of pairs using "{sampling}" method.')
        if sampling == 'head':
            diffs = diffs[:max_output_num]
        elif sampling == 'tail':
            diffs = diffs[-max_output_num:]
        elif sampling == 'random':
            import random
            indices = list(range(len(diffs)))
            indices = sorted(random.sample(indices, max_output_num))
            diffs = [diffs[i] for i in indices]
        else:
            assert False
    result = ''
    for (i, name1, name2, tree1, tree2, tree1_str, tree2_str) in diffs:
        if i > 0:
            result += '<hr>'
        result += f'<p><strong>{name1}</strong> ID={i}: {tree1.word}</p>'
        result += f'<math xmlns="http://www.w3.org/1998/Math/MathML">{tree1_str}</math>'
        result += f'<p><strong>{name2}</strong> ID={i}: {tree2.word}</p>'
        result += f'<math xmlns="http://www.w3.org/1998/Math/MathML">{tree2_str}</math>'
    return _MATHML_MAIN.format(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'show diffs between the structures of two CCG trees')
    parser.add_argument('file1')
    parser.add_argument('file2', nargs='?', default=None)
    parser.add_argument('--max-output-num', default=50, type=int)
    parser.add_argument('--sampling', default='head',
                        choices=['head', 'tail', 'random'])
    args = parser.parse_args()
    if args.file2:
        file1_trees = [(args.file1, tree)
                       for _, _, tree in read_trees_guess_extension(args.file1)]
        file2_trees = [(args.file2, tree)
                       for _, _, tree in read_trees_guess_extension(args.file2)]
        assert len(file1_trees) == len(file2_trees)
        tree_pairs = list(zip(file1_trees, file2_trees))
    else:
        file1_trees = ((args.file1, tree)
                       for _, tree in read_trees_guess_extension(args.file1))
        tree_pairs = list(zip(file1_trees, file1_trees))

    print(to_diffs(tree_pairs, args.max_output_num, args.sampling))
