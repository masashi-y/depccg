from typing import List
from io import StringIO

from depccg.cat import Category
from depccg.tree import ScoredTree, Tree


def _prolog_category_string(cat: Category) -> str:

    def rec(this_cat: Category) -> str:

        if this_cat.is_atomic:
            base = this_cat.base.lower()
            if base == '.':
                return "period"
            elif base == ",":
                return "comma"
            elif base == ":":
                return "colon"
            elif base == ";":
                return "semicolon"
            elif str(this_cat.feature) == "":
                return base
            else:
                return f'{base}:{str(this_cat.feature)}'

        else:
            left = rec(this_cat.left)
            right = rec(this_cat.right)
            return f"({left}{this_cat.slash}{right})"

    return rec(cat)


def _escape_prolog(text: str) -> str:
    return text.replace("'", "\\'")


_op_mapping = {
    'fa': "fa(",
    'ba': "ba(",
    'fx': "fc(",
    'fc': "fc(",
    'bx': "bxc(",
    'gfc': "gfc(",
    'gbx': "gbx(",
    'rp': "rp(",
    'lp': "lx(",
    'conj': "conj(",
    'conj2': "conj(",
}


def _prolog_string(tree: Tree, sentence_index: int) -> str:

    position = 0
    depth = 1

    def indent(output):
        output.write(" " * depth)

    def rec(node: Tree, output):
        nonlocal depth, position

        indent(output)
        if node.is_leaf:
            token = node.token
            result_str = (
                f"t({_prolog_category_string(node.cat)}, "
                f"\'{_escape_prolog(node.word)}\', "
                f"\'{_escape_prolog(token.lemma)}\', "
                f"\'{token.pos}\', "
                f"\'{token.chunk}\', "
                f"\'{token.entity}\')"
            )
            output.write(result_str)
            position += 1
        elif node.is_unary:
            this_cat = _prolog_category_string(node.cat)
            child_cat = _prolog_category_string(node.left_child.cat)
            output.write(f"lx({this_cat}, {child_cat},\n")
            depth += 1
            rec(node.child, output)
            depth -= 1
            output.write(")")
        else:
            output.write(_op_mapping[node.op_string])
            output.write(_prolog_category_string(node.cat))
            output.write(",")

            if node.op_string == 'conj2':
                cat_str = _prolog_category_string(node.right_child.cat)
                output.write(f" {cat_str}\\{cat_str},\n")
                depth += 1
                indent(output)
                output.write(f"conj({cat_str}\\{cat_str}, {cat_str},")

            if node.op_string == 'conj':
                cat_str = _prolog_category_string(node.cat.left)
                output.write(f" {cat_str},")

            if node.op_string == 'lp':
                cat_str = _prolog_category_string(node.right_child.cat)
                output.write(f" {cat_str},\n")
                depth += 1
                indent(output)
                output.write(f"lp({cat_str},")

            output.write("\n")
            depth += 1
            rec(node.left_child, output)
            if not node.is_unary:
                output.write(",\n")
                rec(node.right_child, output)
            output.write(")")
            depth -= 1

            if node.op_string in ('conj2', 'lp'):
                output.write(")")
                depth -= 1

    with StringIO() as output:
        output.write(f"ccg({sentence_index},\n")
        rec(tree, output)
        output.write(").\n")
        return output.getvalue()


_prolog_header = (
    ':- op(601, xfx, (/)).\n'
    ':- op(601, xfx, (\\)).\n'
    ':- multifile ccg/2, id/2.\n'
    ':- discontiguous ccg/2, id/2.\n'
)


def to_prolog_en(
    nbest_trees: List[List[ScoredTree]],
) -> str:
    """convert parsing results to Prolog format used by LangPro.

    Args:
        nbest_trees (List[List[ScoredTree]]): parsing results

    Returns:
        str: Prolog string
    """

    with StringIO() as output:
        print(_prolog_header, file=output)
        for sentence_index, trees in enumerate(nbest_trees, 1):
            for tree, _ in trees:
                print(_prolog_string(tree, sentence_index), file=output)
        result = output.getvalue()
    return result


_ja_combinators = {
    'SSEQ': 'sseq',
    '>': 'fa',
    '<': 'ba',
    '>B': 'fc',
    '<B1': 'bc1',
    '<B2': 'bc2',
    '<B3': 'bc3',
    '<B4': 'bc4',
    '>Bx1': 'fx1',
    '>Bx2': 'fx2',
    '>Bx3': 'fx3',
    "ADNext": 'adnext',
    "ADNint": 'adnint',
    "ADV0": 'adv0',
    "ADV1": 'adv1',
}


def to_prolog_ja(
    nbest_trees: List[List[ScoredTree]],
) -> str:
    """convert parsing results to Prolog format used by LangPro.
    This is specifically used for Japanese sentences.

    Args:
        nbest_trees (List[List[ScoredTree]]): parsing results

    Returns:
        str: Prolog string
    """

    def traverse_cat(node):
        if node.is_functor:
            left = traverse_cat(node.left)
            right = traverse_cat(node.right)
            return f'({left}{node.slash}{right})'
        else:
            feature = dict(node.feature.items())
            base = node.base.lower()
            if 'case' not in feature:
                return base
            else:
                feature_case = feature["case"].lower()
                return f'{base}:{feature_case}'

    def traverse_tree(node, depth=1):
        whitespace = ' ' * depth
        if node.is_leaf:
            cat = traverse_cat(node.cat)
            token = node.token
            surf = _escape_prolog(token.get('surf', node.word))
            base = _escape_prolog(token.get('base', '*'))

            tags = [
                token.get(key, '*')
                for key in ('pos', 'pos1', 'pos2', 'pos3')
            ]

            if all(tag == '*' for tag in tags):
                pos = '*'
            else:
                pos = '/'.join(_escape_prolog(tag) for tag in tags)

            infl_form = _escape_prolog(token.get('inflectionForm', '*'))
            infl_type = _escape_prolog(token.get('inflectionType', '*'))

            output.write(
                f"\n{whitespace}t({cat}, '{surf}', '{base}', '{pos}', '{infl_form}', '{infl_type}')"
            )
        else:
            cat = traverse_cat(node.cat)
            rule = _ja_combinators[node.op_symbol]
            output.write(f"\n{whitespace}{rule}({cat}")

            for i, child in enumerate(node.children):
                if i < len(node.children):
                    output.write(',')
                traverse_tree(child, depth=depth + 1)

            output.write(')')

    output = StringIO()
    print(_prolog_header, file=output)

    for sentence_index, trees in enumerate(nbest_trees, 1):
        for tree, _ in trees:
            output.write(f'ccg({sentence_index},')
            traverse_tree(tree)
            output.write(').\n\n')

    result = output.getvalue()
    output.close()
    return result
