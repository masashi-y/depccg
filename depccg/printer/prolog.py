from typing import List
from io import StringIO

from depccg.py_cat import Category
from depccg.py_tree import ScoredTree, Tree
from depccg.types import Token


def _prolog_category_string(cat: Category) -> str:

    def rec(this_cat: Category) -> str:

        if not this_cat.is_functor:
            base = this_cat.base.lower()
            if base == '.':
                return "period"
            elif base == ",":
                return "comma"
            elif base == ":":
                return "colon"
            elif base == ";":
                return "semicolon"
            elif len(this_cat.feature) == 0:
                return base
            else:
                return f'{base}:{this_cat.feature[1:-1]}'

        else:
            left = rec(this_cat.left)
            right = rec(this_cat.right)
            return f"({left}{this_cat.slash}{right})"

    return rec(cat)


def _escape_prolog(string: str) -> str:
    return string.replace("'", "")


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


def _prolog_string(tree: Tree, tokens: List[Token], sentence_index: int) -> str:

    position = 0
    depth = 0

    def indent(output):
        output.write(" " * depth)

    def rec(node: Tree, tokens: List[Token], output):
        nonlocal depth, position

        if node.is_leaf:
            indent(output)
            result_str = (
                f"t({_prolog_category_string(node.cat)}, "
                f"\'{_escape_prolog(node.word)}\', "
                f"\'{tokens[position].lemma}\', "
                f"\'{tokens[position].pos}\', "
                f"\'{tokens[position].chunk}\', "
                f"\'{tokens[position].entity}\')"
            )
            output.write(result_str)
            position += 1
        else:
            indent(output)
            if node._is_unary:
                output.write("lx(")
            else:
                op_string = _op_mapping[tree.op_string]
                output.write(f"{op_string}(")

            output.write(_prolog_category_string(tree.cat))
            output.write(", ")

            if tree.op_string == 'conj2':
                cat_str = _prolog_category_string(tree.right_child.cat)
                output.write(f"{cat_str}\\{cat_str}, \n")
                depth += 1
                indent(output)
                output.write(f"conj({cat_str}\\{cat_str}, {cat_str}, ")

            if tree.op_string == 'lp':
                cat_str = _prolog_category_string(tree.right_child.cat)
                output.write(f"{cat_str}, \n")
                depth += 1
                indent(output)
                output.write(f"lp({cat_str}, ")

            if node._is_unary:
                cat_str = _prolog_category_string(tree.left_child.cat)
                output.write(f"{cat_str}, ")

            if tree.op_string == 'conj':
                cat_str = _prolog_category_string(tree.cat.left)
                output.write(f"{cat_str}, ")

            output.write("\n")
            rec(tree.left, tokens, output)
            if not tree.is_unary:
                output.write(",\n")
                rec(tree.right, tokens, output)
            output.write(")")
            depth -= 1

            if tree.op_string in ('conj2', 'lp'):
                output.write(")")
                depth -= 1

    with StringIO as output:
        output.write(f"ccg({sentence_index},\n")
        rec(tree, tokens, output)
        output.write(").\n")
        return output.getvalue()


_prolog_header = (
    ':- op(601, xfx, (/)).\n'
    ':- op(601, xfx, (\\)).\n'
    ':- multifile ccg/2, id/2.\n'
    ':- discontiguous ccg/2, id/2.\n'
)


def _normalize_text(text):
    return text.replace("'", "\\'")


def to_prolog_en(
    nbest_trees: List[List[ScoredTree]],
    tagged_doc: List[List[Token]]
) -> str:
    """convert parsing results to Prolog format used by LangPro.

    Args:
        nbest_trees (List[List[ScoredTree]]): parsing results
        tagged_doc (List[List[Token]]): corresponding lists of tokens

    Returns:
        str: Prolog string
    """

    with StringIO() as output:
        print(_prolog_header, file=output)
        for sentence_index, (trees, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
            for token in tokens:
                token.lemma = _normalize_text(token.lemma)
            for tree, _ in trees:
                print(_prolog_string(tree, tokens, sentence_index), file=output)
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
    tagged_doc: List[List[Token]]
) -> str:
    """convert parsing results to Prolog format used by LangPro.
    This is specifically used for Japanese sentences.

    Args:
        nbest_trees (List[List[ScoredTree]]): parsing results
        tagged_doc (List[List[Token]]): corresponding lists of tokens

    Returns:
        str: Prolog string
    """

    def traverse_cat(node):
        if node.is_functor:
            left = traverse_cat(node.left)
            right = traverse_cat(node.right)
            return f'({left}{node.slash}{right})'
        else:
            feature = node.features
            base = node.base.lower()
            if 'case' not in feature:
                return base
            else:
                feature_case = feature["case"].lower()
                return f'{base}:{feature_case}'

    def traverse_tree(node, tokens, depth=1):
        whitespace = ' ' * depth
        if node.is_leaf:
            cat = traverse_cat(node.cat)
            token = tokens.pop(0)
            surf = _normalize_text(token.get('surf', node.word))
            base = _normalize_text(token.get('base', 'XX'))

            tags = [
                token.get(key, 'XX')
                for key in ('pos', 'pos1', 'pos2', 'pos3')
            ]

            if all(tag == 'XX' for tag in tags):
                pos = 'XX'
            else:
                pos = '/'.join(_normalize_text(tag) for tag in tags)

            infl_form = _normalize_text(token.get('inflectionForm', 'XX'))
            infl_type = _normalize_text(token.get('inflectionType', 'XX'))

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
                traverse_tree(child, tokens, depth=depth + 1)

            output.write(')')

    output = StringIO()
    print(_prolog_header, file=output)

    for sentence_index, (trees, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
        for tree, _ in trees:
            output.write(f'ccg({sentence_index},')
            traverse_tree(tree, tokens)
            output.write(').\n\n')

    result = output.getvalue()
    output.close()
    return result
