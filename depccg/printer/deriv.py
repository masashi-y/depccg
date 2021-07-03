
from io import StringIO
from depccg.tree import Tree


def deriv_of(tree: Tree) -> str:
    """ascii art-like derivation tree string.

    Args:
        tree (Tree): tree object

    Returns:
        str: derivation tree string
    """

    catstr = ''
    wordstr = ''
    for leaf in tree.leaves:
        str_cat = str(leaf.cat)
        str_word = leaf.word
        nextlen = 2 + max(len(str_word), len(str_cat))
        lcatlen = (nextlen - len(str_cat)) // 2
        rcatlen = lcatlen + (nextlen - len(str_cat)) % 2
        catstr += ' ' * lcatlen + str_cat + ' ' * rcatlen
        lwordlen = (nextlen - len(str_word)) // 2
        rwordlen = lwordlen + (nextlen - len(str_word)) % 2
        wordstr += ' ' * lwordlen + str_word + ' ' * rwordlen

    def rec(lwidth, node):
        rwidth = lwidth

        if node.is_leaf:
            return max(
                rwidth,
                2 + lwidth + len(str(node.cat)),
                2 + lwidth + len(node.word)
            )

        else:
            for child in node.children:
                rwidth = max(rwidth, rec(rwidth, child))

            print(
                lwidth * ' ' + (rwidth - lwidth) * '-' + str(node.op_symbol),
                file=output
            )

            result = str(node.cat)
            pad_len = (rwidth - lwidth - len(result)) // 2 + lwidth
            print(pad_len * ' ' + result, file=output)

            return rwidth

    with StringIO() as output:
        print(catstr.rstrip(), file=output)
        print(wordstr.rstrip(), file=output)
        rec(0, tree)
        return output.getvalue()
