from typing import List, Union, Optional
import html
import re
from depccg.tree import Tree, ScoredTree


_MATHML_SUBTREE_NONTERMINAL = '''\
<mrow>
  <mfrac {3} linethickness='2px'>
    <mrow>{0}</mrow>
    <mstyle mathcolor='Red'>{1}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>{2}</mtext>
</mrow>
'''


_MATHML_SUBTREE_TERMINAL = '''\
<mrow>
  <mfrac linethickness='2px'>
    <mtext mathsize='1.0' mathcolor='Black'>{0}</mtext>
    <mstyle mathcolor='Red'>{1}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>lex</mtext>
</mrow>
'''


_MATHML_MAIN = '''\
<!doctype html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <style>
    body {{
      font-size: 1em;
    }}
  </style>
  <script type="text/javascript"
     src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
</head>
<body>
  {0}
</body>
</html>
'''


def _mathml_subtree(tree: Tree, bgcolor: Optional[str] = None) -> str:
    bgcolor = f'mathbackground={bgcolor}' if bgcolor else ''
    cat_str = _mathml_cat(str(tree.cat))
    if not tree.is_leaf:
        children_str = ''.join(map(_mathml_subtree, tree.children))
        return _MATHML_SUBTREE_NONTERMINAL.format(
            children_str, cat_str, html.escape(tree.op_string), bgcolor
        )
    else:
        return _MATHML_SUBTREE_TERMINAL.format(html.escape(tree.word), cat_str)


def _mathml_cat(cat: str) -> str:
    cats_feats = re.findall(r'([\w\\/()]+)(\[.+?\])*', cat)
    mathml_str = ''
    for cat, feat in cats_feats:
        cat = html.escape(cat)
        feat = html.escape(feat)
        cat_mathml = f'''\
<mi mathvariant='italic'
  mathsize='1.0' mathcolor='Red'>{cat}</mi>'''
        if feat != '':
            mathml_str += f'''\
<msub>{cat_mathml}
  <mrow>
  <mi mathvariant='italic'
    mathsize='0.8' mathcolor='Purple'>{feat}</mi>
  </mrow>
</msub>'''
        else:
            mathml_str += cat_mathml
    return mathml_str


def to_mathml(nbest_trees: List[List[Union[Tree, ScoredTree]]]) -> str:
    """convert parsing results to a html string containing all the trees.

    Args:
        nbest_trees (List[List[Union[Tree, ScoredTree]]]): parsing results

    Returns:
        str: html string
    """

    result = ''
    for sentence_index, trees in enumerate(nbest_trees, 1):
        if isinstance(trees[0], ScoredTree):
            words = trees[0].tree.word
        else:
            words = trees[0].word

        result += f'<p>ID={sentence_index}: {words}</p>'

        for tree in trees:
            if isinstance(tree, ScoredTree):
                tree, prob = tree
                result += f'<p>Log prob={prob:.5e}</p>'

            tree_str = tree if isinstance(tree, str) else _mathml_subtree(tree)
            result += f'<math xmlns="http://www.w3.org/1998/Math/MathML">{tree_str}</math>'

    return _MATHML_MAIN.format(result)
