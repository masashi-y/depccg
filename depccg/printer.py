import re

class Token:
    def __init__(self, word, lemma, pos, chunk, entity):
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.chunk = chunk
        self.entity = entity

    @staticmethod
    def from_piped(string):
        # WORD|POS|NER or WORD|LEMMA|POS|NER
        # or WORD|LEMMA|POS|NER|CHUCK
        items = string.split('|')
        if len(items) == 5:
            w, l, p, n, c = items
            return Token(w, l, p, c, n)
        elif len(items) == 4:
            w, l, p, n = items
            return Token(w, l, p, 'XX', n)
        else:
            w, p, n = items
            return Token(w, 'XX', p, 'XX', n)


def to_xml(trees, tagged_doc):
    out = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<?xml-stylesheet type="text/xsl" href="candc.xml"?>\n'
           '<candc>\n')
    for i, (tree, tagged) in enumerate(zip(trees, tagged_doc), 1):
        for j, (t, _) in enumerate(tree, 1):
            out += f'<ccg sentence="{i}" id="{j}">'
            out += t.xml.format(*tagged) + '\n'
            out += '</ccg>\n'
    out += '</candc>\n'
    return out


def to_prolog(trees, tagged_doc):
    out = (':- op(601, xfx, (/)).\n'
           ':- op(601, xfx, (\)).\n'
           ':- multifile ccg/2, id/2.\n'
           ':- discontiguous ccg/2, id/2.\n\n')
    for i, (tree, tagged) in enumerate(zip(trees, tagged_doc), 1):
        for tok in tagged:
            if "'" in tok.lemma:
                tok.lemma = tok.lemma.replace("'", "\\'")
        for t, _ in tree:
            out += t.prolog.format(i, *tagged) + '\n'
    return out


def show_mathml_tree(tree):
    cat_str = show_mathml_cat(str(tree.cat))
    if not tree.is_leaf:
        children_str = ''.join(map(show_mathml_tree, tree.children))
        return f"""\
<mrow>
  <mfrac linethickness='2px'>
    <mrow>{children_str}</mrow>
    <mstyle mathcolor='Red'>{cat_str}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>{tree.op_string}</mtext>
</mrow>"""
    else:
        return f"""\
<mrow>
  <mfrac linethickness='2px'>
    <mtext mathsize='1.0' mathcolor='Black'>{tree.word}</mtext>
    <mstyle mathcolor='Red'>{cat_str}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>lex</mtext>
</mrow>"""


def show_mathml_cat(cat):
    cats_feats = re.findall(r'([\w\\/()]+)(\[.+?\])*', cat)
    mathml_str = ''
    for cat, feat in cats_feats:
        cat_mathml = f"""\
<mi mathvariant='italic'
  mathsize='1.0' mathcolor='Red'>{cat}</mi>"""

        if feat != '':
            mathml_str += f"""\
<msub>{cat_mathml}
  <mrow>
  <mi mathvariant='italic'
    mathsize='0.8' mathcolor='Purple'>{feat}</mi>
  </mrow>
</msub>"""
        else:
            mathml_str += cat_mathml
    return mathml_str


def to_mathml(trees):
    def __show(tree):
        res = ''
        for t in tree:
            if isinstance(t, tuple):
                t, prob = t
                res += f'<p>Log prob={prob:.5e}</p>'
            tree_str = show_mathml_tree(t)
            res += f'<math xmlns="http://www.w3.org/1998/Math/MathML">{tree_str}</math>'
        return res

    string = ''
    for i, tree in enumerate(trees):
        words = tree[0][0].word if isinstance(tree[0], tuple) else tree[0].word
        trees_str = __show(tree)
        string += f'<p>ID={i}: {words}</p>{trees_str}'

    results = f"""\
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
<body>{string}
</body></html>"""
    return results


