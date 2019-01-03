import re
import logging
from lxml import etree
from .tree import Tree


logger = logging.getLogger(__name__)


def to_xml(nbest_trees, tagged_doc):
    candc_node = etree.Element('candc')
    for i, (nbest, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
        for j, (tree, _) in enumerate(nbest, 1):
            out = tree.xml(tokens)
            out.set('sentence', str(i))
            out.set('id', str(j))
            candc_node.append(out)
    return etree.tostring(candc_node, encoding='utf-8', pretty_print=True).decode('utf-8')


def to_prolog(nbest_trees, tagged_doc):
    out = (':- op(601, xfx, (/)).\n'
           ':- op(601, xfx, (\)).\n'
           ':- multifile ccg/2, id/2.\n'
           ':- discontiguous ccg/2, id/2.\n\n')
    for i, (nbest, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
        for token in tokens:
            if "'" in token.lemma:
                token.lemma = token.lemma.replace("'", "\\'")
        for t, _ in nbest:
            out += t.prolog().format(i, *tokens) + '\n'
    return out


def mathml_subtree(tree):
    cat_str = mathml_cat(str(tree.cat))
    if not tree.is_leaf:
        children_str = ''.join(map(mathml_subtree, tree.children))
        return f'''\
<mrow>
  <mfrac linethickness='2px'>
    <mrow>{children_str}</mrow>
    <mstyle mathcolor='Red'>{cat_str}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>{tree.op_string}</mtext>
</mrow>'''
    else:
        return f'''\
<mrow>
  <mfrac linethickness='2px'>
    <mtext mathsize='1.0' mathcolor='Black'>{tree.word}</mtext>
    <mstyle mathcolor='Red'>{cat_str}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>lex</mtext>
</mrow>'''


def mathml_cat(cat):
    cats_feats = re.findall(r'([\w\\/()]+)(\[.+?\])*', cat)
    mathml_str = ''
    for cat, feat in cats_feats:
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


def mathml_nbest_trees(tree):
    res = ''
    for t in tree:
        if isinstance(t, tuple):
            t, prob = t
            res += f'<p>Log prob={prob:.5e}</p>'
        tree_str = mathml_subtree(t)
        res += f'<math xmlns="http://www.w3.org/1998/Math/MathML">{tree_str}</math>'
    return res


def to_mathml(trees):
    string = ''
    for i, nbest in enumerate(trees):
        words = nbest[0][0].word if isinstance(nbest[0], tuple) else nbest[0].word
        trees_str = mathml_nbest_trees(nbest)
        string += f'<p>ID={i}: {words}</p>{trees_str}'

    results = f'''\
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
</body></html>'''
    return results


def diff(tree1, tree2):
    def rec(node1, node2):
        pass
    str1, str2 = '', ''
    rec(tree1, tree2)
    return None


class ConvertToJiggXML(object):
    def __init__(self, sid: int):
        self.sid = sid
        self._spid = 0
        self.processed = 0

    @property
    def spid(self) -> int:
        self._spid += 1
        return self._spid

    def process(self, tree: Tree, score: float = None):
        def traverse(node: Tree):
            id = f's{self.sid}_sp{self.spid}'
            xml_node = etree.SubElement(res, 'span')
            xml_node.set('category', str(node.cat.multi_valued))
            xml_node.set('begin', str(node.start_of_span))
            xml_node.set('end', str(node.start_of_span+len(node)))
            xml_node.set('id', id)
            if node.is_leaf:
                xml_node.set('terminal', f't{self.sid}_{node.head_id}')
            else:
                childid = traverse(node.left_child)
                if not node.is_unary:
                    tmp = traverse(node.right_child)
                    childid += ' ' + tmp
                xml_node.set('child', childid)
                xml_node.set('rule', node.op_string)
            return id

        res = etree.Element('ccg')
        res.set('id', f's{self.sid}_ccg{self.processed}')
        id = traverse(tree)
        res.set('root', str(id))
        if score is not None:
            res.set('score', str(score))
        self.processed += 1
        return res


def to_jigg_xml(trees, tagged_doc):
    root_node = etree.Element('root')
    document_node = etree.SubElement(root_node, 'document')
    sentences_node = etree.SubElement(document_node, 'sentences')
    for i, (parsed, tagged) in enumerate(zip(trees, tagged_doc)):
        sentence_node = etree.SubElement(sentences_node, 'sentence')
        tokens_node = etree.SubElement(sentence_node, 'tokens')
        cats = [leaf.cat for leaf in parsed[0][0].leaves]
        assert len(cats) == len(tagged)
        for j, (token, cat) in enumerate(zip(tagged, cats)):
            token_node = etree.SubElement(tokens_node, 'token')
            token_node.set('start', str(j))
            token_node.set('cat', str(cat))
            token_node.set('id', f't{i}_{j}')
            for k, v in token.items():
                token_node.set(k, v)
        converter = ConvertToJiggXML(i)
        for tree, score in parsed:
            sentence_node.append(converter.process(tree, score))
    return etree.tostring(root_node, encoding='utf-8', pretty_print=True).decode('utf-8')

