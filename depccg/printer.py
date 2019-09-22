from typing import List, Tuple, Optional

import re
import sys
import logging
import json
import html
from lxml import etree
from io import StringIO

from .tree import Tree
from .download import SEMANTIC_TEMPLATES
from .semantics.ccg2lambda import parse as ccg2lambda


logger = logging.getLogger(__name__)


def to_xml(nbest_trees, tagged_doc):
    candc_node = etree.Element('candc')
    for i, (nbest, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
        for j, (tree, _) in enumerate(nbest, 1):
            out = tree.xml(tokens)
            out.set('sentence', str(i))
            out.set('id', str(j))
            candc_node.append(out)
    return candc_node


prolog_header = (
    ':- op(601, xfx, (/)).\n'
    ':- op(601, xfx, (\\)).\n'
    ':- multifile ccg/2, id/2.\n'
    ':- discontiguous ccg/2, id/2.\n'
)


def to_prolog_en(nbest_trees, tagged_doc):
    with StringIO() as output:
        print(prolog_header, file=output)
        for i, (nbest, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
            for token in tokens:
                if "'" in token.lemma:
                    token.lemma = token.lemma.replace("'", "\\'")
            for t, _ in nbest:
                print(t.prolog().format(i, *tokens), file=output)
        result = output.getvalue()
    return result


def to_prolog_ja(nbest_trees, tagged_doc):
    ja_combinators = {
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

    def normalize_text(text):
        return text.replace("'", "\\'")

    def traverse_tree(node, tokens, depth=1):
        whitespace = ' ' * depth
        if node.is_leaf:
            cat = traverse_cat(node.cat)
            token = tokens.pop(0)
            surf = normalize_text(token.get('surf', node.word))
            base = normalize_text(token.get('base', 'XX'))
            tags = [token.get(key, 'XX') for key in ('pos', 'pos1', 'pos2', 'pos3')]
            if all(tag == 'XX' for tag in tags):
                pos = 'XX'
            else:
                pos = '/'.join(normalize_text(tag) for tag in tags)
            infl_form = normalize_text(token.get('inflectionForm', 'XX'))
            infl_type = normalize_text(token.get('inflectionType', 'XX'))
            output.write(
                f"\n{whitespace}t({cat}, '{surf}', '{base}', '{pos}', '{infl_form}', '{infl_type}')")
        else:
            cat = traverse_cat(node.cat)
            rule = ja_combinators[node.op_string]
            output.write(f"\n{whitespace}{rule}({cat}")
            for i, child in enumerate(node.children):
                if i < len(node.children):
                    output.write(',')
                traverse_tree(child, tokens, depth=depth+1)
            output.write(')')

    output = StringIO()
    print(prolog_header, file=output)
    for i, (nbest, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
        for tree, _ in nbest:
            output.write(f'ccg({i},')
            traverse_tree(tree, tokens)
            output.write(').\n\n')
    result = output.getvalue()
    output.close()
    return result


MATHML_SUBTREE_NONTERMINAL = '''\
<mrow>
  <mfrac {3} linethickness='2px'>
    <mrow>{0}</mrow>
    <mstyle mathcolor='Red'>{1}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>{2}</mtext>
</mrow>
'''

MATHML_SUBTREE_TERMINAL = '''\
<mrow>
  <mfrac linethickness='2px'>
    <mtext mathsize='1.0' mathcolor='Black'>{0}</mtext>
    <mstyle mathcolor='Red'>{1}</mstyle>
  </mfrac>
  <mtext mathsize='0.8' mathcolor='Black'>lex</mtext>
</mrow>
'''

MATHML_MAIN = '''\
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


def mathml_subtree(tree, bgcolor=None):
    bgcolor = f'mathbackground={bgcolor}' if bgcolor else ''
    cat_str = mathml_cat(str(tree.cat))
    if not tree.is_leaf:
        children_str = ''.join(map(mathml_subtree, tree.children))
        return MATHML_SUBTREE_NONTERMINAL.format(
            children_str, cat_str, html.escape(tree.op_string), bgcolor)
    else:
        return MATHML_SUBTREE_TERMINAL.format(html.escape(tree.word), cat_str)


def mathml_cat(cat):
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


def to_mathml(trees):
    result = ''
    for i, nbest in enumerate(trees):
        words = nbest[0][0].word if isinstance(nbest[0], tuple) else nbest[0].word
        result += f'<p>ID={i}: {words}</p>'
        for tree in nbest:
            if isinstance(tree, tuple):
                tree, prob = tree
                result += f'<p>Log prob={prob:.5e}</p>'
            tree_str = tree if isinstance(tree, str) else mathml_subtree(tree)
            result += f'<math xmlns="http://www.w3.org/1998/Math/MathML">{tree_str}</math>'
    return MATHML_MAIN.format(result)


class ConvertToJiggXML(object):
    def __init__(self, sid: int):
        self.sid = sid
        self._spid = -1
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
        res[0].set('root', 'true')
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
            if 'word' in token:
                token['surf'] = token.pop('word')
            if 'lemma' in token:
                token['base'] = token.pop('lemma')
            for k, v in token.items():
                token_node.set(k, v)
        converter = ConvertToJiggXML(i)
        for tree, score in parsed:
            sentence_node.append(converter.process(tree, score))
    return root_node


def print_(nbest_trees: List[List[Tuple[float, Tree]]],
           tagged_doc: Optional[List[List['Token']]],
           lang: str = 'en',
           format: str = 'auto',
           semantic_templates: Optional[str] = None,
           file = sys.stdout) -> str:
    def process_xml(xml_node):
        return etree.tostring(xml_node, encoding='utf-8', pretty_print=True).decode('utf-8')

    if format == 'xml':
        print(process_xml(to_xml(nbest_trees, tagged_doc)), file=file)
    elif format == 'jigg_xml':
        print(process_xml(to_jigg_xml(nbest_trees, tagged_doc)), file=file)
    elif format == 'prolog':
        if lang == 'en':
            print(to_prolog_en(nbest_trees, tagged_doc), end='', file=file)
        elif lang == 'ja':
            print(to_prolog_ja(nbest_trees, tagged_doc), end='', file=file)
    elif format == 'html':
        print(to_mathml(nbest_trees), file=file)
    elif format == 'jigg_xml_ccg2lambda':
        jigg_xml = to_jigg_xml(nbest_trees, tagged_doc)
        templates = semantic_templates or SEMANTIC_TEMPLATES.get(lang)
        assert templates, f'--semantic-templates must be spcified for language: {lang}'
        result_xml_str, _ = ccg2lambda.parse(jigg_xml, str(templates))
        print(result_xml_str.decode('utf-8'), file=file)
    elif format == 'ccg2lambda':
        jigg_xml = to_jigg_xml(nbest_trees, tagged_doc)
        templates = semantic_templates or SEMANTIC_TEMPLATES.get(lang)
        assert templates, f'--semantic-templates must be spcified for language: {lang}'
        _, formulas_list = ccg2lambda.parse(jigg_xml, str(templates))
        for i, (parsed, formulas) in enumerate(zip(nbest_trees, formulas_list)):
            for (tree, prob), formula in zip(parsed, formulas):
                print(f'ID={i} log probability={prob:.4e}\n{formula}', file=file)
    elif format == 'conll':
        for i, parsed in enumerate(nbest_trees):
            for tree, prob in parsed:
                print(f'# ID={i}\n# log probability={prob:.4e}\n{tree.conll()}', file=file)
    elif format == 'json':
        for i, (parsed, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
            for tree, prob in parsed:
                res = tree.json(tokens=tokens)
                res['id'] = i
                res['prob'] = prob
                print(json.dumps(res), file=file)
    elif format == 'auto':
        for i, (parsed, tokens) in enumerate(zip(nbest_trees, tagged_doc), 1):
            for tree, prob in parsed:
                print(f'ID={i}, log probability={prob}\n{tree.auto(tokens=tokens)}', file=file)
    else:  # deriv, ja, ptb
        for i, parsed in enumerate(nbest_trees, 1):
            for tree, prob in parsed:
                print(f'ID={i}, log probability={prob}\n{getattr(tree, format)()}', file=file)


def to_string(nbest_trees: List[List[Tuple[float, Tree]]],
              tagged_doc: Optional[List[List['Token']]],
              lang: str = 'en',
              format: str = 'auto',
              semantic_templates: Optional[str] = None) -> str:
    with StringIO() as output:
        print_(nbest_trees,
            tagged_doc,
            lang=lang,
            format=format,
            semantic_templates=semantic_templates,
            file=output)
        result = output.getvalue()
    return result