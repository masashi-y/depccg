from typing import List
from lxml import etree

from depccg.tokens import Token
from depccg.tree import ScoredTree, Tree


class _ConvertToJiggXML(object):
    def __init__(self, sid: int, use_symbol: bool) -> None:
        self.sid = sid
        self._spid = -1
        self.processed = 0
        self.use_symbol = use_symbol

    @property
    def spid(self) -> int:
        self._spid += 1
        return self._spid

    def process(self, tree: Tree, score: float = None) -> None:
        counter = 0

        def traverse(node: Tree) -> None:
            nonlocal counter
            id = f's{self.sid}_sp{self.spid}'
            xml_node = etree.SubElement(res, 'span')
            xml_node.set('category', str(node.cat.multi_valued))
            xml_node.set('id', id)
            if node.is_leaf:
                start_of_span = counter
                counter += 1
                xml_node.set('terminal', f's{self.sid}_{start_of_span}')
            else:
                childid, start_of_span = traverse(node.left_child)
                if not node.is_unary:
                    tmp, _ = traverse(node.right_child)
                    childid += ' ' + tmp
                xml_node.set('child', childid)
                xml_node.set(
                    'rule', node.op_symbol if self.use_symbol else node.op_string
                )
            xml_node.set('begin', str(start_of_span))
            xml_node.set('end', str(start_of_span + len(node)))
            return id, start_of_span

        res = etree.Element('ccg')
        res.set('id', f's{self.sid}_ccg{self.processed}')
        id, _ = traverse(tree)
        res.set('root', str(id))
        res[0].set('root', 'true')
        if score is not None:
            res.set('score', str(score))
        self.processed += 1
        return res


def to_jigg_xml(
    trees: List[List[ScoredTree]],
    tagged_doc: List[List[Token]],
    use_symbol: bool = False
) -> etree.Element:
    """generate etree.Element XML object in jigg format
    containing all the parse results

    Args:
        trees (List[List[ScoredTree]]): parsing result
        tagged_doc (List[List[Token]]): list of tokens corresponding to input
        use_symbol (bool, optional): [description]. Defaults to False.

    Returns:
        etree.Element: jigg format etree.Element tree
    """

    root_node = etree.Element('root')
    document_node = etree.SubElement(root_node, 'document')
    sentences_node = etree.SubElement(document_node, 'sentences')

    for sentence_index, (parsed, tagged) in enumerate(zip(trees, tagged_doc)):

        sentence_node = etree.SubElement(sentences_node, 'sentence')
        tokens_node = etree.SubElement(sentence_node, 'tokens')
        cats = [leaf.cat for leaf in parsed[0].tree.leaves]
        assert len(cats) == len(tagged)

        for token_index, (token, cat) in enumerate(zip(tagged, cats)):
            token_node = etree.SubElement(tokens_node, 'token')
            token_node.set('start', str(token_index))
            token_node.set('cat', str(cat))
            token_node.set('id', f's{sentence_index}_{token_index}')

            if 'word' in token:
                token['surf'] = token.pop('word')
            if 'lemma' in token:
                token['base'] = token.pop('lemma')
            for k, v in token.items():
                token_node.set(k, v)

        converter = _ConvertToJiggXML(sentence_index, use_symbol)
        for tree, score in parsed:
            sentence_node.append(converter.process(tree, score))

    return root_node
