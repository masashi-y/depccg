
from ..tree import Tree
from ..cat import Category
from ..combinator import unknown_combinator
from ..token import Token
from lxml import etree
import logging

logger = logging.getLogger(__name__)


def read_auto(filename, lang='en'):
    for line in open(filename):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith("ID"):
            name = line
        else:
            tree, tokens = Tree.of_auto(line, lang)
            yield name, tokens, tree


def read_xml(filename, lang='en'):
    unknown_op = unknown_combinator()  # TODO

    def parse(tree):
        def rec(node):
            attrib = node.attrib
            if node.tag == 'rule':
                cat = Category.parse(attrib['cat'])
                children = [rec(child) for child in node.getchildren()]
                if len(children) == 1:
                    return Tree.make_unary(cat, children[0], lang)
                else:
                    assert len(children) == 2
                    left, right = children
                    return Tree.make_binary(
                        cat, True, left, right, unknown_op, lang)
            else:
                assert node.tag == 'lf'
                cat = Category.parse(attrib['cat'])
                word = attrib['word']
                position = len(tokens)
                token = Token(word=attrib['word'],
                              pos=attrib['pos'],
                              entity=attrib['entity'],
                              lemma=attrib['lemma'],
                              chunk=attrib['chunk'])
                tokens.append(token)
                return Tree.make_terminal(word, cat, position, lang)
        tokens = []
        tree = rec(tree)
        return tree, tokens

    trees = etree.parse(filename).getroot().xpath('ccg')
    for tree in trees:
        name = '_'.join(f'{k}={v}' for k, v in tree.items())
        yield (name,) + parse(tree[0])


def read_jigg_xml(filename, lang='en'):
    unknown_op = unknown_combinator()  # TODO

    def parse(tree, tokens):
        def rec(node):
            attrib = node.attrib
            if 'terminal' not in attrib:
                cat = Category.parse(attrib['category'])
                children = [rec(spans[child]) for child in attrib['child'].split(' ')]
                if len(children) == 1:
                    return Tree.make_unary(cat, children[0], lang)
                else:
                    assert len(children) == 2
                    left, right = children
                    return Tree.make_binary(
                        cat, True, left, right, unknown_op, lang)
            else:
                cat = Category.parse(attrib['category'])
                word = tokens[attrib['terminal']].word
                position = int(attrib['begin'])
                return Tree.make_terminal(word, cat, position, lang)

        spans = {span.attrib['id']: span for span in tree.xpath('./span')}
        return rec(spans[tree.attrib['root']])

    trees = etree.parse(filename).getroot()
    sentences = trees[0][0].xpath('sentence')
    for sentence in sentences:
        token_and_ids = []
        for token in sentence.xpath('.//token'):
            token_attribs = dict(token.attrib)
            token_id = token_attribs['id']
            for no_need in ['id', 'start', 'cat']:
                if no_need in token_attribs:
                    del token_attribs[no_need]
            token_and_ids.append((token_id, Token(**token_attribs)))
        tokens = [token for _, token in token_and_ids]
        for ccg in sentence.xpath('./ccg'):
            tree = parse(ccg, dict(token_and_ids))
            yield ccg.attrib['id'], tokens, tree


def read_trees_guess_extension(filename, lang='en'):
    logger.info(f'reading trees from: {filename}')
    if filename.endswith('.jigg.xml'):
        logger.info(f'read it as jigg XML file')
        yield from read_jigg_xml(filename, lang=lang)
    elif filename.endswith('.xml'):
        logger.info(f'read it as C&C XML file')
        yield from read_xml(filename, lang=lang)
    else:
        logger.info(f'read it as AUTO file')
        yield from read_auto(filename, lang=lang)
