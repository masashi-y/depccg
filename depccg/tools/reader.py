
from typing import Tuple, List, Iterator, NamedTuple
from depccg.tree import Tree
from depccg.cat import Category
from depccg.combinator import guess_combinator_by_triplet, UNKNOWN_COMBINATOR
from depccg.lang import BINARY_RULES, GLOBAL_LANG_NAME
from depccg.types import Token
from lxml import etree
import logging

logger = logging.getLogger(__name__)


class ReaderResult(NamedTuple):
    name: str
    tokens: List[Token]
    tree: Tree


def read_auto(filename: str) -> Iterator[ReaderResult]:
    """read traditional AUTO file used for CCGBank
    English CCGbank contains some unwanted categories such as (S\\NP)\\(S\\NP)[conj].
    This reads the treebank while taking care of those categories.

    Args:
        filename (str): file name string

    Yields:
        Iterator[ReaderResult]: iterator object containing parse results
    """

    for line in open(filename):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith("ID"):
            name = line
        else:
            tokens = []
            for token in line.split(' '):
                if token.endswith(')[conj]'):
                    token = token[:-6]
                tokens.append(token)
            line = ' '.join(tokens)
            tree, tokens = Tree.of_auto(line)
            yield ReaderResult(name, tokens, tree)


def read_xml(filename: str) -> Iterator[ReaderResult]:
    """read XML format file commonly used by C&C.

    Args:
        filename (str): file name string

    Yields:
        Iterator[ReaderResult]: iterator object containing parse results
    """

    binary_rules = BINARY_RULES[GLOBAL_LANG_NAME]

    def parse(tree):
        def rec(node):
            attrib = node.attrib
            if node.tag == 'rule':
                cat = Category.parse(attrib['cat'])
                children = [rec(child) for child in node.getchildren()]
                if len(children) == 1:
                    return Tree.make_unary(cat, children[0])
                else:
                    assert len(children) == 2
                    left, right = children
                    combinator = guess_combinator_by_triplet(
                        binary_rules, cat, left.cat, right.cat
                    )
                    combinator = combinator or UNKNOWN_COMBINATOR
                    return Tree.make_binary(cat, left, right, combinator)
            else:
                assert node.tag == 'lf'
                cat = Category.parse(attrib['cat'])
                word = attrib['word']
                token = Token(
                    word=attrib['word'],
                    pos=attrib['pos'],
                    entity=attrib['entity'],
                    lemma=attrib['lemma'],
                    chunk=attrib['chunk']
                )
                tokens.append(token)
                return Tree.make_terminal(word, cat)
        tokens = []
        tree = rec(tree)
        return tokens, tree

    trees = etree.parse(filename).getroot().xpath('ccg')
    for tree in trees:
        name = '_'.join(f'{k}={v}' for k, v in tree.items())
        yield ReaderResult(name, *parse(tree[0]))


def read_jigg_xml(filename: str) -> Iterator[ReaderResult]:
    """read XML format file used by Jigg.

    Args:
        filename (str): file name string

    Yields:
        Iterator[ReaderResult]: iterator object containing parse results
    """

    binary_rules = BINARY_RULES[GLOBAL_LANG_NAME]
    # TODO

    def try_get_surface(token):
        if 'word' in token:
            return token.word
        elif 'surf' in token:
            return token.surf
        else:
            raise RuntimeError(
                'the attribute for the token\'s surface form is unknown'
            )

    def parse(tree, tokens):
        def rec(node):
            attrib = node.attrib
            if 'terminal' not in attrib:
                cat = Category.parse(attrib['category'])
                children = [
                    rec(spans[child])
                    for child in attrib['child'].split(' ')
                ]
                if len(children) == 1:
                    return Tree.make_unary(cat, children[0])
                else:
                    assert len(children) == 2
                    left, right = children
                    combinator = guess_combinator_by_triplet(
                        binary_rules, cat, left.cat, right.cat
                    )
                    combinator = combinator or UNKNOWN_COMBINATOR
                    return Tree.make_binary(cat, left, right, combinator)
            else:
                cat = Category.parse(attrib['category'])
                word = try_get_surface(tokens[attrib['terminal']])
                return Tree.make_terminal(word, cat)

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
            yield ReaderResult(ccg.attrib['id'], tokens, tree)


def _parse_ptb(tree_string: str) -> Tuple[Tree, List[Token]]:
    """parse a S-expression like PTB-format tree

    Args:
        tree_string (str): S-expression

    Raises:
        RuntimeError: when parsing fails.

    Returns:
        Tuple[Tree, List[Token]]: Tree object and tokens
    """
    binary_rules = BINARY_RULES[GLOBAL_LANG_NAME]
    assert tree_string.startswith('(ROOT ')
    buf = list(reversed(tree_string[6:-1].split(' ')))
    stack = []
    tokens = []
    position = 0

    def reduce(item: str) -> None:
        nonlocal position
        if item[-1] != ')':
            token = Token(word=item)
            tokens.append(token)
            stack.append(item)
            return

        reduce(item[:-1])
        if isinstance(stack[-1], str):
            word = stack.pop()
            category = stack.pop()
            tree = Tree.make_terminal(word, category)
            position += 1
        else:
            assert isinstance(stack[-1], Tree)
            children = []
            while isinstance(stack[-1], Tree):
                tree = stack.pop()
                children.append(tree)
            category = stack.pop()
            if len(children) == 1:
                tree = Tree.make_unary(category, children[0])
            elif len(children) == 2:
                right, left = children
                combinator = guess_combinator_by_triplet(
                    binary_rules, category, left.cat, right.cat
                )
                combinator = combinator or UNKNOWN_COMBINATOR
                tree = Tree.make_binary(
                    category, left, right, combinator
                )
            else:
                assert False
        stack.append(tree)

    def rec() -> None:
        if len(buf) == 0:
            return
        item = buf.pop()
        assert item[0] == '(' or item[-1] == ')'
        if item[0] == '(':
            stack.append(Category.parse(item[1:]))
        elif item[-1] == ')':
            reduce(item)
        rec()

    try:
        rec()
        assert len(stack) == 1 and isinstance(stack[0], Tree)
    except AssertionError:
        raise RuntimeError('Parse failed on an invalid CCG tree')
    return stack[0], tokens


def read_ptb(filename: str) -> Iterator[ReaderResult]:
    """parse PTB-formatted file

    Args:
        filename (str): file name string

    Yields:
        Iterator[ReaderResult]: iterator object containing parse results
    """
    name0 = None
    for i, line in enumerate(open(filename)):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith("ID"):
            name0 = line
        else:
            tree, tokens = _parse_ptb(line)
            name = name0 or f'ID={i}'
            yield ReaderResult(name, tokens, tree)


def read_trees_guess_extension(filename: str) -> Iterator[ReaderResult]:
    """guess the file format based on the extension and parse it

    Args:
        filename (str): file name string

    Yields:
        Iterator[ReaderResult]: iterator object containing parse results
    """

    logger.info(f'reading trees from: {filename}')

    if filename.endswith('.jigg.xml'):
        logger.info('read it as jigg XML file')
        yield from read_jigg_xml(filename)

    elif filename.endswith('.xml'):
        logger.info('read it as C&C XML file')
        yield from read_xml(filename)

    elif filename.endswith('.ptb'):
        logger.info('read it as PTB format file')
        yield from read_ptb(filename)

    else:
        logger.info('read it as AUTO file')
        yield from read_auto(filename)
