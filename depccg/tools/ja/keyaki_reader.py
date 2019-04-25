import json
import sys
import logging
import re

from collections import defaultdict, OrderedDict
from typing import NamedTuple, Union, List
from pathlib import Path


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


test = """
(TOP (Sm."L" (Sm."L" (CPt."L" (IP-MS."L" (IP-MS."L" (IP-MS."L" (IP-MS *pro*) (<IP-MS\IP-MS> 「)) (<IP-MS\IP-MS>."R" (<<IP-MS\IP-MS>/<IP-MS\IP-MS>>."L" (<<IP-MS\IP-MS>/<IP-MS\IP-MS>>."L" (<<IP-MS\IP-MS>/<IP-MS\IP-MS>>."L" (PPs *pro*) (<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>."R" (<<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>/<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>>."L" (<<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>/<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>> さあ) (<<<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>/<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>>\<<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>/<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>>> 、)) (<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>."FCLeft1" (<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>>."FCLeft1" (<PPs\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>> 流れ) (<<<IP-MS\IP-MS>/<IP-MS\IP-MS>>\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>> て)) (<<<IP-MS\IP-MS>/<IP-MS\IP-MS>>\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>> 来る)))) (<<<IP-MS\IP-MS>/<IP-MS\IP-MS>>\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>> ぞ)) (<<<IP-MS\IP-MS>/<IP-MS\IP-MS>>\<<IP-MS\IP-MS>/<IP-MS\IP-MS>>> 。)) (<IP-MS\IP-MS>."FCLeft1" (<IP-MS\IP-MS>."L" (PPs みんな) (<PPs\<IP-MS\IP-MS>> とれ)) (<IP-MS\IP-MS> 。)))) (<IP-MS\IP-MS> 」)) (<IP-MS\CPt> と)) (<CPt\Sm>."L" (PPs."L" (NP 一郎) (<NP\PPs> が)) (<PPs\<CPt\Sm>>."FCLeft2" (<PPs\<CPt\Sm>>."FCLeft2" (<PPs\<CPt\Sm>> 言い) (<Sm\Sm> まし)) (<Sm\Sm> た)))) (<Sm\Sm> 。)) (ID 771_aozora_Miyazawa-1934;JP))
"""


Category = Union['Atomic', 'Functor']
Node = Union['Leaf', 'Tree']


class Atomic(NamedTuple):
    string: str

    def __str__(self):
        return self.string


class Functor(NamedTuple):
    left: str
    slash: str
    right: str

    def __str__(self):
        """
        convert to CCG version
        """
        def rec(cat):
            if isinstance(cat, Atomic):
                return str(cat)
            right = rec(cat.right if cat.slash == '\\' else cat.left)
            left = rec(cat.left if cat.slash == '\\' else cat.right)
            return f'({right}{self.slash}{left})'
        return rec(self)[1:-1]


class Leaf(NamedTuple):
    cat: str
    word: str
    pos: int

    def __str__(self):
        return f'({self.cat}, {self.word})'


class Tree(NamedTuple):
    cat: Category
    children: List[Node]

    def __str__(self):
        children = ', '.join(str(child) for child in self.children)
        return f'({self.cat}, ({children}))'


def drop_brackets(cat):
    if cat.startswith('<') and \
        cat.endswith('>') and \
        find_closing_bracket(cat) == len(cat) - 1:
        return cat[1:-1]
    else:
        return cat


def find_closing_bracket(source):
    open_brackets = 0
    for i, char in enumerate(source):
        open_brackets += char == '<'
        open_brackets -= char == '>'
        if open_brackets == 0:
            return i
    raise Exception("Mismatched brackets in string: " + source)


def find_non_nested_char(haystack, needles):
    open_brackets = 0
    for i in range(len(haystack) -1, -1, -1):
        char = haystack[i]
        open_brackets += char == '<'
        open_brackets -= char == '>'
        if open_brackets == 0 and char in needles:
            return i
    return -1


FEATURE_PATTERN = re.compile(r'([a-z][a-z1-9]*)')
def parse_cat(cat):
    """
    ex. PPo1\<PPs\Sm>.h --> PPo1\(PPs\Sm)
    """
    rule = cat.rfind('.')
    if rule >= 0:
        cat = cat[:rule]
    cat = drop_brackets(cat)
    op_idx = find_non_nested_char(cat, '/\\')
    if op_idx == -1:
        return Atomic(FEATURE_PATTERN.sub('[\\1]', cat))
    else:
        left = parse_cat(cat[:op_idx])
        slash = cat[op_idx:op_idx + 1]
        right = parse_cat(cat[op_idx + 1:])
        return Functor(left, slash, right)


class KeyakiParser(object):
    def __init__(self, line):
        self.index = 0
        self.line = line
        self.items = self.lex(line)

    def lex(self, line):
        def is_bracket(string):
            return string in ['(', ')']

        flag = False
        res = []
        pos = 0

        line = line.replace('(', '( ').replace(')', ' )').split()
        for a, b in zip(line, line[1:]):
            if flag:
                flag = False
                continue
            elif not is_bracket(a) and not is_bracket(b):
                res.append(Leaf(parse_cat(a), b, pos))
                flag = True
                pos += 1
            else:
                res.append(a)
                flag = False
        res.append(')')
        return res

    def next(self):
        res = self.items[self.index]
        self.index += 1
        return res

    def peek(self):
        return self.items[self.index]

    def peek_next(self):
        return self.items[self.index+1]

    def peek_prev(self):
        return self.items[self.index-1]

    def parse(self):
        """
        parse (IP-MAT (INTJ ..) (INTJ ..))
        """
        if isinstance(self.peek_next(), Leaf):
            return self.parse_terminal()
        else:
            return self.parse_nonterminal()

    def parse_terminal(self):
        """
        parse ( INTJ えーっと )
        """
        assert self.next() == '('
        res = self.next()
        assert self.next() == ')'
        return res

    def parse_nonterminal(self):
        """
        parse ( IP-MAT ( INTJ .. ) ( INTJ .. ) )
        """
        assert self.next() == '(', self.index
        tag = parse_cat(self.next())
        children = []
        while self.peek() != ')':
            children.append(self.parse())
        assert self.next() == ')'
        res = Tree(tag, children)
        return res


def tree_is_to_be_used(tree):
    """
    discard the input tree if:
    1) the root is FRAG
    2) it contains empty category such as *T*
    3) a node with more than two children
    """
    def rec(node):
        if isinstance(node, Tree):
            if len(node.children) > 2:
                # logger.warn(f'more than 2 children: {tree}')
                return False
            return all(rec(child) for child in node.children)
        else:
            return not (node.word.startswith('*') and node.word.endswith('*'))
    if str(tree.cat) == 'FRAG':
        return False
    else:
        return rec(tree)


def read_keyaki(keyakipath, remove_top=True):
    for line in open(keyakipath):
        try:
            tree = KeyakiParser(line.strip()).parse()
        except AssertionError:
            continue
        if isinstance(tree, Tree) and remove_top:
            tree = tree.children[0]
        if tree_is_to_be_used(tree):
            yield tree


def get_leaves(tree):
    res = []
    def rec(tree):
        for child in tree.children:
            if isinstance(child, Tree):
                rec(child)
            else:
                assert isinstance(child, Leaf)
                res.append(child)

    if isinstance(tree, Tree):
        rec(tree)
    else:
        res.append(tree)
    return res


UNK = "*UNKNOWN*"
START = "*START*"
END = "*END*"
IGNORE = -1


class TrainingDataCreator(object):
    """
    create train & validation data
    """
    def __init__(self, filepath, word_freq_cut, char_freq_cut, cat_freq_cut):
        self.filepath = filepath
        # those categories whose frequency < freq_cut are discarded.
        self.word_freq_cut = word_freq_cut
        self.char_freq_cut = char_freq_cut
        self.cat_freq_cut  = cat_freq_cut
        self.seen_rules = defaultdict(int)  # seen binary rules
        self.unary_rules = defaultdict(int)  # seen unary rules
        self.cats = defaultdict(int, {
            START: cat_freq_cut,
            END: cat_freq_cut
        })  # all cats
        self.words = defaultdict(int, {
            UNK: word_freq_cut,
            START: word_freq_cut,
            END: word_freq_cut
        })
        self.chars = defaultdict(int, {
            UNK: char_freq_cut,
            START: char_freq_cut,
            END: char_freq_cut
        })
        self.samples = []
        self.sents = []

    def _traverse(self, tree):
        if isinstance(tree, Leaf):
            self.cats[str(tree.cat)] += 1
            word = tree.word
            self.words[word] += 1
            for char in word:
                self.chars[char] += 1
        else:
            children = tree.children
            if len(children) == 1:
                rule = str(tree.cat), str(children[0].cat)
                self.unary_rules[rule] += 1
                self._traverse(children[0])
            else:
                if len(children) == 2:  # TODO
                    rule = str(children[0].cat), str(children[1].cat)
                    self.seen_rules[rule] += 1
                    self._traverse(children[0])
                    self._traverse(children[1])

    @staticmethod
    def _write(dct, filename):
        with open(filename, 'w') as f:
            logger.info(f'writing to {f.name}')
            for key, value in dct.items():
                print(f'{key} # {str(value)}', file=f)

    def _get_dependencies(self, tree, sent_len):
        def rec(subtree):
            if isinstance(subtree, Tree):
                children = subtree.children
                if len(children) == 2:
                    head = rec(children[1])
                    dep  = rec(children[0])
                    res[dep] = head
                else:
                    head = rec(children[0])
                return head
            else:
                return subtree.pos

        res = [-1 for _ in range(sent_len)]
        rec(tree)
        res = [i+ 1 for i in res]
        assert len(list(filter(lambda i:i == 0, res))) == 1, (res, str(tree))
        return res

    def _to_conll(self, out):
        for sent, (cats, deps) in self.samples:
            words = sent.split(' ')
            for i, (word, cat, dep) in enumerate(zip(words, cats, deps), 1):
                print(f'{i}\t{word}\t{cat}\t{dep}', file=out)
            print('', file=out)

    def _create_samples(self, trees):
        for tree in trees:
            tokens = get_leaves(tree)
            words = [token.word for token in tokens]
            cats = [str(token.cat) for token in tokens]
            deps = self._get_dependencies(tree, len(tokens))
            sent = ' '.join(words)
            self.sents.append(sent)
            self.samples.append((sent, [cats, deps]))

    @staticmethod
    def create_traindata(args):
        self = TrainingDataCreator(args.PATH,
                                   args.word_freq_cut,
                                   args.char_freq_cut,
                                   args.cat_freq_cut)

        trees = [tree for tree in read_keyaki(self.filepath)]
        # trees = [] # TODO
        # for line in open(self.filepath):
        #     try:
        #         trees.append(KeyakiParser(line.strip()).parse())
        #     except Exception:
        #         continue

        for tree in trees:
            self._traverse(tree)
        self._create_samples(trees)

        cats = {k: v for k, v in self.cats.items() if v >= self.cat_freq_cut}
        self._write(cats, args.OUT / 'target.txt')

        words = {k: v for k, v in self.words.items() if v >= self.word_freq_cut}
        self._write(words, args.OUT / 'words.txt')

        chars = {k: v for k, v in self.chars.items() if v >= self.char_freq_cut}
        self._write(chars, args.OUT / 'chars.txt')

        seen_rules = {f'{c1} {c2}': v for (c1, c2), v in self.seen_rules.items()
                      if c1 in cats and c2 in cats}
        self._write(seen_rules, args.OUT / 'seen_rules.txt')

        unary_rules = {f'{c1} {c2}': v for (c1, c2), v in self.unary_rules.items()}
        self._write(unary_rules, args.OUT / 'unary_rules.txt')

        with open(args.OUT / 'traindata.json', 'w') as f:
            logger.info(f'writing to {f.name}')
            json.dump(self.samples, f)

        with open(args.OUT / 'trainsents.txt', 'w') as f:
            logger.info(f'writing to {f.name}')
            for sent in self.sents:
                print(sent, file=f)

        with open(args.OUT / 'trainsents.conll', 'w') as f:
            logger.info(f'writing to {f.name}')
            self._to_conll(f)

    @staticmethod
    def create_testdata(args):
        self = TrainingDataCreator(args.PATH,
                                   args.word_freq_cut,
                                   args.char_freq_cut,
                                   args.cat_freq_cut)

        trees = [tree for tree in read_keyaki(self.filepath)]
        # trees = [] # TODO
        # for line in open(self.filepath):
        #     try:
        #         trees.append(KeyakiParser(line.strip()).parse())
        #     except Exception:
        #         continue

        self._create_samples(trees)
        with open(args.OUT / 'testdata.json', 'w') as f:
            logger.info(f'writing to {f.name}')
            json.dump(self.samples, f)

        with open(args.OUT / 'testsents.txt', 'w') as f:
            logger.info(f'writing to {f.name}')
            for sent in self.sents:
                print(sent, file=f)

        with open(args.OUT / 'testsents.conll', 'w') as f:
            logger.info(f'writing to {f.name}')
            self._to_conll(f)

    @staticmethod
    def convert_json(autopath):
        self = TrainingDataCreator(autopath, None, None, None)
        trees = [tree for tree in read_keyaki(self.filepath)]
        logger.info(f'loaded {len(trees)} trees')
        self._create_samples(trees)
        return self.samples


def convert_keyaki_to_json(keyakipath):
    return TrainingDataCreator.convert_json(keyakipath)


def main():
    import argparse
    parser = argparse.ArgumentParser('keyaki')

    parser.add_argument('PATH',
                        type=Path,
                        help='path to ccgbank data file')
    parser.add_argument('OUT',
                        type=Path,
                        help='output directory path')
    parser.add_argument('--cat-freq-cut',
                        type=int,
                        default=10,
                        help='only allow categories which appear >= freq-cut')
    parser.add_argument('--word-freq-cut',
                        type=int,
                        default=5,
                        help='only allow words which appear >= freq-cut')
    parser.add_argument('--afix-freq-cut',
                        type=int,
                        default=5,
                        help='only allow afixes which appear >= freq-cut')
    parser.add_argument('--char-freq-cut',
                        type=int,
                        default=5,
                        help='only allow characters which appear >= freq-cut')
    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train')

    args = parser.parse_args()

    if args.mode == 'train':
        TrainingDataCreator.create_traindata(args)
    else:
        TrainingDataCreator.create_testdata(args)


if __name__ == '__main__':
    main()