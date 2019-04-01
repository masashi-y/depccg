
import json
from depccg.tools.reader import read_auto
from depccg.utils import normalize
from collections import defaultdict, Counter
from pathlib import Path
import logging
import sys


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


UNK = "*UNKNOWN*"
OOR2 = "OOR2"
OOR3 = "OOR3"
OOR4 = "OOR4"
START = "*START*"
END = "*END*"
IGNORE = -1
MISS = -2


def get_suffix(word):
    return [word[-1],
           word[-2:] if len(word) > 1 else OOR2,
           word[-3:] if len(word) > 2 else OOR3,
           word[-4:] if len(word) > 3 else OOR4]


def get_prefix(word):
    return [word[0],
            word[:2] if len(word) > 1 else OOR2,
            word[:3] if len(word) > 2 else OOR3,
            word[:4] if len(word) > 3 else OOR4]


class TrainingDataCreator(object):
    def __init__(self, filepath, word_freq_cut, cat_freq_cut, afix_freq_cut):
        self.filepath = filepath
         # those categories whose frequency < freq_cut are discarded.
        self.word_freq_cut = word_freq_cut
        self.cat_freq_cut  = cat_freq_cut
        self.afix_freq_cut = afix_freq_cut
        self.seen_rules = defaultdict(int) # seen binary rules
        self.unary_rules = defaultdict(int) # seen unary rules
        self.cats = defaultdict(int) # all cats
        self.words = defaultdict(int, {UNK: word_freq_cut, START: word_freq_cut, END: word_freq_cut})
        afix_defaults = {UNK: afix_freq_cut, START: afix_freq_cut, END: afix_freq_cut,
                            OOR2: afix_freq_cut, OOR3: afix_freq_cut, OOR4: afix_freq_cut}
        self.prefixes = defaultdict(int, afix_defaults)
        self.suffixes = defaultdict(int, afix_defaults)
        self.samples = []
        self.sents = []

    def _traverse(self, tree):
        if tree.is_leaf:
            self.cats[str(tree.cat)] += 1
            word = normalize(tree.word)
            self.words[word.lower()] += 1
            for f in get_suffix(word):
                self.suffixes[f] += 1
            for f in get_prefix(word):
                self.prefixes[f] += 1
        else:
            children = tree.children
            if len(children) == 1:
                rule = str(tree.cat), str(children[0].cat)
                self.unary_rules[rule] += 1
                self._traverse(children[0])
            else:
                rule = str(children[0].cat), str(children[1].cat)
                self.seen_rules[rule] += 1
                self._traverse(children[0])
                self._traverse(children[1])

    @staticmethod
    def _write(dct, filename):
        with open(args.OUT / filename, 'w') as f:
            logger.info(f'writing to {f.name}')
            for key, value in dct.items():
                print(f'{key} # {str(value)}', file=f)

    def _get_dependencies(self, tree, sent_len):
        def rec(subtree):
            if not subtree.is_leaf:
                children = subtree.children
                if len(children) == 2:
                    head = rec(children[0])
                    dep = rec(children[1])
                    res[dep] = head
                else:
                    head = rec(children[0])
                return head
            else:
                return subtree.head_id

        res = [-1 for _ in range(sent_len)]
        rec(tree)
        res = [i + 1 for i in res]
        assert len(list(filter(lambda i:i == 0, res))) == 1
        return res

    def _to_conll(self, out):
        for sent, [cats, deps] in self.samples:
            sent = sent.split(' ')
            for i, vs in enumerate(zip(sent, cats, deps), 1):
                out.write('{0}\t{1}\t{1}\tPOS\tPOS\t_\t{3}\tnone\t_\t{2}\n'.format(i, *vs))
            out.write('\n')

    def _create_samples(self, trees):
        for tree in trees:
            tokens = tree.leaves
            words = [normalize(token.word) for token in tokens]
            cats = [str(token.cat) for token in tokens]
            deps = self._get_dependencies(tree, len(tokens))
            sent = ' '.join(words)
            self.sents.append(sent)
            self.samples.append((sent, [cats, deps]))

    @staticmethod
    def create_traindata(args):
        self = TrainingDataCreator(
            args.PATH, args.word_freq_cut, args.cat_freq_cut, args.afix_freq_cut)

        trees = [tree for _, _, tree in read_auto(self.filepath) if tree.word != 'FAILED']
        logger.info(f'loaded {len(trees)} trees')
        for tree in trees:
            self._traverse(tree)
        self._create_samples(trees)

        cats = {k: v for k, v in self.cats.items() if v >= self.cat_freq_cut}
        self._write(cats, 'target.txt')

        words = {k: v for k, v in self.words.items() if v >= self.word_freq_cut}
        self._write(words, 'words.txt')

        suffixes = {k: v for k, v in self.suffixes.items() if v >= self.afix_freq_cut}
        self._write(suffixes, 'suffixes.txt')

        prefixes = {k: v for k, v in self.prefixes.items() if v >= self.afix_freq_cut}
        self._write(prefixes, 'prefixes.txt')

        seen_rules = {f'{c1} {c2}': v for (c1, c2), v in self.seen_rules.items()
                      if c1 in cats and c2 in cats}
        self._write(seen_rules, 'seen_rules.txt')

        unary_rules = {f'{c1} {c2}': v for (c1, c2), v in self.unary_rules.items()
                       if c1 in cats and c2 in cats}
        self._write(unary_rules, 'unary_rules.txt')

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
        self = TrainingDataCreator(
            args.PATH, args.word_freq_cut, args.cat_freq_cut, args.afix_freq_cut)

        trees = [tree for _, _, tree in read_auto(self.filepath)]
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Creating training data
    parser.add_argument('PATH',
            type=Path,
            help='path to conll data file')
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
    parser.add_argument('--mode',
            choices=['train', 'test'],
            default='train')

    args = parser.parse_args()
    if args.mode == 'train':
        TrainingDataCreator.create_traindata(args)
    else:
        TrainingDataCreator.create_testdata(args)

