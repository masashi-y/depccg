import logging
import json
from collections import defaultdict
from pathlib import Path

from depccg.tools.ja.reader import read_ccgbank

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


UNK = "*UNKNOWN*"
START = "*START*"
END = "*END*"
IGNORE = -1


class TrainingDataCreator(object):
    def __init__(self, filepath, word_freq_cut, char_freq_cut, cat_freq_cut):
        self.filepath = filepath
        # those categories whose frequency < freq_cut are discarded.
        self.word_freq_cut = word_freq_cut
        self.char_freq_cut = char_freq_cut
        self.cat_freq_cut = cat_freq_cut
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
        if tree.is_leaf:
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
        counter = 0

        def rec(subtree):
            nonlocal counter
            if not subtree.is_leaf:
                children = subtree.children
                if len(children) == 2:
                    head = rec(children[0 if subtree.head_is_left else 1])
                    dep = rec(children[1 if subtree.head_is_left else 0])
                    res[dep] = head
                else:
                    head = rec(children[0])
                return head
            else:
                head = counter
                counter += 1
                return head

        res = [-1 for _ in range(sent_len)]
        rec(tree)
        res = [i + 1 for i in res]
        assert len(list(filter(lambda i: i == 0, res))) == 1
        return res

    def _to_conll(self, out):
        for sent, (cats, deps) in self.samples:
            words = sent.split(' ')
            for i, (word, cat, dep) in enumerate(zip(words, cats, deps), 1):
                print(f'{i}\t{word}\t{cat}\t{dep}', file=out)
            print('', file=out)

    def _create_samples(self, trees):
        for tree in trees:
            tokens = tree.leaves
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

        trees = [tree for _, _, tree in read_ccgbank(self.filepath)]
        for tree in trees:
            self._traverse(tree)
        self._create_samples(trees)

        cats = {k: v for k, v in self.cats.items() if v >= self.cat_freq_cut}
        self._write(cats, args.OUT / 'target.txt')

        words = {k: v for k, v in self.words.items() if v >=
                 self.word_freq_cut}
        self._write(words, args.OUT / 'words.txt')

        chars = {k: v for k, v in self.chars.items() if v >=
                 self.char_freq_cut}
        self._write(chars, args.OUT / 'chars.txt')

        seen_rules = {f'{c1} {c2}': v for (c1, c2), v in self.seen_rules.items()
                      if c1 in cats and c2 in cats}
        self._write(seen_rules, args.OUT / 'seen_rules.txt')

        unary_rules = {f'{c1} {c2}': v for (
            c1, c2), v in self.unary_rules.items()}
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
                                   args.cat_freq_cut,
                                   args.char_freq_cut)

        trees = [tree for _, _, tree in read_ccgbank(self.filepath)]
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
        trees = [tree for _, _, tree in read_ccgbank(self.filepath)]
        logger.info(f'loaded {len(trees)} trees')
        self._create_samples(trees)
        return self.samples


def convert_ccgbank_to_json(ccgbankpath):
    return TrainingDataCreator.convert_json(ccgbankpath)


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
