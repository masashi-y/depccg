
import sys
import json
import chainer
from collections import defaultdict
from japanese_ccg import JaCCGReader
from tree import Leaf, Tree, get_leaves

# filepath = "/home/masashi-y/japanese-ccgbank/ccgbank-20150216/test10.ccgbank"
filepath = "/home/masashi-y/japanese-ccgbank/ccgbank-20150216/train.ccgbank"

class JaCCGInspector(object):
    """
    create train & validation data
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.seen_rules = defaultdict(int) # seen binary rules
        self.unary_rules = defaultdict(int) # seen unary rules
        self.cats = defaultdict(int) # all cats
        self.words = defaultdict(int)
        self.sents = {}

    def _traverse(self, tree):
        if isinstance(tree, Leaf):
            self.cats[tree.cat.without_semantics] += 1
            self.words[tree.word.encode("utf-8")] += 1
        else:
            children = tree.children
            if len(children) == 1:
                rule = tree.cat.without_semantics + \
                        " " + children[0].cat.without_semantics
                self.unary_rules[rule] += 1
                self._traverse(children[0])
            else:
                rule = children[0].cat.without_semantics + \
                        " " + children[1].cat.without_semantics
                self.seen_rules[rule] += 1
                self._traverse(children[0])
                self._traverse(children[1])

    @staticmethod
    def _write(dct, out, comment_out_value=False):
        for key, value in dct.items():
            out.write(str(key) + " ")
            if comment_out_value:
                out.write("# ")
            out.write(str(value) + "\n")

    def create_traindata(self, outdir):
        trees = JaCCGReader(self.filepath).readall()
        for tree in trees:
            self._traverse(tree)
            tokens = get_leaves(tree)
            words = " ".join([token.word for token in tokens])
            cats = " ".join([token.cat.without_semantics for token in tokens])
            self.sents[words] = cats
        with open(outdir + "/unary_rules.txt", "w") as f:
            self._write(self.unary_rules, f, comment_out_value=True)
        with open(outdir + "/seen_rules.txt", "w") as f:
            self._write(self.seen_rules, f, comment_out_value=True)
        with open(outdir + "/targets.txt", "w") as f:
            self._write(self.cats, f, comment_out_value=False)
        with open(outdir + "/words.txt", "w") as f:
            self._write(self.words, f, comment_out_value=False)
        with open(outdir + "traindata.json", "w") as f:
            json.dump(self.sents, f)

    def create_testdata(self, outdir):
        trees = JaCCGReader(self.filepath).readall()
        for tree in trees:
            tokens = get_leaves(tree)
            words = " ".join([token.word for token in tokens])
            cats = " ".join([token.cat.without_semantics for token in tokens])
            self.sents[words] = cats
        with open(outdir + "testdata.json", "w") as f:
            json.dump(self.sents, f)


class JaCCGTaggerDataset(chainer.dataset.DatasetMixin):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def get_example(self, i):
        pass

class JaCCGEmbeddingTagger(object):
    def __init__(self, model_path, word_dim=None, caps_dim=None, suffix_dim=None):
        pass

    def __call__(self, xs, ts):
        pass

    def predict(self, tokens):
        pass

    def predict_doc(self, doc, batchsize=100):
        pass

    @property
    def cats(self):
        pass
        # return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]

outdir = "ja_model/"
JaCCGInspector(filepath).create_traindata(outdir)
