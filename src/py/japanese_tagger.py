
from __future__ import print_function
import sys
import numpy as np
import json
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training, Variable
from chainer.training import extensions
from chainer.dataset import convert
from chainer.optimizer import WeightDecay
from py.py_utils import get_context_by_window, read_pretrained_embeddings, read_model_defs
from py.tagger import MultiProcessTaggerMixin
from collections import defaultdict
from py.japanese_ccg import JaCCGReader
from py.tree import Leaf, Tree, get_leaves

WINDOW_SIZE = 7
CONTEXT = (WINDOW_SIZE - 1) / 2
IGNORE = -1
UNK = "*UNKNOWN*"
LPAD, RPAD, PAD = "LPAD", "RPAD", "PAD"

class JaCCGInspector(object):
    """
    create train & validation data
    """
    def __init__(self, filepath, word_freq_cut, cat_freq_cut):
        self.filepath = filepath
         # those categories whose frequency < freq_cut are discarded.
        self.word_freq_cut = word_freq_cut
        self.cat_freq_cut = cat_freq_cut
        self.seen_rules = defaultdict(int) # seen binary rules
        self.unary_rules = defaultdict(int) # seen unary rules
        self.cats = defaultdict(int) # all cats
        self.words = defaultdict(int)
        self.chars = defaultdict(int)
        self.samples = {}
        self.sents = []

        self.words[UNK] = 10000
        self.words[LPAD] = 10000
        self.words[RPAD] = 10000
        self.chars[UNK] = 10000
        self.chars[PAD] = 10000

    def _traverse(self, tree):
        if isinstance(tree, Leaf):
            self.cats[tree.cat.without_semantics] += 1
            w = tree.word
            self.words[w] += 1
            for c in tree.word:
                self.chars[c] += 1
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
        print("writing to", out.name, file=sys.stderr)
        for key, value in dct.items():
            out.write(key.encode("utf-8") + " ")
            if comment_out_value:
                out.write("# ")
            out.write(str(value) + "\n")

    def create_traindata(self, outdir):
        trees = JaCCGReader(self.filepath).readall()
        # first construct dictionaries only
        for tree in trees:
            self._traverse(tree)
        # construct training samples with
        # categories whose frequency >= freq_cut.
        for tree in trees:
            tokens = get_leaves(tree)
            words = [token.word for token in tokens]
            self.sents.append(" ".join(words))
            cats = [token.cat.without_semantics for token in tokens]
            samples = get_context_by_window(
                    words, CONTEXT, lpad=LPAD, rpad=RPAD)
            assert len(samples) == len(cats)
            for cat, sample in zip(cats, samples):
                if self.cats[cat] >= self.cat_freq_cut:
                    self.samples[" ".join(sample)] = cat

        self.cats = {k: v for (k, v) in self.cats.items() \
                        if v >= self.cat_freq_cut}
        self.words = {k: v for (k, v) in self.words.items() \
                        if v >= self.word_freq_cut}
        with open(outdir + "/unary_rules.txt", "w") as f:
            self._write(self.unary_rules, f, comment_out_value=True)
        with open(outdir + "/seen_rules.txt", "w") as f:
            self._write(self.seen_rules, f, comment_out_value=True)
        with open(outdir + "/target.txt", "w") as f:
            self._write(self.cats, f, comment_out_value=False)
        with open(outdir + "/words.txt", "w") as f:
            self._write(self.words, f, comment_out_value=False)
        with open(outdir + "/chars.txt", "w") as f:
            self._write(self.chars, f, comment_out_value=False)
        with open(outdir + "/traindata.json", "w") as f:
            json.dump(self.samples, f)
        with open(outdir + "/trainsents.txt", "w") as f:
            for sent in self.sents:
                f.write(sent.encode("utf-8") + "\n")

    def create_testdata(self, outdir):
        trees = JaCCGReader(self.filepath).readall()
        for tree in trees:
            tokens = get_leaves(tree)
            words = [token.word for token in tokens]
            self.sents.append(" ".join(words))
            cats = [token.cat.without_semantics for token in tokens]
            samples = get_context_by_window(
                    words, CONTEXT, lpad=LPAD, rpad=RPAD)
            assert len(samples) == len(cats)
            for cat, sample in zip(cats, samples):
                self.samples[" ".join(sample)] = cat
        with open(outdir + "/testdata.json", "w") as f:
            json.dump(self.samples, f)
        with open(outdir + "/testsents.txt", "w") as f:
            for sent in self.sents:
                f.write(sent.encode("utf-8") + "\n")


class FeatureExtractor(object):
    def __init__(self, model_path):
        self.words = read_model_defs(model_path + "/words.txt")
        self.chars = read_model_defs(model_path + "/chars.txt")
        self.unk_word = self.words[UNK]
        self.unk_char = self.chars[UNK]
        self.max_char_len = max(len(w) for w in self.words if w != UNK)

    def __call__(self, tokens, max_len=None):
        if max_len is None:
            max_len = max(len(w) for w in tokens)
        w = np.zeros((WINDOW_SIZE,), 'i')
        c = -np.ones((WINDOW_SIZE, max_len + 1), 'i')
        l = np.zeros((WINDOW_SIZE,), 'f')
        for i, word in enumerate(tokens):
            w[i] = self.words.get(word, self.unk_word)
            l[i] = len(word)
            if word == LPAD or word == RPAD:
                c[i, 0] = self.chars[PAD]
            else:
                for j, char in enumerate(word):
                    c[i, j] = self.chars.get(char, self.unk_char)
        return w, c, l

class JaCCGTaggerDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.targets = read_model_defs(model_path + "/target.txt")
        self.extractor = FeatureExtractor(model_path)
        with open(samples_path) as f:
            self.samples = json.load(f).items()
        assert isinstance(self.samples[0][0], unicode)

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        """
        `line`: word1 word2 ,.., wordN target\n
        Returns:
            np.ndarray shape(WINDOW_SIZE, 1+max_char_len)
            with first column id for each word in the window,
            second till the last columns are filled with character id.
        """
        line, target = self.samples[i]
        items = line.strip().split(" ")
        x, c, l = self.extractor(items)
        t = np.asarray(self.targets.get(target, IGNORE), 'i')
        return x, c, l, t


class JaCCGEmbeddingTagger(chainer.Chain, MultiProcessTaggerMixin):
    def __init__(self, model_path, word_dim=None, char_dim=None):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            # use as supertagger
            with open(defs_file) as f:
                defs = json.load(f)
            self.word_dim = defs["word_dim"]
            self.char_dim = defs["char_dim"]
        else:
            # training
            self.word_dim = word_dim
            self.char_dim = char_dim
            with open(defs_file, "w") as f:
                json.dump({"model": self.__class__.__name__,
                           "word_dim": self.word_dim,
                           "char_dim": self.char_dim}, f)

        self.extractor = FeatureExtractor(model_path)
        self.targets = read_model_defs(model_path + "/target.txt")
        self.train = True

        hidden_dim = 1000
        in_dim = WINDOW_SIZE * (self.word_dim + self.char_dim)
        super(JaCCGEmbeddingTagger, self).__init__(
                emb_word=L.EmbedID(len(self.extractor.words), self.word_dim),
                emb_char=L.EmbedID(len(self.extractor.chars),
                            self.char_dim, ignore_label=IGNORE),
                linear1=L.Linear(in_dim, hidden_dim),
                linear2=L.Linear(hidden_dim, len(self.targets)),
                )

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def __call__(self, ws, cs, ls, ts):
        h_w = self.emb_word(ws) #_(batchsize, windowsize, word_dim)
        h_c = self.emb_char(cs) # (batchsize, windowsize, max_char_len, char_dim)
        batchsize, windowsize, _, _ = h_c.data.shape
        # (batchsize, windowsize, char_dim)
        h_c = F.sum(h_c, 2)
        h_c, ls = F.broadcast(h_c, F.reshape(ls, (batchsize, windowsize, 1)))
        h_c = h_c / ls
        h = F.concat([h_w, h_c], 2)
        h = F.reshape(h, (batchsize, -1))
        # ys = self.linear1(h)
        h = F.relu(self.linear1(h))
        h = F.dropout(h, ratio=.5, train=self.train)
        ys = self.linear2(h)

        loss = F.softmax_cross_entropy(ys, ts)
        acc = F.accuracy(ys, ts)
        chainer.report({
            "loss": loss,
            "accuracy": acc
            }, self)
        return loss

    def feature_extract(self, tokens):
        max_len = max(len(w) for w in tokens)
        contexts = get_context_by_window(
                    tokens, CONTEXT, lpad=LPAD, rpad=RPAD)
        return [self.extractor(c, max_len) for c in contexts]

    def predict(self, tokens):
        self.train = False
        contexts = self.feature_extract(tokens) \
                if isinstance(tokens[0], unicode) else tokens

        # contexts [(w, c, l), (w, c, l)]
        ws, cs, ls = zip(*contexts)
        max_cs_size = max(c.shape[1] for c in cs)
        new_cs = []
        for c in cs:
            c = np.pad(c, ((0, 0), (0, max_cs_size - c.shape[1])),
                    mode='constant', constant_values=-1)
            new_cs.append(c)
        ws = np.asarray(ws, 'i')
        cs = np.asarray(new_cs, 'i')
        ls = np.asarray(ls, 'f')
        h_w = self.emb_word(ws) #_(batchsize, windowsize, word_dim)
        h_c = self.emb_char(cs) # (batchsize, windowsize, max_char_len, char_dim)
        batchsize, windowsize, _, _ = h_c.data.shape
        # (batchsize, windowsize, char_dim)
        h_c = F.sum(h_c, 2)
        h_c, ls = F.broadcast(h_c, F.reshape(ls, (batchsize, windowsize, 1)))
        h_c = h_c / ls
        h = F.concat([h_w, h_c], 2)
        h = F.reshape(h, (batchsize, -1))
        # ys = self.linear(h)
        h = F.relu(self.linear1(h))
        h = F.dropout(h, ratio=.5, train=self.train)
        ys = self.linear2(h)
        return ys.data

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]

def train(args):
    model = JaCCGEmbeddingTagger(args.model,
                args.word_emb_size, args.char_emb_size)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    if args.pretrained:
        print('Load pretrained word embeddings from', args.pretrained)
        model.load_pretrained_embeddings(args.pretrained)

    train = JaCCGTaggerDataset(args.model, args.train)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val = JaCCGTaggerDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.AdaGrad()
    optimizer.setup(model)
    # optimizer.add_hook(WeightDecay(1e-8))
    my_converter = lambda x, dev: convert.concat_examples(x, dev, (None,-1,None,None))
    updater = training.StandardUpdater(train_iter, optimizer, converter=my_converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 1000, 'iteration'
    log_interval = 200, 'iteration'

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, my_converter), trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                "Japanese CCG parser's supertag tagger")
    subparsers = parser.add_subparsers()

    # Creating training data
    parser_c = subparsers.add_parser(
            "create", help="create tagger input data")
    parser_c.add_argument("path",
            help="path to ccgbank data file")
    parser_c.add_argument("out",
            help="output directory path")
    parser_c.add_argument("--cat-freq-cut",
            type=int, default=10,
            help="only allow categories which appear >= freq-cut")
    parser_c.add_argument("--word-freq-cut",
            type=int, default=5,
            help="only allow words which appear >= freq-cut")
    parser_c.add_argument("--windowsize",
            type=int, default=3,
            help="window size to extract features")
    parser_c.add_argument("--subset",
            choices=["train", "test"],
            default="train")

    parser_c.set_defaults(func=
            (lambda args:
                (lambda x=JaCCGInspector(args.path,
                    args.word_freq_cut, args.cat_freq_cut) :
                    x.create_traindata(args.out)
                        if args.subset == "train"
                    else  x.create_testdata(args.out))()
                ))

    # Do training using training data created through `create`
    parser_t = subparsers.add_parser(
            "train", help="train supertagger model")
    parser_t.add_argument("model",
            help="path to model directory")
    # parser_t.add_argument("embed",
            # help="path to embedding file")
    # parser_t.add_argument("vocab",
            # help="path to embedding vocab file")
    parser_t.add_argument("train",
            help="training data file path")
    parser_t.add_argument("val",
            help="validation data file path")
    parser_t.add_argument("--batchsize",
            type=int, default=1000, help="batch size")
    parser_t.add_argument("--epoch",
            type=int, default=20, help="epoch")
    parser_t.add_argument("--word-emb-size",
            type=int, default=50,
            help="word embedding size")
    parser_t.add_argument("--char-emb-size",
            type=int, default=50,
            help="character embedding size")
    parser_t.add_argument("--initmodel",
            help="initialize model with `initmodel`")
    parser_t.add_argument("--pretrained",
            help="pretrained word embeddings")
    parser_t.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)
