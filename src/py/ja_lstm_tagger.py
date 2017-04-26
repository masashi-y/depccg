
from __future__ import print_function
import sys
import numpy as np
import json
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import training, Variable
from chainer.training import extensions
from chainer.optimizer import WeightDecay, GradientClipping
from py.japanese_ccg import JaCCGReader
from collections import defaultdict
from py.py_utils import read_pretrained_embeddings, read_model_defs
from py.tree import Leaf, Tree, get_leaves

UNK = "*UNKNOWN*"
START = "*START*"
END = "*END*"
IGNORE = -1

def log(args, out):
    for k, v in vars(args).items():
        out.write("{}: {}\n".format(k, v))

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
        self.seen_rules = defaultdict(int) # seen binary rules
        self.unary_rules = defaultdict(int) # seen unary rules
        self.cats = defaultdict(int) # all cats
        self.words = defaultdict(int)
        self.chars = defaultdict(int)
        self.samples = {}
        self.sents = []

        self.words[UNK]      = 10000
        self.words[START]    = 10000
        self.words[END]      = 10000
        self.chars[UNK]      = 10000
        self.chars[START]    = 10000
        self.chars[END]      = 10000
        self.cats[START]     = 10000
        self.cats[END]       = 10000

    def _traverse(self, tree):
        if isinstance(tree, Leaf):
            self.cats[tree.cat.without_semantics] += 1
            w = tree.word
            self.words[w] += 1
            for c in w:
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

    def _create_samples(self, trees):
        for tree in trees:
            tokens = get_leaves(tree)
            words = [token.word for token in tokens]
            cats = [token.cat.without_semantics for token in tokens]
            sent = " ".join(words)
            self.sents.append(sent)
            self.samples[sent] = " ".join(cats)

    @staticmethod
    def create_traindata(args):
        self = TrainingDataCreator(args.path,
                args.word_freq_cut, args.char_freq_cut, args.cat_freq_cut)
        with open(args.out + "/log_create_traindata", "w") as f:
            log(args, f)

        trees = JaCCGReader(self.filepath).readall()
        for tree in trees:
            self._traverse(tree)
        self._create_samples(trees)

        self.cats = {k: v for (k, v) in self.cats.items() \
                        if v >= self.cat_freq_cut}
        self.words = {k: v for (k, v) in self.words.items() \
                        if v >= self.word_freq_cut}
        self.chars = {k: v for (k, v) in self.chars.items() \
                        if v >= self.char_freq_cut}
        with open(args.out + "/unary_rules.txt", "w") as f:
            self._write(self.unary_rules, f, comment_out_value=True)
        with open(args.out + "/seen_rules.txt", "w") as f:
            self._write(self.seen_rules, f, comment_out_value=True)
        with open(args.out + "/target.txt", "w") as f:
            self._write(self.cats, f, comment_out_value=False)
        with open(args.out + "/words.txt", "w") as f:
            self._write(self.words, f, comment_out_value=False)
        with open(args.out + "/chars.txt", "w") as f:
            self._write(self.chars, f, comment_out_value=False)
        with open(args.out + "/traindata.json", "w") as f:
            json.dump(self.samples, f)
        with open(args.out + "/trainsents.txt", "w") as f:
            for sent in self.sents:
                f.write(sent.encode("utf-8") + "\n")

    @staticmethod
    def create_testdata(args):
        self = TrainingDataCreator(args.path,
                args.word_freq_cut, args.char_freq_cut, args.cat_freq_cut)
        with open(args.out + "/log_create_testdata", "w") as f:
            log(args, f)
        trees = JaCCGReader(self.filepath).readall()
        self._create_samples(trees)
        with open(args.out + "/testdata.json", "w") as f:
            json.dump(self.samples, f)
        with open(args.out + "/testsents.txt", "w") as f:
            for sent in self.sents:
                f.write(sent.encode("utf-8") + "\n")


class FeatureExtractor(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.words = read_model_defs(model_path + "/words.txt")
        self.chars = read_model_defs(model_path + "/chars.txt")
        self.targets = read_model_defs(model_path + "/target.txt")
        self.unk_word = self.words[UNK]
        self.start_word = self.words[START]
        self.end_word = self.words[END]
        self.unk_char = self.chars[UNK]
        self.start_char = self.chars[START]
        self.end_char = self.chars[END]

    def process(self, words):
        """
        words: list of unicode tokens
        """
        w = np.array([self.start_word] + [self.words.get(
            x, self.unk_word) for x in words] + [self.end_word], 'i')
        l = max(len(x) for x in words)
        c = -np.ones((len(words) + 2, l), 'i')
        c[0, 0] = self.start_char
        c[-1, 0] = self.end_char
        for i, word in enumerate(words, 1):
            for j in range(len(word)):
                c[i, j] = self.chars.get(word[j], self.unk_char)
        return w, c, l


class LSTMTaggerDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.extractor = FeatureExtractor(model_path)
        with open(samples_path) as f:
            self.samples = json.load(f).items()

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        words, y = self.samples[i]
        words = words.split(" ")
        y = [START] + y.split(" ") + [END]
        w, c, l = self.extractor.process(words)
        y = np.array([self.targets.get(x, IGNORE) for x in y], 'i')
        return w, c, l, y


class JaLSTMTagger(chainer.Chain):
    def __init__(self, model_path, word_dim=None, char_dim=None,
            nlayers=2, hidden_dim=128, relu_dim=64, dropout_ratio=0.5):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            # use as supertagger
            with open(defs_file) as f:
                defs = json.load(f)
            self.word_dim   = defs["word_dim"]
            self.char_dim   = defs["char_dim"]
            self.hidden_dim = defs["hidden_dim"]
            self.relu_dim   = defs["relu_dim"]
            self.nlayers    = defs["nlayers"]
            self.train = False
            self.extractor = FeatureExtractor(model_path)
        else:
            # training
            self.word_dim = word_dim
            self.char_dim = char_dim
            self.hidden_dim = hidden_dim
            self.relu_dim = relu_dim
            self.nlayers = nlayers
            self.train = True
            with open(defs_file, "w") as f:
                json.dump({"model": self.__class__.__name__,
                           "word_dim": self.word_dim, "char_dim": self.char_dim,
                           "hidden_dim": hidden_dim, "relu_dim": relu_dim,
                           "nlayers": nlayers}, f)

        self.targets = read_model_defs(model_path + "/target.txt")
        self.words = read_model_defs(model_path + "/words.txt")
        self.chars = read_model_defs(model_path + "/chars.txt")
        self.in_dim = self.word_dim + self.char_dim
        self.dropout_ratio = dropout_ratio
        super(JaLSTMTagger, self).__init__(
                emb_word=L.EmbedID(len(self.words), self.word_dim),
                emb_char=L.EmbedID(len(self.chars), 50, ignore_label=IGNORE),
                conv_char=L.Convolution2D(1, self.char_dim,
                    (3, 50), stride=1, pad=(1, 0)),
                lstm_f=L.NStepLSTM(nlayers, self.in_dim,
                    self.hidden_dim, 0.),
                lstm_b=L.NStepLSTM(nlayers, self.in_dim,
                    self.hidden_dim, 0.),
                conv1=L.Convolution2D(1, 2 * self.hidden_dim,
                    (7, 2 * self.hidden_dim), stride=1, pad=(3, 0)),
                linear1=L.Linear(2 * self.hidden_dim, self.relu_dim),
                linear2=L.Linear(self.relu_dim, len(self.targets)),
                )

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def __call__(self, xs):
        """
        xs [(w,s,p,y), ..., ]
        w: word, c: char, l: length, y: label
        """
        batchsize = len(xs)
        ws, cs, ls, ts = zip(*xs)
        # cs: [(sentence length, max word length)]
        ws = map(self.emb_word, ws)
        # ls: [(sentence length, char dim)]
        # cs = map(lambda (c, l): F.sum(self.emb_char(c), 1) / l, zip(cs, ls))
        # cs = [F.reshape(F.average_pooling_2d(
        #         F.expand_dims(self.emb_char(c), 0), (l, 1)), (-1, self.char_dim))
        #             for c, l in zip(cs, ls)]
        # before conv: (sent len, 1, max word len, char_size)
        # after conv: (sent len, char_size, max word len, 1)
        # after max_pool: (sent len, char_size, 1, 1)
        cs = [F.squeeze(
            F.max_pooling_2d(
                self.conv_char(
                    F.expand_dims(
                        self.emb_char(c), 1)), (l, 1)))
                    for c, l in zip(cs, ls)]
        # [(sentence length, (word_dim + char_dim))]
        xs_f = [F.dropout(F.concat([w, c]),
            self.dropout_ratio, train=self.train) for w, c in zip(ws, cs)]
        xs_b = [x[::-1] for x in xs_f]
        cx_f, hx_f, cx_b, hx_b = self._init_state(batchsize)
        _, _, hs_f = self.lstm_f(hx_f, cx_f, xs_f, train=self.train)
        _, _, hs_b = self.lstm_b(hx_b, cx_b, xs_b, train=self.train)
        hs_b = [x[::-1] for x in hs_b]
        # ys: [(sentence length, number of category)]
        ys = [self.linear2(F.relu(self.linear1(F.concat([h_f, h_b]))))
                    for h_f, h_b in zip(hs_f, hs_b)]
        # ys = [self.linear2(F.relu(
        #         self.linear1(
        #             F.squeeze(
        #                 F.transpose(
        #                     F.relu(self.conv1(
        #                         F.reshape(
        #                             F.concat([h_f, h_b]),
        #                             (1, 1, -1, 2 * self.hidden_dim))), (0, 3, 2, 1))
        #                 )))))
        #             for h_f, h_b in zip(hs_f, hs_b)]
        loss = reduce(lambda x, y: x + y,
            [F.softmax_cross_entropy(y, t) for y, t in zip(ys, ts)])
        acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(ys, ts)])

        acc /= batchsize
        chainer.report({
            "loss": loss,
            "accuracy": acc
            }, self)
        return loss

    def predict(self, xs):
        """
        batch: list of splitted sentences
        """
        xs = [self.extractor.process(x) for x in xs]
        batchsize = len(xs)
        ws, cs, ls = zip(*xs)
        ws = map(self.emb_word, ws)
        cs = [F.squeeze(
            F.max_pooling_2d(
                self.conv_char(
                    F.expand_dims(
                        self.emb_char(c), 1)), (l, 1)))
                    for c, l in zip(cs, ls)]
        xs_f = [F.dropout(F.concat([w, c]),
            self.dropout_ratio, train=self.train) for w, c in zip(ws, cs)]
        xs_b = [x[::-1] for x in xs_f]
        cx_f, hx_f, cx_b, hx_b = self._init_state(batchsize)
        _, _, hs_f = self.lstm_f(hx_f, cx_f, xs_f, train=self.train)
        _, _, hs_b = self.lstm_b(hx_b, cx_b, xs_b, train=self.train)
        hs_b = [x[::-1] for x in hs_b]
        ys = [self.linear2(F.relu(self.linear1(F.concat([h_f, h_b]))))
                    for h_f, h_b in zip(hs_f, hs_b)]
        return [y.data[1:-1] for y in ys]

    def predict_doc(self, doc, batchsize=16):
        """
        doc list of splitted sentences
        """
        res = []
        for i in range(0, len(doc), batchsize):
            res.extend([(i + j, 0, y)
                for j, y in enumerate(self.predict(doc[i:i + batchsize]))])
        return res

    def _init_state(self, batchsize):
        res = [Variable(np.zeros(( # forward cx, hx, backward cx, hx
                self.nlayers, batchsize, self.hidden_dim), 'f')) for _ in range(4)]
        return res

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]


def converter(x, device):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device, cuda.Stream.null)


def train(args):
    model = LSTMTagger(args.model, args.word_emb_size, args.char_emb_size,
            args.nlayers, args.hidden_dim, args.relu_dim, args.dropout_ratio)
    with open(args.model + "/params", "w") as f:
            log(args, f)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    if args.pretrained:
        print('Load pretrained word embeddings from', args.pretrained)
        model.load_pretrained_embeddings(args.pretrained)

    train = LSTMTaggerDataset(args.model, args.train)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val = LSTMTaggerDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.MomentumSGD(momentum=0.7)
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(1e-6))
    # optimizer.add_hook(GradientClipping(5.))
    updater = training.StandardUpdater(train_iter, optimizer, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 1000, 'iteration'
    log_interval = 200, 'iteration'

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, converter), trigger=val_interval)
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
                "CCG parser's LSTM supertag tagger")
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
    parser_c.add_argument("--char-freq-cut",
            type=int, default=5,
            help="only allow characters which appear >= freq-cut")
    parser_c.add_argument("--mode",
            choices=["train", "test"],
            default="train")

    parser_c.set_defaults(func=
            (lambda args:
                TrainingDataCreator.create_traindata(args)
                    if args.mode == "train"
                else  TrainingDataCreator.create_testdata(args)))

            #TODO updater
    # Do training using training data created through `create`
    parser_t = subparsers.add_parser(
            "train", help="train supertagger model")
    parser_t.add_argument("model",
            help="path to model directory")
    parser_t.add_argument("train",
            help="training data file path")
    parser_t.add_argument("val",
            help="validation data file path")
    parser_t.add_argument("--batchsize",
            type=int, default=16, help="batch size")
    parser_t.add_argument("--epoch",
            type=int, default=20, help="epoch")
    parser_t.add_argument("--word-emb-size",
            type=int, default=50,
            help="word embedding size")
    parser_t.add_argument("--char-emb-size",
            type=int, default=32,
            help="character embedding size")
    parser_t.add_argument("--nlayers",
            type=int, default=1,
            help="number of layers for each LSTM")
    parser_t.add_argument("--hidden-dim",
            type=int, default=128,
            help="dimensionality of hidden layer")
    parser_t.add_argument("--relu-dim",
            type=int, default=64,
            help="dimensionality of relu layer")
    parser_t.add_argument("--dropout-ratio",
            type=float, default=0.5,
            help="dropout ratio")
    parser_t.add_argument("--initmodel",
            help="initialize model with `initmodel`")
    parser_t.add_argument("--pretrained",
            help="pretrained word embeddings")
    parser_t.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)
