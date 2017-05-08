
from __future__ import print_function, unicode_literals
import sys
import random
import numpy as np
import json
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import training, Variable
from chainer.training import extensions
from chainer.optimizer import WeightDecay, GradientClipping
from py.ccgbank import walk_autodir
from py.japanese_ccg import JaCCGReader
from collections import defaultdict, OrderedDict
from py.py_utils import read_pretrained_embeddings, read_model_defs
from py.tree import Leaf, Tree, get_leaves
from py.biaffine import Biaffine
from py.param import Param

from py.lstm_tagger import UNK, OOR2, OOR3, OOR4, START, END, IGNORE, MISS
from py.lstm_tagger import log, get_suffix, get_prefix, normalize

class TrainingDataCreator(object):
    """
    create train & validation data
    """
    def __init__(self, filepath, word_freq_cut, cat_freq_cut, afix_freq_cut):
        self.filepath = filepath
         # those categories whose frequency < freq_cut are discarded.
        self.word_freq_cut = word_freq_cut
        self.cat_freq_cut  = cat_freq_cut
        self.afix_freq_cut = afix_freq_cut
        self.seen_rules = defaultdict(int) # seen binary rules
        self.unary_rules = defaultdict(int) # seen unary rules
        self.cats = defaultdict(int) # all cats
        self.words = defaultdict(int,
                {UNK: word_freq_cut, START: word_freq_cut, END: word_freq_cut})
        afix_defaults = {UNK: afix_freq_cut, START: afix_freq_cut, END: afix_freq_cut,
                            OOR2: afix_freq_cut, OOR3: afix_freq_cut, OOR4: afix_freq_cut}
        self.prefixes = defaultdict(int, afix_defaults)
        self.suffixes = defaultdict(int, afix_defaults)
        self.samples = []
        self.sents = []

    def _traverse(self, tree):
        if isinstance(tree, Leaf):
            self.cats[str(tree.cat)] += 1
            w = normalize(tree.word)
            self.words[w.lower()] += 1
            for f in get_suffix(w):
                self.suffixes[f] += 1
            for f in get_prefix(w):
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
    def _write(dct, out, comment_out_value=False):
        print("writing to", out.name, file=sys.stderr)
        for key, value in dct.items():
            out.write(key.encode("utf-8") + " ")
            if comment_out_value:
                out.write("# ")
            out.write(str(value) + "\n")

    def _get_dependencies(self, tree, sent_len):
        def rec(subtree):
            if isinstance(subtree, Tree):
                children = subtree.children
                if len(children) == 2:
                    head = rec(children[0 if subtree.left_is_head else 1])
                    dep  = rec(children[1 if subtree.left_is_head else 0])
                    res[dep] = head
                else:
                    head = rec(children[0])
                return head
            else:
                return subtree.pos

        res = [-1 for _ in range(sent_len)]
        rec(tree)
        res = [i + 1 for i in res]
        assert len(filter(lambda i:i == 0, res)) == 1
        return res

    def _to_conll(self, out):
        for sent, tags, (cats, deps) in self.samples:
            for i, (w, t, c, d) in enumerate(zip(sent.split(" "), tags, cats, deps), 1):
                out.write("{0}\t{1}\t{1}\t{2}\t{2}\t_\t{4}\tnone\t_\t{3}\n"
                        .format(i, w.encode("utf-8"), t, c, d))
            out.write("\n")

    def _create_samples(self, trees):
        for tree in trees:
            tokens = get_leaves(tree)
            words = [normalize(token.word) for token in tokens]
            tags = ["POS" for token in tokens] # dummy
            cats = [str(token.cat) for token in tokens]
            deps = self._get_dependencies(tree, len(tokens))
            sent = " ".join(words)
            self.sents.append(sent)
            self.samples.append((sent, tags, (cats, deps)))

    @staticmethod
    def create_traindata(args):
        self = TrainingDataCreator(args.path,
                args.word_freq_cut, args.cat_freq_cut, args.afix_freq_cut)
        with open(args.out + "/log_create_traindata", "w") as f:
            log(args, f)

        trees = walk_autodir(self.filepath, args.subset)
        for tree in trees:
            self._traverse(tree)
        self._create_samples(trees)

        self.cats = {k: v for (k, v) in self.cats.items() \
                        if v >= self.cat_freq_cut}
        self.words = {k: v for (k, v) in self.words.items() \
                        if v >= self.word_freq_cut}
        self.suffixes = {k: v for (k, v) in self.suffixes.items() \
                        if v >= self.afix_freq_cut}
        self.prefixes = {k: v for (k, v) in self.prefixes.items() \
                        if v >= self.afix_freq_cut}
        self.seen_rules = {c1 + " " + c2: v
                for (c1, c2), v in self.seen_rules.items()
                    if c1 in self.cats and c2 in self.cats}
        self.unary_rules = {c1 + " " + c2: v
                for (c1, c2), v in self.unary_rules.items()
                    if c1 in self.cats and c2 in self.cats}
        with open(args.out + "/unary_rules.txt", "w") as f:
            self._write(self.unary_rules, f, comment_out_value=True)
        with open(args.out + "/seen_rules.txt", "w") as f:
            self._write(self.seen_rules, f, comment_out_value=True)
        with open(args.out + "/target.txt", "w") as f:
            self._write(self.cats, f, comment_out_value=False)
        with open(args.out + "/words.txt", "w") as f:
            self._write(self.words, f, comment_out_value=False)
        with open(args.out + "/suffixes.txt", "w") as f:
            self._write(self.suffixes, f, comment_out_value=False)
        with open(args.out + "/prefixes.txt", "w") as f:
            self._write(self.prefixes, f, comment_out_value=False)
        with open(args.out + "/traindata.json", "w") as f:
            json.dump([(s, t) for (s, _, t) in self.samples], f) # no need for tags
        with open(args.out + "/trainsents.txt", "w") as f:
            for sent in self.sents: f.write(sent.encode("utf-8") + "\n")
        with open(args.out + "/trainsents.conll", "w") as f:
            self._to_conll(f)

    @staticmethod
    def create_testdata(args):
        self = TrainingDataCreator(args.path,
                args.word_freq_cut, args.cat_freq_cut, args.afix_freq_cut)
        with open(args.out + "/log_create_{}data".format(args.subset), "w") as f:
            log(args, f)
        trees = walk_autodir(self.filepath, args.subset)
        self._create_samples(trees)
        with open(args.out + "/{}data.json".format(args.subset), "w") as f:
            json.dump([(s, t) for (s, _, t) in self.samples], f)
        with open(args.out + "/{}sents.txt".format(args.subset), "w") as f:
            for sent in self.sents: f.write(sent.encode("utf-8") + "\n")
        with open(args.out + "/{}sents.conll".format(args.subset), "w") as f:
            self._to_conll(f)


class FeatureExtractor(object):
    def __init__(self, model_path, length=False):
        self.words = read_model_defs(model_path + "/words.txt")
        self.suffixes = read_model_defs(model_path + "/suffixes.txt")
        self.prefixes = read_model_defs(model_path + "/prefixes.txt")
        self.unk_word = self.words[UNK]
        self.start_word = self.words[START]
        self.end_word = self.words[END]
        self.unk_suf = self.suffixes[UNK]
        self.unk_prf = self.prefixes[UNK]
        self.start_pre = [[self.prefixes[START]] + [IGNORE] * 3]
        self.start_suf = [[self.suffixes[START]] + [IGNORE] * 3]
        self.end_pre = [[self.prefixes[END]] + [IGNORE] * 3]
        self.end_suf = [[self.suffixes[END]] + [IGNORE] * 3]
        self.length = length

    def process(self, words):
        """
        words: list of unicode tokens
        """
        words = list(map(normalize, words))
        w = np.array([self.start_word] + [self.words.get(
            x.lower(), self.unk_word) for x in words] + [self.end_word], 'i')
        s = np.asarray(self.start_suf + [[self.suffixes.get(
            f, self.unk_suf) for f in get_suffix(x)] for x in words] + self.end_suf, 'i')
        p = np.asarray(self.start_pre + [[self.prefixes.get(
            f, self.unk_prf) for f in get_prefix(x)] for x in words] + self.end_pre, 'i')
        if not self.length:
            return w, s, p
        else:
            return w, s, p, w.shape[0]


class LSTMParserDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path, length=False):
        self.model_path = model_path
        self.targets = read_model_defs(model_path + "/target.txt")
        self.extractor = FeatureExtractor(model_path, length)
        with open(samples_path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        words, [cats, deps] = self.samples[i]
        feat = self.extractor.process(words.split(" "))
        cats = np.array([IGNORE] + [self.targets.get(x, IGNORE) for x in cats] + [IGNORE], 'i')
        deps = np.array([IGNORE] + deps + [IGNORE], 'i')
        return feat + (cats, deps)


class LSTMParserTriTrainDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, ccgbank_path, tritrain_path, weight, length=False):
        self.model_path = model_path
        self.targets = read_model_defs(model_path + "/target.txt")
        self.extractor = FeatureExtractor(model_path, length)
        self.weight = weight
        self.ncopies = 15
        with open(ccgbank_path) as f:
            self.ccgbank_samples = json.load(f)
            self.ccgbank_size = len(self.ccgbank_samples)
        with open(tritrain_path) as f:
            self.tritrain_samples = json.load(f)
            self.tritrain_size = len(self.tritrain_samples)

        print("len(ccgbank):", self.ccgbank_size, file=sys.stderr)
        print("len(ccgbank) * # copies:", self.ccgbank_size * self.ncopies, file=sys.stderr)
        print("len(tritrain):", self.tritrain_size, file=sys.stderr)

    def __len__(self):
        # some copies of ccgbank corpus plus tritrain dataset
        return self.ccgbank_size * self.ncopies + self.tritrain_size

    def get_example(self, i):
        if i < self.tritrain_size:
            sent, [cats, deps] = self.tritrain_samples[i]
            words = sent.split(" ")
            # tri-train dataset is noisy and contains unwanted zero-length word ...
            if any(len(w) == 0 for w in words):
                print("ignore sentence:", sent, file=sys.stderr)
                return self.get_example(random.randint(0, len(self)))
            weight = np.array(self.weight, 'f')
        else:
            words, [cats, deps] = self.ccgbank_samples[(i - self.tritrain_size) % self.ccgbank_size]
            words = words.split(" ")
            weight = np.array(1., 'f')

        feat = self.extractor.process(words)
        cats = np.array([IGNORE] + [self.targets.get(x, IGNORE) for x in cats] + [IGNORE], 'i')
        deps = np.array([IGNORE] + deps + [IGNORE], 'i')
        return feat + (cats, deps, weight)


class LSTMParser(chainer.Chain):
    def __init__(self, model_path, word_dim=None, afix_dim=None, nlayers=2,
            hidden_dim=128, elu_dim=64, dep_dim=100, dropout_ratio=0.5):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            self.train = False
            Param.load(self, defs_file)
            self.extractor = FeatureExtractor(model_path)
        else:
            # training
            self.train = True
            p = Param(self)
            p.dep_dim = dep_dim
            p.word_dim = word_dim
            p.afix_dim = afix_dim
            p.hidden_dim = hidden_dim
            p.elu_dim = elu_dim
            p.nlayers = nlayers
            p.n_words = len(read_model_defs(model_path + "/words.txt"))
            p.n_suffixes = len(read_model_defs(model_path + "/suffixes.txt"))
            p.n_prefixes = len(read_model_defs(model_path + "/prefixes.txt"))
            p.targets = read_model_defs(model_path + "/target.txt")
            p.dump(defs_file)

        self.in_dim = self.word_dim + 8 * self.afix_dim
        self.dropout_ratio = dropout_ratio
        super(LSTMParser, self).__init__(
                emb_word=L.EmbedID(self.n_words, self.word_dim),
                emb_suf=L.EmbedID(self.n_suffixes, self.afix_dim, ignore_label=IGNORE),
                emb_prf=L.EmbedID(self.n_prefixes, self.afix_dim, ignore_label=IGNORE),
                lstm_f=L.NStepLSTM(nlayers, self.in_dim,
                    self.hidden_dim, self.dropout_ratio),
                lstm_b=L.NStepLSTM(nlayers, self.in_dim,
                    self.hidden_dim, self.dropout_ratio),
                linear_cat1=L.Linear(2 * self.hidden_dim, self.elu_dim),
                linear_cat2=L.Linear(self.elu_dim, len(self.targets)),
                linear_dep=L.Linear(2 * self.hidden_dim, self.dep_dim),
                linear_head=L.Linear(2 * self.hidden_dim, self.dep_dim),
                biaffine=Biaffine(self.dep_dim)
                )

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def __call__(self, xs):
        """
        xs [(w,s,p,y), ..., ]
        w: word, c: char, l: length, y: label
        """
        batchsize = len(xs)
        ws, ss, ps, cat_ts, dep_ts = zip(*xs)
        cat_ys, dep_ys = self.forward(ws, ss, ps)

        cat_loss = reduce(lambda x, y: x + y,
            [F.softmax_cross_entropy(y, t) for y, t in zip(cat_ys, cat_ts)])
        cat_acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(cat_ys, cat_ts)])

        dep_loss = reduce(lambda x, y: x + y,
            [F.softmax_cross_entropy(y, t) for y, t in zip(dep_ys, dep_ts)])
        dep_acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(dep_ys, dep_ts)])


        cat_acc /= batchsize
        dep_acc /= batchsize
        chainer.report({
            "tagging_loss": cat_loss,
            "tagging_accuracy": cat_acc,
            "parsing_loss": dep_loss,
            "parsing_accuracy": dep_acc
            }, self)
        return cat_loss + dep_loss

    def forward(self, ws, ss, ps):
        batchsize = len(ws)
        xp = chainer.cuda.get_array_module(ws[0])
        ws = map(self.emb_word, ws)
        ss = [F.reshape(self.emb_suf(s), (s.shape[0], 4 * self.afix_dim)) for s in ss]
        ps = [F.reshape(self.emb_prf(s), (s.shape[0], 4 * self.afix_dim)) for s in ps]
        xs_f = [F.dropout(F.concat([w, s, p]),
            self.dropout_ratio, train=self.train) for w, s, p in zip(ws, ss, ps)]
        xs_b = [x[::-1] for x in xs_f]
        cx_f, hx_f, cx_b, hx_b = self._init_state(xp, batchsize)
        _, _, hs_f = self.lstm_f(hx_f, cx_f, xs_f, train=self.train)
        _, _, hs_b = self.lstm_b(hx_b, cx_b, xs_b, train=self.train)
        hs_b = [x[::-1] for x in hs_b]
        # ys: [(sentence length, number of category)]
        hs = [F.concat([h_f, h_b]) for h_f, h_b in zip(hs_f, hs_b)]

        cat_ys = [self.linear_cat2(
            F.dropout(F.elu(self.linear_cat1(h)), 0.5, train=self.train)) for h in hs]

        dep_ys = [self.biaffine(
            F.elu(F.dropout(self.linear_dep(h), 0.32, train=self.train)),
            F.elu(F.dropout(self.linear_head(h), 0.32, train=self.train))) for h in hs]

        return cat_ys, dep_ys

    def predict(self, xs):
        """
        batch: list of splitted sentences
        """
        xs = [self.extractor.process(x) for x in xs]
        ws, ss, ps = zip(*xs)
        cat_ys, dep_ys = self.forward(ws, ss, ps)
        return zip([y.data[1:-1] for y in cat_ys],
                [F.log_softmax(y[1:-1, :-1]).data for y in dep_ys])

    def predict_doc(self, doc, batchsize=16):
        """
        doc list of splitted sentences
        """
        res = []
        for i in range(0, len(doc), batchsize):
            res.extend([(i + j, 0, y)
                for j, y in enumerate(self.predict(doc[i:i + batchsize]))])
        return res

    def _init_state(self, xp, batchsize):
        res = [Variable(xp.zeros(( # forward cx, hx, backward cx, hx
                self.nlayers, batchsize, self.hidden_dim), 'f')) for _ in range(4)]
        return res

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]


def converter(xs, device):
    if device is None:
        return xs
    elif device < 0:
        return map(lambda x: map(lambda m: cuda.to_cpu(m), x), xs)
    else:
        return map(lambda x: map(
            lambda m: cuda.to_gpu(m, device, cuda.Stream.null), x), xs)


def train(args):
    model = LSTMParser(args.model, args.word_emb_size, args.afix_emb_size, args.nlayers,
            args.hidden_dim, args.elu_dim, args.dep_dim, args.dropout_ratio)
    with open(args.model + "/params", "w") as f: log(args, f)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    if args.pretrained:
        print('Load pretrained word embeddings from', args.pretrained)
        model.load_pretrained_embeddings(args.pretrained)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    train = LSTMParserDataset(args.model, args.train)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val = LSTMParserDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.Adam(beta2=0.9)
    # optimizer = chainer.optimizers.MomentumSGD(momentum=0.7)
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(1e-6))
    # optimizer.add_hook(GradientClipping(5.))
    updater = training.StandardUpdater(train_iter, optimizer,
            device=args.gpu, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 1000, 'iteration'
    log_interval = 200, 'iteration'

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(val_iter, eval_model,
                    converter, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/tagging_accuracy', 'main/tagging_loss',
        'main/parsing_accuracy', 'main/parsing_loss',
        'validation/main/tagging_accuracy',
        'validation/main/parsing_accuracy'
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
    parser_c.add_argument("--afix-freq-cut",
            type=int, default=5,
            help="only allow afixes which appear >= freq-cut")
    parser_c.add_argument("--subset",
            choices=["train", "test", "dev", "all"],
            default="train")
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
    parser_t.add_argument("--gpu", type=int, default=-1,
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
    parser_t.add_argument("--afix-emb-size",
            type=int, default=32,
            help="character embedding size")
    parser_t.add_argument("--nlayers",
            type=int, default=1,
            help="number of layers for each LSTM")
    parser_t.add_argument("--hidden-dim",
            type=int, default=128,
            help="dimensionality of hidden layer")
    parser_t.add_argument("--elu-dim",
            type=int, default=64,
            help="dimensionality of elu layer")
    parser_t.add_argument("--dep-dim",
            type=int, default=100,
            help="dim")
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
