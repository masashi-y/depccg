
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
from ccgbank import walk_autodir
from japanese_ccg import JaCCGReader
from collections import defaultdict
from py_utils import read_pretrained_embeddings, read_model_defs
from tree import Leaf, Tree, get_leaves

UNK = "*UNKNOWN*"
START = "*START*"
END = "*END*"
IGNORE = -1
MISS = -2

def log(args, out):
    for k, v in vars(args).items():
        out.write("{}: {}\n".format(k, v))

def get_suffix(word):
    if word == START or word == END: return word
    return (word[-2:] if len(word) > 1 else "_" + word[-1]).lower()


def get_prefix(word):
    if word == START or word == END: return word
    return (word[:2] if len(word) > 1 else "_" + word[0]).lower()


def normalize(word):
    if word == "-LRB-":
        return "("
    elif word == "-RRB-":
        return ")"
    else:
        return word


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
        self.words = defaultdict(int)
        self.prefixes = defaultdict(int)
        self.suffixes = defaultdict(int)
        self.samples = {}
        self.sents = []

        self.words[UNK]      = 10000
        self.words[START]    = 10000
        self.words[END]      = 10000
        self.suffixes[UNK]   = 10000
        self.suffixes[START] = 10000
        self.suffixes[END]   = 10000
        self.prefixes[UNK]   = 10000
        self.prefixes[START] = 10000
        self.prefixes[END]   = 10000
        self.cats[START]     = 10000
        self.cats[END]       = 10000

    def _traverse(self, tree):
        if isinstance(tree, Leaf):
            self.cats[str(tree.cat)] += 1
            w = normalize(tree.word)
            self.words[w] += 1
            self.suffixes[get_suffix(w)] += 1
            self.prefixes[get_prefix(w)] += 1
        else:
            children = tree.children
            if len(children) == 1:
                rule = str(tree.cat) + \
                        " " + str(children[0].cat)
                self.unary_rules[rule] += 1
                self._traverse(children[0])
            else:
                rule = str(children[0].cat) + \
                        " " + str(children[1].cat)
                self.seen_rules[rule] += 1
                self._traverse(children[0])
                self._traverse(children[1])

    @staticmethod
    def _write(dct, out, comment_out_value=False):
        print >> sys.stderr, "writing to", out.name
        for key, value in dct.items():
            out.write(key.encode("utf-8") + " ")
            if comment_out_value:
                out.write("# ")
            out.write(str(value) + "\n")

    def _create_samples(self, trees):
        for tree in trees:
            tokens = get_leaves(tree)
            words = [normalize(token.word) for token in tokens]
            cats = [token.cat.without_semantics for token in tokens]
            sent = " ".join(words)
            self.sents.append(sent)
            self.samples[sent] = " ".join(cats)

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
            json.dump(self.samples, f)
        with open(args.out + "/trainsents.txt", "w") as f:
            for sent in self.sents:
                f.write(sent.encode("utf-8") + "\n")

    @staticmethod
    def create_testdata(args):
        self = TrainingDataCreator(args.path,
                args.word_freq_cut, args.cat_freq_cut, args.afix_freq_cut)
        with open(args.out + "/log_create_testdata", "w") as f:
            log(args, f)
        trees = walk_autodir(self.filepath, args.subset)
        self._create_samples(trees)
        with open(args.out + "/testdata.json", "w") as f:
            json.dump(self.samples, f)
        with open(args.out + "/testsents.txt", "w") as f:
            for sent in self.sents:
                f.write(sent.encode("utf-8") + "\n")


class LSTMTaggerDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.words = read_model_defs(model_path + "/words.txt")
        self.suffixes = read_model_defs(model_path + "/suffixes.txt")
        self.prefixes = read_model_defs(model_path + "/prefixes.txt")
        self.targets = read_model_defs(model_path + "/target.txt")
        self.unk_word = self.words[UNK]
        self.unk_suf = self.suffixes[UNK]
        self.unk_prf = self.prefixes[UNK]
        with open(samples_path) as f:
            self.samples = json.load(f).items()

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        words, y = self.samples[i]
        words = [START] + words.split(" ") + [END]
        y = [START] + y.split(" ") + [END]
        w = np.array([self.words.get(x,
            self.words.get(x.lower(), self.unk_word)) for x in words], 'i')
        s = np.array([self.suffixes.get(get_suffix(x), self.unk_suf) for x in words], 'i')
        p = np.array([self.prefixes.get(get_prefix(x), self.unk_prf) for x in words], 'i')
        y = np.array([self.targets.get(x, IGNORE) for x in y], 'i')
        return w, s, p, y


class LSTMTagger(chainer.Chain):
    def __init__(self, model_path, word_dim, afix_dim, nlayers,
            hidden_dim, relu_dim, dropout_ratio=0.5):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            # use as supertagger
            with open(defs_file) as f:
                defs = json.load(f)
            self.word_dim = defs["word_dim"]
            self.afix_dim = defs["afix_dim"]
        else:
            # training
            self.word_dim = word_dim
            self.afix_dim = afix_dim
            with open(defs_file, "w") as f:
                json.dump({"model": self.__class__.__name__,
                           "word_dim": self.word_dim,
                           "afix_dim": self.afix_dim}, f)

        self.targets = read_model_defs(model_path + "/target.txt")
        self.words = read_model_defs(model_path + "/words.txt")
        self.suffixes = read_model_defs(model_path + "/suffixes.txt")
        self.prefixes = read_model_defs(model_path + "/prefixes.txt")
        self.in_dim = self.word_dim + 2 * self.afix_dim
        self.hidden_dim = hidden_dim
        self.relu_dim = relu_dim
        self.nlayers = nlayers
        self.dropout_ratio = dropout_ratio
        self.train = True
        super(LSTMTagger, self).__init__(
                emb_word=L.EmbedID(len(self.words), self.word_dim),
                emb_suf=L.EmbedID(len(self.suffixes), self.afix_dim),
                emb_prf=L.EmbedID(len(self.prefixes), self.afix_dim),
                lstm_f=L.NStepLSTM(nlayers, self.in_dim,
                    self.hidden_dim, 0.),
                lstm_b=L.NStepLSTM(nlayers, self.in_dim,
                    self.hidden_dim, 0.),
                linear1=L.Linear(2 * self.hidden_dim, self.relu_dim),
                linear2=L.Linear(self.relu_dim, len(self.targets)),
                )

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def __call__(self, xs):
        """
        xs [(w,s,p,y), ..., ]
        w: word, s: suffix, p: prefix, y: label
        """
        batchsize = len(xs)
        ws, ss, ps, ts = zip(*xs)
        ws = map(self.emb_word, ws)
        ss = map(self.emb_suf, ss)
        ps = map(self.emb_prf, ps)
        # [(sentence length, (word_dim + suf_dim + prf_dim))]
        xs_f = [F.dropout(F.concat([w, s, p]),
            self.dropout_ratio, train=self.train) for w, s, p in zip(ws, ss, ps)]
        xs_b = list(reversed(xs_f))
        cx_f, hx_f, cx_b, hx_b = self._init_state(batchsize)
        _, _, hs_f = self.lstm_f(hx_f, cx_f, xs_f, train=self.train)
        _, _, hs_b = self.lstm_b(hx_b, cx_b, xs_b, train=self.train)
        hs_b = list(reversed(hs_b))
        # ys: [(sentence length, number of category)]
        ys = [self.linear2(F.relu(
                self.linear1(F.concat([h_f, h_b]))))
                    for h_f, h_b in zip(hs_f, hs_b)]
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

    def _init_state(self, batchsize):
        res = [Variable(np.zeros(( # forward cx, hx, backward cx, hx
                self.nlayers, batchsize, self.hidden_dim), 'f')) for _ in range(4)]
        return res


def converter(x, device):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device, cuda.Stream.null)


def train(args):
    model = LSTMTagger(args.model, args.word_emb_size, args.afix_emb_size,
            args.nlayers, args.hidden_dim, args.relu_dim, args.dropout_ratio)
    with open(args.model + "/params", "w") as f:
            log(args, f)
    if args.initmodel:
        print 'Load model from', args.initmodel
        chainer.serializers.load_npz(args.initmodel, model)

    if args.pretrained:
        print 'Load pretrained word embeddings from', args.pretrained
        model.load_pretrained_embeddings(args.pretrained)

    train = LSTMTaggerDataset(args.model, args.train)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val = LSTMTaggerDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.MomentumSGD(momentum=0.7)
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(1e-6))
    optimizer.add_hook(GradientClipping(5.))
    updater = training.StandardUpdater(train_iter, optimizer, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 2000, 'iteration'
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
        'main/accuracy', 'validation/main/accuracy', 'validation/main/ignore_unseen',
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
