
import sys
import numpy as np
import json
import chainer
import chainer.links as L
import chainer.functions as F
from my_iterator import SerialIterator
from chainer import cuda
from chainer import training, Variable
from chainer.training import extensions
from chainer.optimizer import WeightDecay, GradientClipping
from chainer.dataset import concat_examples
from japanese_ccg import JaCCGReader
from collections import defaultdict, OrderedDict
from py_utils import read_pretrained_embeddings, read_model_defs
from tree import Leaf, Tree, get_leaves
from biaffine import Biaffine
from dyer_lstm import DyerLSTM
from param import Param

from ja_lstm_parser import UNK, START, END, IGNORE, log
from ja_lstm_parser import TrainingDataCreator, FeatureExtractor


class LSTMParserDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.extractor = FeatureExtractor(model_path)
        self.targets = read_model_defs(model_path + "/target.txt")
        with open(samples_path) as f:
            self.samples = sorted(
                    json.load(f).items(), key=lambda x: len(x[1][0]))

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        words, [cats, deps] = self.samples[i]
        words = words.split(" ")
        w, c, l = self.extractor.process(words)
        cats = np.array([self.targets.get(x, IGNORE) for x in cats], 'i')
        deps = np.array(deps, 'i')
        return w, c, cats, deps


class PeepHoleJaLSTMParser(chainer.Chain):
    def __init__(self, model_path, word_dim=None, char_dim=None, nlayers=2,
            hidden_dim=128, relu_dim=64, dep_dim=100, dropout_ratio=0.5):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            # use as supertagger
            self.train = False
            Param.load(self, defs_file)
            self.extractor = FeatureExtractor(model_path)
        else:
            # training
            self.train = True
            p = Param(self)
            p.dep_dim = dep_dim
            p.word_dim = word_dim
            p.char_dim = char_dim
            p.hidden_dim = hidden_dim
            p.relu_dim = relu_dim
            p.nlayers = nlayers
            p.n_words = len(read_model_defs(model_path + "/words.txt"))
            p.n_chars = len(read_model_defs(model_path + "/chars.txt"))
            p.targets = read_model_defs(model_path + "/target.txt")
            p.dump(defs_file)

        self.in_dim = self.word_dim + self.char_dim
        self.dropout_ratio = dropout_ratio
        super(PeepHoleJaLSTMParser, self).__init__(
                emb_word=L.EmbedID(self.n_words, self.word_dim),
                emb_char=L.EmbedID(self.n_chars, 50, ignore_label=IGNORE),
                conv_char=L.Convolution2D(1, self.char_dim,
                    (3, 50), stride=1, pad=(1, 0)),
                lstm_f1=DyerLSTM(self.in_dim, self.hidden_dim),
                lstm_f2=DyerLSTM(self.hidden_dim, self.hidden_dim),
                lstm_b1=DyerLSTM(self.in_dim, self.hidden_dim),
                lstm_b2=DyerLSTM(self.hidden_dim, self.hidden_dim),
                linear_cat1=L.Linear(2 * self.hidden_dim, self.relu_dim),
                linear_cat2=L.Linear(self.relu_dim, len(self.targets)),
                linear_dep=L.Linear(2 * self.hidden_dim, self.dep_dim),
                linear_head=L.Linear(2 * self.hidden_dim, self.dep_dim),
                biaffine=Biaffine(self.dep_dim)
                )

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def forward(self, ws, cs):
        batchsize, length, max_word_len = cs.shape
        ws = self.emb_word(ws) # (batch, length, word_dim)
        cs = F.reshape(
            F.max_pooling_2d(
                self.conv_char(
                    F.reshape(
                        self.emb_char(cs),
                        (batchsize * length, 1, max_word_len, 50))), (max_word_len, 1)),
                    (batchsize, length, self.char_dim))

        hs = F.transpose(F.concat([ws, cs], 2), (1, 0, 2))
        hs = F.dropout(hs, self.dropout_ratio, train=self.train)
        hs = F.split_axis(hs, length, 0)
        hs_f = []
        hs_b = []
        self._init_state()
        for h_in_f, h_in_b in zip(hs, reversed(hs)):
            h_f = self.lstm_f2(self.lstm_f1(F.reshape(h_in_f, (batchsize, -1))))
            hs_f.append(h_f)
            h_b = self.lstm_b2(self.lstm_b1(F.reshape(h_in_b, (batchsize, -1))))
            hs_b.append(h_b)

        hs = [F.concat([h_f, h_b]) for h_f, h_b in zip(hs_f, reversed(hs_b))]

        cat_ys = [self.linear_cat2(F.dropout(
            F.elu(self.linear_cat1(h)), 0.5, train=self.train)) for h in hs]

        hs = [F.reshape(h, (length, -1)) for h in \
                F.split_axis(F.transpose(F.stack(hs, 2), (0, 2, 1)), batchsize, 0)]

        dep_ys = [self.biaffine(
            F.relu(F.dropout(self.linear_dep(h), 0.32, train=self.train)),
            F.relu(F.dropout(self.linear_head(h), 0.32, train=self.train))) for h in hs]
        return cat_ys, dep_ys

    def __call__(self, ws, cs, cat_ts, dep_ts):
        batchsize, length = cat_ts.shape
        cat_ys, dep_ys = self.forward(ws, cs)
        cat_ys = cat_ys[1:-1]
        cat_ts = [F.reshape(x, (batchsize,)) for x \
                in F.split_axis(F.transpose(cat_ts), length, 0)]
        assert len(cat_ys) == len(cat_ts)
        cat_loss = reduce(lambda x, y: x + y,
            [F.softmax_cross_entropy(y, t) for y, t in zip(cat_ys, cat_ts)])
        cat_acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(cat_ys, cat_ts)])


        # hs [(length, hidden_dim), ...]
        dep_ys = [x[1:-1] for x in dep_ys]
        dep_ts = [F.reshape(x, (length,)) for x in F.split_axis(dep_ts, batchsize, 0)]

        dep_loss = reduce(lambda x, y: x + y,
            [F.softmax_cross_entropy(y, t) for y, t in zip(dep_ys, dep_ts)])
        dep_acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(dep_ys, dep_ts)])

        cat_acc /= length
        dep_acc /= batchsize
        chainer.report({
            "tagging_loss": cat_loss,
            "tagging_accuracy": cat_acc,
            "parsing_loss": dep_loss,
            "parsing_accuracy": dep_acc
            }, self)
        return cat_loss + dep_loss

    def predict(self, xs):
        """
        batch: list of splitted sentences
        """
        batchsize = len(xs)
        fs = [self.extractor.process(x)[:2] for x in xs]
        ws, cs = concat_examples(fs, padding=IGNORE)
        cat_ys, dep_ys = self.forward(ws, cs)
        cat_ys = F.transpose(F.stack(cat_ys, 2), (0, 2, 1))
        # dep_ys = F.transpose(F.stack(dep_ys, 2), (0, 2, 1))

        cat_ys = [F.log_softmax(
            F.reshape(y, (y.shape[1], -1))[1:len(x) + 1]).data for x, y in \
                zip(xs, F.split_axis(cat_ys, batchsize, 0))]

        dep_ys = [F.log_softmax(y[1:len(x) + 1, :len(x) + 1]).data \
                for x, y in zip(xs, dep_ys)]
        assert len(cat_ys) == len(dep_ys)
        return zip(cat_ys, dep_ys)

    def predict_doc(self, doc, batchsize=16):
        """
        doc list of splitted sentences
        """
        res = []
        for i in xrange(0, len(doc), batchsize):
            res.extend([(i + j, 0, y)
                for j, y in enumerate(self.predict(doc[i:i + batchsize]))])
        return res

    def _init_state(self):
        self.lstm_f1.reset_state()
        self.lstm_f2.reset_state()
        self.lstm_b1.reset_state()
        self.lstm_b2.reset_state()

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]


def train(args):
    model = PeepHoleJaLSTMParser(args.model, args.word_emb_size, args.char_emb_size,
            args.nlayers, args.hidden_dim, args.relu_dim, args.dep_dim, args.dropout_ratio)

    with open(args.model + "/params", "w") as f: log(args, f)

    if args.initmodel:
        print 'Load model from', args.initmodel
        chainer.serializers.load_npz(args.initmodel, model)

    if args.pretrained:
        print 'Load pretrained word embeddings from', args.pretrained
        model.load_pretrained_embeddings(args.pretrained)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()


    converter = lambda x, device: \
            concat_examples(x, device=device, padding=-1)

    train = LSTMParserDataset(args.model, args.train)
    train_iter = SerialIterator(train, args.batchsize)
    val = LSTMParserDataset(args.model, args.val)
    val_iter = SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.Adam(beta2=0.9)
    # optimizer = chainer.optimizers.MomentumSGD(momentum=0.7)
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(1e-6))
    # optimizer.add_hook(GradientClipping(5.))
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 2000, 'iteration'
    log_interval = 200, 'iteration'

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(val_iter, eval_model,
                    converter, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/tagging_loss',
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
    parser_t.add_argument("--gpu", type=int, default=-1,
            help="path to model directory")
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
