
from __future__ import print_function
import sys
import numpy as np
import json
import chainer
import chainer.links as L
from py.my_iterator import SerialIterator
import chainer.functions as F
from chainer import cuda
from chainer import training, Variable
from chainer.dataset import concat_examples
from chainer.training import extensions
from chainer.optimizer import WeightDecay, GradientClipping
from py.ccgbank import walk_autodir
from py.japanese_ccg import JaCCGReader
from collections import defaultdict
from py.py_utils import read_pretrained_embeddings, read_model_defs
from py.tree import Leaf, Tree, get_leaves
from py.dyer_lstm import DyerLSTM
from py.lstm_tagger import UNK, OOR2, OOR3, OOR4, START, END, IGNORE, MISS
from py.lstm_tagger import log, get_suffix, get_prefix, normalize
from py.lstm_tagger import TrainingDataCreator, FeatureExtractor
from py.param import Param

class LSTMTaggerDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.targets = read_model_defs(model_path + "/target.txt")
        self.extractor = FeatureExtractor(model_path)
        with open(samples_path) as f:
            self.samples = sorted(
                    json.load(f).items(), key=lambda x: len(x[1]))

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        words, y = self.samples[i]
        w, s, p = self.extractor.process(words.split(" "))
        y = np.array([self.targets.get(x, IGNORE) for x in y], 'i')
        return w, s, p, y


class PeepHoleLSTMTagger(chainer.Chain):
    def __init__(self, model_path, word_dim=None, afix_dim=None,
            nlayers=2, hidden_dim=128, relu_dim=64, dropout_ratio=0.5):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            self.train = False
            Param.load(self, defs_file)
            self.extractor = FeatureExtractor(model_path)
        else:
            self.train = True
            p = Param(self)
            p.word_dim = word_dim
            p.afix_dim = afix_dim
            p.hidden_dim = hidden_dim
            p.relu_dim = relu_dim
            p.nlayers = nlayers
            p.dropout_ratio = dropout_ratio
            p.in_dim = self.word_dim + 8 * self.afix_dim
            p.n_words = len(read_model_defs(model_path + "/words.txt"))
            p.n_suffixes = len(read_model_defs(model_path + "/suffixes.txt"))
            p.n_prefixes = len(read_model_defs(model_path + "/prefixes.txt"))
            p.targets = read_model_defs(model_path + "/target.txt")
            p.dump(defs_file)

        super(PeepHoleLSTMTagger, self).__init__(
                emb_word=L.EmbedID(self.n_words, self.word_dim, ignore_label=IGNORE),
                emb_suf=L.EmbedID(self.n_suffixes, self.afix_dim, ignore_label=IGNORE),
                emb_prf=L.EmbedID(self.n_prefixes, self.afix_dim, ignore_label=IGNORE),
                lstm_f1=DyerLSTM(self.in_dim, self.hidden_dim),
                lstm_f2=DyerLSTM(self.hidden_dim, self.hidden_dim),
                lstm_b1=DyerLSTM(self.in_dim, self.hidden_dim),
                lstm_b2=DyerLSTM(self.hidden_dim, self.hidden_dim),
                linear1=L.Linear(2 * self.hidden_dim, self.relu_dim),
                linear2=L.Linear(self.relu_dim, len(self.targets)),
                )

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def __call__(self, ws, ss, ps, ts):
        """
        xs [(w,s,p,y), ..., ]
        w: word, s: suffix, p: prefix, y: label
        """
        batchsize, length = ts.shape
        ys = self.forward(ws, ss, ps)[1:-1]
        ts = [F.squeeze(x, 0) for x in F.split_axis(F.transpose(ts), length, 0)]
        loss = reduce(lambda x, y: x + y,
            [F.softmax_cross_entropy(y, t) for y, t in zip(ys, ts)])

        acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(ys, ts)])

        acc /= length
        chainer.report({
            "loss": loss,
            "accuracy": acc
            }, self)
        return loss

    def forward(self, ws, ss, ps):
        batchsize, length = ws.shape
        xp = chainer.cuda.get_array_module(ws[0])
        ws = self.emb_word(ws) # (batch, length, word_dim)
        ss = F.reshape(self.emb_suf(ss), (batchsize, length, -1))
        ps = F.reshape(self.emb_prf(ps), (batchsize, length, -1))
        hs = F.transpose(F.concat([ws, ss, ps], 2), (1, 0, 2))
        hs = F.dropout(hs, self.dropout_ratio, train=self.train)
        hs = F.split_axis(hs, length, 0)
        hs_f = []
        hs_b = []
        self._init_state()
        for h_in_f, h_in_b in zip(hs, reversed(hs)):
            h_f = self.lstm_f2(self.lstm_f1(F.squeeze(h_in_f, 0)))
            hs_f.append(h_f)
            h_b = self.lstm_b2(self.lstm_b1(F.squeeze(h_in_b, 0)))
            hs_b.append(h_b)

        ys = [self.linear2(F.relu(self.linear1(F.concat([h_f, h_b]))))
                for h_f, h_b in zip(hs_f, reversed(hs_b))]
        return ys

    def _init_state(self):
        self.lstm_f1.reset_state()
        self.lstm_f2.reset_state()
        self.lstm_b1.reset_state()
        self.lstm_b2.reset_state()

    def predict(self, xs):
        """
        batch: list of splitted sentences
        """
        batchsize = len(xs)
        fs = [self.extractor.process(x) for x in xs]
        ws, ss, ps = concat_examples(fs, padding=-1)
        ys = self.forward(ws, ss, ps)
        ys = F.transpose(F.stack(ys, 2), (0, 2, 1))
        return [F.squeeze(y, 0).data[1:len(x) + 1] for x, y in \
                zip(xs, F.split_axis(ys, batchsize, 0))]

    def predict_doc(self, doc, batchsize=16):
        """
        doc list of splitted sentences
        """
        res = []
        for i in range(0, len(doc), batchsize):
            res.extend([(i + j, 0, y)
                for j, y in enumerate(self.predict(doc[i:i + batchsize]))])
        return res

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]


def train(args):
    model = PeepHoleLSTMTagger(args.model, args.word_emb_size, args.afix_emb_size,
            args.nlayers, args.hidden_dim, args.relu_dim, args.dropout_ratio)
    with open(args.model + "/params", "w") as f:
            log(args, f)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    if args.pretrained:
        print('Load pretrained word embeddings from', args.pretrained)
        model.load_pretrained_embeddings(args.pretrained)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    converter = lambda x, device: \
            concat_examples(x, device=device, padding=-1)

    train = LSTMTaggerDataset(args.model, args.train)
    train_iter = SerialIterator(train, args.batchsize)
    val = LSTMTaggerDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.MomentumSGD(momentum=0.7)
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(1e-6))
    optimizer.add_hook(GradientClipping(5.))
    updater = training.StandardUpdater(train_iter, optimizer,
            device=args.gpu, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 1000, 'iteration'
    log_interval = 200, 'iteration'

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, converter, device=args.gpu), trigger=val_interval)
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
