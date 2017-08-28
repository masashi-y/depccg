
from __future__ import print_function
import sys
import numpy as np
import json
import random
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import training, Variable
from chainer.training import extensions
from chainer.optimizer import WeightDecay, GradientClipping
from chainer.dataset.convert import _concat_arrays
from py.ccgbank import walk_autodir
from py.japanese_ccg import JaCCGReader
from collections import defaultdict, OrderedDict
from py.py_utils import read_pretrained_embeddings, read_model_defs
from py.tree import Leaf, Tree, get_leaves
from py.biaffine import Biaffine, Bilinear
from py.param import Param
from py.fixed_length_n_step_lstm import FixedLengthNStepLSTM

from py.lstm_parser_bi import IGNORE, log, TrainingDataCreator, \
        FeatureExtractor, LSTMParserDataset, LSTMParserTriTrainDataset


class Linear(L.Linear):

    def __call__(self, x):
        shape = x.shape
        if len(shape) == 3:
            x = F.reshape(x, (-1, shape[2]))
        y = super(Linear, self).__call__(x)
        if len(shape) == 3:
            y = F.reshape(y, (shape[0], shape[1], -1))
        return y


def concat_examples(batch, device=None):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    result = [to_device(_concat_arrays([s[0] for s in batch], -1)), # ws
              to_device(_concat_arrays([s[1] for s in batch], -1)), # ps
              to_device(_concat_arrays([s[2] for s in batch], -1)), # ss
              [s[3] for s in batch]]                                # ls

    if len(batch[0]) == 7:
        result.append([to_device(s[4]) for s in batch])            # cat_ts
        result.append([to_device(s[5]) for s in batch])            # dep_ts
        result.append(to_device(_concat_arrays([s[6] for s in batch], None))) # weights

    return tuple(result)


class FastBiaffineLSTMParser(chainer.Chain):
    """
    chainer.links.Bilinear may have some problem with GPU
    and results in nan with batches with big size
    this implementation uses different implementation of bilinear
    and does not run into nan.
    """
    def __init__(self, model_path, word_dim=None, afix_dim=None, nlayers=2,
            hidden_dim=128, dep_dim=100, dropout_ratio=0.5):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            self.train = False
            Param.load(self, defs_file)
            self.extractor = FeatureExtractor(model_path, length=True)
        else:
            # training
            self.train = True
            p = Param(self)
            p.dep_dim = dep_dim
            p.word_dim = word_dim
            p.afix_dim = afix_dim
            p.hidden_dim = hidden_dim
            p.nlayers = nlayers
            p.n_words = len(read_model_defs(model_path + "/words.txt"))
            p.n_suffixes = len(read_model_defs(model_path + "/suffixes.txt"))
            p.n_prefixes = len(read_model_defs(model_path + "/prefixes.txt"))
            p.targets = read_model_defs(model_path + "/target.txt")
            p.dump(defs_file)

        self.in_dim = self.word_dim + 8 * self.afix_dim
        self.dropout_ratio = dropout_ratio
        super(FastBiaffineLSTMParser, self).__init__(
                emb_word=L.EmbedID(self.n_words, self.word_dim, ignore_label=IGNORE),
                emb_suf=L.EmbedID(self.n_suffixes, self.afix_dim, ignore_label=IGNORE),
                emb_prf=L.EmbedID(self.n_prefixes, self.afix_dim, ignore_label=IGNORE),
                lstm_f=FixedLengthNStepLSTM(self.nlayers, self.in_dim, self.hidden_dim, 0.32),
                lstm_b=FixedLengthNStepLSTM(self.nlayers, self.in_dim, self.hidden_dim, 0.32),
                arc_dep=Linear(2 * self.hidden_dim, self.dep_dim),
                arc_head=Linear(2 * self.hidden_dim, self.dep_dim),
                rel_dep=Linear(2 * self.hidden_dim, self.dep_dim),
                rel_head=Linear(2 * self.hidden_dim, self.dep_dim),
                biaffine_arc=Biaffine(self.dep_dim),
                biaffine_tag=Bilinear(self.dep_dim, self.dep_dim, len(self.targets)))

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def __call__(self, xs):
        """
        xs [(w,s,p,y), ..., ]
        w: word, c: char, l: length, y: label
        """
        batchsize = len(xs)

        if len(xs[0]) == 6:
            ws, ss, ps, ls, cat_ts, dep_ts = zip(*xs)
            xp = chainer.cuda.get_array_module(ws[0])
            weights = [xp.array(1, 'f') for _ in xs]
        else:
            ws, ss, ps, ls, cat_ts, dep_ts, weights = zip(*xs)

        cat_ys, dep_ys = self.forward(ws, ss, ps, ls, dep_ts if self.train else None)

        cat_loss = reduce(lambda x, y: x + y,
            [we * F.softmax_cross_entropy(y, t) \
                    for y, t, we  in zip(cat_ys, cat_ts, weights)])
        cat_acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(cat_ys, cat_ts)]) / batchsize

        dep_loss = reduce(lambda x, y: x + y,
            [we * F.softmax_cross_entropy(y, t) \
                    for y, t, we in zip(dep_ys, dep_ts, weights)])
        dep_acc = reduce(lambda x, y: x + y,
            [F.accuracy(y, t, ignore_label=IGNORE) for y, t in zip(dep_ys, dep_ts)]) / batchsize

        chainer.report({
            "tagging_loss": cat_loss,
            "tagging_accuracy": cat_acc,
            "parsing_loss": dep_loss,
            "parsing_accuracy": dep_acc
            }, self)
        return cat_loss + dep_loss

    def forward(self, ws, ss, ps, ls, dep_ts=None):
        batchsize, slen = ws.shape
        xp = chainer.cuda.get_array_module(ws[0])

        wss = self.emb_word(ws)
        sss = F.reshape(self.emb_suf(ss), (batchsize, slen, 4 * self.afix_dim))
        pss = F.reshape(self.emb_prf(ps), (batchsize, slen, 4 * self.afix_dim))
        ins = F.dropout(F.concat([wss, sss, pss], 2), self.dropout_ratio, train=self.train)
        xs_f = F.transpose(ins, (1, 0, 2))
        xs_b = xs_f[::-1]

        cx_f, hx_f, cx_b, hx_b = self._init_state(xp, batchsize)
        _, _, hs_f = self.lstm_f(hx_f, cx_f, xs_f, train=self.train)
        _, _, hs_b = self.lstm_b(hx_b, cx_b, xs_b, train=self.train)

        # (batch, length, hidden_dim)
        hs = F.transpose(F.concat([hs_f, hs_b[::-1]], 2), (1, 0, 2))

        dep_ys = self.biaffine_arc(
            F.elu(F.dropout(self.arc_dep(hs), 0.32, train=self.train)),
            F.elu(F.dropout(self.arc_head(hs), 0.32, train=self.train)))

        if dep_ts is not None and random.random >= 0.5:
            heads = dep_ts
        else:
            heads = F.flatten(F.argmax(dep_ys, axis=2)) + \
                    xp.repeat(xp.arange(0, batchsize * slen, slen), slen)

        hs = F.reshape(hs, (batchsize * slen, -1))
        heads = F.permutate(
                    F.elu(F.dropout(
                        self.rel_head(hs), 0.32, train=self.train)), heads)

        childs = F.elu(F.dropout(self.rel_dep(hs), 0.32, train=self.train))
        cat_ys = self.biaffine_tag(childs, heads)

        dep_ys = F.split_axis(dep_ys, batchsize, 0) if batchsize > 1 else [dep_ys]
        dep_ys = [F.reshape(v, v.shape[1:])[:l, :l] for v, l in zip(dep_ys, ls)]

        cat_ys = F.split_axis(cat_ys, batchsize, 0) if batchsize > 1 else [cat_ys]
        cat_ys = [v[:l] for v, l in zip(cat_ys, ls)]

        return cat_ys, dep_ys

    def predict(self, xs):
        """
        batch: list of splitted sentences
        """
        xs = [self.extractor.process(x) for x in xs]
        ws, ss, ps, ls = concat_examples(xs)
        cat_ys, dep_ys = self.forward(ws, ss, ps, ls)
        return zip([F.log_softmax(y[1:-1]).data for y in cat_ys],
                [F.log_softmax(y[1:-1, :-1]).data for y in dep_ys])

    def predict_doc(self, doc, batchsize=32):
        """
        doc list of splitted sentences
        """
        res = []
        doc = sorted(enumerate(doc), key=lambda x: len(x[1]))
        for i in range(0, len(doc), batchsize):
            ids, batch = zip(*doc[i:i + batchsize])
            pred = self.predict(batch)
            res.extend([(j, 0, y) for j, y in zip(ids, pred)])
        return res

    def _init_state(self, xp, batchsize):
        res = [Variable(xp.zeros(( # forward cx, hx, backward cx, hx
                self.nlayers, batchsize, self.hidden_dim), 'f')) for _ in range(4)]
        return res

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]


# def converter(xs, device):
#     if device is None:
#         return xs
#     elif device < 0:
#         return map(lambda x: map(lambda m: cuda.to_cpu(m), x), xs)
#     else:
#         return map(lambda x: map(
#             lambda m: cuda.to_gpu(m, device, cuda.Stream.null), x), xs)
#
#
# def train(args):
#     model = FastBiaffineLSTMParser(
#             args.model, args.word_emb_size, args.afix_emb_size, args.nlayers,
#             args.hidden_dim, args.dep_dim, args.dropout_ratio)
#     with open(args.model + "/params", "w") as f: log(args, f)
#
#     if args.initmodel:
#         print('Load model from', args.initmodel)
#         chainer.serializers.load_npz(args.initmodel, model)
#
#     if args.pretrained:
#         print('Load pretrained word embeddings from', args.pretrained)
#         model.load_pretrained_embeddings(args.pretrained)
#
#     if args.gpu >= 0:
#         chainer.cuda.get_device(args.gpu).use()
#         model.to_gpu()
#
#     if args.tritrain is not None:
#         train = LSTMParserTriTrainDataset(
#                 args.model, args.train, args.tritrain, args.tri_weight)
#     else:
#         train = LSTMParserDataset(args.model, args.train)
#
#     train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
#     val = LSTMParserDataset(args.model, args.val)
#     val_iter = chainer.iterators.SerialIterator(
#             val, args.batchsize, repeat=False, shuffle=False)
#     optimizer = chainer.optimizers.Adam(beta2=0.9)
#     # optimizer = chainer.optimizers.MomentumSGD(momentum=0.7)
#     optimizer.setup(model)
#     optimizer.add_hook(WeightDecay(2e-6))
#     optimizer.add_hook(GradientClipping(15.))
#     updater = training.StandardUpdater(train_iter, optimizer,
#             device=args.gpu, converter=converter)
#     trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)
#
#     val_interval = 1000, 'iteration'
#     log_interval = 200, 'iteration'
#
#     eval_model = model.copy()
#     eval_model.train = False
#
#     trainer.extend(extensions.ExponentialShift(
#                     "eps", .75, init=2e-3, optimizer=optimizer), trigger=(2500, 'iteration'))
#     trainer.extend(extensions.Evaluator(val_iter, eval_model,
#                     converter, device=args.gpu), trigger=val_interval)
#     trainer.extend(extensions.snapshot_object(
#         model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
#     trainer.extend(extensions.LogReport(trigger=log_interval))
#     trainer.extend(extensions.observe_lr(observation_key="eps"), trigger=log_interval)
#     trainer.extend(extensions.PrintReport([
#         'epoch', 'iteration',
#         'main/tagging_accuracy', 'main/tagging_loss',
#         'main/parsing_accuracy', 'main/parsing_loss',
#         'validation/main/tagging_accuracy',
#         'validation/main/parsing_accuracy',
#         'eps'
#     ]), trigger=log_interval)
#     trainer.extend(extensions.ProgressBar(update_interval=10))
#
#     trainer.run()
#
#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(
#                 "CCG parser's LSTM supertag tagger")
#     subparsers = parser.add_subparsers()
#
#     # Creating training data
#     parser_c = subparsers.add_parser(
#             "create", help="create tagger input data")
#     parser_c.add_argument("path",
#             help="path to ccgbank data file")
#     parser_c.add_argument("out",
#             help="output directory path")
#     parser_c.add_argument("--cat-freq-cut",
#             type=int, default=10,
#             help="only allow categories which appear >= freq-cut")
#     parser_c.add_argument("--word-freq-cut",
#             type=int, default=5,
#             help="only allow words which appear >= freq-cut")
#     parser_c.add_argument("--afix-freq-cut",
#             type=int, default=5,
#             help="only allow afixes which appear >= freq-cut")
#     parser_c.add_argument("--subset",
#             choices=["train", "test", "dev", "all"],
#             default="train")
#     parser_c.add_argument("--mode",
#             choices=["train", "test"],
#             default="train")
#
#     parser_c.set_defaults(func=
#             (lambda args:
#                 TrainingDataCreator.create_traindata(args)
#                     if args.mode == "train"
#                 else  TrainingDataCreator.create_testdata(args)))
#
#             #TODO updater
#     # Do training using training data created through `create`
#     parser_t = subparsers.add_parser(
#             "train", help="train supertagger model")
#     parser_t.add_argument("model",
#             help="path to model directory")
#     parser_t.add_argument("--gpu", type=int, default=-1,
#             help="path to model directory")
#     parser_t.add_argument("train",
#             help="training data file path")
#     parser_t.add_argument("--tritrain",
#             help="tri-training data file path")
#     parser_t.add_argument("--tri-weight",
#             type=float, default=0.4,
#             help="multiply tri-training sample losses")
#     parser_t.add_argument("val",
#             help="validation data file path")
#     parser_t.add_argument("--batchsize",
#             type=int, default=16, help="batch size")
#     parser_t.add_argument("--epoch",
#             type=int, default=20, help="epoch")
#     parser_t.add_argument("--word-emb-size",
#             type=int, default=50,
#             help="word embedding size")
#     parser_t.add_argument("--afix-emb-size",
#             type=int, default=32,
#             help="character embedding size")
#     parser_t.add_argument("--nlayers",
#             type=int, default=1,
#             help="number of layers for each LSTM")
#     parser_t.add_argument("--hidden-dim",
#             type=int, default=128,
#             help="dimensionality of hidden layer")
#     parser_t.add_argument("--dep-dim",
#             type=int, default=100,
#             help="dim")
#     parser_t.add_argument("--dropout-ratio",
#             type=float, default=0.5,
#             help="dropout ratio")
#     parser_t.add_argument("--initmodel",
#             help="initialize model with `initmodel`")
#     parser_t.add_argument("--pretrained",
#             help="pretrained word embeddings")
#     parser_t.set_defaults(func=train)
#
#     args = parser.parse_args()
#
#     args.func(args)
