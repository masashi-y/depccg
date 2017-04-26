
import copy
import sys
import numpy as np
import json
import chainer
import chainer.links as L
import chainer.functions as F
from my_iterator import SerialIterator
from chainer import cuda
from chainer import training, Variable, ChainList
from chainer.training import extensions
from chainer.optimizer import WeightDecay, GradientClipping
from chainer.dataset.convert import _concat_arrays
from ccgbank import walk_autodir
from japanese_ccg import JaCCGReader
from collections import defaultdict, OrderedDict
from py_utils import read_pretrained_embeddings, read_model_defs
from tree import Leaf, Tree, get_leaves
from biaffine import Biaffine, Bilinear
from dyer_lstm import DyerLSTM
from param import Param
from qrnn import QRNNLayer

from lstm_tagger import UNK, OOR2, OOR3, OOR4, START, END, IGNORE, MISS
from lstm_tagger import log, get_suffix, get_prefix, normalize
from lstm_parser import TrainingDataCreator, FeatureExtractor
from lstm_parser import LSTMParser, LSTMParserTriTrainDataset

def scanl(f, base, l):
    res = [base]
    acc = base
    for x in l:
        acc = f(acc, x)
        res += [acc]
    return res


class LSTMParserDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.targets = read_model_defs(model_path + "/target.txt")
        self.extractor = FeatureExtractor(model_path)
        with open(samples_path) as f:
            self.samples = sorted(
                    json.load(f), key=lambda x: len(x[1][0]))

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        words, [cats, deps] = self.samples[i]
        splitted = words.split(" ")
        w, s, p = self.extractor.process(splitted)
        cats = np.array([-1] + [self.targets.get(x, IGNORE) for x in cats] + [-1], 'i')
        deps = np.array([-1] + deps + [-1], 'i')
        l = len(splitted) + 2
        weight = np.array(1, 'f')
        return w, s, p, l, cats, deps, weight


class QRNNTriTrainDataset(LSTMParserTriTrainDataset):
    def __init__(self, model_path, ccgbank_path, tritrain_path, weight):
        self.model_path = model_path
        self.targets = read_model_defs(model_path + "/target.txt")
        self.extractor = FeatureExtractor(model_path)
        self.weight = weight
        self.ncopies = 15
        with open(ccgbank_path) as f:
            self.ccgbank_samples = sorted(
                    json.load(f), key=lambda x: len(x[1][0]))
            self.ccgbank_size = len(self.ccgbank_samples)
        with open(tritrain_path) as f:
            self.tritrain_samples = sorted(
                    json.load(f), key=lambda x: len(x[1][0]))
            self.tritrain_size = len(self.tritrain_samples)

        print >> sys.stderr, "len(ccgbank):", self.ccgbank_size
        print >> sys.stderr, "len(ccgbank) * # copies:", self.ccgbank_size * self.ncopies
        print >> sys.stderr, "len(tritrain):", self.tritrain_size

    def get_example(self, i):
        w, s, p, cats, deps, weight = super(QRNNTriTrainDataset, self).get_example(i)
        l = w.shape[0]
        return w, s, p, l, cats, deps, weight


class QRNNParser(chainer.Chain):
    def __init__(self, model_path, word_dim=None, afix_dim=None, nlayers=2,
            hidden_dim=128, elu_dim=64, dep_dim=100, dropout_ratio=0.5, use_cudnn=False):
        self.model_path = model_path
        defs_file = model_path + "/tagger_defs.txt"
        if word_dim is None:
            self.train = False
            Param.load(self, defs_file)
            self.extractor = FeatureExtractor(model_path)
        else:
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
        super(QRNNParser, self).__init__(
                emb_word=L.EmbedID(self.n_words, self.word_dim, ignore_label=IGNORE),
                emb_suf=L.EmbedID(self.n_suffixes, self.afix_dim, ignore_label=IGNORE),
                emb_prf=L.EmbedID(self.n_prefixes, self.afix_dim, ignore_label=IGNORE),
                qrnn_fs=ChainList(),
                qrnn_bs=ChainList(),
                arc_dep=L.Linear(2 * self.hidden_dim, self.dep_dim),
                arc_head=L.Linear(2 * self.hidden_dim, self.dep_dim),
                rel_dep=L.Linear(2 * self.hidden_dim, self.dep_dim),
                rel_head=L.Linear(2 * self.hidden_dim, self.dep_dim),
                biaffine_arc=Biaffine(self.dep_dim),
                biaffine_tag=Bilinear(self.dep_dim, self.dep_dim, len(self.targets))
                )
        in_dim = self.in_dim
        for _ in range(self.nlayers):
            self.qrnn_fs.add_link(QRNNLayer(in_dim, self.hidden_dim))
            self.qrnn_bs.add_link(QRNNLayer(in_dim, self.hidden_dim))
            in_dim = self.hidden_dim
            # in_dim += self.hidden_dim

    def load_pretrained_embeddings(self, path):
        self.emb_word.W.data = read_pretrained_embeddings(path)

    def __call__(self, ws, ss, ps, ls, cat_ts, dep_ts, weights):
        """
        xs [(w,s,p,y), ..., ]
        w: word, s: suffix, p: prefix, y: label
        """
        try:
            batchsize, length = ws.shape
            cat_ys, dep_ys = self.forward(ws, ss, ps, ls, dep_ts if self.train else None)

            cat_loss = reduce(lambda x, y: x + y,
                        [we * F.softmax_cross_entropy(y, t) \
                            for y, t, we in zip(cat_ys, cat_ts, weights)])
            cat_acc = reduce(lambda x, y: x + y,
                        [F.accuracy(y, t, ignore_label=IGNORE) \
                                for y, t in zip(cat_ys, cat_ts)]) / batchsize

            dep_loss = reduce(lambda x, y: x + y,
                        [we * F.softmax_cross_entropy(y, t) \
                            for y, t, we in zip(dep_ys, dep_ts, weights)])
            dep_acc = reduce(lambda x, y: x + y,
                        [F.accuracy(y, t, ignore_label=IGNORE) \
                                for y, t in zip(dep_ys, dep_ts)]) / batchsize
        except:
            print "caught erroneous example ignoring..."
            print [w.shape for w in ws]
            print [w.shape for w in ss]
            print [w.shape for w in ps]
            print ls
            print [w.shape for w in cat_ts]
            print [w.shape for w in dep_ts]
            xp = chainer.cuda.get_array_module(ws[0])
            return Variable(xp.array(0, 'f'))


        chainer.report({
            "tagging_loss": cat_loss,
            "tagging_accuracy": cat_acc,
            "parsing_loss": dep_loss,
            "parsing_accuracy": dep_acc
            }, self)
        return cat_loss + dep_loss

    def forward(self, ws, ss, ps, ls, dep_ts=None):
        batchsize, length = ws.shape
        split = scanl(lambda x,y: x+y, 0, ls)[1:-1]
        xp = chainer.cuda.get_array_module(ws[0])
        ws = self.emb_word(ws) # (batch, length, word_dim)
        ss = F.reshape(self.emb_suf(ss), (batchsize, length, -1))
        ps = F.reshape(self.emb_prf(ps), (batchsize, length, -1))
        hs = F.concat([ws, ss, ps], 2)
        hs = F.dropout(hs, self.dropout_ratio, train=self.train)
        fs = hs
        for qrnn_f in self.qrnn_fs:
            inp = fs
            fs = qrnn_f(inp)

        bs = hs[:, ::-1, :]
        for qrnn_b in self.qrnn_bs:
            inp = bs
            bs = qrnn_b(inp)

        # fs = [hs]
        # for qrnn_f in self.qrnn_fs:
        #     inp = F.concat(fs, 2)
        #     fs.append(F.dropout(qrnn_f(inp), 0.32, train=self.train))
        # fs = fs[-1]
        #
        # bs = [hs[:, ::-1, :]]
        # for qrnn_b in self.qrnn_bs:
        #     inp = F.concat(bs, 2)
        #     bs.append(F.dropout(qrnn_b(inp), 0.32, train=self.train))
        # bs = bs[-1]
        #
        hs = F.concat([fs, bs[:, ::-1, :]], 2)

        _, hs_len, hidden = hs.shape
        hs = [F.reshape(var, (hs_len, hidden))[:l] for l, var in \
                zip(ls, F.split_axis(hs, batchsize, 0))]

        dep_ys = [self.biaffine_arc(
            F.elu(F.dropout(self.arc_dep(h), 0.32, train=self.train)),
            F.elu(F.dropout(self.arc_head(h), 0.32, train=self.train))) for h in hs]

        if dep_ts is not None:
            heads = dep_ts
        else:
            heads = [F.argmax(y, axis=1) for y in dep_ys]

        heads = F.elu(F.dropout(
            self.rel_head(
                F.vstack([F.embed_id(t, h, ignore_label=IGNORE) \
                        for h, t in zip(hs, heads)])),
            0.32, train=self.train))

        childs = F.elu(F.dropout(self.rel_dep(F.vstack(hs)), 0.32, train=self.train))
        cat_ys = self.biaffine_tag(childs, heads)

        cat_ys = list(F.split_axis(cat_ys, split, 0))

        return cat_ys, dep_ys

    def predict(self, xs):
        """
        batch: list of splitted sentences
        """
        fs = [self.extractor.process(x) for x in xs]
        ws, ss, ps = concat_examples(fs)
        ls = [len(x)+2 for x in xs]
        cat_ys, dep_ys = self.forward(ws, ss, ps, ls)
        return zip([F.log_softmax(y[1:-1]).data for y in cat_ys],
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

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]


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
              to_device(_concat_arrays([s[2] for s in batch], -1))] # ss

    if len(batch[0]) == 7:
        result.append([s[3] for s in batch])                       # ls
        result.append([to_device(s[4]) for s in batch])            # cat_ts
        result.append([to_device(s[5]) for s in batch])            # dep_ts
        result.append(to_device(_concat_arrays([s[6] for s in batch], None))) # weights

    return tuple(result)


class MyUpdater(training.StandardUpdater):
    def update_core(self):
        batch = self._iterators['main'].next()
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target
        optimizer.update(loss_func, *self.converter(batch, self.device))


from chainer import reporter as reporter_module

class MyEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target
        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                eval_func(*self.converter(batch, self.device))
            summary.add(observation)
        return summary.compute_mean()


def train(args):
    model = QRNNParser(args.model, args.word_emb_size, args.afix_emb_size, args.nlayers,
            args.hidden_dim, args.elu_dim, args.dep_dim, args.dropout_ratio, args.gpu >= 0)
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

    converter = lambda x, device: concat_examples(x, device=device)

    if args.tritrain is not None:
        train = QRNNTriTrainDataset(
                args.model, args.train, args.tritrain, args.tri_weight)
    else:
        train = LSTMParserDataset(args.model, args.train)

    train_iter = SerialIterator(train, args.batchsize)
    val = LSTMParserDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, 32, repeat=False, shuffle=False)
            # val, args.batchsize, repeat=False, shuffle=False)
    # optimizer = chainer.optimizers.Adam(beta2=0.9)
    # optimizer = chainer.optimizers.MomentumSGD(momentum=0.7)
    optimizer = chainer.optimizers.RMSprop(0.001, 0.9, 1e-8)
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(4e-6))
    # optimizer.add_hook(GradientClipping(5.))
    updater = MyUpdater(train_iter, optimizer,
            device=args.gpu, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 1000, 'iteration'
    log_interval = 200, 'iteration'

    eval_model = model.copy()
    eval_model.train = False

    # trainer.extend(extensions.ExponentialShift(
    #                 "eps", .75, init=2e-3, optimizer=optimizer), trigger=(2500, 'iteration'))
    trainer.extend(MyEvaluator(val_iter, eval_model,
                    converter, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/tagging_accuracy', 'main/tagging_loss',
        'main/parsing_accuracy', 'main/parsing_loss',
        'validation/main/tagging_accuracy', 'validation/main/parsing_accuracy'
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
    parser_t.add_argument("--tritrain",
            help="tri-training data file path")
    parser_t.add_argument("--tri-weight",
            type=float, default=0.4,
            help="multiply tri-training sample losses")
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
