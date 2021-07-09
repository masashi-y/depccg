
import os
import random
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer import Variable
from chainer.dataset.convert import _concat_arrays
from depccg.utils import read_model_defs, normalize
from .biaffine import Biaffine, Bilinear
from depccg.chainer.param import Param
from .fixed_length_n_step_lstm import FixedLengthNStepLSTM
from depccg.types import ScoringResult


UNK = "*UNKNOWN*"
OOR2 = "OOR2"
OOR3 = "OOR3"
OOR4 = "OOR4"
START = "*START*"
END = "*END*"
IGNORE = -1
MISS = -2


def get_suffix(word):
    return [word[-1],
            word[-2:] if len(word) > 1 else OOR2,
            word[-3:] if len(word) > 2 else OOR3,
            word[-4:] if len(word) > 3 else OOR4]


def get_prefix(word):
    return [word[0],
            word[:2] if len(word) > 1 else OOR2,
            word[:3] if len(word) > 2 else OOR3,
            word[:4] if len(word) > 3 else OOR4]


class FeatureExtractor(object):
    def __init__(self, model_path, length=False):
        self.words = read_model_defs(os.path.join(model_path, "words.txt"))
        self.suffixes = read_model_defs(
            os.path.join(model_path, "suffixes.txt"))
        self.prefixes = read_model_defs(
            os.path.join(model_path, "prefixes.txt"))
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

    def process(self, words, xp):
        """
        words: list of unicode tokens
        """
        words = list(map(normalize, words))
        w = xp.array([self.start_word] + [self.words.get(
            x.lower(), self.unk_word) for x in words] + [self.end_word], 'i')
        s = xp.asarray(self.start_suf + [[self.suffixes.get(
            f, self.unk_suf) for f in get_suffix(x)] for x in words] + self.end_suf, 'i')
        p = xp.asarray(self.start_pre + [[self.prefixes.get(
            f, self.unk_prf) for f in get_prefix(x)] for x in words] + self.end_pre, 'i')
        if not self.length:
            return w, s, p
        else:
            return w, s, p, w.shape[0]


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

    result = [to_device(_concat_arrays([s[0] for s in batch], -1)),  # ws
              to_device(_concat_arrays([s[1] for s in batch], -1)),  # ps
              to_device(_concat_arrays([s[2] for s in batch], -1)),  # ss
              [s[3] for s in batch]]                                # ls

    if len(batch[0]) == 7:
        result.append([to_device(s[4]) for s in batch])            # cat_ts
        result.append([to_device(s[5]) for s in batch])            # dep_ts
        result.append(to_device(_concat_arrays(
            [s[6] for s in batch], None)))  # weights

    return tuple(result)


class FastBiaffineLSTMParser(chainer.Chain):
    def __init__(self, model_path, word_dim=None, afix_dim=None, nlayers=2,
                 hidden_dim=128, dep_dim=100, dropout_ratio=0.5):
        Param.load(self, os.path.join(model_path, 'tagger_defs.txt'))
        self.extractor = FeatureExtractor(model_path, length=True)
        self.in_dim = self.word_dim + 8 * self.afix_dim
        self.dropout_ratio = dropout_ratio
        super(FastBiaffineLSTMParser, self).__init__(
            emb_word=L.EmbedID(self.n_words, self.word_dim,
                               ignore_label=IGNORE),
            emb_suf=L.EmbedID(self.n_suffixes, self.afix_dim,
                              ignore_label=IGNORE),
            emb_prf=L.EmbedID(self.n_prefixes, self.afix_dim,
                              ignore_label=IGNORE),
            lstm_f=FixedLengthNStepLSTM(
                self.nlayers, self.in_dim, self.hidden_dim, 0.32),
            lstm_b=FixedLengthNStepLSTM(
                self.nlayers, self.in_dim, self.hidden_dim, 0.32),
            arc_dep=Linear(2 * self.hidden_dim, self.dep_dim),
            arc_head=Linear(2 * self.hidden_dim, self.dep_dim),
            rel_dep=Linear(2 * self.hidden_dim, self.dep_dim),
            rel_head=Linear(2 * self.hidden_dim, self.dep_dim),
            biaffine_arc=Biaffine(self.dep_dim),
            biaffine_tag=Bilinear(self.dep_dim, self.dep_dim, len(self.targets)))

    def forward(self, ws, ss, ps, ls, dep_ts=None):
        batchsize, slen = ws.shape

        wss = self.emb_word(ws)
        sss = F.reshape(self.emb_suf(ss), (batchsize, slen, 4 * self.afix_dim))
        pss = F.reshape(self.emb_prf(ps), (batchsize, slen, 4 * self.afix_dim))
        ins = F.dropout(F.concat([wss, sss, pss], 2), self.dropout_ratio)
        xs_f = F.transpose(ins, (1, 0, 2))
        xs_b = xs_f[::-1]

        cx_f, hx_f, cx_b, hx_b = self._init_state(batchsize)
        _, _, hs_f = self.lstm_f(hx_f, cx_f, xs_f, train=False)
        _, _, hs_b = self.lstm_b(hx_b, cx_b, xs_b, train=False)

        # (batch, length, hidden_dim)
        hs = F.transpose(F.concat([hs_f, hs_b[::-1]], 2), (1, 0, 2))

        dep_ys = self.biaffine_arc(
            F.elu(F.dropout(self.arc_dep(hs), 0.32)),
            F.elu(F.dropout(self.arc_head(hs), 0.32)))

        if dep_ts is not None and random.random >= 0.5:
            heads = dep_ts
        else:
            heads = F.flatten(F.argmax(dep_ys, axis=2)) + \
                self.xp.repeat(self.xp.arange(0, batchsize * slen, slen), slen)

        hs = F.reshape(hs, (batchsize * slen, -1))
        heads = F.permutate(
            F.elu(F.dropout(
                self.rel_head(hs), 0.32)), heads)

        childs = F.elu(F.dropout(self.rel_dep(hs), 0.32))
        cat_ys = self.biaffine_tag(childs, heads)

        dep_ys = F.split_axis(
            dep_ys, batchsize, 0) if batchsize > 1 else [dep_ys]
        dep_ys = [F.reshape(v, v.shape[1:])[:l, :l]
                  for v, l in zip(dep_ys, ls)]

        cat_ys = F.split_axis(
            cat_ys, batchsize, 0) if batchsize > 1 else [cat_ys]
        cat_ys = [v[:l] for v, l in zip(cat_ys, ls)]
        return cat_ys, dep_ys

    def _predict(self, xs):
        xs = [self.extractor.process(x, self.xp) for x in xs]
        ws, ss, ps, ls = concat_examples(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            cat_ys, dep_ys = self.forward(ws, ss, ps, ls)
        cat_ys = [F.log_softmax(y[1:-1]).data for y in cat_ys]
        cat_ys = [chainer.cuda.to_cpu(y) for y in cat_ys]
        dep_ys = [F.log_softmax(y[1:-1, :-1]).data for y in dep_ys]
        dep_ys = [chainer.cuda.to_cpu(y) for y in dep_ys]
        return list(zip(cat_ys, dep_ys))

    def predict_doc(self, doc, batchsize=32, gpu=-1):
        res = []
        doc = sorted(enumerate(doc), key=lambda x: len(x[1]))
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
        for i in range(0, len(doc), batchsize):
            ids, batch = zip(*doc[i:i + batchsize])
            pred = self._predict(batch)
            res.extend((j, y) for j, y in zip(ids, pred))
        res = [ScoringResult(*pred) for _, pred in sorted(res)]
        return res, self.cats

    def _init_state(self, batchsize):
        res = [Variable(self.xp.zeros(
            (self.nlayers, batchsize, self.hidden_dim), 'f')) for _ in range(4)]
        return res

    @property
    def cats(self):
        return list(zip(*sorted(self.targets.items(), key=lambda x: x[1])))[0]
