
import os
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda

from depccg.utils import read_model_defs
from depccg.biaffine import Biaffine, Bilinear
from depccg.param import Param

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


def normalize(word):
    if word == "-LRB-":
        return "("
    elif word == "-RRB-":
        return ")"
    elif word == "-LCB-":
        return "("
    elif word == "-RCB-":
        return ")"
    else:
        return word


def scanl(f, base, l):
    res = [base]
    acc = base
    for x in l:
        acc = f(acc, x)
        res += [acc]
    return res


class FeatureExtractor(object):
    def __init__(self, model_path, length=False):
        self.words = read_model_defs(os.path.join(model_path, 'words.txt'))
        self.suffixes = read_model_defs(os.path.join(model_path, 'suffixes.txt'))
        self.prefixes = read_model_defs(os.path.join(model_path, 'prefixes.txt'))
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


class Linear(L.Linear):
    def __call__(self, x):
        shape = x.shape
        if len(shape) == 3:
            x = F.reshape(x, (-1, shape[2]))
        y = super(Linear, self).__call__(x)
        if len(shape) == 3:
            y = F.reshape(y, (shape[0], shape[1], -1))
        return y


def concat_examples(xs, device=None):
    if device is None:
        return xs
    elif device < 0:
        return map(lambda x: map(lambda m: cuda.to_cpu(m), x), xs)
    else:
        return map(lambda x: map(
            lambda m: cuda.to_gpu(m, device, cuda.Stream.null), x), xs)


class FastBiaffineLSTMParser(chainer.Chain):
    def __init__(self, model_path):
        Param.load(self, os.path.join(model_path, 'tagger_defs.txt'))
        self.extractor = FeatureExtractor(model_path, length=True)
        self.in_dim = self.word_dim + 8 * self.afix_dim
        super(FastBiaffineLSTMParser, self).__init__(
                emb_word=L.EmbedID(self.n_words, self.word_dim, ignore_label=IGNORE),
                emb_suf=L.EmbedID(self.n_suffixes, self.afix_dim, ignore_label=IGNORE),
                emb_prf=L.EmbedID(self.n_prefixes, self.afix_dim, ignore_label=IGNORE),
                lstm_f=L.NStepLSTM(self.nlayers, self.in_dim, self.hidden_dim, 0.32),
                lstm_b=L.NStepLSTM(self.nlayers, self.in_dim, self.hidden_dim, 0.32),
                arc_dep=Linear(2 * self.hidden_dim, self.dep_dim),
                arc_head=Linear(2 * self.hidden_dim, self.dep_dim),
                rel_dep=Linear(2 * self.hidden_dim, self.dep_dim),
                rel_head=Linear(2 * self.hidden_dim, self.dep_dim),
                biaffine_arc=Biaffine(self.dep_dim),
                biaffine_tag=Bilinear(self.dep_dim, self.dep_dim, len(self.targets)))

    def forward(self, ws, ss, ps, ls, dep_ts=None):
        split = scanl(lambda x,y: x+y, 0, [w.shape[0] for w in ws])[1:-1]

        wss = self.emb_word(F.hstack(ws))
        sss = F.reshape(self.emb_suf(F.vstack(ss)), (-1, 4 * self.afix_dim))
        pss = F.reshape(self.emb_prf(F.vstack(ps)), (-1, 4 * self.afix_dim))
        ins = F.dropout(F.concat([wss, sss, pss]), 0.5)

        xs_f = list(F.split_axis(ins, split, 0))
        xs_b = [x[::-1] for x in xs_f]
        _, _, hs_f = self.lstm_f(None, None, xs_f)
        _, _, hs_b = self.lstm_b(None, None, xs_b)

        hs_b = [x[::-1] for x in hs_b]
        hs = [F.concat([h_f, h_b]) for h_f, h_b in zip(hs_f, hs_b)]

        dep_ys = [self.biaffine_arc(
            F.elu(F.dropout(self.arc_dep(h), 0.32)),
            F.elu(F.dropout(self.arc_head(h), 0.32))) for h in hs]

        if dep_ts is not None:
            heads = dep_ts
        else:
            heads = [F.argmax(y, axis=1) for y in dep_ys]

        heads = F.elu(F.dropout(
            self.rel_head(
                F.vstack([F.embed_id(t, h, ignore_label=IGNORE)
                          for h, t in zip(hs, heads)])), 0.32))

        childs = F.elu(F.dropout(self.rel_dep(F.vstack(hs)), 0.32))
        cat_ys = self.biaffine_tag(childs, heads)

        cat_ys = list(F.split_axis(cat_ys, split, 0))

        return cat_ys, dep_ys

    def predict(self, xs):
        ws, ss, ps, ls = zip(*[self.extractor.process(x) for x in xs])
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            cat_ys, dep_ys = self.forward(ws, ss, ps, ls)
        return zip([F.log_softmax(y[1:-1]).data for y in cat_ys],
                   [F.log_softmax(y[1:-1, :-1]).data for y in dep_ys])

    def predict_doc(self, doc, batchsize=32):
        res = []
        for i in range(0, len(doc), batchsize):
            res.extend(self.predict(doc[i:i + batchsize]))
        return res

    @property
    def cats(self):
        return list(zip(*sorted(self.targets.items(), key=lambda x: x[1])))[0]

