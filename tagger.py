
import os
import re
import json
from collections import defaultdict, OrderedDict
import numpy as np
from ccgbank import Leaf, AutoReader, get_leaves
from utils import get_context_by_window, read_pretrained_embeddings, read_model_defs
import chainer
import multiprocessing
import chainer.functions as F
import chainer.links as L
from chainer import training, Variable
from chainer.training import extensions


lpad = "LPAD", "PAD", "0"
rpad = "RPAD", "PAD", "0"

re_subset = {"train": re.compile(r"wsj_(0[2-9]|1[0-9]|20|21)..\.auto"),
            "test": re.compile(r"wsj_23..\.auto"),
            "val": re.compile(r"wsj_00..\.auto"),
            "all": re.compile(r"wsj_....\.auto") }

num = re.compile(r"[0-9]+")


class CCGBankDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.words = read_model_defs(os.path.join(model_path, "words.txt"))
        self.suffixes = read_model_defs(os.path.join(model_path, "suffixes.txt"))
        self.caps = read_model_defs(os.path.join(model_path, "caps.txt"))
        self.targets = read_model_defs(os.path.join(model_path, "target.txt"))
        self.samples = open(samples_path).readlines()
        self.unk_word = self.words["*UNKNOWN*"]
        self.unk_suffix = self.suffixes["UNK"]

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        line = self.samples[i]
        items = line.strip().split(" ")
        t = np.asarray(self.targets.get(items[-1], -1), 'i')

        x = np.zeros((3 * 7), "i")
        for i, item in enumerate(items[:-1]):
            word, suffix, cap = item.split("|")
            x[i] = self.words.get(word, self.unk_word)
            x[7 + i] = self.suffixes.get(suffix, self.unk_suffix)
            x[14 + i] = self.caps[cap]
        return x, t


class EmbeddingTagger(chainer.Chain):
    """
    model proposed in:
    A* CCG Parsing with a Supertag-factored Model, Lewis and Steedman, EMNLP 2014
    """
    def __init__(self, model_path, word_dim=None, caps_dim=None, suffix_dim=None):
        self.model_path = model_path
        if word_dim is None:
            # use as supertagger
            with open(os.path.join(model_path, "tagger_defs.txt")) as defs_file:
                defs = json.load(defs_file)
            self.word_dim = defs["word_dim"]
            self.caps_dim = defs["caps_dim"]
            self.suffix_dim = defs["suffix_dim"]
        else:
            # training
            self.word_dim = word_dim
            self.caps_dim = caps_dim
            self.suffix_dim = suffix_dim

        self.words = read_model_defs(os.path.join(model_path, "words.txt"))
        self.suffixes = read_model_defs(os.path.join(model_path, "suffixes.txt"))
        self.caps = read_model_defs(os.path.join(model_path, "caps.txt"))
        self.targets = read_model_defs(os.path.join(model_path, "target.txt"))

        self.unk_word = self.words["*UNKNOWN*"]
        self.unk_suffix = self.suffixes["UNK"]

        in_dim = 7 * (self.word_dim + self.caps_dim + self.suffix_dim)
        super(EmbeddingTagger, self).__init__(
                emb_word=L.EmbedID(len(self.words), self.word_dim),
                emb_caps=L.EmbedID(len(self.caps), self.caps_dim),
                emb_suffix=L.EmbedID(len(self.suffixes), self.suffix_dim),
                linear=L.Linear(in_dim, len(self.targets)),
                )

    def setup_training(self, pretrained_path):
        emb_w = read_pretrained_embeddings(pretrained_path)
        new_emb_w = 0.02 * np.random.random_sample(
                (len(self.words), emb_w.shape[1])).astype('f') - 0.01
        for i in xrange(len(emb_w)):
            new_emb_w[i] = emb_w[i]
        self.emb_word.W.data = new_emb_w
        with open(os.path.join(self.model_path, "tagger_defs.txt"), "w") as out:
            json.dump({"word_dim": self.word_dim,
                       "suffix_dim": self.suffix_dim,
                       "caps_dim": self.caps_dim}, out)

    def __call__(self, xs, ts):
        """
        Inputs:
            xs (tuple(Variable, Variable, Variable)):
                each of Variables is of dim (batchsize,)
            ts Variable:
                (batchsize)
        """
        words, suffixes, caps = xs[:,:7], xs[:, 7:14], xs[:, 14:]
        h_w = self.emb_word(words)
        h_c = self.emb_caps(caps)
        h_s = self.emb_suffix(suffixes)
        h = F.concat([h_w, h_c, h_s], 2)
        batchsize, ntokens, hidden = h.data.shape
        h = F.reshape(h, (batchsize, ntokens * hidden))
        ys = self.linear(h)

        loss = F.softmax_cross_entropy(ys, ts)
        acc = F.accuracy(ys, ts)

        chainer.report({
            "loss": loss,
            "accuracy": acc
            }, self)
        return loss

    def predict(self, tokens):
        """
        Inputs:
            tokens (list[str])
        Returns
            list[Leaf]
        """
        contexts = get_context_by_window(
                map(feature_extract, tokens), 3, lpad=lpad, rpad=rpad)

        words = np.zeros((len(tokens), 7), 'i')
        suffixes = np.zeros((len(tokens), 7), 'i')
        caps = np.zeros((len(tokens), 7), 'i')
        for i, context in enumerate(contexts):
            w, s, c = zip(*context)
            words[i] = map(lambda x:
                    self.words.get(x, self.unk_word), w)
            suffixes[i] = map(lambda x:
                    self.suffixes.get(x, self.unk_suffix), s)
            caps[i] = map(lambda x:
                    self.caps[x], c)

        h_w = self.emb_word(np.asarray(words, "i"))
        h_c = self.emb_caps(np.asarray(caps, "i"))
        h_s = self.emb_suffix(np.asarray(suffixes, "i"))
        h = F.concat([h_w, h_c, h_s], 2)
        batchsize, ntokens, hidden = h.data.shape
        h = F.reshape(h, (batchsize, ntokens * hidden))
        ys = self.linear(h)
        ys = F.softmax(ys)
        return ys.data

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]

class ScoredLeaf(Leaf):
    def __init__(self, word, cat, score):
        super(ScoredLeaf, self).__init__(
                word, cat, None)
        self.score = score


def feature_extract(word_str):
    isupper = "1" if word_str.isupper() else "0"
    normalizd = word_str.lower()
    normalizd = num.sub("#", normalizd)
    if normalizd == "-lrb-":
        normalizd = "("
    elif normalizd == "-rrb-":
        normalizd = ")"
    suffix = normalizd.ljust(2, "_")[-2:]
    return normalizd, suffix, str(isupper)


def compress_traindata(args):
    words = OrderedDict()
    print "reading embedding vocabulary"
    for word in open(args.vocab):
        words[word.strip()] = 1
    suffixes = defaultdict(int)
    suffixes["UNK"] = 1
    caps = defaultdict(int)
    target = defaultdict(int)
    traindata = open(args.path)
    len_traindata = 0
    print "reading training file"
    for line in traindata:
        len_traindata += 1
        items = line.strip().split(" ")
        target[items[-1]] += 1
        for item in items[:-1]:
            word, suffix, cap = item.split("|")
            if words.has_key(word):
                words[word] += 1
            else:
                words[word] = 1
            suffixes[suffix] += 1
            caps[cap] += 1
    def out_dict(d, outfile, freq_cut=-1):
        print "writing to {}".format(outfile)
        res = {}
        with open(outfile, "w") as out:
            i = 0
            for item, n in d.items():
                if freq_cut <= n:
                    out.write("{} {}\n".format(item, n))
                    res[item] = i
                    i += 1
        return res
    word2id = out_dict(words, os.path.join(args.out, "words.txt"))
    suffix2id = out_dict(suffixes, os.path.join(args.out, "suffixes.txt"))
    cap2id = out_dict(caps, os.path.join(args.out, "caps.txt"))
    target2id = out_dict(target, os.path.join(args.out, "target.txt"), freq_cut=10)
    traindata.seek(0)
    new_traindata = os.path.join(args.out, "traindata.txt")
    print "writing to {}".format(new_traindata)
    with open(new_traindata, "w") as out:
        for i, line in enumerate(traindata):
            items = line.strip().split(" ")
            if not target2id.has_key(items[-1]):
                continue
            target =items[-1]
            new_line = ""
            for j, item in enumerate(items[:-1]):
                word, suffix, cap = item.split("|")
                if not word2id.has_key(word):
                    word = "*UNKNOWN*"
                if not suffix2id.has_key(suffix):
                    suffix = "UNK"
                new_line += "|".join([word, suffix, cap]) + " "
            out.write(new_line + target + "\n")


def create_traindata(args):
    matcher = re_subset[args.subset]
    autos = []
    for root, dirs, files in os.walk(args.path):
        for autofile in files:
            if matcher.match(autofile):
                autos.append(
                        (os.path.join(root, autofile), args.windowsize))
    n_process = multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_process)
    with open(args.out, "w") as out:
        for lines in p.map(_worker, autos):
            for line in lines:
                out.write(line)


def _worker(inp):
    autofile, window_size = inp
    res = []
    for tree in AutoReader(autofile).readall(suppress_error=True):
        leaves = get_leaves(tree)
        feats = map(feature_extract, [leaf.word for leaf in leaves])
        contexts = get_context_by_window(
                feats, window_size, lpad=lpad, rpad=rpad)
        for leaf, context in zip(leaves, contexts):
            res.append(" ".join(map(lambda c: "|".join(c), context)) + \
                    " " + str(leaf.cat) + "\n")
    return res


def train(args):
    model = EmbeddingTagger(args.model, 50, 20, 30)
    model.setup_training(args.embed)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    train = CCGBankDataset(args.model, args.train)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val = CCGBankDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.model)

    val_interval = 5000, 'iteration'
    log_interval = 200, 'iteration'
    val_model = model.copy()

    trainer.extend(extensions.Evaluator(val_iter, val_model), trigger=val_interval)
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
                "CCG parser's supertag tagger")
    subparsers = parser.add_subparsers()

    # Creating training data from CCGBank AUTO files
    parser_c = subparsers.add_parser(
            "create", help="create tagger input data")
    parser_c.add_argument("path",
            help="path to AUTO directory")
    parser_c.add_argument("out",
            help="output file path")
    parser_c.add_argument("--windowsize",
            type=int, default=3,
            help="window size to extract features")
    parser_c.add_argument("--subset",
            choices=["train", "val", "test", "all"],
            default="train",
            help="train: 02-21, val: 00, test: 23, (default: train)")
    parser_c.set_defaults(func=create_traindata)

    # Do training using training data created through `create`
    parser_t = subparsers.add_parser(
            "train", help="train supertagger model")
    parser_t.add_argument("model",
            help="path to model directory")
    parser_t.add_argument("embed",
            help="path to embedding file")
    parser_t.add_argument("vocab",
            help="path to embedding vocab file")
    parser_t.add_argument("train",
            help="training data file path")
    parser_t.add_argument("val",
            help="validation data file path")
    parser_t.add_argument("--batchsize",
            type=int, default=1000, help="batch size")
    parser_t.add_argument("--epoch",
            type=int, default=20, help="epoch")
    parser_t.add_argument("--initmodel",
            help="initialize model with `initmodel`")
    parser_t.set_defaults(func=train)

    # Compress
    parser_co = subparsers.add_parser(
            "compress", help="compress tagger input data")
    parser_co.add_argument("path",
            help="path to a file to compress")
    parser_co.add_argument("out",
            help="output directory path")
    parser_co.add_argument("vocab",
            help="embedding vocabulary file")
    parser_co.set_defaults(func=compress_traindata)

    args = parser.parse_args()
    args.func(args)
