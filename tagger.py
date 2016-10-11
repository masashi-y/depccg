
import os
import re
from collections import defaultdict, OrderedDict
import numpy as np
from ccgbank import Leaf, AutoReader, get_leaves
from utils import get_context_by_window
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Variable
from chainer.training import extensions


lpad = "LPAD", "PAD", "0"
rpad = "RPAD", "PAD", "0"

class CCGBankDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path):
        self.model_path = model_path
        self.dataset = np.load(os.path.join(model_path, "traindata.npz"))
        self.xs = self.dataset["xs"]
        self.ts = self.dataset["ts"]

    def __len__(self):
        return len(self.xs)

    def get_example(self, i):
        return self.xs[i], self.ts[i]

class EmbeddingTagger(chainer.Chain):
    """
    model proposed in:
    A* CCG Parsing with a Supertag-factored Model, Lewis and Steedman, EMNLP 2014
    """
    def __init__(self, model_path, embed_path, caps_dim, suffix_dim):
        emb_w = self._read_pretrained_embeddings(embed_path)
        nwords = sum(1 for line in open(os.path.join(model_path, "words.txt")))
        new_emb_w = 0.02 * np.random.random_sample((nwords, emb_w.shape[1])).astype('f') - 0.01
        for i in xrange(len(emb_w)):
            new_emb_w[i] = emb_w[i]

        ncaps = sum(1 for line in open(os.path.join(model_path, "caps.txt")))
        nsuffixes = sum(1 for line in open(os.path.join(model_path, "suffixes.txt")))
        ntargets  = sum(1 for line in open(os.path.join(model_path, "target.txt")))
        super(EmbeddingTagger, self).__init__(
                emb_word=L.EmbedID(*new_emb_w.shape, initialW=new_emb_w),
                emb_caps=L.EmbedID(ncaps, caps_dim),
                emb_suffix=L.EmbedID(nsuffixes, suffix_dim),
                linear=L.Linear(7 * (new_emb_w.shape[1] + caps_dim + suffix_dim), ntargets),
                )

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
            })
        return loss

    @staticmethod
    def _read_pretrained_embeddings(filepath):
        io = open(filepath)
        dim = len(io.readline().split())
        io.seek(0)
        nvocab = sum(1 for line in io)
        io.seek(0)
        res = np.empty((nvocab, dim), dtype=np.float32)
        for i, line in enumerate(io):
            line = line.strip()
            if len(line) == 0: continue
            res[i] = line.split()
        io.close()
        return res

def compress_traindata(trainfile, outdir, vocabfile):
    words = OrderedDict()
    print "reading embedding vocabulary"
    for word in open(vocabfile):
        words[word.strip()] = 1
    suffixes = defaultdict(int)
    caps = defaultdict(int)
    target = defaultdict(int)
    traindata = open(trainfile)
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
            for i, (item, n) in enumerate(d.items()):
                if freq_cut <= n:
                    out.write("{} {} {}\n".format(i, item, n))
                    res[item] = i
        return res
    word2id = out_dict(words, os.path.join(outdir, "words.txt"))
    suffix2id = out_dict(suffixes, os.path.join(outdir, "suffixes.txt"))
    cap2id = out_dict(caps, os.path.join(outdir, "caps.txt"))
    target2id = out_dict(target, os.path.join(outdir, "target.txt"), freq_cut=10)
    traindata.seek(0)
     # (word_id, suffix_id, cap_id) * 7 tokens
    xs = np.zeros(
            (len_traindata + 1, 3 * 7), dtype='i')
    ts = np.zeros(len_traindata + 1, dtype='i')
    print "creating traindata.npz"
    for i, line in enumerate(traindata):
        items = line.strip().split(" ")
        target_id = target2id[items[-1]]
        ts[i] = target_id
        for j, item in enumerate(items[:-1]):
            word, suffix, cap = item.split("|")
            xs[i, j] = word2id[word]
            xs[i, 7 + j] = suffix2id[suffix]
            xs[i, 14 + j] = cap2id[cap]
    np.savez(
            os.path.join(outdir, "traindata.npz"), xs=xs, ts=ts)


def create_traindata(autofile, outdir, window_size=3):
    outpath = os.path.join(outdir, os.path.basename(autofile))
    with open(outpath, "w") as out:
        for tree in AutoReader(autofile).readall(suppress_error=True):
            leaves = get_leaves(tree)
            feats = map(feature_extract, leaves)
            contexts = get_context_by_window(
                    feats, window_size, lpad=lpad, rpad=rpad)
            for leaf, context in zip(leaves, contexts):
                out.write(" ".join(map(lambda c: "|".join(c), context)))
                out.write(" {}\n".format(leaf.cat))

num = re.compile(r"[0-9]+")
def feature_extract(leaf):
    if leaf is lpad or leaf is rpad:
        return leaf.word, "PAD", "0"
    word_str = leaf.word
    isupper = "1" if word_str.isupper() else "0"
    normalizd = word_str.lower()
    normalizd = num.sub("#", normalizd)
    if normalizd == "-lrb-":
        normalizd = "("
    elif normalizd == "-rrb-":
        normalizd = ")"
    suffix = normalizd.ljust(2, "_")[-2:]
    return normalizd, suffix, str(isupper)

def train(args):
    embed_path = os.path.join(args.path, "embeddings-scaled.EMBEDDING_SIZE=50.vectors")
    model = EmbeddingTagger(args.path, embed_path, 20, 30)
    train = CCGBankDataset(args.path)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.outdir)

    val_interval = 100000, 'iteration'
    log_interval = 1000, 'iteration'

    # trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                "CCG parser's supertag tagger")
    parser.add_argument("path", help="Path to auto file")
    parser.add_argument("--create", action='store_true',
            help="create training data")
    parser.add_argument("--train", action='store_true',
            help="train")
    parser.add_argument("--compress", action='store_true',
            help="compress training data")
    parser.add_argument("--vocab",
            help="embedding vocab file")
    parser.add_argument("--outdir",
            help="output directory path")
    parser.add_argument("--batchsize",
            default=1000, help="batch size")
    parser.add_argument("--epoch",
            default=20, help="epoch")
    parser.set_defaults(train=True)
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.create:
        create_traindata(args.path, args.outdir)
    elif args.compress:
        compress_traindata(args.path, args.outdir, args.vocab)
