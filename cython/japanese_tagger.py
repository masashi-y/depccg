
import sys
import numpy as np
import os
import json
import chainer
from utils import get_context_by_window, read_pretrained_embeddings, read_model_defs
from collections import defaultdict
from japanese_ccg import JaCCGReader
from tree import Leaf, Tree, get_leaves

filepath = "/home/masashi-y/japanese-ccgbank/ccgbank-20150216/test10.ccgbank"
# filepath = "/home/masashi-y/japanese-ccgbank/ccgbank-20150216/train.ccgbank"

WINDOW_SIZE = 7

class JaCCGInspector(object):
    """
    create train & validation data
    """
    def __init__(self, filepath, freq_cut=10):
        self.filepath = filepath
         # those categories whose frequency < freq_cut are discarded.
        self.freq_cut = freq_cut
        self.seen_rules = defaultdict(int) # seen binary rules
        self.unary_rules = defaultdict(int) # seen unary rules
        self.cats = defaultdict(int) # all cats
        self.words = defaultdict(int)
        self.chars = defaultdict(int)
        self.samples = {}

        self.words["*UNKNOWN*"] = 10000
        self.chars["*UNKNOWN*"] = 10000

    def _traverse(self, tree):
        self.cats[tree.cat.without_semantics] += 1
        if isinstance(tree, Leaf):
            w = tree.word.encode("utf-8")
            self.words[w] += 1
            for c in w.decode("utf-8"):
                self.chars[c.encode("utf-8")] += 1
        else:
            children = tree.children
            if len(children) == 1:
                rule = tree.cat.without_semantics + \
                        " " + children[0].cat.without_semantics
                self.unary_rules[rule] += 1
                self._traverse(children[0])
            else:
                rule = children[0].cat.without_semantics + \
                        " " + children[1].cat.without_semantics
                self.seen_rules[rule] += 1
                self._traverse(children[0])
                self._traverse(children[1])

    @staticmethod
    def _write(dct, out, comment_out_value=False):
        for key, value in dct.items():
            out.write(str(key) + " ")
            if comment_out_value:
                out.write("# ")
            out.write(str(value) + "\n")

    def create_traindata(self, outdir):
        trees = JaCCGReader(self.filepath).readall()
        # first construct dictionaries only
        for tree in trees:
            self._traverse(tree)
        # construct training samples with
        # categories whose frequency >= freq_cut.
        for tree in trees:
            tokens = get_leaves(tree)
            words = [token.word for token in tokens]
            cats = [token.cat.without_semantics for token in tokens]
            samples = get_context_by_window(words, 3, lpad="LPAD", rpad="RPAD")
            assert len(samples) == len(cats)
            for cat, sample in zip(cats, samples):
                if self.cats[cat] >= self.freq_cut:
                    self.samples[" ".join(sample)] = cat

        with open(outdir + "/unary_rules.txt", "w") as f:
            self._write(self.unary_rules, f, comment_out_value=True)
        with open(outdir + "/seen_rules.txt", "w") as f:
            self._write(self.seen_rules, f, comment_out_value=True)
        with open(outdir + "/targets.txt", "w") as f:
            self._write(self.cats, f, comment_out_value=False)
        with open(outdir + "/words.txt", "w") as f:
            self._write(self.words, f, comment_out_value=False)
        with open(outdir + "/chars.txt", "w") as f:
            self._write(self.chars, f, comment_out_value=False)
        with open(outdir + "traindata.json", "w") as f:
            json.dump(self.samples, f)

    def create_testdata(self, outdir):
        trees = JaCCGReader(self.filepath).readall()
        for tree in trees:
            tokens = get_leaves(tree)
            words = [token.word for token in tokens]
            cats = [token.cat.without_semantics for token in tokens]
            samples = get_context_by_window(words, 3, lpad="LPAD", rpad="RPAD")
            assert len(samples) == len(cats)
            for cat, sample in zip(cats, samples):
                self.samples[" ".join(sample)] = cat
        with open(outdir + "testdata.json", "w") as f:
            json.dump(self.samples, f)


class JaCCGTaggerDataset(chainer.dataset.DatasetMixin):
    def __init__(self, model_path, samples_path):
        self.model_path = model_path
        self.words = read_model_defs(os.path.join(model_path, "words.txt"))
        self.chars = read_model_defs(os.path.join(model_path, "chars.txt"))
        self.targets = read_model_defs(os.path.join(model_path, "targets.txt"))
        with open(samples_path) as f:
            self.samples = json.load(f).items()
        self.unk_word = self.words["*UNKNOWN*"]
        self.unk_char = self.chars["*UNKNOWN*"]
        self.max_char_len = max(len(w) for w in self.words if w != "*UNKNOWN*")

    def __len__(self):
        return len(self.samples)

    def get_example(self, i):
        """
        `line`: word1 word2 ,.., wordN target\n
        Returns:
            np.ndarray shape(WINDOW_SIZE, 1+max_char_len)
            with first column id for each word in the window,
            second till the last columns are filled with character id.
        """
        line, target = self.samples[i]
        items = line.strip().split(" ")
        x = -np.ones((WINDOW_SIZE, self.max_char_len + 1), 'i')
        l = np.zeros((WINDOW_SIZE,), 'f')
        for i, word in enumerate(items):
            x[i, 0] = self.words.get(word, self.unk_word)
            l[i] = len(word)
            for j, char in enumerate(word, 1):
                x[i, j] = self.chars.get(char, self.unk_char)
        t = np.asarray(self.targets[target], 'i')
        return x, l, t


class JaCCGEmbeddingTagger(chainer.Chain):
    def __init__(self, model_path, word_dim=None, char_dim=None):
        self.model_path = model_path
        if word_dim is None:
            # use as supertagger
            with open(os.path.join(model_path, "tagger_defs.txt")) as defs_file:
                defs = json.load(defs_file)
            self.word_dim = defs["word_dim"]
            self.char_dim = defs["char_dim"]
        else:
            # training
            self.word_dim = word_dim
            self.char_dim = char_dim

        self.words = read_model_defs(os.path.join(model_path, "words.txt"))
        self.chars = read_model_defs(os.path.join(model_path, "chars.txt"))
        self.targets = read_model_defs(os.path.join(model_path, "target.txt"))

        in_dim = WINDOW_SIZE * (self.word_dim + self.char_dim)
        super(JaCCGEmbeddingTagger, self).__init__(
                emb_word=L.EmbedID(len(self.words), self.word_dim),
                emb_caps=L.EmbedID(len(self.caps), self.caps_dim),
                linear=L.Linear(in_dim, len(self.targets)),
                )

    def __call__(self, xs, ls, ts):
        """
        xs: Variable of shape(batchsize, windowsize, 1 + max_char_len)
        ls: lengths of each word
        ts:
        """
        words, chars = xs[:, :, 0], xs[:, :, 1:]
        h_w = self.emb_word(words) #_(batchsize, windowsize, word_dim)
        h_c = self.emb_chars(chars) # (batchsize, windowsize, max_char_len, char_dim)
        batchsize, windowsize, _, _ = h_c.shape
        # (batchsize, windowsize, char_dim)
        h_c = F.sum(h_c, 2) / ls.reshape((batchsize, windowsize, 1))
        h = F.concat([h_w, h_c], 2)
        h = F.reshape(h, (batchsize, -1))
        ys = self.linear(h)

        loss = F.softmax_cross_entropy(ys, ts)
        acc = F.accuracy(ys, ts)

        chainer.report({
            "loss": loss,
            "accuracy": acc
            }, self)
        return loss

    def predict(self, tokens):
        pass

    def predict_doc(self, doc, batchsize=100):
        pass

    @property
    def cats(self):
        return zip(*sorted(self.targets.items(), key=lambda x: x[1]))[0]

def train(args):
    model = JaCCGEmbeddingTagger(args.model, 50, 50)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    train = JaCCGTaggerDataset(args.model, args.train)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val = JaCCGTaggerDataset(args.model, args.val)
    val_iter = chainer.iterators.SerialIterator(
            val, args.batchsize, repeat=False, shuffle=False)
    optimizer = chainer.optimizers.SGD(lr=0.01)
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
def main():
    outdir = "ja_model/"
    JaCCGInspector(filepath).create_traindata(outdir)
main()
