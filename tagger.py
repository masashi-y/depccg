
import os
import re
from ccgbank import Leaf, AutoReader, get_leaves
from utils import get_context_by_window
import chainer
import chainer.functions as F
import chainer.links as L

class EmbeddingTagger(chainer.Chain):
    """
    model proposed in:
    A* CCG Parsing with a Supertag-factored Model, Lewis and Steedman, EMNLP 2014
    """
    def __init__(self, embed_path):
        self._read_pretrained_embeddings(embed_path)
        super(EmbeddingTagger, self).__init__(
                emb_word=F.EmbedId(),
                emb_caps=F.EmbedId(),
                emb_suffix=F.EmbedId(),
                linear=F.Linear(),
                )

    def __call__(self, xs, ts):
        """
        Inputs:
            xs (tuple(Variable, Variable, Variable)):
                each of Variables is of dim (batchsize,)
            ts Variable:
                (batchsize)
        """
        words, caps, suffixes = xs
        h_w = self.emb_word(words)
        h_c = self.emb_cap(caps)
        h_s = self.emb_suffix(suffixes)
        h = F.concat([h_w, h_c, h_s])
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

lpad = "LPAD", "PAD", "0"
rpad = "RPAD", "PAD", "0"

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                "CCG parser's supertag tagger")
    parser.add_argument("path", help="Path to auto file")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--outdir", help="output directory path")
    parser.set_defaults(train=True)
    args = parser.parse_args()

    if args.train:
        create_traindata(args.path, args.outdir)
