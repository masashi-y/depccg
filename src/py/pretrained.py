
import os
import sys
import numpy as np
from py_utils import read_pretrained_embeddings, read_model_defs
from collections import OrderedDict

def augment_pretrained_with_random_initialization(args):
    words = OrderedDict()
    # words in pretrained word embedding
    for word in open(args.pretrained_vocab):
        words[word.strip()] = 1

    # words in specials e.g. PAD, START, END
    for word in args.specials:
        words[word] = 1

    # words found in training data
    for word, freq in read_model_defs(args.new_words).items():
        if freq >= args.freq_cut:
            words[word] = freq

    new_pretrained_vocab = os.path.join(args.out, "new_words.txt")
    print >> sys.stderr, "writing to", new_pretrained_vocab
    with open(new_pretrained_vocab, "w") as f:
        for word, freq in words.items():
            f.write("{} {}\n".format(word, freq))

    embeddings = read_pretrained_embeddings(args.pretrained)
    assert embeddings.shape[1] <= len(words)
    new_embeddings = 0.02 * np.random.random_sample(
            (len(words), embeddings.shape[1])).astype('f') - 0.01
    for i in xrange(len(embeddings)):
        new_embeddings[i] = embeddings[i]

    new_pretrained = os.path.join(args.out, "new_embeddings.txt")
    print >> sys.stderr, "writing to", new_pretrained
    np.savetxt(new_pretrained, new_embeddings)
    print >> sys.stderr, "vocabulary size", len(embeddings), "-->", len(new_embeddings)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                "augment pretrained word embeddings with new entries.")
    parser.add_argument("pretrained",
            help="path to pretrained word embedding file")
    parser.add_argument("pretrained_vocab",
            help="path to pretrained embedding vocabulary")
    parser.add_argument("new_words",
            help="path to file with new entries")
    parser.add_argument("--freq-cut", type=int, default=3,
            help="cut words in new-words with frequency less than this value")
    parser.add_argument("--specials", nargs="*", default=[],
            help="special tokens e.g. PAD, UNK")
    parser.add_argument("out",
            help="output directory")

    args = parser.parse_args()
    print args
    augment_pretrained_with_random_initialization(args)
