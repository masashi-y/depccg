import sys
import os
import pickle

def _print(string="", newline=True):
    print >> sys.stderr, "[precomputed]", string,
    if newline:
        print >> sys.stderr

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

class PrecomputedParser(object):
    def __init__(self, modeldir):
        self.modeldir = modeldir

    def predict_doc(self, doc, batchsize=None):
        files = dict(
                enumerate(
                    sorted([f for f in os.listdir(self.modeldir) \
                            if not f.endswith(".txt")])))
        _print("which file to use:")
        for i, f in files.items():
            _print("{}\t{}".format(i, f))
        _print("> ", False),
        d = raw_input()
        _print()
        choice = files[int(d)]
        pickled = os.path.join(self.modeldir, choice)
        _print("using pickled file" + pickled)
        preds = pickle.load(open(pickled))
        res = []
        for i, sent in enumerate(doc):
            _, j, (cat, dep) = preds[" ".join(map(normalize, sent))]
            res.append((i,j,(cat,dep)))
        _print("done")
        return res
