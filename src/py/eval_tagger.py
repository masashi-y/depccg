
import chainer
import json
import argparse
import pickle
from chainer import cuda
import chainer.functions as F

from collections import OrderedDict

from lstm_tagger import LSTMTagger
from lstm_parser import LSTMParser
from lstm_tagger_ph import PeepHoleLSTMTagger
from lstm_parser_ph import PeepHoleLSTMParser
from ja_lstm_tagger import JaLSTMTagger
from ja_lstm_parser import JaLSTMParser
from ja_lstm_parser_ph import PeepHoleJaLSTMParser
from lstm_parser_bi import BiaffineLSTMParser
from ja_lstm_parser_bi import BiaffineJaLSTMParser
from lstm_parser_bi_fast import FastBiaffineLSTMParser

Parser = (LSTMParser,
        PeepHoleLSTMParser,
        JaLSTMParser,
        PeepHoleJaLSTMParser,
        BiaffineLSTMParser,
        BiaffineJaLSTMParser,
        FastBiaffineLSTMParser)

def calc_acc(preds, targets, name="tagging"):
    corrects = 0
    totals = len([i for a in preds for i in a])
    assert len(preds) == len(targets)
    for y, t in zip(preds, targets):
        assert len(y) == len(t), "{}: len(y) = {} != len(t) = {}".format(name, len(y), len(t))
        for yy, tt in zip(y, t):
            if yy == tt: corrects += 1

    print "{} accuracy: {}".format(name, corrects / float(totals))

parser = argparse.ArgumentParser("evaluate lstm tagger")
parser.add_argument("model",     help="model_iter_****")
parser.add_argument("defs_dir",  help="directory which contains tagger_defs.txt")
parser.add_argument("test_data", help="test data json file")
parser.add_argument("--save", default=None, help="save result in .npz")
args = parser.parse_args()



tagger = eval(json.load(open(args.defs_dir + "/tagger_defs.txt"))["model"])(args.defs_dir)
chainer.serializers.load_npz(args.model, tagger)

data = json.load(open(args.test_data), object_pairs_hook=OrderedDict)

sents, target = zip(*[(s.split(" "),
    t.split(" ") if isinstance(t, (str, unicode)) else t) for s, t in data])

res = tagger.predict_doc(sents)

if args.save is not None:
    print "saving to", args.save
    out = {" ".join(s): r for s, r in zip(sents, res)}
    with open(args.save, 'w') as f:
        pickle.dump(out, f)

if isinstance(tagger, Parser):
    pred_tags = [[tagger.cats[i] for i in y.argmax(1)] for _,_,(y,_) in res]
    pred_deps = [y.argmax(1) for _,_,(_,y) in res]
    tags = [target[i][0] for i,_,_ in res]
    deps = [target[i][1] for i,_,_ in res]
    calc_acc(pred_tags, tags, name="tagging")
    calc_acc(pred_deps, deps, name="parsing")
else:
    preds = [[tagger.cats[i] for i in y.argmax(1)] for _,_,y in res]
    tags = target
    tags = [target[i] for i,_,_ in res]
    calc_acc(preds, tags, name="tagging")

# from chainer.training import extension
# from chainer.dataset import convert
# from chainer import reporter as reporter_module
# from chainer import variable
#
# import six
# import copy
#
# class TaggerEvaluator(extension.Extension):
#     def __init__(self, data, target, extractor, device=None):
#
#         self.target = target
#         self.data = json.load(open(data), object_pairs_hook=OrderedDict)
#         self.sents, self.targets = zip(*[(s.split(" "),
#             t.split(" ") if isinstance(t, (str, unicode)) else t) \
#                     for s, t in self.data])
#         self.extractor = extractor
#         self.device = device
#
#     def __call__(self, trainer=None):
#         # set up a reporter
#         reporter = reporter_module.Reporter()
#         with reporter:
#             result = self.evaluate()
#
#         reporter_module.report(result)
#         return result
#
#
#     def converter(self, xs, device):
#         if device is None:
#             return xs
#         elif device < 0:
#             return map(lambda x: map(lambda m: cuda.to_cpu(m), x), xs)
#         else:
#             return map(lambda x: map(
#                 lambda m: cuda.to_gpu(m, device, cuda.Stream.null), x), xs)
#
#     def calc_acc(self, preds, targets):
#         corrects = 0
#         totals = len([i for a in preds for i in a])
#         assert len(preds) == len(targets)
#         for y, t in zip(preds, targets):
#             assert len(y) == len(t)
#             for yy, tt in zip(y, t):
#                 if yy == tt: corrects += 1
#
#         return corrects / float(totals)
#
#     def evaluate(self):
#         res = []
#         for batch in self.sents:
#             batch = [self.extractor.process(x) for x in batch]
#             in_var = self.converter(batch, self.device)
#             tup = zip(*in_var)
#             cat_ys, dep_ys = self.target.forward(*tup)
#             out = zip([F.log_softmax(y[1:-1]).data for y in cat_ys],
#                     [F.log_softmax(y[1:-1, :-1]).data for y in dep_ys])
#             res.append(out)
#
#         if isinstance(res[0][2], (list,tuple)):
#             pred_tags = [[tagger.cats[i] for i in y.argmax(1)] for y,_ in res]
#             pred_deps = [y.argmax(1) for _,y in res]
#             tags = [t for t, _ in self.targets]
#             deps = [t for _, t in self.targets]
#             res = {"tagging": calc_acc(pred_tags, tags),
#                     "dependency": calc_acc(pred_deps, deps)}
#             print res
#             return res
#
#         else:
#             preds = [[tagger.cats[i] for i in y.argmax(1)] for y in res]
#             tags = self.targets
#             return {"tagging": calc_acc(pred, tags)}
