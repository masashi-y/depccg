
from astar import AStarParser
import sys
import re
import multiprocessing
import os
import argparse
from pathlib import Path
from time import time
from ccgbank import AutoReader, get_leaves, AutoLineReader


re_subset = {"train": re.compile(r"wsj_(0[2-9]|1[0-9]|20|21)..\.auto"),
            "test": re.compile(r"wsj_23..\.auto"),
            "val": re.compile(r"wsj_00..\.auto"),
            "all": re.compile(r"wsj_....\.auto") }

# def _worker(autofile):
#     res = []
#     for tree in AutoReader(autofile).readall(suppress_error=True):
#         sent = " ".join(map(lambda x: x.word, get_leaves(tree)))
#         res.append(sent)
#     return res


def ccgbank2raw_text(args):
    matcher = re_subset[args.subset]
    autos = []
    for root, dirs, files in os.walk(args.path):
        for autofile in files:
            if matcher.match(autofile):
                autos.append(os.path.join(root, autofile))
    with open(args.out_raw, "w") as out_raw:
        with open(args.out_tree, "w") as out_tree:
            for auto in sorted(autos):
                for tree in AutoReader(auto). \
                        readall(suppress_error=True):
                    sent = " ".join(map(lambda x: x.word, get_leaves(tree)))
                    out_raw.write(sent + "\n")
                    out_tree.write(str(tree) + "\n")


def read_generated(filepath, trees):
    leaves = map(get_leaves, trees)
    res = [set()]
    io = open(filepath)
    for _ in range(3):
        io.readline()
    for line in io:
        line = line.strip()
        if len(line) == 0:
            res.append(set())
            continue
        items = line.split(" ")
        idx = items[0].find("_")
        predicate = int(items[0].split("_")[-1])
        try:
            cat = str(leaves[len(res)-1][predicate-1].cat)
        except:
            print predicate
            print tree
            print line
            raise Exception()
        argid = int(items[2])
        argument = int(items[3].split("_")[-1])
        res[-1].add((cat, predicate, argid, argument))
    return res[:-1]



# def read_generated(filepath):
#     res = [set()]
#     io = open(filepath)
#     for _ in range(3):
#         io.readline()
#     for line in io:
#         line = line.strip()
#         if len(line) == 0:
#             res.append(set())
#             continue
#         items = line.split(" ")
#         idx = items[0].find("_")
#         predicate = int(items[0].split("_")[-1])
#         argid = int(items[2])
#         argument = int(items[3].split("_")[-1])
#         res[-1].add((predicate, argid, argument))
#     return res[:-1]



def predict_auto(args):
    outdir = Path(args.out)
    candc = Path(args.candc)
    scripts = (candc / "src" / "scripts" / "ccg")
    catsdir = (candc / "src" / "data" / "ccg" / "cats")
    markedup = (catsdir / "markedup")
    pred = (outdir / "auto_predicts.txt")
    pred_processed = (outdir / "auto_predicts_processed.txt")
    deps = (outdir / "auto_predicts.deps")

    print "parsing sentences in {}".format(args.testfile)
    astarp = AStarParser(args.model)
    with open(str(pred), "w") as f:
        sents = [sent.strip().split(" ") for sent in open(args.testfile)]
        res = astarp.parse_doc(sents)
        for sent in res:
            f.write(str(sent) + "\n")
    cmd1 = "cat {0} | {1}/convert_auto | sed -f {1}/convert_brackets > {2}" \
        .format(pred, scripts, pred_processed)
    print cmd1
    os.system(cmd1)

    cmd2 = "{0}/generate -j {1} {2} {3} > {4}" \
            .format((candc / "bin"), catsdir, markedup, pred_processed, deps)
    print cmd2
    os.system(cmd2)
    return read_generated(str(deps), [AutoLineReader(sent).parse() for sent in res])


def evaluate(args):
    preddeps = predict_auto(args)
    gold_trees = AutoReader(args.gold_tree).readall()
    golddeps = read_generated(args.gold_raw, gold_trees)
    # preddeps = read_generated(args.pred)
    # golddeps = read_generated(args.gold)
    assert len(golddeps) == len(preddeps)
    predlen, goldlen, correct = 0, 0, 0
    for gold, pred in zip(golddeps, preddeps):
        predlen += len(pred)
        goldlen += len(gold)
        correct += (len(pred.intersection(gold)))
    precision = float(correct) / float(predlen)
    recall = float(correct) / float(goldlen)
    f1 = (2 * precision * recall) / (precision + recall)
    print "precision: {}, recall: {}, F1: {}".format(precision, recall, f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                "CCG parser's evaulator")
    subparsers = parser.add_subparsers()

# Creating training data from CCGBank AUTO files
    parser_c = subparsers.add_parser(
            "create", help="create test data")
    parser_c.add_argument("path",
            help="path to AUTO directory")
    parser_c.add_argument("out_raw",
            help="raw text output file path")
    parser_c.add_argument("out_tree",
            help="auto tree output file path")
    parser_c.add_argument("--subset",
            choices=["train", "val", "test", "all"],
            default="train",
            help="train: 02-21, val: 00, test: 23, (default: train)")
    parser_c.set_defaults(func=ccgbank2raw_text)

    parser_e = subparsers.add_parser(
            "parse-eval", help="evaluate on test data")
    parser_e.add_argument("gold_raw",
            help="gold deps file path")
    parser_e.add_argument("gold_tree",
            help="gold auto file path")
    parser_e.add_argument("out",
            help="output temporary directory path")
    parser_e.add_argument("candc",
            help="path to candc parser")
    parser_e.add_argument("model",
            help="supertagger model directory")
    parser_e.add_argument("testfile",
            help="path to a file with test sentences")
    parser_e.set_defaults(func=evaluate)

    # parser_et = subparsers.add_parser(
    #         "eval", help="evaluate on test data")
    # parser_et.add_argument("gold",
    #         help="gold deps file path")
    # parser_et.add_argument("pred",
    #         help="pred deps file path")
    # parser_et.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)

