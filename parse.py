
from astar import AStarParser
import sys
import argparse
from tqdm import tqdm
from time import time

parser = argparse.ArgumentParser(
            "parse sentence")
parser.add_argument("model",
        help="model to use")

parser.add_argument("sents",
        type=argparse.FileType('r'),
        default=sys.stdin,
        nargs='?',
        help="sentences")

args = parser.parse_args()

a_parser = AStarParser(args.model)
fr = time()
# sents = [inp.strip().split(" ") for inp in args.sents]
# for res in a_parser.parse_doc(sents):
#     print res
sents = a_parser.parse_doc([sent.strip().split(" ") for sent in args.sents])
with open("result.txt", "w") as f:
    for i, res in enumerate([sent for sent in sents]):
        # print res
        continue
print len(sents)
to = time()

print "time elapsed: ", to - fr


