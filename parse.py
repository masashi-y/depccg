
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
# sents = [inp.strip().split(" ") for inp in open("test.txt")]
# for res in a_parser.parse_doc(sents):
#     print res
sents = [sent.strip() for sent in args.sents]
with open("result.txt", "w") as f:
    for inp in tqdm(sents):
        # print inp
        res = a_parser.parse(inp)
        # res.show_derivation()
        f.write(str(res) + "\n")
        # print res
to = time()

print "time elapsed: ", to - fr


