
from astar import AStarParser
import sys
import argparse
from time import time

parser = argparse.ArgumentParser(
            "parse sentence")
parser.add_argument("model",
        help="model to use")

# parser.add_argument("sents",
#         type=argparse.FileType('r'),
#         default=sys.stdin,
#         nargs='?',
#         help="sentences")

args = parser.parse_args()

a_parser = AStarParser(args.model)
fr = time()
for inp in open("test.txt"):
    res = a_parser.parse(inp.strip())
    res.show_derivation()
to = time()

print "time elapsed: ", to - fr
