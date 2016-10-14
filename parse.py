
from astar import AStarParser
import sys
import argparse

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
for inp in args.sents:
    res, _ = a_parser.parse(inp.strip())
    print res

