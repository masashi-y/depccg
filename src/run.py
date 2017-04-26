
from __future__ import print_function, unicode_literals
import argparse
import fileinput
import codecs
import sys

if sys.version_info.major == 2:
    sys.stdin = codecs.getreader('utf-8')(sys.stdin)

from depccg import PyAStarParser, PyJaAStarParser

Parsers = {"en": PyAStarParser, "ja": PyJaAStarParser}

def to_xml(trees):
    print("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    print("<?xml-stylesheet type=\"text/xsl\" href=\"candc.xml\"?>")
    print("<candc>")
    for tree in trees:
        print("<ccg>\n{}\n</ccg>".format(tree.xml))
    print("</candc>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A* CCG parser")
    parser.add_argument("model", help="model directory")
    parser.add_argument("lang", help="language", choices=["en", "ja"])
    parser.add_argument("--input", default=None,
            help="a file with tokenized sentences in each line")
    parser.add_argument("--batchsize", type=int, default=32,
            help="batchsize in supertagger")
    parser.add_argument("--format", default="auto",
            choices=["auto", "deriv", "xml", "ja", "conll"],
            help="output format")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    fin = sys.stdin if args.input is None else codecs.open(args.input, encoding="utf-8")
    doc = [l.strip() for l in fin]

    parser = Parsers[args.lang](args.model,
                               batchsize=args.batchsize,
                               loglevel=1 if args.verbose else 3)


    res = parser.parse_doc(doc)
    if args.format == "xml":
        to_xml(res)
    else:
        for i, r in enumerate(res):
            print("ID={}".format(i))
            r.suppress_feat = True
            print(getattr(r, args.format))
