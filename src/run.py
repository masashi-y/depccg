
from __future__ import print_function, unicode_literals
import argparse
import codecs
import sys

if sys.version_info.major == 2:
    sys.stdin = codecs.getreader('utf-8')(sys.stdin)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

from depccg import PyAStarParser, PyJaAStarParser, to_mathml

Parsers = {"en": PyAStarParser, "ja": PyJaAStarParser}

class Token:
    def __init__(self, word, lemma, pos, chunk, entity):
        self.word   = word
        self.lemma  = lemma
        self.pos    = pos
        self.chunk  = chunk
        self.entity = entity

    @staticmethod
    def from_piped(string):
        # WORD|POS|NER or WORD|LEMMA|POS|NER
        # or WORD|LEMMA|POS|NER|CHUCK
        items = string.split("|")
        if len(items) == 5:
            w, l, p, n, c = items
            return Token(w, l, p, c, n)
        elif len(items) == 4:
            w, l, p, n = items
            return Token(w, l, p, "XX", n)
        else:
            w, p, n = items
            return Token(w, "XX", p, "XX", n)

def to_xml(trees, tagged_doc, file=sys.stdout):
    print("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    print("<?xml-stylesheet type=\"text/xsl\" href=\"candc.xml\"?>")
    print("<candc>")
    for i, (tree, tagged) in enumerate(zip(trees, tagged_doc), 1):
        for j, (t, _) in enumerate(tree, 1):
            print("<ccg sentence=\"{}\" id=\"{}\">".format(i, j))
            print(t.xml.format(*tagged))
            print("</ccg>")
    print("</candc>")

def to_prolog(trees, tagged_doc, file=sys.stdout):
    print(":- op(601, xfx, (/)).")
    print(":- op(601, xfx, (\)).")
    print(":- multifile ccg/2, id/2.")
    print(":- discontiguous ccg/2, id/2.\n")
    for i, (tree, tagged) in enumerate(zip(trees, tagged_doc), 1):
        for tok in tagged:
            if "'" in tok.lemma:
                tok.lemma = tok.lemma.replace("'", "\\'")
        for t, _ in tree:
            print(t.prolog.format(i, *tagged))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A* CCG parser")
    parser.add_argument("model", help="model directory")
    parser.add_argument("lang", help="language", choices=["en", "ja"])
    parser.add_argument("--input", default=None,
            help="a file with tokenized sentences in each line")
    parser.add_argument("--batchsize", type=int, default=32,
            help="batchsize in supertagger")
    parser.add_argument("--nbest", type=int, default=1,
            help="output N best parses")
    parser.add_argument("--input-format", default="raw",
            choices=["raw", "POSandNERtagged"],
            help="input format")
    parser.add_argument("--format", default="auto",
            choices=["auto", "deriv", "xml", "ja", "conll", "html", "prolog"],
            help="output format")
    parser.add_argument("--root-cats", default=None,
            help="allow only these categories to be at the root of a tree. If None, use default setting.")

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    fin = sys.stdin if args.input is None else codecs.open(args.input, encoding="utf-8")
    if args.input_format == "POSandNERtagged":
        tagged_doc = [[Token.from_piped(token) for token in sent.strip().split(" ")]for sent in fin]
        doc = [" ".join([token.word for token in sent]) for sent in tagged_doc]
    else:
        assert args.format not in ["xml", "prolog"], \
                "XML and Prolog output format is supported only with --input-format POSandNERtagged."
        doc = [l.strip() for l in fin]

    if args.root_cats is not None:
        args.root_cats = args.root_cats.split(",")

    parser = Parsers[args.lang](args.model,
                               batchsize=args.batchsize,
                               nbest=args.nbest,
                               possible_root_cats=args.root_cats,
                               loglevel=1 if args.verbose else 3)
    res = parser.parse_doc(doc)


    if args.format == "xml":
        to_xml(res, tagged_doc)
    elif args.format == "prolog":
        to_prolog(res, tagged_doc)
    elif args.format == "html":
        to_mathml(res, file=sys.stdout)
    elif args.format == "conll":
        for i, parsed in enumerate(res):
            for tree, prob in parsed:
                print("# ID={}\n# log probability={:.4e}\n{}".format(
                        i, prob, tree.conll))
    else: # "auto", "deriv", "ja"
        for i, parsed in enumerate(res, 1):
            for tree, _ in parsed:
                print("ID={}".format(i))
                print(getattr(tree, args.format))


