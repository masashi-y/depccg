
import argparse
import sys
import logging

from .parser import EnglishCCGParser, JapaneseCCGParser
from .printer import to_mathml, to_prolog, to_xml, Token


logging.basicConfig(level=logging.DEBUG)

Parsers = {'en': EnglishCCGParser, 'ja': JapaneseCCGParser}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A* CCG parser')
    parser.add_argument('lang',
                        help='language',
                        choices=['en', 'ja'])
    parser.add_argument('--model',
                        help='path to model directory')
    parser.add_argument('--input',
                        default=None,
                        help='a file with tokenized sentences in each line')
    parser.add_argument('--batchsize',
                        type=int,
                        default=32,
                        help='batchsize in supertagger')
    parser.add_argument('--nbest',
                        type=int,
                        default=1,
                        help='output N best parses')
    parser.add_argument('--input-format',
                        default='raw',
                        choices=['raw', 'POSandNERtagged', 'json'],
                        help='input format')
    parser.add_argument('--format',
                        default='auto',
                        choices=['auto', 'deriv', 'xml', 'ja', 'conll', 'html', 'prolog'],
                        help='output format')
    parser.add_argument('--root-cats',
                        default=None,
                        help='allow only these categories to be at the root of a tree. If None, use default setting.')
    parser.add_argument('--verbose',
                        action='store_true')
    args = parser.parse_args()

    fin = sys.stdin if args.input is None else open(args.input)

    if args.input_format == 'POSandNERtagged':
        tagged_doc = [[Token.from_piped(token) for token in sent.strip().split(' ')] for sent in fin]
        doc = [' '.join(token.word for token in sent) for sent in tagged_doc]
    else:
        assert args.format not in ['xml', 'prolog'], \
                'XML and Prolog output format is supported only with --input-format POSandNERtagged.'
        doc = [l.strip() for l in fin]
        tagged_doc = None

    if args.root_cats is not None:
        args.root_cats = args.root_cats.split(',')

    load_tagger = args.model is not None

    parser = Parsers[args.lang].from_dir(args.model,
                                         load_tagger=load_tagger,
                                         nbest=args.nbest,
                                         possible_root_cats=args.root_cats,
                                         loglevel=1 if args.verbose else 3)
    res = parser.parse_doc(doc, batchsize=args.batchsize)

    if args.format == 'xml':
        print(to_xml(res, tagged_doc))
    elif args.format == 'prolog':
        print(to_prolog(res, tagged_doc))
    elif args.format == 'html':
        print(to_mathml(res))
    elif args.format == 'conll':
        for i, parsed in enumerate(res):
            for tree, prob in parsed:
                print(f'# ID={i}\n# log probability={prob:.4e}\n{tree.conll}')
    else:  # 'auto', 'deriv', 'ja'
        for i, parsed in enumerate(res, 1):
            for tree, _ in parsed:
                print(f'ID={i}')
                print(getattr(tree, args.format))


