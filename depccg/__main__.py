
import argparse
import sys
import logging
import json

from .parser import EnglishCCGParser, JapaneseCCGParser
from .printer import to_mathml, to_prolog, to_xml, Token
from .download import download, load_model_directory
from .utils import read_partial_tree, read_weights

Parsers = {'en': EnglishCCGParser, 'ja': JapaneseCCGParser}


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)

    if args.weights is not None:
        probs, tag_list = read_weights(args.weights)
    else:
        probs, tag_list = None, None

    if args.root_cats is not None:
        args.root_cats = args.root_cats.split(',')

    load_tagger = True  # args.input_format != 'json'
    model = args.model or load_model_directory(args.lang)
    parser = Parsers[args.lang].from_dir(model,
                                         load_tagger=load_tagger,
                                         nbest=args.nbest,
                                         possible_root_cats=args.root_cats,
                                         pruning_size=args.pruning_size,
                                         beta=args.beta,
                                         use_beta=not args.disable_beta,
                                         use_seen_rules=not args.disable_seen_rules,
                                         use_category_dict=not args.disable_category_dictionary)

    fin = sys.stdin if args.input is None else open(args.input)

    tagged_doc = None
    if args.input_format == 'POSandNERtagged':
        tagged_doc = [[Token.from_piped(token) for token in sent.strip().split(' ')] for sent in fin]
        doc = [' '.join(token.word for token in sent) for sent in tagged_doc]
        res = parser.parse_doc(doc,
                               probs=probs,
                               tag_list=tag_list,
                               batchsize=args.batchsize)
    elif args.input_format == 'json':
        res = parser.parse_json([json.loads(line) for line in fin])
    elif args.input_format == 'partial':
        doc, constraints = zip(*[read_partial_tree(l.strip()) for l in fin])
        res = parser.parse_doc(doc,
                               probs=probs,
                               tag_list=tag_list,
                               batchsize=args.batchsize,
                               constraints=constraints)
    else:
        assert args.format not in ['xml', 'prolog'], \
            'XML and Prolog output format is supported only with --input-format POSandNERtagged.'
        doc = [l.strip() for l in fin]
        res = parser.parse_doc(doc,
                               probs=probs,
                               tag_list=tag_list,
                               batchsize=args.batchsize)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A* CCG parser')
    parser.add_argument('lang',
                        help='language',
                        choices=['en', 'ja'])
    parser.add_argument('-m', '--model',
                        help='path to model directory')
    parser.add_argument('-i', '--input',
                        default=None,
                        help='a file with tokenized sentences in each line')
    parser.add_argument('-w', '--weights',
                        default=None,
                        help='a file that contains weights (p_tag, p_dep)')
    parser.add_argument('--batchsize',
                        type=int,
                        default=32,
                        help='batchsize in supertagger')
    parser.add_argument('--nbest',
                        type=int,
                        default=1,
                        help='output N best parses')
    parser.add_argument('-I', '--input-format',
                        default='raw',
                        choices=['raw', 'POSandNERtagged', 'json', 'partial'],
                        help='input format')
    parser.add_argument('-f', '--format',
                        default='auto',
                        choices=['auto', 'deriv', 'xml', 'ja', 'conll', 'html', 'prolog'],
                        help='output format')
    parser.add_argument('--root-cats',
                        default=None,
                        help='allow only these categories to be at the root of a tree. If None, use default setting.')
    parser.add_argument('--beta',
                        default=0.00001,
                        type=float,
                        help='parameter used to filter categories with lower probabilities')
    parser.add_argument('--pruning-size',
                        default=50,
                        type=int,
                        help='use only the most probable supertags per word')
    parser.add_argument('--disable-beta',
                        action='store_true',
                        help='disable the use of the beta value')
    parser.add_argument('--disable-category-dictionary',
                        action='store_true',
                        help='disable a category dictionary that maps words to most likely supertags')
    parser.add_argument('--disable-seen-rules',
                        action='store_true',
                        help='')
    parser.add_argument('--verbose',
                        action='store_true')
    parser.set_defaults(func=main)

    subparsers = parser.add_subparsers()
    download_parser = subparsers.add_parser('download')
    download_parser.add_argument('lang', choices=['en', 'ja'])
    download_parser.set_defaults(func=lambda args: download(args.lang))
    args = parser.parse_args()
    args.func(args)



