
import argparse
import sys
import logging
import json

from .parser import JapaneseCCGParser
from .printer import print_
from depccg.token import japanese_annotator, annotate_XX
from depccg.download import MODEL_DIRECTORY
from .combinator import (headfinal_combinator,
                         ja_forward_application,
                         ja_backward_application,
                         ja_generalized_forward_composition0,
                         ja_generalized_backward_composition0,
                         ja_generalized_backward_composition1,
                         ja_generalized_backward_composition2,
                         ja_generalized_backward_composition3,
                         ja_generalized_forward_composition0,
                         ja_generalized_forward_composition1,
                         ja_generalized_forward_composition2)



def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.CRITICAL if args.silent else logging.INFO)

    binary_rules = [
        headfinal_combinator(ja_forward_application()),
        headfinal_combinator(ja_backward_application()),
        headfinal_combinator(ja_generalized_forward_composition0('/', '/', '/', '>B')),
        headfinal_combinator(ja_generalized_backward_composition0('\\', '\\', '\\', '<B1')),
        headfinal_combinator(ja_generalized_backward_composition1('\\', '\\', '\\', '<B2')),
        headfinal_combinator(ja_generalized_backward_composition2('\\', '\\', '\\', '<B3')),
        headfinal_combinator(ja_generalized_backward_composition3('\\', '\\', '\\', '<B4')),
    ]

    annotate_fun = japanese_annotator['janome'] if args.tokenize else annotate_XX

    kwargs = dict(
        unary_penalty=args.unary_penalty,
        nbest=args.nbest,
        binary_rules=binary_rules,
        possible_root_cats=["S[m]", "FRAG", "INTJP", "CP[f]", "CP[q]", "S[imp]", "CP[t]", "LST", "CP-EXL"],
        pruning_size=args.pruning_size,
        beta=args.beta,
        use_beta=not args.disable_beta,
        use_seen_rules=False,
        use_category_dict=False,
        max_length=args.max_length,
        max_steps=args.max_steps,
        gpu=args.gpu
    )

    config = MODEL_DIRECTORY / 'config_abc.json'
    parser = JapaneseCCGParser.from_json(config, args.model, **kwargs)

    fin = sys.stdin if args.input is None else open(args.input)

    doc = [l.strip() for l in fin]
    doc = [sentence for sentence in doc if len(sentence) > 0]
    tagged_doc = annotate_fun([[word for word in sent.split(' ')] for sent in doc],
                              tokenize=args.tokenize)
    if args.tokenize:
        tagged_doc, doc = tagged_doc
    res = parser.parse_doc(doc, batchsize=args.batchsize)

    print_(res, tagged_doc, format=args.format, lang='ja')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A* CCG parser')
    parser.set_defaults(func=lambda _: parser.print_help())

    parser.add_argument('-m',
                        '--model',
                        help='path to model directory')
    parser.add_argument('-i',
                        '--input',
                        default=None,
                        help='a file with tokenized sentences in each line')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='specify gpu id')
    parser.add_argument('--batchsize',
                        type=int,
                        default=32,
                        help='batchsize in supertagger')
    parser.add_argument('--nbest',
                        type=int,
                        default=1,
                        help='output N best parses')
    parser.add_argument('--unary-penalty',
                        default=0.1,
                        type=float,
                        help='penalty to use a unary rule')
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
    parser.add_argument('--max-length',
                        default=250,
                        type=int,
                        help='give up parsing a sentence that contains more words than this value')
    parser.add_argument('--max-steps',
                        default=10000000,
                        type=int,
                        help='give up parsing when the number of times of popping agenda items exceeds this value')
    parser.add_argument('-f',
                        '--format',
                        default='auto',
                        choices=['auto', 'deriv', 'xml', 'conll', 'html', 'prolog', 'jigg_xml', 'ptb', 'json'],
                        help='output format')
    parser.add_argument('--tokenize',
                        action='store_true',
                        help='tokenize input sentences')
    parser.add_argument('--silent',
                        action='store_true')

    main(parser.parse_args())



